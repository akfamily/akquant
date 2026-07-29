# -*- coding: utf-8 -*-
"""引擎每 bar 固定开销基准,带噪声闸门.

用法:
    # 基线(改动前)
    uv run python scripts/bench_engine_perbar.py --profile wide \
        --json base_wide.json
    # 候选(改动后)
    uv run python scripts/bench_engine_perbar.py --profile wide \
        --baseline base_wide.json

设计要点:
1. 固定 bar 总数、只变宇宙宽度,使不同 profile 可比.
2. bars/sec 取 min 而非 median.
3. 噪声闸门: |效应| < 噪声带 时判定 INCONCLUSIVE, 不得声称收益.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

TOTAL_BARS = 180_000
TZ = "Asia/Shanghai"

PROFILES: Dict[str, Dict[str, int]] = {
    # O(N) 项主导(实测占 92.8%): PR-1 的主战场
    "wide": {"symbols": 4500, "days": 40},
    # 常数项主导(Python 侧实测占 27.4%): PR-2 的主战场
    "narrow": {"symbols": 300, "days": 600},
    # 覆盖 O(P) 路径(账户指标/frozen_cash/entry prices)
    "positions": {"symbols": 4500, "days": 40},
}


@dataclass
class RunStats:
    """单次基准运行的统计量."""

    profile: str
    bars: int
    rounds: List[float] = field(default_factory=list)

    @property
    def min(self) -> float:
        """各轮次最快耗时(秒)."""
        return min(self.rounds)

    @property
    def median(self) -> float:
        """各轮次耗时中位数(秒)."""
        return statistics.median(self.rounds)

    @property
    def max(self) -> float:
        """各轮次最慢耗时(秒)."""
        return max(self.rounds)

    @property
    def spread(self) -> float:
        """离散度 = max/min - 1."""
        return self.max / self.min - 1.0

    @property
    def bars_per_sec(self) -> float:
        """基于最快耗时折算的吞吐(bars/秒)."""
        return self.bars / self.min

    @property
    def us_per_bar(self) -> float:
        """基于最快耗时折算的单 bar 均摊耗时(微秒)."""
        return self.min / self.bars * 1e6

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可写入 JSON 的字典."""
        return {
            "profile": self.profile,
            "bars": self.bars,
            "rounds": self.rounds,
            "min": self.min,
            "median": self.median,
            "max": self.max,
            "spread": self.spread,
            "bars_per_sec": self.bars_per_sec,
            "us_per_bar": self.us_per_bar,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunStats":
        """从 JSON 反序列化的字典重建 RunStats."""
        return cls(
            profile=str(data["profile"]),
            bars=int(data["bars"]),
            rounds=[float(x) for x in data["rounds"]],
        )


def verdict(base: RunStats, cand: RunStats) -> Tuple[str, float, float]:
    """比较两次运行, 在噪声带内拒绝声称收益.

    噪声带 = max(两侧离散度); 效应 = min_base/min_cand - 1.
    效应为正表示候选更快.
    """
    noise_band = max(base.spread, cand.spread)
    effect = base.min / cand.min - 1.0
    if abs(effect) < noise_band:
        return "INCONCLUSIVE", effect, noise_band
    return ("IMPROVED" if effect > 0 else "REGRESSED"), effect, noise_band


def _make_data(n_sym: int, n_day: int) -> Any:
    """合成 OHLCV, 固定种子以保证跨运行可比."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    days = pd.date_range("2020-01-02", periods=n_day, freq="B", tz=TZ)
    n = n_sym * n_day
    close = (
        100.0 * np.exp(rng.normal(0, 0.01, n).cumsum().reshape(n_day, n_sym) * 0.1)
    ).ravel()
    return pd.DataFrame(
        {
            "date": np.repeat(days.values, n_sym),
            "symbol": np.tile([f"S{i:05d}" for i in range(n_sym)], n_day),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 1e6),
        }
    )


def _build_strategy_class(hold_positions: int) -> Any:
    """构造只覆写 on_cross_section 的基准策略类.

    hold_positions == 0: 全程不下单, 测纯每 bar 固定开销.
    hold_positions > 0 : 首个截面一次性建仓并全程持有(此后不调仓),
                         使 O(P) 路径进入热路径而不引入撮合噪声.
    """
    import akquant as aq

    class BenchStrategy(aq.Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.cs_calls = 0

        def on_cross_section(self, trading_date: object, timestamp: int) -> None:
            self.cs_calls += 1
            if hold_positions <= 0 or self.cs_calls != 1:
                return
            weight = 0.5 / hold_positions
            for i in range(hold_positions):
                self.order_target_percent(target_percent=weight, symbol=f"S{i:05d}")

    return BenchStrategy


def run_profile(profile: str, rounds: int, positions: int) -> RunStats:
    """运行指定 profile 若干轮并返回耗时统计."""
    import akquant as aq

    cfg = PROFILES[profile]
    n_sym, n_day = cfg["symbols"], cfg["days"]
    hold = positions if profile == "positions" else 0
    data = _make_data(n_sym, n_day)
    symbols = sorted(data["symbol"].unique().tolist())
    strategy_cls = _build_strategy_class(hold)
    stats = RunStats(profile=profile, bars=len(data))
    for _ in range(rounds):
        gc.collect()
        start = time.perf_counter()
        aq.run_backtest(
            data=data,
            strategy=strategy_cls,
            symbols=symbols,
            initial_cash=1e8,
            commission_rate=0.0003,
            show_progress=False,
        )
        stats.rounds.append(time.perf_counter() - start)
    return stats


def main(argv: List[str] | None = None) -> int:
    """CLI 入口: 运行基准, 可选写 JSON / 与基线比较判定."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="wide", help="负载档位"
    )
    parser.add_argument("--rounds", type=int, default=5, help="轮次(取 min)")
    parser.add_argument(
        "--positions", type=int, default=500, help="positions 档的持仓数"
    )
    parser.add_argument("--json", type=Path, help="把本次统计写入 JSON")
    parser.add_argument("--baseline", type=Path, help="与该 JSON 基线比较")
    args = parser.parse_args(argv)

    stats = run_profile(args.profile, args.rounds, args.positions)

    print(f"profile={stats.profile} bars={stats.bars} rounds={args.rounds}")
    print(
        f"  min={stats.min:.3f}s median={stats.median:.3f}s max={stats.max:.3f}s"
        f"  离散度={stats.spread * 100:.1f}%"
    )
    print(f"  {stats.bars_per_sec:.1f} bars/sec   {stats.us_per_bar:.1f} us/bar")

    if args.json:
        args.json.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
        print(f"  已写入 {args.json}")

    if args.baseline:
        base = RunStats.from_dict(json.loads(args.baseline.read_text(encoding="utf-8")))
        if base.profile != stats.profile:
            print(
                f"错误: 基线 profile={base.profile} 与本次 {stats.profile} 不一致",
                file=sys.stderr,
            )
            return 2
        name, effect, noise = verdict(base, stats)
        print(
            f"\n基线 {base.bars_per_sec:.1f} -> 候选 {stats.bars_per_sec:.1f} bars/sec"
        )
        print(f"  噪声带 {noise * 100:.1f}%")
        if name == "INCONCLUSIVE":
            print(
                f"  判定 INCONCLUSIVE: 效应 {effect * 100:+.1f}% 落在噪声带内,"
                f"不得声称收益"
            )
        else:
            print(f"  判定 {name}: 效应 {effect * 100:+.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
