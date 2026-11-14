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
    # 以下四档走 on_bar(前三档走 on_cross_section), 对应 2026-11 与 backtrader
    # 对比时拆出的四类开销; bars_per_day > 1 时生成日内分钟线。
    # self.I() 预计算指标的逐 bar 取值
    "indicators": {"symbols": 1, "days": 750, "bars_per_day": 240},
    # 100 个持仓只持有不交易: 每个 bar 事件按持仓数重算的部分
    "holdings": {"symbols": 100, "days": 1800},
    # 单标的只买不卖: 每根 bar 遍历未平仓批次的部分(批次数随时间线性增长)
    "accumulate": {"symbols": 1, "days": 750, "bars_per_day": 240},
    # 单标的反复开平: 订单/成交路径, 看单笔成本是否随累计笔数上升
    "churn": {"symbols": 1, "days": 750, "bars_per_day": 240},
}

#: 会下单的档位: 每轮跑完核对订单确实成交(见 `_require_fills`)。
FILL_PROFILES = frozenset({"positions", "holdings", "accumulate", "churn"})

#: accumulate / churn 档每隔多少根 bar 下一笔单。
ORDER_EVERY = 25


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
    if abs(effect) <= noise_band:
        return "INCONCLUSIVE", effect, noise_band
    return ("IMPROVED" if effect > 0 else "REGRESSED"), effect, noise_band


def _make_data(n_sym: int, n_day: int, bars_per_day: int = 1) -> Any:
    """合成 OHLCV, 固定种子以保证跨运行可比.

    bars_per_day == 1 为日线; 否则每个交易日从 09:31 起连续 bars_per_day 根分钟线
    (上午 120 根, 其余顺延到 13:01 之后), 时间戳落在 A 股交易时段内。
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    # 先按本地墙上时间造, 最后再挂时区: `带时区索引.values` 会变成不带时区的 UTC 时刻
    # (00:00 上海 → 前一天 16:00), 落在闭市时段, 订单永远不成交。
    days = pd.date_range("2020-01-02", periods=n_day, freq="B")
    if bars_per_day > 1:
        offsets = np.arange(bars_per_day)
        minutes = np.where(
            offsets < 120, 9 * 60 + 31 + offsets, 13 * 60 + 1 + offsets - 120
        )
        days = pd.DatetimeIndex(
            np.repeat(days.values, bars_per_day)
            + np.tile(minutes, n_day).astype("timedelta64[m]")
        )
        n_day = len(days)
    n = n_sym * n_day
    close = (
        100.0 * np.exp(rng.normal(0, 0.01, n).cumsum().reshape(n_day, n_sym) * 0.1)
    ).ravel()
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(np.repeat(days.values, n_sym)).tz_localize(TZ),
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


def _build_on_bar_strategy_class(profile: str) -> Any:
    """构造走 on_bar 的基准策略类(indicators / holdings / accumulate / churn)."""
    import akquant as aq

    if profile == "indicators":

        class IndicatorStrategy(aq.Strategy):
            def on_start(self) -> None:
                self.fast = self.I(
                    aq.PrecomputedIndicator(
                        "fast", lambda df: df["close"].rolling(10).mean()
                    ),
                    name="fast",
                )
                self.slow = self.I(
                    aq.PrecomputedIndicator(
                        "slow", lambda df: df["close"].rolling(30).mean()
                    ),
                    name="slow",
                )
                self.above = 0

            def on_bar(self, bar: Any) -> None:
                fast, slow = self.fast[0], self.slow[0]
                if fast is not None and slow is not None and fast > slow:
                    self.above += 1

        return IndicatorStrategy

    if profile == "holdings":

        class HoldingStrategy(aq.Strategy):
            def on_start(self) -> None:
                self.bought: set[str] = set()

            def on_bar(self, bar: Any) -> None:
                if bar.symbol not in self.bought:
                    self.bought.add(bar.symbol)
                    self.buy(symbol=bar.symbol, quantity=100)

        return HoldingStrategy

    sell_back = profile == "churn"

    class OrderStrategy(aq.Strategy):
        def on_start(self) -> None:
            self.seen = 0

        def on_bar(self, bar: Any) -> None:
            self.seen += 1
            phase = self.seen % (2 * ORDER_EVERY)
            if phase == 0 or (phase == ORDER_EVERY and not sell_back):
                self.buy(symbol=bar.symbol, quantity=100)
            elif phase == ORDER_EVERY:
                self.sell(symbol=bar.symbol, quantity=100)

    return OrderStrategy


def run_profile(profile: str, rounds: int, positions: int) -> RunStats:
    """运行指定 profile 若干轮并返回耗时统计."""
    import akquant as aq

    cfg = PROFILES[profile]
    n_sym, n_day = cfg["symbols"], cfg["days"]
    hold = positions if profile == "positions" else 0
    data = _make_data(n_sym, n_day, cfg.get("bars_per_day", 1))
    symbols = sorted(data["symbol"].unique().tolist())
    if profile in ("indicators", "holdings", "accumulate", "churn"):
        strategy_cls = _build_on_bar_strategy_class(profile)
    else:
        strategy_cls = _build_strategy_class(hold)
    stats = RunStats(profile=profile, bars=len(data))
    for _ in range(rounds):
        gc.collect()
        start = time.perf_counter()
        result = aq.run_backtest(
            data=data,
            strategy=strategy_cls,
            symbols=symbols,
            initial_cash=1e8,
            commission_rate=0.0003,
            show_progress=False,
        )
        stats.rounds.append(time.perf_counter() - start)
        _require_fills(profile, result)
    return stats


def _require_fills(profile: str, result: Any) -> None:
    """下单档位必须真的成交, 否则测到的是挂单堆积而不是档位声称的路径."""
    if profile not in FILL_PROFILES:
        return
    orders = result.orders_df
    filled = int((orders["status"] == "filled").sum()) if len(orders) else 0
    if filled * 2 < len(orders) or filled == 0:
        raise RuntimeError(
            f"profile={profile}: {len(orders)} 笔订单只成交 {filled} 笔, "
            "基准数据或策略有问题(常见原因: 行情时间戳落在闭市时段)"
        )


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

    if args.rounds < 2:
        print(
            "错误: --rounds 必须 >= 2。噪声带定义为各轮次耗时的离散度"
            "(max/min - 1),单轮运行没有多个样本可比较,无法计算离散度,"
            "噪声闸门也就形同虚设(spread 恒为 0.0,任何效应都会被判定为"
            "IMPROVED/REGRESSED 而不是 INCONCLUSIVE)。请提高 --rounds"
            "(建议 >= 3)后重试。",
            file=sys.stderr,
        )
        return 2

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
