"""引擎每 bar 固定开销基准脚本（性能优化系列 PR 的统一度量衡）.

目的：用可复现的合成宽宇宙负载（默认 4500 标的 × 120 个交易日 ≈ 54 万根日频 bar）
量化引擎每 bar 固定开销，为"削减每 bar 无条件重建全量快照"系列优化 PR 提供
前后对比基线，避免凭感觉优化。仿 ``scripts/profile_backtest.py``（issue #288）模式。

负载形态对齐真实宽宇宙 cross-section 研究：
T 日收盘出 topk 名单 → ``on_cross_section`` 等权下单 → T+1 next-open 成交。
策略回调只做名单查表与下单，CPU 占比极低，测得的主要是引擎每 bar 固定开销。

用法（在仓库根目录）:

    uv run python scripts/bench_engine_perbar.py
    uv run python scripts/bench_engine_perbar.py --symbols 4500 --days 120
    uv run python scripts/bench_engine_perbar.py --no-orders  # 纯管线开销
    uv run python scripts/bench_engine_perbar.py --selftest   # 小负载自校验

输出：数据生成耗时、各轮墙钟、中位 bars/sec、订单/成交笔数；末行输出机器可读 JSON。
自校验：``--selftest`` 断言输出 schema 完整、两轮成交笔数完全一致（结果确定性），
失败以非零码退出。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Optional

import akquant
import numpy as np
import pandas as pd
from akquant import Strategy, run_backtest
from akquant.log import register_logger

# Windows 控制台默认非 UTF-8，中文会乱码；尽力切到 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass

# 基准关注墙钟与吞吐：预注册 ERROR 级 logger，run_backtest 检测到已配置的
# handler 后不会再回退注册 INFO 级（拒单/延延迟 WARNING 会刷屏干扰计时输出）。
register_logger(console=True, level="ERROR")

TZ = "Asia/Shanghai"
LOOKBACK = 5  # 动量打分窗口（交易日）
SUMMARY_KEYS = (
    "symbols",
    "days",
    "topk",
    "no_orders",
    "total_bars",
    "data_gen_sec",
    "wall_runs_sec",
    "wall_median_sec",
    "bars_per_sec",
    "orders",
    "fills",
    "akquant_version",
)


def _build_data(
    n_symbols: int, n_days: int, seed: int
) -> tuple[dict[str, pd.DataFrame], np.ndarray, pd.DatetimeIndex]:
    """构造 ``n_symbols`` 个标的、各 ``n_days`` 根日频 Bar 的随机游走行情.

    返回 (data, close 矩阵[days × symbols], dates)。数据生成与回测分离计时。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B", tz=TZ)
    # 向量化随机游走，保证价格为正且各标的走势不同。
    rets = rng.normal(loc=0.0003, scale=0.02, size=(n_days, n_symbols))
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.full_like(close, 1_000_000.0)

    data: dict[str, pd.DataFrame] = {}
    for i in range(n_symbols):
        df = pd.DataFrame(
            {
                "date": dates,
                "open": open_[:, i],
                "high": high[:, i],
                "low": low[:, i],
                "close": close[:, i],
                "volume": volume[:, i],
                "symbol": f"S{i:04d}",
            }
        )
        data[f"S{i:04d}"] = df
    return data, close, dates


def _build_schedule(
    close: np.ndarray,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    topk: int,
    lookback: int,
) -> dict[str, list[str]]:
    """预计算每日 topk 名单（动量 = close[t] / close[t-lookback] - 1）.

    key 为 "YYYY-MM-DD" 字符串；不足 lookback 的预热期无名单。
    """
    n_days, n_symbols = close.shape
    schedule: dict[str, list[str]] = {}
    symbol_arr = np.array(symbols)
    for t in range(lookback, n_days):
        scores = close[t] / close[t - lookback] - 1.0
        k = min(topk, n_symbols)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        # 名次稳定（分数降序），保证跨轮确定性。
        top_idx = top_idx[np.argsort(-scores[top_idx], kind="stable")]
        schedule[dates[t].strftime("%Y-%m-%d")] = symbol_arr[top_idx].tolist()
    return schedule


class _TopKRotationStrategy(Strategy):
    """cross-section 每日等权轮动到 topk 名单；``no_orders`` 时完全空转."""

    def __init__(self, schedule: dict[str, list[str]], no_orders: bool) -> None:
        super().__init__()
        self.picks_map = schedule
        self.no_orders = no_orders
        self.rebalances = 0

    def on_cross_section(self, trading_date: Any, timestamp: int) -> None:
        _ = timestamp
        if self.no_orders:
            return
        picks = self.picks_map.get(str(trading_date)[:10])
        if not picks:
            return
        pick_set = set(picks)
        for symbol, qty in list(self.positions.items()):
            if qty != 0 and symbol not in pick_set:
                self.order_target_percent(target_percent=0.0, symbol=symbol)
        # 留 2% 现金余量覆盖手续费，避免满仓下单被保证金检查拒绝。
        weight = 0.98 / len(picks)
        for symbol in picks:
            self.order_target_percent(target_percent=weight, symbol=symbol)
        self.rebalances += 1


def _run_once(
    data: dict[str, pd.DataFrame],
    schedule: dict[str, list[str]],
    no_orders: bool,
) -> tuple[float, int, int]:
    """跑一次回测，返回 (墙钟秒, 订单笔数, 成交笔数)."""
    symbols = sorted(data.keys())
    strat = _TopKRotationStrategy(schedule=schedule, no_orders=no_orders)
    start = time.perf_counter()
    result = run_backtest(
        data=data,
        strategy=strat,
        symbols=symbols,
        initial_cash=100_000_000.0,
        commission_rate=0.0003,
        show_progress=False,
    )
    wall = time.perf_counter() - start
    orders = len(result.orders_df) if hasattr(result, "orders_df") else -1
    fills = len(result.trades_df) if hasattr(result, "trades_df") else -1
    return wall, orders, fills


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    """执行基准并返回 summary 字典."""
    t0 = time.perf_counter()
    data, close, dates = _build_data(args.symbols, args.days, seed=args.seed)
    data_gen = time.perf_counter() - t0
    symbols = sorted(data.keys())
    schedule = _build_schedule(close, dates, symbols, args.topk, LOOKBACK)

    walls: list[float] = []
    orders = fills = 0
    for _ in range(args.runs):
        wall, orders, fills = _run_once(data, schedule, args.no_orders)
        walls.append(wall)

    median_wall = float(np.median(walls))
    total_bars = args.symbols * args.days
    summary: dict[str, Any] = {
        "symbols": args.symbols,
        "days": args.days,
        "topk": args.topk,
        "no_orders": args.no_orders,
        "total_bars": total_bars,
        "data_gen_sec": round(data_gen, 3),
        "wall_runs_sec": [round(w, 3) for w in walls],
        "wall_median_sec": round(median_wall, 3),
        "bars_per_sec": round(total_bars / median_wall, 1),
        "orders": orders,
        "fills": fills,
        "akquant_version": akquant.__version__,
    }
    return summary


def _selftest() -> int:
    """小负载自校验：schema 完整 + 两轮结果确定（订单/成交笔数一致且 > 0）."""
    args = argparse.Namespace(
        symbols=60, days=25, topk=5, runs=2, seed=0, no_orders=False
    )
    summary = _execute(args)
    missing = [k for k in SUMMARY_KEYS if k not in summary]
    if missing:
        print(f"SELFTEST FAIL: summary 缺少字段 {missing}")
        return 1
    if summary["fills"] <= 0 or summary["orders"] <= 0:
        print(
            "SELFTEST FAIL: 应有订单与成交，实际 "
            f"orders={summary['orders']} fills={summary['fills']}"
        )
        return 1
    # 同参数再跑一轮，结果必须逐笔一致（确定性）。
    summary2 = _execute(args)
    if (summary["orders"], summary["fills"]) != (
        summary2["orders"],
        summary2["fills"],
    ):
        print(
            "SELFTEST FAIL: 两轮结果不一致 "
            f"({summary['orders']}/{summary['fills']} vs "
            f"{summary2['orders']}/{summary2['fills']})"
        )
        return 1
    print(f"SELFTEST OK: {json.dumps(summary, ensure_ascii=False)}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 入口."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=4500, help="标的数量")
    parser.add_argument("--days", type=int, default=120, help="交易日数量")
    parser.add_argument("--topk", type=int, default=50, help="每日持仓只数")
    parser.add_argument("--runs", type=int, default=3, help="重复轮数（取中位）")
    parser.add_argument("--seed", type=int, default=0, help="行情随机种子")
    parser.add_argument(
        "--no-orders",
        action="store_true",
        help="策略不下单，测纯管线每 bar 开销",
    )
    parser.add_argument("--selftest", action="store_true", help="小负载自校验")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    summary = _execute(args)
    print(
        f"负载: {summary['symbols']} 标的 × {summary['days']} 日 "
        f"= {summary['total_bars']} bar | topk={summary['topk']} "
        f"no_orders={summary['no_orders']}"
    )
    print(f"数据生成: {summary['data_gen_sec']}s")
    print(f"各轮墙钟: {summary['wall_runs_sec']}")
    print(
        f"中位墙钟: {summary['wall_median_sec']}s | "
        f"bars/sec: {summary['bars_per_sec']} | "
        f"orders: {summary['orders']} | fills: {summary['fills']}"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
