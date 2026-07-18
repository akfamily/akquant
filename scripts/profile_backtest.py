"""回测性能剖析脚本 (issue #288 配套).

目的:用可复现的合成数据量化"历史数据访问"在整体回测耗时中的占比,
并对比逐字段 ``get_history`` 与批量 ``get_history_multi`` 的边界开销,
为后续优化提供基线,避免"凭感觉"优化错地方。

用法(在 workspace 根,`P` 记为 akquant/scripts/profile_backtest.py):

    uv run --package akquant python P
    uv run --package akquant python P --bars 20000 --symbols 5 --count 60
    uv run --package akquant python P --cprofile --top 25

输出:
1. 三档回测墙钟耗时(不取历史 / 每 bar get_history / 每 bar get_history_df),
   及历史访问在总耗时中的占比。
2. ``5×get_history`` vs ``1×get_history_multi`` 微基准(同取 OHLCV 五字段)。
3. 可选:cProfile 累计耗时 Top-N 函数,定位真正热点。
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from typing import Optional

import akquant
import numpy as np
import pandas as pd
from akquant import Strategy, run_backtest

# Windows 控制台默认非 UTF-8,中文会乱码;尽力切到 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass

OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


def _build_data(n_bars: int, n_symbols: int) -> dict[str, pd.DataFrame]:
    """构造 ``n_symbols`` 个标的、各 ``n_bars`` 根 Bar 的随机游走行情."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-04", periods=n_bars, freq="B", tz="Asia/Shanghai")
    data: dict[str, pd.DataFrame] = {}
    for i in range(n_symbols):
        close = 100.0 + np.cumsum(rng.standard_normal(n_bars)) * 0.1
        df = pd.DataFrame(index=dates)
        df.index.name = "date"
        df["open"] = close
        df["high"] = close + 0.1
        df["low"] = close - 0.1
        df["close"] = close
        df["volume"] = 1_000_000.0
        df["symbol"] = f"S{i}"
        data[f"S{i}"] = df
    return data


class _ProfileStrategy(Strategy):
    """每根 Bar 按 ``mode`` 访问一次历史数据,并累计其耗时."""

    def __init__(self, mode: str, count: int, symbol: str) -> None:
        super().__init__()
        self.set_history_depth(max(count, 1))
        self.mode = mode  # "none" | "single" | "df"
        self.count = count
        self.target_symbol = symbol
        self.calls = 0
        self.hist_ns = 0

    def on_bar(self, bar: akquant.Bar) -> None:
        if self.mode == "none":
            return
        start = time.perf_counter_ns()
        if self.mode == "single":
            self.get_history(self.count, self.target_symbol, "close")
        elif self.mode == "df":
            self.get_history_df(count=self.count, symbol=self.target_symbol)
        self.hist_ns += time.perf_counter_ns() - start
        self.calls += 1


def _run_once(
    data: dict[str, pd.DataFrame], mode: str, count: int
) -> tuple[float, int, int]:
    """跑一次回测,返回 (墙钟秒, 历史访问调用数, 历史访问累计纳秒)."""
    symbols = sorted(data.keys())
    strat = _ProfileStrategy(mode=mode, count=count, symbol=symbols[0])
    start = time.perf_counter()
    run_backtest(
        data=data,
        strategy=strat,
        symbols=symbols,
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        show_progress=False,
    )
    return time.perf_counter() - start, strat.calls, strat.hist_ns


def _micro_bench_multi(
    data: dict[str, pd.DataFrame], count: int, iterations: int
) -> None:
    """在一个已积累历史的回调里,对比 5×get_history 与 1×get_history_multi."""
    symbols = sorted(data.keys())
    target = symbols[0]
    results: dict[str, float] = {}

    class _BenchStrategy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.set_history_depth(max(count, 1))
            self._done = False

        def on_bar(self, bar: akquant.Bar) -> None:
            if self._done or self.get_history(count, target, "close").size < count:
                return
            self._done = True

            t0 = time.perf_counter_ns()
            for _ in range(iterations):
                {f: self.get_history(count, target, f) for f in OHLCV_FIELDS}
            results["per_field"] = (time.perf_counter_ns() - t0) / iterations

            t0 = time.perf_counter_ns()
            for _ in range(iterations):
                self.get_history_multi(count, target, OHLCV_FIELDS)
            results["multi"] = (time.perf_counter_ns() - t0) / iterations

    run_backtest(
        data=data,
        strategy=_BenchStrategy(),
        symbols=symbols,
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        show_progress=False,
    )

    if "per_field" in results and "multi" in results:
        per = results["per_field"]
        multi = results["multi"]
        speedup = per / multi if multi > 0 else float("nan")
        print("\n[微基准] 取回 OHLCV 五字段 (每次调用平均耗时):")
        print(f"  5×get_history      : {per:8.0f} ns")
        print(f"  1×get_history_multi: {multi:8.0f} ns  (约 {speedup:.2f}× 更快)")
    else:
        print("\n[微基准] 历史不足,未能采样(尝试增大 --bars 或减小 --count)。")


def _run_cprofile(data: dict[str, pd.DataFrame], count: int, top: int) -> None:
    """对'每 bar 取一次 get_history'的回测做 cProfile,打印累计耗时 Top-N."""
    profiler = cProfile.Profile()
    profiler.enable()
    _run_once(data, mode="single", count=count)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(top)
    print("\n[cProfile] 累计耗时 Top 函数:")
    print(stream.getvalue())


def main(argv: Optional[list[str]] = None) -> None:
    """命令行入口:跑三档回测 + 微基准,可选 cProfile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=10_000, help="每个标的的 Bar 数量")
    parser.add_argument("--symbols", type=int, default=1, help="标的数量")
    parser.add_argument("--count", type=int, default=60, help="每次取历史的窗口长度")
    parser.add_argument(
        "--iterations", type=int, default=5_000, help="微基准的重复次数"
    )
    parser.add_argument(
        "--cprofile", action="store_true", help="额外输出 cProfile 热点"
    )
    parser.add_argument("--top", type=int, default=20, help="cProfile 打印的函数数量")
    args = parser.parse_args(argv)

    data = _build_data(args.bars, args.symbols)
    total_bars = args.bars * args.symbols
    print(
        f"配置: bars/标的={args.bars}, 标的数={args.symbols}, "
        f"总 Bar={total_bars}, 历史窗口 count={args.count}"
    )

    base_s, _, _ = _run_once(data, mode="none", count=args.count)
    single = _run_once(data, mode="single", count=args.count)
    df = _run_once(data, mode="df", count=args.count)

    def _line(label: str, total_s: float, calls: int, hist_ns: int) -> str:
        pct = 100.0 * (hist_ns / 1e9) / total_s if total_s > 0 else 0.0
        per = hist_ns / max(calls, 1)
        return (
            f"  {label}: {total_s * 1000:9.1f} ms | 历史访问 {hist_ns / 1e6:7.2f} ms"
            f" / {calls} 次 / {per:6.0f} ns 每次 | 占比 {pct:4.1f}%"
        )

    print("\n[墙钟耗时]")
    print(f"  基线(不取历史)         : {base_s * 1000:9.1f} ms")
    print(_line("每 bar get_history   ", single[0], single[1], single[2]))
    print(_line("每 bar get_history_df", df[0], df[1], df[2]))

    _micro_bench_multi(data, args.count, args.iterations)

    if args.cprofile:
        _run_cprofile(data, args.count, args.top)


if __name__ == "__main__":
    main()
