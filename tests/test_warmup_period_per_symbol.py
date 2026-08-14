"""多标的下 warmup_period 门槛必须按 symbol 独立计数.

根因: strategy_events.py 里驱动 warmup 门槛的 ``strategy._bar_count`` 是跨
symbol 的全局计数器, 但历史缓冲区是 per-symbol 的。M 个 symbol 交替产生 bar
事件时, 每个 symbol 实际只攒到约 ``ceil(warmup_period / M)`` 根真实历史就被
放行, 官方示例里 ``warmup_period = long_window`` 这类写法在多标的下全部失效
(参见 docs/zh/guide/examples.md、strategy.md、quant_basics.md)。

这里覆盖三个场景:
1. 多标的且各标的数据条数相同: 首次 on_bar 时该标的历史必须真正攒够
   warmup_period 根, 且值与从数据序列独立推导的结果精确匹配(不是拿程序
   输出反推)。
2. 单标的: 原本正确的行为不能变(回归底线)。
3. 标的上市时间不同(数据条数不等): 更极端的场景 —— 后来者不能"蹭"到先行
   者攒下的全局计数, 必须等自己攒够。
"""

from __future__ import annotations

import math
from typing import cast

import pandas as pd
from akquant import Bar, Strategy, run_backtest


def _df(closes: list[float], start: str, periods: int, symbol: str) -> pd.DataFrame:
    """构造单标的日线 DataFrame, 时间戳与 close 一一对应."""
    dates = pd.date_range(start, periods=periods, freq="D", tz="Asia/Shanghai")
    assert len(closes) == periods
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * periods,
            "symbol": [symbol] * periods,
        }
    )


class _FirstBarHistoryCollector(Strategy):
    """每个 symbol 首次触发 on_bar 时, 记录当时的历史窗口(count=warmup_period)."""

    warmup_period = 3

    def on_start(self) -> None:
        """重置采集状态."""
        self.seen: set[str] = set()
        self.captured: dict[str, list[float]] = {}

    def on_bar(self, bar: Bar) -> None:
        """每个 symbol 首次收到 bar 时取一次历史(不覆盖)."""
        if bar.symbol in self.seen:
            return
        self.seen.add(bar.symbol)
        self.captured[bar.symbol] = [
            float(x)
            for x in self.get_history(
                count=self.warmup_period, symbol=bar.symbol, field="close"
            )
        ]


def test_warmup_period_reaches_full_depth_per_symbol_with_multiple_symbols() -> None:
    """两个标的、条数相同、时间戳对齐: 各自应在攒够 3 根真实历史后才首次触发.

    独立推导的期望值: X 的前 3 根收盘价是 [10, 11, 12], Y 的前 3 根是
    [100, 101, 102]。修复前, 全局计数器在两个 symbol 交替下 3 次 bar 事件
    (即 X 的第 2 根、随后 Y 的第 2 根) 就会跨过门槛, 此时历史里会混入 nan
    占位, 且值不精确匹配 [10, 11, 12] / [100, 101, 102]。
    """
    x_closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    y_closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    data = {
        "X": _df(x_closes, "2024-01-01", 10, "X"),
        "Y": _df(y_closes, "2024-01-01", 10, "Y"),
    }

    result = run_backtest(
        data=data,
        strategy=_FirstBarHistoryCollector,
        symbols=["X", "Y"],
        initial_cash=1e5,
        show_progress=False,
    )
    strategy = cast(_FirstBarHistoryCollector, result.strategy)

    assert strategy.captured["X"] == [10.0, 11.0, 12.0]
    assert strategy.captured["Y"] == [100.0, 101.0, 102.0]
    for values in strategy.captured.values():
        assert all(not math.isnan(v) for v in values)


def test_warmup_period_single_symbol_unaffected() -> None:
    """单标的场景下, 修复前后行为必须一致(回归底线)."""
    x_closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    data = {"X": _df(x_closes, "2024-01-01", 10, "X")}

    result = run_backtest(
        data=data,
        strategy=_FirstBarHistoryCollector,
        symbols=["X"],
        initial_cash=1e5,
        show_progress=False,
    )
    strategy = cast(_FirstBarHistoryCollector, result.strategy)

    assert strategy.captured["X"] == [10.0, 11.0, 12.0]


def test_warmup_period_per_symbol_independent_with_staggered_listing() -> None:
    """标的上市时间不同(数据条数不等)时, 后来者不能蹭到先行者攒下的全局计数.

    X 从第 1 天开始有 10 根 bar; Y 从第 6 天才开始上市, 只有 5 根 bar。修复前
    全局计数器在 Y 上市前就已被 X 推过 warmup_period=3 的门槛, Y 的第 1 根
    bar 就会直接触发 on_bar, 历史几乎全是 nan。独立推导的期望: Y 必须等到
    自己攒够 3 根(第 6、7、8 天), 历史精确等于 [200, 201, 202]。
    """
    x_closes = [10.0 + i for i in range(10)]  # 10..19, day1..day10
    y_closes = [200.0, 201.0, 202.0, 203.0, 204.0]  # day6..day10

    data = {
        "X": _df(x_closes, "2024-01-01", 10, "X"),
        "Y": _df(y_closes, "2024-01-06", 5, "Y"),
    }

    result = run_backtest(
        data=data,
        strategy=_FirstBarHistoryCollector,
        symbols=["X", "Y"],
        initial_cash=1e5,
        show_progress=False,
    )
    strategy = cast(_FirstBarHistoryCollector, result.strategy)

    assert strategy.captured["Y"] == [200.0, 201.0, 202.0]
    assert all(not math.isnan(v) for v in strategy.captured["Y"])
