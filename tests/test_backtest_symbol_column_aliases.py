"""回测 DataFrame 入口的标的列识别必须用项目统一的别名表.

``run_backtest(data=DataFrame)`` 的多标的分支判据此前是 ``if "symbol" in df.columns``,
**只认英文列名**。而项目自己的别名表(``schema.COLUMN_ALIASES``)包含
``股票代码``/``symbol``/``code``/``ticker``, 实盘侧的 ``dataframe_to_bars``
更是**只认** ``股票代码``(``normalize.py``) —— 两侧要求的列名正好相反, 且都静默
退化为单标的。

AKShare 的标准输出列名就是 ``股票代码``, 于是"从 AKShare 取多标的数据直接丢进
run_backtest"这个最自然的用法会被**静默**压成单标的: 所有标的的 bar 混进一条序列,
``instruments_config`` 按真实 symbol 配的合约参数(tick_size/lot_size)全部失效,
下单真实 symbol 报 ``Instrument not found``。

参见 ``docs/zh/meta/columnar-rfc.md`` 第 335 行(已挂账的"取并集"统一项)。
"""

from typing import Any, List, Set

import pandas as pd
import pytest
from akquant import BacktestConfig, Strategy, StrategyConfig, run_backtest

SYMBOLS = ("600000.SH", "000012.SZ")


class _SymbolRecorder(Strategy):
    """记录 on_bar 实际看到的 symbol 集合."""

    def __init__(self) -> None:
        self.seen: Set[str] = set()

    def on_bar(self, bar: Any) -> None:
        self.seen.add(str(bar.symbol))


def _multi_symbol_df(symbol_col: str) -> pd.DataFrame:
    rows: List[dict] = []
    for day in (3, 4, 5):
        for symbol in SYMBOLS:
            rows.append(
                {
                    "date": pd.Timestamp(f"2023-01-{day:02d}", tz="UTC"),
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.8,
                    "close": 10.0,
                    "volume": 1_000_000.0,
                    symbol_col: symbol,
                }
            )
    return pd.DataFrame(rows)


def _single_symbol_df() -> pd.DataFrame:
    """真正的单标的数据: 每个时间戳一行, 无标的列(合法且常见的用法)."""
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(f"2023-01-{day:02d}", tz="UTC"),
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 1_000_000.0,
            }
            for day in (3, 4, 5)
        ]
    )


def _run(data: pd.DataFrame) -> _SymbolRecorder:
    strategy = _SymbolRecorder()
    run_backtest(
        strategy=strategy,
        data=data,
        config=BacktestConfig(
            strategy_config=StrategyConfig(initial_cash=1_000_000.0),
        ),
        show_progress=False,
    )
    return strategy


def test_chinese_symbol_column_is_recognized_as_multi_symbol() -> None:
    """``股票代码`` 列(AKShare 标准列名)必须被识别为多标的."""
    strategy = _run(_multi_symbol_df("股票代码"))
    assert strategy.seen == set(SYMBOLS), (
        f"多标的未被识别, on_bar 实际看到: {strategy.seen}"
    )


@pytest.mark.parametrize("symbol_col", ["symbol", "code", "ticker"])
def test_other_alias_columns_are_recognized(symbol_col: str) -> None:
    """别名表里的其余列名同样被识别(``symbol`` 是既有行为, 不能回退)."""
    strategy = _run(_multi_symbol_df(symbol_col))
    assert strategy.seen == set(SYMBOLS), (
        f"列名 {symbol_col!r} 下多标的未被识别: {strategy.seen}"
    )


def test_unrecognized_symbol_column_warns_about_degradation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """同一时间戳多行却识别不到标的列时必须告警(此前完全静默)."""
    df = _multi_symbol_df("我的标的列")
    with caplog.at_level("WARNING"):
        _run(df)
    assert "标的列" in caplog.text or "symbol" in caplog.text.lower(), (
        f"退化为单标的未告警, 日志为: {caplog.text[:500]}"
    )


def test_genuine_single_symbol_data_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """真单标的数据(每时间戳一行)不能被告警刷屏."""
    with caplog.at_level("WARNING"):
        _run(_single_symbol_df())
    assert "退化为单标的" not in caplog.text, f"单标的用法被误告警: {caplog.text[:500]}"
