"""缺省最小变动价位按资产类型分流.

ETF/基金/可转债的 tick 是 0.001, 一律缺省 0.01 会让合法委托被 tick 校验误拒
——QuantConnect Lean 正因"查不到就默认 0.01"出过止损单被挪走的事故。
"""

from typing import Any, cast

import pandas as pd
import pytest
from akquant import AssetType, Instrument, Strategy, run_backtest


def test_stock_default_tick_is_one_cent() -> None:
    """A 股股票 0.01."""
    assert Instrument("600008.SH", AssetType.Stock).tick_size == pytest.approx(0.01)


def test_fund_default_tick_is_one_thousandth() -> None:
    """ETF/基金/可转债 0.001(深交所交易规则 3.3.13 条)."""
    assert Instrument("511990.SH", AssetType.Fund).tick_size == pytest.approx(0.001)


def test_explicit_tick_size_wins_over_default() -> None:
    """显式传值优先于缺省."""
    inst = Instrument("511990.SH", AssetType.Fund, tick_size=0.01)
    assert inst.tick_size == pytest.approx(0.01)


def test_futures_default_tick_unchanged() -> None:
    """期货缺省保持 0.01(其真实 tick 由用户按品种传入)."""
    assert Instrument("IF2601", AssetType.Futures).tick_size == pytest.approx(0.01)


def _fund_feed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
            "open": 2.83,
            "high": 2.85,
            "low": 2.81,
            "close": 2.83,
            "volume": 100000.0,
            "symbol": "511990.SH",
        }
    )


class _TickProbe(Strategy):
    """记录 on_start 时刻引擎里该 symbol 的 tick_size, 供断言用."""

    def on_start(self) -> None:
        self.observed_tick_size = float(self.get_instrument("511990.SH").tick_size)

    def on_bar(self, bar: Any) -> None:
        _ = bar


def test_backtest_blanket_fund_asset_type_defaults_tick_to_one_thousandth() -> None:
    """全局 asset_type=FUND、无逐 symbol InstrumentConfig、无 tick_size 时也要是 0.001.

    对应 run_backtest 里"无 InstrumentConfig、无期货模板匹配"的缺省分支
    (engine.py 的 default_tick_size/preliminary_default_tick_size 曾经对
    asset_type 视而不见, 一律回退 0.01)——这条路径不经过 Instrument() 的
    Rust 侧缺省逻辑, 必须在 Python 侧单独按 asset_type 分流。

    ``asset_type`` 是通过 run_backtest 的 **kwargs 传给引擎的全局缺省值, 同一个
    kwargs 字典也会拿去对策略构造参数做严格校验(strict_strategy_params);
    _TickProbe 没有声明 asset_type 这个字段, 因此这里显式关掉严格校验,
    只是为了让用例聚焦在 tick 缺省分流本身, 与该校验开关无关。
    """
    result = run_backtest(
        data=_fund_feed(),
        strategy=_TickProbe,
        symbols="511990.SH",
        asset_type=AssetType.Fund,
        strict_strategy_params=False,
        show_progress=False,
    )
    strategy = cast(_TickProbe, result.strategy)
    assert strategy.observed_tick_size == pytest.approx(0.001)
