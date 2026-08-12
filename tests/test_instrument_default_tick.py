"""缺省最小变动价位按资产类型分流.

ETF/基金/可转债的 tick 是 0.001, 一律缺省 0.01 会让合法委托被 tick 校验误拒
——QuantConnect Lean 正因"查不到就默认 0.01"出过止损单被挪走的事故。
"""

import pytest
from akquant import AssetType, Instrument


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
