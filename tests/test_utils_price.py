"""委托价按最小变动价位取整(对标 vnpy round_to, 但显式区分方向)."""

import pytest
from akquant.utils.price import round_to_tick


def test_rounds_to_nearest_tick() -> None:
    """缺省四舍五入到最近的 tick 倍数."""
    assert round_to_tick(2.8314, 0.01) == pytest.approx(2.83)
    assert round_to_tick(2.8364, 0.01) == pytest.approx(2.84)


def test_rounds_down_for_buy_protection() -> None:
    """direction='down': 买单向下取整, 不会把价格抬到不利方向."""
    assert round_to_tick(2.8364, 0.01, direction="down") == pytest.approx(2.83)


def test_rounds_up_for_sell_protection() -> None:
    """direction='up': 卖单向上取整, 不会把价格压到不利方向."""
    assert round_to_tick(2.8314, 0.01, direction="up") == pytest.approx(2.84)


def test_handles_float_error_free_of_binary_artifacts() -> None:
    """必须走 Decimal: 二进制浮点会让 0.1+0.2 这类值产生 0.30000000000000004."""
    assert round_to_tick(0.1 + 0.2, 0.001) == pytest.approx(0.3)
    assert round_to_tick(1.005, 0.01) == pytest.approx(1.01)


def test_fund_tick_of_one_thousandth() -> None:
    """ETF/基金/可转债 tick 是 0.001."""
    assert round_to_tick(1.2344, 0.001) == pytest.approx(1.234)
    assert round_to_tick(1.2346, 0.001) == pytest.approx(1.235)


def test_non_positive_tick_returns_price_unchanged() -> None:
    """tick<=0 时原样返回, 不抛除零(vnpy 的 pricetick=0 -> DivisionByZero 教训)."""
    assert round_to_tick(2.8314, 0.0) == pytest.approx(2.8314)
    assert round_to_tick(2.8314, -1.0) == pytest.approx(2.8314)


def test_already_aligned_price_is_unchanged() -> None:
    """已对齐的价格不能被改动."""
    assert round_to_tick(2.83, 0.01) == pytest.approx(2.83)


def test_rejects_unknown_direction() -> None:
    """非法 direction 要显式报错, 不能静默当成 nearest."""
    with pytest.raises(ValueError, match="direction"):
        round_to_tick(2.83, 0.01, direction="sideways")
