"""实盘报单前的 tick 校验(本地拒单, 不发给柜台)."""

from typing import Any

import pytest
from akquant.gateway.order_submitter import _validate_price_tick


class _Snap:
    def __init__(self, tick: float) -> None:
        self.tick_size = tick


class _Strategy:
    def __init__(self, tick: float | None) -> None:
        self._tick = tick

    def get_instrument(self, symbol: str) -> Any:
        if self._tick is None:
            raise KeyError(symbol)
        return _Snap(self._tick)


def test_misaligned_price_rejected_with_actionable_message() -> None:
    """非 tick 倍数的委托价本地拒单, 报错要给 tick 和买/卖两个建议价."""
    with pytest.raises(ValueError) as exc:
        _validate_price_tick(_Strategy(0.01), "600008.SH", 2.8314)
    message = str(exc.value)
    assert "2.8314" in message
    assert "0.01" in message
    assert "2.83" in message and "2.84" in message


def test_aligned_price_passes() -> None:
    """对齐的价格放行."""
    _validate_price_tick(_Strategy(0.01), "600008.SH", 2.83)


def test_fund_tick_of_one_thousandth_passes() -> None:
    """基金 0.001 下 2.831 合法(用股票 0.01 判会误拒)."""
    _validate_price_tick(_Strategy(0.001), "511990.SH", 2.831)


def test_market_order_without_price_passes() -> None:
    """市价单没有 price, 不参与校验."""
    _validate_price_tick(_Strategy(0.01), "600008.SH", None)


def test_unregistered_instrument_skips_validation() -> None:
    """标的未登记时跳过校验, 不能因为拿不到 tick 就拦掉所有单.

    这里与 Strategy.round_to_tick 的取舍不同: 那里是用户主动求取整, 报错是
    帮助; 这里是下单路径, 拿不到元数据就拦单会让没配 instruments 的用户完全
    无法下单——柜台自己也会校验, 交给它。
    """
    _validate_price_tick(_Strategy(None), "600008.SH", 2.8314)


def test_non_positive_tick_skips_validation() -> None:
    """tick<=0(柜台未提供)时跳过, 不抛除零."""
    _validate_price_tick(_Strategy(0.0), "600008.SH", 2.8314)
