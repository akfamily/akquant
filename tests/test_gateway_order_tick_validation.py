"""实盘报单前的 tick 校验(本地拒单, 不发给柜台)."""

from typing import Any

import pytest
from akquant.gateway.order_submitter import _validate_price_tick

# `InstrumentSnapshot.asset_type`(strategy.get_instrument() 的真实返回类型)
# 是大写字符串字面量, 不是 `akquant.AssetType` 枚举——两者是不同类型, 这里
# 用真实形状构造测试桩, 避免用错类型让校验"看起来测过、其实从没生效"。
_STOCK = "STOCK"
_FUND = "FUND"
_OPTION = "OPTION"


class _Snap:
    def __init__(self, tick: float, asset_type: Any = _STOCK) -> None:
        self.tick_size = tick
        self.asset_type = asset_type


class _Strategy:
    def __init__(self, tick: float | None, asset_type: Any = _STOCK) -> None:
        self._tick = tick
        self._asset_type = asset_type

    def get_instrument(self, symbol: str) -> Any:
        if self._tick is None:
            raise KeyError(symbol)
        return _Snap(self._tick, self._asset_type)


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
    _validate_price_tick(_Strategy(0.001, _FUND), "511990.SH", 2.831)


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


@pytest.mark.parametrize(
    "price",
    [
        0.1 + 0.2,  # 0.30000000000000004
        100.1 + 0.1,  # 100.19999999999999
        0.07 * 3,  # 0.21000000000000002
    ],
)
def test_float_noise_price_passes_like_rust_is_multiple(price: float) -> None:
    """f64 运算噪声价必须放行, 不能回测能过、实盘却本地拒.

    这三个价格都是"看起来对齐、算出来带二进制噪声"的典型例子, 与 Rust 侧
    `execution::validation::is_multiple` 的"商距最近整数 1e-6 容差"判定必须
    完全一致, 否则同一笔算术生成的价格会出现回测成交、实盘拒单的分裂。
    """
    _validate_price_tick(_Strategy(0.001, _FUND), "511990.SH", price)


def test_option_asset_type_skips_validation() -> None:
    """期权不做本地 tick 校验.

    `Instrument` 缺省 tick 目前只按 stock/fund 分流, Option 仍缺省 0.01,
    但 SSE/SZSE ETF 期权实际最小变动价位是 0.0001; 而回测侧
    `execution/option.rs` 本就不校验期权 tick。若在此对期权做校验, 会出现
    "回测能成交、实盘被本地误拒"且无软开关可关的问题, 因此直接跳过。
    """
    _validate_price_tick(_Strategy(0.01, _OPTION), "10004532.SH", 2.6543)
