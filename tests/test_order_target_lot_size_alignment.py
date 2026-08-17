"""order_target 系列的取整口径必须与撮合层校验口径一致.

此前下单侧取整读的是**策略属性** ``self.lot_size``(默认 1), 而撮合层校验读的是
**标的登记值** ``Instrument.lot_size``。于是登记了 ``lot_size=100`` 的 A 股标的,
``order_target_percent`` 仍按 1 股取整, 算出的非整百数量必然被自己的风控拒掉
(``Quantity 18099 is not a multiple of lot size 100``)。实盘尤其无解:
``run_live`` 没有 ``lot_size`` 参数, 除了手写 ``self.lot_size = 100`` 没有任何途径
让取整逻辑知道登记值。
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar

SYM = "600016.SH"
PRICE = 10.13  # 刻意选让 0.2 * 权益 / 价格 不落在整百上的价格


def _instrument(lot_size: float) -> Instrument:
    return Instrument(
        symbol=SYM,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=lot_size,
    )


def _bars() -> List[Bar]:
    out = []
    for day in (3, 4, 5):
        ts = int(pd.Timestamp(f"2023-01-{day:02d} 14:00:00", tz="Asia/Shanghai").value)
        out.append(
            Bar(
                timestamp=ts,
                open=PRICE,
                high=PRICE + 0.2,
                low=PRICE - 0.2,
                close=PRICE,
                volume=1_000_000.0,
                symbol=SYM,
            )
        )
    return out


class _TargetProbe(Strategy):
    """第一根 bar 上调一次 order_target_percent, 记录产生的订单."""

    def __init__(self, explicit_lot_size: Optional[int] = None) -> None:
        self._explicit_lot_size = explicit_lot_size
        # 按订单 id 去重: on_order 会为同一订单的每次状态变化各触发一次。
        self.seen_orders: Dict[str, float] = {}
        self.lot_rejects: List[str] = []
        self._done = False

    @property
    def quantities(self) -> List[float]:
        """本次会话产生的各订单数量(下单侧取整后的结果)."""
        return list(self.seen_orders.values())

    def on_start(self) -> None:
        if self._explicit_lot_size is not None:
            self.lot_size = self._explicit_lot_size

    def on_order(self, order: Any) -> None:
        self.seen_orders[str(order.id)] = float(order.quantity)
        reason = str(order.reject_reason or "")
        if "lot size" in reason:
            self.lot_rejects.append(reason)

    def on_bar(self, bar: Bar) -> None:
        if self._done:
            return
        self._done = True
        self.order_target_percent(SYM, 0.2, price=PRICE)


def _run(strategy: Strategy, lot_size: float) -> None:
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument(lot_size)],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars()},
        cash=1_000_000.0,
        show_progress=False,
        duration="60s",
    )


def test_order_target_percent_rounds_to_registered_lot_size() -> None:
    """标的登记 lot_size=100 时, 下单数量必须是 100 的倍数(策略未设 self.lot_size)."""
    strategy = _TargetProbe()
    _run(strategy, lot_size=100.0)
    assert strategy.quantities, "order_target_percent 未产生订单"
    assert not strategy.lot_rejects, f"出现 lot size 拒单: {strategy.lot_rejects}"
    qty = strategy.quantities[0]
    assert qty % 100 == 0, f"数量 {qty} 不是登记 lot_size=100 的倍数"


def test_explicit_coarser_lot_size_is_respected() -> None:
    """策略显式设的更粗粒度(200)要被尊重, 不能被登记值 100 覆盖掉."""
    strategy = _TargetProbe(explicit_lot_size=200)
    _run(strategy, lot_size=100.0)
    assert strategy.quantities, "order_target_percent 未产生订单"
    assert not strategy.lot_rejects, f"出现 lot size 拒单: {strategy.lot_rejects}"
    qty = strategy.quantities[0]
    assert qty % 200 == 0, f"数量 {qty} 未按显式 lot_size=200 取整"


def test_lot_size_one_still_allows_odd_quantities() -> None:
    """登记 lot_size=1 的标的(美股/加密)不能被凭空取整到 100."""
    strategy = _TargetProbe()
    _run(strategy, lot_size=1.0)
    assert strategy.quantities, "order_target_percent 未产生订单"
    assert not strategy.lot_rejects, f"出现 lot size 拒单: {strategy.lot_rejects}"
    qty = strategy.quantities[0]
    assert qty % 100 != 0, f"数量 {qty} 被误取整到整百(该标的 lot_size=1)"
