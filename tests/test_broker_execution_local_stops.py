"""BrokerExecution 本地止损: 拦截/撤单/列单/触发提交底层单."""

from typing import Any, cast

from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_state_cache import BrokerStateCache


class _Cache:
    def positions(self) -> dict[str, float]:
        return {}

    def available_positions(self) -> dict[str, float]:
        return {}

    def open_orders(self) -> list[object]:
        return []

    def account(self) -> None:
        return None


class _Gw:
    def __init__(self) -> None:
        self.canceled: list[str] = []

    def cancel_order(self, bid: str) -> None:
        self.canceled.append(bid)

    def sync_open_orders(self) -> list[object]:
        return []


class _Submitter:
    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []

    def submit_order(self, **kw: Any) -> str:
        self.orders.append(kw)
        return "BID-1"

    def _get_execution_capabilities(self) -> dict[str, bool]:
        return {"broker_live": True}


class _S:
    current_bar = None
    current_tick = None


def _exec() -> BrokerExecution:
    return BrokerExecution(
        _S(),
        _Gw(),
        cast(BrokerStateCache, _Cache()),
        _Submitter(),
    )


def test_conditional_order_registered_not_submitted() -> None:
    """条件单(trigger_price)应入本地簿, 不下发柜台."""
    ex = _exec()
    oid = ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    assert oid.startswith("LSTOP-")
    assert ex._submitter.orders == []  # not sent to broker yet
    assert len(ex.get_open_orders("X")) == 1


def test_plain_order_passes_through() -> None:
    """普通单(无 trigger_price)路径不变, 直接走 submitter."""
    ex = _exec()
    oid = ex.submit_order(
        symbol="X", side="Buy", quantity=100, price=10.0, order_type="Limit"
    )
    assert oid == "BID-1"
    assert len(ex._submitter.orders) == 1


def test_cancel_local_stop() -> None:
    """撤销本地止损单只操作本地簿, 不调柜台撤单."""
    ex = _exec()
    oid = ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    ex.cancel_order(oid)
    assert ex.get_open_orders() == []
    assert ex._gw.canceled == []  # local cancel, no broker call


def test_check_triggers_submits_underlying_market() -> None:
    """StopMarket 触发后应提交底层 Market 单, 不带 trigger_price."""
    ex = _exec()
    ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    ex.check_stop_triggers("X", last=9.4, high=9.6, low=9.3)
    assert len(ex._submitter.orders) == 1
    sub = ex._submitter.orders[0]
    assert sub["order_type"] == "Market"
    assert sub.get("trigger_price") is None
    assert sub["side"] == "Sell" and sub["quantity"] == 100
    assert ex.get_open_orders() == []  # consumed


def test_check_triggers_submits_underlying_limit() -> None:
    """StopLimit 触发后应提交底层 Limit 单, 保留原 price."""
    ex = _exec()
    ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopLimit",
        trigger_price=9.5,
        price=9.4,
    )
    ex.check_stop_triggers("X", last=9.3, high=9.5, low=9.3)
    sub = ex._submitter.orders[0]
    assert sub["order_type"] == "Limit" and sub["price"] == 9.4


def test_check_triggers_carries_time_in_force() -> None:
    """带 time_in_force 的 stop 触发后, 底层单应带上同样的 time_in_force."""
    ex = _exec()
    ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
        time_in_force="IOC",
    )
    ex.check_stop_triggers("X", last=9.4, high=9.6, low=9.3)
    sub = ex._submitter.orders[0]
    assert sub["time_in_force"] == "IOC"


def test_check_triggers_without_time_in_force_omits_key() -> None:
    """未指定 time_in_force 的 stop 触发后, 底层单不应携带该 key(或为 None)."""
    ex = _exec()
    ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    ex.check_stop_triggers("X", last=9.4, high=9.6, low=9.3)
    sub = ex._submitter.orders[0]
    assert sub.get("time_in_force") is None
