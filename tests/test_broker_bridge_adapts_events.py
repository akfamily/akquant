"""BrokerEventBridge 派发前把 Unified* 适配为 StrategyOrder/StrategyTrade."""

import threading

from akquant.akquant import OrderSide, OrderStatus
from akquant.gateway.broker_event_adapter import (
    StrategyOrder,
    StrategyTrade,
    map_order_snapshot,
    map_trade,
)
from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.gateway.broker_models import (
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedTrade,
)


class _Strat:
    def __init__(self):
        self.orders = []
        self.trades = []

    def on_order(self, o):
        self.orders.append(o)

    def on_trade(self, t):
        self.trades.append(t)


def _bridge(adapt):
    store = []
    return BrokerEventBridge(
        event_lock=threading.Lock(),
        event_store=store,
        event_keys=set(),
        get_on_broker_event=lambda: None,
        make_event_key=lambda n, p: f"{n}:{id(p)}",
        update_broker_state=lambda n, p: None,
        resolve_owner_strategy_id=lambda p: "",
        payload_to_dict=lambda p: {},
        safe_strategy_callback=lambda s, name, p: getattr(s, name)(p),
        adapt_strategy_payload=adapt,
    )


def test_bridge_dispatches_adapted_objects() -> None:
    """on_order/on_trade should get StrategyOrder/StrategyTrade, not raw Unified*."""

    def adapt(name, payload):
        if name == "order":
            return map_order_snapshot(payload)
        if name == "trade":
            return map_trade(payload)
        return payload

    b = _bridge(adapt)
    s = _Strat()
    b.queue_event(
        "order",
        UnifiedOrderSnapshot(
            client_order_id="c1",
            broker_order_id="B1",
            symbol="X",
            status=UnifiedOrderStatus.FILLED,
            filled_quantity=1.0,
            avg_fill_price=2.0,
        ),
    )
    b.queue_event(
        "trade",
        UnifiedTrade(
            trade_id="T1",
            broker_order_id="B1",
            client_order_id="c1",
            symbol="X",
            side="Buy",
            quantity=1.0,
            price=2.0,
            timestamp_ns=1,
        ),
    )
    b.drain_events(s)
    assert isinstance(s.orders[0], StrategyOrder)
    assert s.orders[0].status is OrderStatus.Filled
    assert isinstance(s.trades[0], StrategyTrade)
    assert s.trades[0].side is OrderSide.Buy
