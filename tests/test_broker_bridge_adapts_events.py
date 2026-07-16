"""BrokerEventBridge 派发前把 Unified* 适配为 StrategyOrder/StrategyTrade."""

import threading
from typing import Any, cast

from akquant.akquant import OrderSide, OrderStatus
from akquant.gateway.broker_event_adapter import (
    StrategyOrder,
    StrategyTrade,
    map_order_snapshot,
    map_trade,
)
from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.gateway.broker_models import (
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedTrade,
)
from akquant.live import LiveRunner


class _Strat:
    def __init__(self) -> None:
        self.orders: list[StrategyOrder] = []
        self.trades: list[StrategyTrade] = []

    def on_order(self, o: StrategyOrder) -> None:
        self.orders.append(o)

    def on_trade(self, t: StrategyTrade) -> None:
        self.trades.append(t)

    def _process_order_groups(self, t: StrategyTrade) -> None:
        # broker_live 成交后桥会额外驱动协调器; 本桩无 OCO/bracket 组, no-op。
        pass


def _bridge(adapt: Any) -> BrokerEventBridge:
    store: list[tuple[str, Any]] = []
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

    def adapt(name: str, payload: Any) -> Any:
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


def test_terminal_order_dispatch_still_backfills_request_fields() -> None:
    """终态(FILLED)order 派发给 on_order 时,side/quantity/price 仍须来自 request.

    用真实 LiveRunner 组装的 drain_events 管线(真实 _adapt_strategy_payload +
    真实 _update_broker_state,后者在终态会 pop 请求缓存)来复现:
    若适配发生在状态清理之后,side/quantity/price 会全部退化成 None。
    """
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()

    request = UnifiedOrderRequest(
        client_order_id="c-term",
        symbol="600000.SH",
        side="Buy",
        quantity=100.0,
        price=12.5,
    )
    runner._record_order_request("c-term", request)
    runner._sync_order_id_mapping("c-term", "b-term")

    strategy = _Strat()
    snapshot = UnifiedOrderSnapshot(
        client_order_id="c-term",
        broker_order_id="b-term",
        symbol="600000.SH",
        status=UnifiedOrderStatus.FILLED,
        filled_quantity=100.0,
        avg_fill_price=12.5,
    )
    runner._queue_broker_event("order", snapshot)
    runner._drain_broker_events(cast(Any, strategy))

    assert len(strategy.orders) == 1
    order = strategy.orders[0]
    assert isinstance(order, StrategyOrder)
    assert order.side is OrderSide.Buy
    assert order.quantity == 100.0
    assert order.price == 12.5

    # State cleanup on terminal status must still have happened (no leak).
    assert runner._order_requests == {}
    assert runner._lookup_order_request(snapshot) is None
