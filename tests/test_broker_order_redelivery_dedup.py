"""broker_live 委托重放去重: 同状态每会话只派发一次, 状态推进立刻放行."""

import threading

from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.live._payload_utils import payload_field


class _Strat:
    def __init__(self) -> None:
        self.orders: list = []

    def on_order(self, o: object) -> None:
        self.orders.append(o)

    def on_trade(self, t: object) -> None:
        pass


def _bridge(store: list) -> BrokerEventBridge:
    def safe(strategy: object, name: str, payload: object) -> None:
        fn = getattr(strategy, name, None)
        if fn is not None:
            fn(payload)

    def make_key(name: str, payload: object) -> str:
        # 镜像 _runner._make_event_key 改后的口径(order 键不含 timestamp_ns)
        if name == "order":
            return (
                f"order:{payload_field(payload, 'broker_order_id')}"
                f":{payload_field(payload, 'status')}"
                f":{payload_field(payload, 'filled_quantity')}"
            )
        return f"{name}:{id(payload)}"

    return BrokerEventBridge(
        event_lock=threading.Lock(),
        event_store=store,
        event_keys=set(),
        get_on_broker_event=lambda: None,
        make_event_key=make_key,
        update_broker_state=lambda n, p: None,
        resolve_owner_strategy_id=lambda p: "",
        payload_to_dict=lambda p: dict(p) if isinstance(p, dict) else {},
        safe_strategy_callback=safe,
        adapt_strategy_payload=lambda n, p: p,
        payload_field=payload_field,
    )


def _order(status: str, filled: float = 0.0, oid: str = "O1") -> dict:
    return {
        "broker_order_id": oid,
        "symbol": "600008.SH",
        "status": status,
        "filled_quantity": filled,
        "avg_fill_price": 0.0,
        "reject_reason": "",
        "timestamp_ns": 0,
    }


def test_same_state_dispatched_once_across_drains() -> None:
    """Recovery 每轮重放同一状态的挂单, on_order 只触发一次."""
    store: list = []
    b, s = _bridge(store), _Strat()

    b.queue_event("order", _order("submitted"))
    b.drain_events(s)
    for _ in range(5):  # 模拟 5 轮 recovery 重放
        b.queue_event("order", _order("submitted"))
        b.drain_events(s)

    assert len(s.orders) == 1


def test_state_progression_always_dispatched() -> None:
    """New -> PartiallyFilled -> Filled 是真实推进, 三次都必须放行."""
    store: list = []
    b, s = _bridge(store), _Strat()

    progression = (("new", 0.0), ("partially_filled", 100.0), ("filled", 200.0))
    for status, filled in progression:
        b.queue_event("order", _order(status, filled))
        b.drain_events(s)

    assert len(s.orders) == 3
    assert [str(o["status"]) for o in s.orders] == ["new", "partially_filled", "filled"]


def test_changing_timestamp_does_not_defeat_dedupe() -> None:
    """同状态但时间戳每帧变化(ctp 等会填真实时间), 仍应只派发一次."""
    store: list = []
    b, s = _bridge(store), _Strat()

    for ts in (1_000, 2_000, 3_000):
        payload = _order("submitted")
        payload["timestamp_ns"] = ts
        b.queue_event("order", payload)
        b.drain_events(s)

    assert len(s.orders) == 1


def test_terminal_state_not_redispatched() -> None:
    """终态单在后续轮次仍被重放时不再派发(终态不从表中移除)."""
    store: list = []
    b, s = _bridge(store), _Strat()

    b.queue_event("order", _order("filled", 200.0))
    b.drain_events(s)
    b.queue_event("order", _order("filled", 200.0))
    b.drain_events(s)

    assert len(s.orders) == 1


def test_distinct_orders_are_independent() -> None:
    """不同委托各自独立去重, 不互相干扰."""
    store: list = []
    b, s = _bridge(store), _Strat()

    b.queue_event("order", _order("submitted", oid="O1"))
    b.queue_event("order", _order("submitted", oid="O2"))
    b.drain_events(s)

    assert len(s.orders) == 2


def test_dropped_counter_records_duplicates() -> None:
    """被去重丢弃的事件要计数(供收尾摘要暴露过度去重)."""
    store: list = []
    b, s = _bridge(store), _Strat()

    b.queue_event("order", _order("submitted"))
    b.drain_events(s)
    b.queue_event("order", _order("submitted"))

    assert b.dropped_event_counts()["duplicate_order"] == 1
