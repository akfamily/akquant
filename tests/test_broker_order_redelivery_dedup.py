"""broker_live 委托重放去重: 同状态每会话只派发一次, 状态推进立刻放行."""

import threading

from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.live._payload_utils import payload_field


class _Strat:
    def __init__(self) -> None:
        self.orders: list = []
        self.reports: list = []

    def on_order(self, o: object) -> None:
        self.orders.append(o)

    def on_trade(self, t: object) -> None:
        pass

    def on_execution_report(self, r: object) -> None:
        self.reports.append(r)


def _bridge(store: list) -> BrokerEventBridge:
    def safe(strategy: object, name: str, payload: object) -> None:
        fn = getattr(strategy, name, None)
        if fn is not None:
            fn(payload)

    def make_key(name: str, payload: object) -> str:
        # 镜像 _runner._make_event_key 改后的口径(order/execution_report 键
        # 都含 filled_quantity、不含 timestamp_ns)。
        if name in ("order", "execution_report"):
            return (
                f"{name}:{payload_field(payload, 'broker_order_id')}"
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


def _order(
    status: str, filled: float = 0.0, oid: str = "O1", avg_fill_price: float = 0.0
) -> dict:
    return {
        "broker_order_id": oid,
        "symbol": "600008.SH",
        "status": status,
        "filled_quantity": filled,
        "avg_fill_price": avg_fill_price,
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


def test_order_and_execution_report_dedupe_independently() -> None:
    """内置 broker 对同一状态成对派发 order+execution_report, 两者互不吞对方.

    ctp/miniqmt/ptrade 的 ingest 路径用**同一个** payload 先触发
    order_callback、紧接着触发 execution_callback, 四个指纹字段(status/
    filled_quantity/avg_fill_price/reject_reason)逐字相同。去重键必须带
    事件类型前缀, 否则先入队的 order 会让随后同指纹的 execution_report
    被误判"已派发过"而永久丢弃, on_execution_report 收不到任何回报.
    """
    store: list = []
    b, s = _bridge(store), _Strat()

    payload = _order("submitted")
    b.queue_event("order", payload)
    b.queue_event("execution_report", payload)
    b.drain_events(s)

    assert len(s.orders) == 1
    assert len(s.reports) == 1


def test_avg_fill_price_correction_survives_batch_key_and_replay() -> None:
    """同批内均价修正不应永久锁死指纹, 重放要能追上修正值.

    批内键(``order:{id}:{status}:{filled_quantity}``)不含 ``avg_fill_price``,
    而会话指纹多含它 —— 若指纹提交发生在批内键否决之前, 同批内第二条
    (均价 0.0 -> 10.2)会把指纹写成 10.2 却被批内键丢弃, 之后柜台重放同一
    修正值时指纹命中, 永久吞掉。指纹提交必须与真正入队的事件绑定。
    """
    store: list = []
    b, s = _bridge(store), _Strat()

    # 同一批内先后到达: 均价 0.0 -> 10.2, status/filled_quantity 不变。
    b.queue_event("order", _order("filled", 200.0, avg_fill_price=0.0))
    b.queue_event("order", _order("filled", 200.0, avg_fill_price=10.2))
    b.drain_events(s)

    # 批内键否决第二条是预期行为(同批只留一条); 关键在于指纹不能被第二条
    # 抢先提交——下一轮重放同一条修正值必须还能追上。
    b.queue_event("order", _order("filled", 200.0, avg_fill_price=10.2))
    b.drain_events(s)

    assert len(s.orders) == 2
    assert float(s.orders[-1]["avg_fill_price"]) == 10.2


def test_dropped_counter_records_duplicates() -> None:
    """被去重丢弃的事件要计数(供收尾摘要暴露过度去重)."""
    store: list = []
    b, s = _bridge(store), _Strat()

    b.queue_event("order", _order("submitted"))
    b.drain_events(s)
    b.queue_event("order", _order("submitted"))

    assert b.dropped_event_counts()["duplicate_order"] == 1
