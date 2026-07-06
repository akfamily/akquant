"""broker_live 成交重放去重: 同 trade_id 每会话只派发一次, 补漏保留, 无 id 不去重."""

import threading

from akquant.gateway.broker_event_bridge import BrokerEventBridge


class _Strat:
    def __init__(self) -> None:
        self.trades: list = []
        self.grouped: list = []

    def on_trade(self, t: object) -> None:
        self.trades.append(t)

    def _process_order_groups(self, t: object) -> None:
        self.grouped.append(t)

    def on_order(self, o: object) -> None:
        pass


def _bridge(store: list) -> BrokerEventBridge:
    def safe(strategy: object, name: str, payload: object) -> None:
        fn = getattr(strategy, name, None)
        if fn is not None:
            fn(payload)

    return BrokerEventBridge(
        event_lock=threading.Lock(),
        event_store=store,
        event_keys=set(),
        get_on_broker_event=lambda: None,
        make_event_key=lambda n, p: (
            f"trade:{p.get('trade_id')}"
            if n == "trade" and p.get("trade_id")
            else f"{n}:{id(p)}"
        ),
        update_broker_state=lambda n, p: None,
        resolve_owner_strategy_id=lambda p: "",
        payload_to_dict=lambda p: dict(p),
        safe_strategy_callback=safe,
        adapt_strategy_payload=lambda n, p: p,
    )


def test_same_trade_id_dispatched_once_across_drains() -> None:
    """跨多次 drain 重放同一 trade_id, on_trade/_process_order_groups 只触发一次."""
    store: list = []
    b = _bridge(store)
    s = _Strat()
    b.queue_event("trade", {"trade_id": "T1", "symbol": "X"})
    b.drain_events(s)  # 派发一次
    b.queue_event("trade", {"trade_id": "T1", "symbol": "X"})  # 恢复重放
    b.drain_events(s)
    assert len(s.trades) == 1  # on_trade 只一次
    assert len(s.grouped) == 1  # _process_order_groups 只一次


def test_same_trade_id_dropped_within_batch() -> None:
    """同一 trade_id 在同一批(drain 之前)重复 queue_event, 第二次不入队."""
    store: list = []
    b = _bridge(store)
    b.queue_event("trade", {"trade_id": "T1", "symbol": "X"})
    b.queue_event("trade", {"trade_id": "T1", "symbol": "X"})
    assert len(store) == 1  # 第二次未入队


def test_missed_trade_still_dispatched_recovery_catch_up() -> None:
    """断线漏推、恢复补上的新 trade_id 仍会派发(补漏不受会话级去重影响)."""
    store: list = []
    b = _bridge(store)
    s = _Strat()
    # T1 已派发; T2 是断线漏推、恢复补上的新成交
    b.queue_event("trade", {"trade_id": "T1", "symbol": "X"})
    b.drain_events(s)
    b.queue_event("trade", {"trade_id": "T2", "symbol": "X"})  # 补漏
    b.drain_events(s)
    assert [t["trade_id"] for t in s.trades] == ["T1", "T2"]


def test_trade_without_id_not_deduped() -> None:
    """无 trade_id 的成交无法参与去重, 每次都照常入队."""
    store: list = []
    b = _bridge(store)
    b.queue_event("trade", {"symbol": "X"})  # 无 trade_id
    b.queue_event("trade", {"symbol": "X"})
    assert len(store) == 2  # 无 id 无法去重, 都入队


def test_distinct_trade_ids_not_dropped() -> None:
    """不同 trade_id 互不影响, 都正常入队."""
    store: list = []
    b = _bridge(store)
    b.queue_event("trade", {"trade_id": "A", "symbol": "X"})
    b.queue_event("trade", {"trade_id": "B", "symbol": "X"})
    assert len(store) == 2
