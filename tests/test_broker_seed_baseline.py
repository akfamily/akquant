"""冷启动基线: bridge.mark_trades_seen 灌 dedup; recovery 成交重放门控."""

import threading

from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.gateway.broker_recovery import BrokerRecovery


def _bridge(store: list) -> BrokerEventBridge:
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
        payload_to_dict=lambda p: dict(p) if isinstance(p, dict) else {},
        safe_strategy_callback=lambda s, n, p: None,
        adapt_strategy_payload=lambda n, p: p,
    )


def test_mark_trades_seen_drops_subsequent_queue() -> None:
    """已标记的 trade_id 再次入队时被丢弃, 未标记的正常入队."""
    store: list = []
    b = _bridge(store)
    b.mark_trades_seen(["T", "", None])  # 空/None 跳过
    b.queue_event("trade", {"trade_id": "T", "symbol": "X"})  # 已 seen -> 丢
    b.queue_event("trade", {"trade_id": "U", "symbol": "X"})  # 新 -> 入队
    assert [p["trade_id"] for _, p in store] == ["U"]


class _Gw:
    def __init__(self) -> None:
        self.trades = [type("T", (), {"trade_id": "t1"})()]

    def heartbeat(self) -> bool:
        return True

    def sync_open_orders(self):
        return []

    def sync_today_trades(self):
        return list(self.trades)

    def query_account(self):
        return None


def _recovery(gw, should_replay):
    queued: list = []
    rec = BrokerRecovery(
        get_trader_gateway=lambda: gw,
        queue_broker_event=lambda n, p: queued.append((n, p)),
        notify_strategy_error=lambda *a: None,
        get_on_broker_event=lambda: None,
        get_recovery_mode=lambda: "compatible",
        get_last_error_key=lambda: "",
        set_last_error_key=lambda k: None,
        should_replay_trades=should_replay,
    )
    return rec, queued


def test_recovery_gates_trade_replay_until_baseline() -> None:
    """should_replay_trades 为 False 时不重放成交, 为 True 时重放."""
    gw = _Gw()
    rec, queued = _recovery(gw, should_replay=lambda: False)  # 未激活
    rec.run_cycle()
    assert not any(n == "trade" for n, _ in queued)  # 不重放成交
    rec2, queued2 = _recovery(gw, should_replay=lambda: True)  # 已激活
    rec2.run_cycle()
    assert any(n == "trade" for n, _ in queued2)  # 重放


def test_recovery_default_replays_when_no_gate() -> None:
    """未传 should_replay_trades 时默认重放成交(后向兼容)."""
    gw = _Gw()
    queued: list = []
    rec = BrokerRecovery(
        get_trader_gateway=lambda: gw,
        queue_broker_event=lambda n, p: queued.append((n, p)),
        notify_strategy_error=lambda *a: None,
        get_on_broker_event=lambda: None,
        get_recovery_mode=lambda: "compatible",
        get_last_error_key=lambda: "",
        set_last_error_key=lambda k: None,
    )  # 无 should_replay_trades -> 默认 True(后向兼容)
    rec.run_cycle()
    assert any(n == "trade" for n, _ in queued)
