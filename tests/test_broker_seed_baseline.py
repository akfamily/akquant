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


def test_cold_start_pending_trade_discarded_no_double_count() -> None:
    """激活前入队的成交(已烘进快照)在激活时被丢弃, drain 不再 apply_fill 双计."""
    from akquant.gateway.broker_models import UnifiedPosition
    from akquant.gateway.broker_state_cache import BrokerStateCache
    from akquant.gateway.broker_strategy_api import wrap_state_invalidation

    class _G:
        def query_positions(self):
            return [
                UnifiedPosition(symbol="X", quantity=100.0, available_quantity=100.0)
            ]

    cache = BrokerStateCache(_G())
    caches = [cache]
    store: list = []
    # bridge 的 update_broker_state 经 wrap → 真正会 apply_fill 到 cache
    wrapped_update = wrap_state_invalidation(lambda n, p: None, lambda: caches)
    b = BrokerEventBridge(
        event_lock=threading.Lock(),
        event_store=store,
        event_keys=set(),
        get_on_broker_event=lambda: None,
        make_event_key=lambda n, p: (
            f"trade:{p.get('trade_id')}"
            if n == "trade" and p.get("trade_id")
            else f"{n}:{id(p)}"
        ),
        update_broker_state=wrapped_update,
        resolve_owner_strategy_id=lambda p: "",
        payload_to_dict=lambda p: dict(p) if isinstance(p, dict) else {},
        safe_strategy_callback=lambda s, n, p: None,
        adapt_strategy_payload=lambda n, p: p,
    )
    # 盘中重启: 激活前一笔 live push T 已入队
    b.queue_event(
        "trade", {"trade_id": "T", "symbol": "X", "side": "Buy", "quantity": 100.0}
    )
    # 激活: 先丢弃待派发成交(T 已在快照里), 再 seed
    b.discard_pending_trades()
    assert cache.positions()["X"] == 100.0  # seed 快照(含 T)
    # drain: T 已被丢弃, 不会 apply_fill
    b.drain_events(object())
    assert cache.positions()["X"] == 100.0  # 未双计(改前: 200)


def test_baseline_broker_state_seeds_and_marks_seen() -> None:
    """_baseline_broker_state: seed 各 cache + 用 sync_today_trades 灌 seen + 置标志."""
    from akquant.live import LiveRunner

    class _Cache:
        def __init__(self) -> None:
            self.seeded = 0

        def positions(self):
            self.seeded += 1
            return {}

    class _GwB:
        def sync_today_trades(self):
            return [type("T", (), {"trade_id": "s1"})()]

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ptrade"
    runner._init_broker_bridge_state()
    assert runner._broker_baseline_done is False
    cache = _Cache()
    runner._broker_runtime._broker_state_caches.append(cache)
    runner._baseline_broker_state(_GwB())
    assert cache.seeded == 1  # 急切 seed
    assert "s1" in runner._broker_event_bridge._seen_trade_ids  # 灌基线
    assert runner._broker_baseline_done is True
