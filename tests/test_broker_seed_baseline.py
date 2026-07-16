"""Cold-start baseline tests for broker bridge dedup and recovery trade replay."""

import threading
from typing import Any, Callable

from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.gateway.broker_recovery import BrokerRecovery


def _bridge(store: list[tuple[str, dict[str, Any]]]) -> BrokerEventBridge:
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
    """Already-seen trades should be dropped while unseen ones still queue."""
    store: list[tuple[str, dict[str, Any]]] = []
    b = _bridge(store)
    b.mark_trades_seen(["T", ""])
    b.queue_event("trade", {"trade_id": "T", "symbol": "X"})
    b.queue_event("trade", {"trade_id": "U", "symbol": "X"})
    assert [p["trade_id"] for _, p in store] == ["U"]


class _Gw:
    def __init__(self) -> None:
        self.trades = [type("T", (), {"trade_id": "t1"})()]

    def heartbeat(self) -> bool:
        return True

    def sync_open_orders(self) -> list[object]:
        return []

    def sync_today_trades(self) -> list[object]:
        return list(self.trades)

    def query_account(self) -> None:
        return None


def _recovery(
    gw: _Gw, should_replay: Callable[[], bool]
) -> tuple[BrokerRecovery, list[tuple[str, Any]]]:
    queued: list[tuple[str, Any]] = []
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
    """Trade replay should follow the should_replay_trades gate."""
    gw = _Gw()
    rec, queued = _recovery(gw, should_replay=lambda: False)
    rec.run_cycle()
    assert not any(n == "trade" for n, _ in queued)

    rec2, queued2 = _recovery(gw, should_replay=lambda: True)
    rec2.run_cycle()
    assert any(n == "trade" for n, _ in queued2)


def test_recovery_default_replays_when_no_gate() -> None:
    """Without an explicit gate, recovery should still replay trades."""
    gw = _Gw()
    queued: list[tuple[str, Any]] = []
    rec = BrokerRecovery(
        get_trader_gateway=lambda: gw,
        queue_broker_event=lambda n, p: queued.append((n, p)),
        notify_strategy_error=lambda *a: None,
        get_on_broker_event=lambda: None,
        get_recovery_mode=lambda: "compatible",
        get_last_error_key=lambda: "",
        set_last_error_key=lambda k: None,
    )
    rec.run_cycle()
    assert any(n == "trade" for n, _ in queued)


def test_cold_start_pending_trade_discarded_no_double_count() -> None:
    """Discarded pending trades should not apply their fill twice after seeding."""
    from akquant.gateway.broker_models import UnifiedPosition
    from akquant.gateway.broker_state_cache import BrokerStateCache
    from akquant.gateway.broker_strategy_api import wrap_state_invalidation

    class _G:
        def query_positions(self) -> list[UnifiedPosition]:
            return [
                UnifiedPosition(symbol="X", quantity=100.0, available_quantity=100.0)
            ]

    cache = BrokerStateCache(_G())
    caches = [cache]
    store: list[tuple[str, dict[str, Any]]] = []
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
    b.queue_event(
        "trade", {"trade_id": "T", "symbol": "X", "side": "Buy", "quantity": 100.0}
    )
    b.discard_pending_trades()
    assert cache.positions()["X"] == 100.0
    b.drain_events(object())
    assert cache.positions()["X"] == 100.0


def test_baseline_broker_state_seeds_and_marks_seen() -> None:
    """Baseline seeding should warm caches, mark seen trades, and set the flag."""
    from akquant.live import LiveRunner

    class _Cache:
        def __init__(self) -> None:
            self.seeded = 0

        def positions(self) -> dict[str, float]:
            self.seeded += 1
            return {}

    class _GwB:
        def sync_today_trades(self) -> list[object]:
            return [type("T", (), {"trade_id": "s1"})()]

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ptrade"
    runner._init_broker_bridge_state()
    assert runner._broker_baseline_done is False
    cache = _Cache()
    runner._broker_runtime._broker_state_caches.append(cache)
    runner._baseline_broker_state(_GwB())
    assert cache.seeded == 1
    assert "s1" in runner._broker_event_bridge._seen_trade_ids
    assert runner._broker_baseline_done is True
