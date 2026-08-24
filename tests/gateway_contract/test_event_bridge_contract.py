import threading
from typing import Any

from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.live._payload_utils import payload_field


class _DummyStrategy:
    def __init__(self) -> None:
        self.orders: list[Any] = []
        self.trades: list[Any] = []
        self.portfolio_updates: list[Any] = []

    def on_order(self, order: Any) -> None:
        self.orders.append(order)

    def on_trade(self, trade: Any) -> None:
        self.trades.append(trade)

    def _process_order_groups(self, trade: Any) -> None:
        # broker_live 成交后桥会额外驱动协调器; 本契约桩无 OCO/bracket 组, no-op。
        pass

    def on_portfolio_update(self, payload: Any) -> None:
        self.portfolio_updates.append(payload)


def _build_event_bridge() -> tuple[
    BrokerEventBridge,
    _DummyStrategy,
    list[tuple[str, Any]],
    list[dict[str, Any]],
]:
    event_store: list[tuple[str, Any]] = []
    update_calls: list[tuple[str, Any]] = []
    observed_events: list[dict[str, Any]] = []
    strategy = _DummyStrategy()
    bridge = BrokerEventBridge(
        event_lock=threading.Lock(),
        event_store=event_store,
        event_keys=set(),
        get_on_broker_event=lambda: observed_events.append,
        make_event_key=lambda event_name, payload: (
            f"{event_name}:{payload.get('id', '')}:{payload.get('status', '')}"
        ),
        update_broker_state=lambda event_name, payload: update_calls.append(
            (event_name, payload)
        ),
        resolve_owner_strategy_id=lambda payload: payload.get(
            "owner_strategy_id", "_default"
        ),
        payload_to_dict=lambda payload: dict(payload),
        safe_strategy_callback=lambda target, callback_name, payload: getattr(
            target,
            callback_name,
        )(payload),
        adapt_strategy_payload=lambda event_name, payload: payload,
        payload_field=payload_field,
    )
    return bridge, strategy, update_calls, observed_events


def test_event_bridge_contract_deduplicates_and_dispatches_semantic_events() -> None:
    """Event bridge should deduplicate semantic duplicates before dispatch."""
    bridge, strategy, update_calls, observed_events = _build_event_bridge()
    duplicate_order = {
        "id": "b-1",
        "status": "Submitted",
        "owner_strategy_id": "alpha",
    }

    bridge.queue_event("order", duplicate_order)
    bridge.queue_event("order", dict(duplicate_order))
    bridge.queue_event(
        "trade",
        {"id": "t-1", "owner_strategy_id": "alpha", "symbol": "IF2406"},
    )
    bridge.drain_events(strategy)

    assert len(update_calls) == 2
    assert len(strategy.orders) == 1
    assert len(strategy.trades) == 1
    assert len(observed_events) == 2
    assert observed_events[0]["event_type"] == "order"
    assert observed_events[1]["event_type"] == "trade"


def test_event_bridge_contract_emits_owner_and_account_updates() -> None:
    """Event bridge should normalize observer snapshots and account callbacks."""
    bridge, strategy, _, observed_events = _build_event_bridge()
    account_payload = {
        "id": "acct-1",
        "account_id": "acct-1",
        "owner_strategy_id": "beta",
        "equity": 120000.0,
    }

    bridge.queue_event("account", account_payload)
    bridge.drain_events(strategy)

    assert len(strategy.portfolio_updates) == 1
    assert strategy.portfolio_updates[0]["account_id"] == "acct-1"
    assert observed_events == [
        {
            "event_type": "account",
            "owner_strategy_id": "beta",
            "payload": account_payload,
        }
    ]
