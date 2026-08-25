"""BrokerEventBridge trade 派发后驱动 _process_order_groups(异常隔离)."""

import threading

from akquant.gateway.broker_event_bridge import BrokerEventBridge
from akquant.live._payload_utils import payload_field


class _Strat:
    def __init__(self) -> None:
        self.trades: list = []
        self.grouped: list = []
        self.errors: list = []

    def on_trade(self, t: object) -> None:
        self.trades.append(t)

    def _process_order_groups(self, t: object) -> None:
        self.grouped.append(t)

    def on_order(self, o: object) -> None:
        pass

    def on_error(self, exc: object, source: str, payload: object = None) -> None:
        self.errors.append((source, payload))


def _bridge() -> BrokerEventBridge:
    def safe(strategy: object, name: str, payload: object) -> None:
        try:
            getattr(strategy, name)(payload)
        except Exception as exc:  # noqa: BLE001
            on_err = getattr(strategy, "on_error", None)
            if on_err:
                on_err(exc, name, payload)

    return BrokerEventBridge(
        event_lock=threading.Lock(),
        event_store=[],
        event_keys=set(),
        get_on_broker_event=lambda: None,
        make_event_key=lambda n, p: f"{n}:{id(p)}",
        update_broker_state=lambda n, p: None,
        resolve_owner_strategy_id=lambda p: "",
        payload_to_dict=lambda p: {},
        safe_strategy_callback=safe,
        adapt_strategy_payload=lambda n, p: p,
        payload_field=payload_field,
    )


def test_trade_drives_process_order_groups() -> None:
    """成交派发后同一 payload 驱动 on_trade 与 _process_order_groups."""
    b = _bridge()
    s = _Strat()
    b.queue_event("trade", {"broker_order_id": "B1"})
    b.queue_event("order", {"broker_order_id": "B2"})
    b.drain_events(s)
    assert len(s.trades) == 1 and len(s.grouped) == 1  # trade drives both
    assert s.grouped[0] is s.trades[0]  # same adapted payload


def test_process_order_groups_error_isolated() -> None:
    """协调器抛错经 on_error 隔离, drain 不崩."""
    b = _bridge()

    class _BadStrat(_Strat):
        def _process_order_groups(self, t: object) -> None:
            raise RuntimeError("boom")

    s = _BadStrat()
    b.queue_event("trade", {"broker_order_id": "B1"})
    b.drain_events(s)  # must not raise
    assert s.errors and s.errors[0][0] == "_process_order_groups"
