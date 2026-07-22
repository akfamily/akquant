"""Helpers for bridging order/trade stream events to frontend-friendly messages.

The engine already emits ``order`` and ``trade`` stream events (see
``src/pipeline/stages/shared.rs``). This module is the trade-side counterpart of
:mod:`akquant.indicator_stream`: it normalizes those raw events into the same
message envelope the indicator bridge uses, so a single frontend consumer can
draw entry/exit markers on the price pane alongside recorded indicators.

Envelope keys (shared with the indicator bridge): ``channel``, ``type``,
``run_id``, ``seq``, ``ts``, ``symbol``, ``level``. Trade-specific payload lives
under ``fill`` (for ``trade`` events) or ``order`` (for ``order`` events), plus a
second-precision ``time`` field ready for chart marker placement.
"""

from typing import Any, Iterable, Optional

from .backtest import BacktestStreamEvent
from .stream_schema import STREAM_SCHEMA_VERSION

_TRADE_EVENT_TYPES = {"order", "trade"}


def _to_int(value: Any, default: int = 0) -> int:
    """Convert values into integers with a safe fallback."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert values into floats with a safe fallback."""
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _normalize_symbol(value: Any) -> Optional[str]:
    """Normalize unknown or empty symbols into None."""
    if value in (None, "", "_unknown"):
        return None
    return str(value)


def _event_time_seconds(event: BacktestStreamEvent) -> int:
    """Second-precision time for chart markers, derived from the event ns ts."""
    ts_ns = _to_int(event.get("ts", 0))
    return ts_ns // 1_000_000_000


def is_trade_stream_event(event: BacktestStreamEvent) -> bool:
    """Return whether the event is an order or trade stream event."""
    return str(event.get("event_type", "")) in _TRADE_EVENT_TYPES


def to_trade_message(event: BacktestStreamEvent) -> Optional[dict[str, Any]]:
    """Convert one order/trade stream event into a frontend-friendly message.

    :param event: A backtest stream event.
    :return: A normalized message dict, or ``None`` if the event is neither an
        ``order`` nor a ``trade`` event.
    """
    event_type = str(event.get("event_type", ""))
    if event_type not in _TRADE_EVENT_TYPES:
        return None

    payload = event.get("payload", {})
    symbol = _normalize_symbol(event.get("symbol"))

    base_message: dict[str, Any] = {
        "channel": "trade",
        "type": event_type,
        "run_id": str(event.get("run_id", "")),
        "seq": _to_int(event.get("seq", 0)),
        "ts": _to_int(event.get("ts", 0)),
        "symbol": symbol,
        "level": str(event.get("level", "info")),
        "schema_version": STREAM_SCHEMA_VERSION,
        "time": _event_time_seconds(event),
    }

    if event_type == "trade":
        base_message["fill"] = {
            "trade_id": str(payload.get("trade_id", "")),
            "order_id": str(payload.get("order_id", "")),
            "owner_strategy_id": str(payload.get("owner_strategy_id", "")),
            "symbol": _normalize_symbol(payload.get("symbol")) or symbol,
            "side": str(payload.get("side", "")),
            "price": _to_float(payload.get("price")),
            "quantity": _to_float(payload.get("quantity")),
        }
        return base_message

    base_message["order"] = {
        "order_id": str(payload.get("order_id", "")),
        "owner_strategy_id": str(payload.get("owner_strategy_id", "")),
        "symbol": _normalize_symbol(payload.get("symbol")) or symbol,
        "status": str(payload.get("status", "")),
        "filled_qty": _to_float(payload.get("filled_qty")),
    }
    return base_message


def to_trade_messages(
    events: Iterable[BacktestStreamEvent],
) -> list[dict[str, Any]]:
    """Convert an event iterable into order/trade-only frontend messages.

    :param events: An iterable of backtest stream events.
    :return: The normalized order/trade messages, preserving input order.
    """
    messages: list[dict[str, Any]] = []
    for event in events:
        message = to_trade_message(event)
        if message is not None:
            messages.append(message)
    return messages
