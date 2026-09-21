"""Helpers for bridging indicator stream events to frontend-friendly messages."""

import json
from typing import Any, Iterable, Optional

from .backtest import BacktestStreamEvent
from .stream_schema import STREAM_SCHEMA_VERSION


def _to_int(value: Any, default: int = 0) -> int:
    """Convert values into integers with a safe fallback."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _to_float_or_text(value: Any) -> Any:
    """Prefer numeric values for charts while keeping unparseable text intact."""
    if value is None:
        return None
    text = str(value)
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def _to_bool(value: Any) -> bool:
    """Parse common string boolean forms."""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _json_loads_or_default(value: Any, default: Any) -> Any:
    """Decode JSON strings and fall back to the provided default."""
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _normalize_symbol(value: Any) -> Optional[str]:
    """Normalize unknown or empty symbols into None."""
    if value in (None, "", "_unknown"):
        return None
    return str(value)


def _timestamp_ms(payload: Any, timestamp_ns: int) -> int:
    """Resolve the millisecond timestamp, deriving it when the payload omits it.

    The collection layers always emit ``timestamp_ms``, but events captured
    before the field existed — and third-party :class:`IndicatorSink`
    implementations — only carry the nanosecond ``timestamp``. Deriving keeps the
    bridge aligned with ``BacktestResult._indicator_points_df``, which fills the
    same gap for legacy payloads.
    """
    if isinstance(payload, dict) and payload.get("timestamp_ms") not in (None, ""):
        return _to_int(payload.get("timestamp_ms"), timestamp_ns // 1_000_000)
    return timestamp_ns // 1_000_000


def is_indicator_stream_event(event: BacktestStreamEvent) -> bool:
    """Return whether the event belongs to indicator streaming."""
    return str(event.get("event_type", "")) in {"indicator_point", "indicator_snapshot"}


def to_indicator_message(event: BacktestStreamEvent) -> Optional[dict[str, Any]]:
    """Convert one indicator stream event into a frontend-friendly message."""
    event_type = str(event.get("event_type", ""))
    if event_type not in {"indicator_point", "indicator_snapshot"}:
        return None

    payload = event.get("payload", {})
    symbol = _normalize_symbol(event.get("symbol"))

    base_message: dict[str, Any] = {
        "channel": "indicator",
        "type": "point" if event_type == "indicator_point" else "snapshot",
        "run_id": str(event.get("run_id", "")),
        "seq": _to_int(event.get("seq", 0)),
        "ts": _to_int(event.get("ts", 0)),
        "symbol": symbol,
        "level": str(event.get("level", "info")),
        "schema_version": STREAM_SCHEMA_VERSION,
    }

    if event_type == "indicator_point":
        timestamp_ns = _to_int(payload.get("timestamp", 0))
        base_message["indicator"] = {
            "owner_strategy_id": str(payload.get("owner_strategy_id", "")),
            "indicator_key": str(payload.get("indicator_key", "")),
            "display_name": str(payload.get("display_name", "")),
            "pane": _to_int(payload.get("pane", 0)),
            "render_type": str(payload.get("render_type", "")),
            "symbol": _normalize_symbol(payload.get("symbol")),
            "timestamp": timestamp_ns,
            # ``timestamp_ms`` mirrors ``timestamp`` for frontend charting
            # libraries that expect epoch milliseconds, matching the DataFrame
            # and export exits of the same ``record_indicator`` call.
            "timestamp_ms": _timestamp_ms(payload, timestamp_ns),
            "value": _to_float_or_text(payload.get("value")),
            "scale_group": str(payload.get("scale_group", "")),
            "warmup": _to_bool(payload.get("warmup", False)),
            # 缺失时默认 True: 早期事件与第三方 IndicatorSink 都只在 bar
            # 闭合后产点, 把它们当作已确认是正确的语义。
            "confirmed": _to_bool(payload.get("confirmed", "true")),
            "meta": _json_loads_or_default(payload.get("meta_json"), {}),
        }
        return base_message

    raw_items = _json_loads_or_default(payload.get("items_json"), [])
    items: list[dict[str, Any]] = []
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            items.append(
                {
                    "indicator_key": str(raw_item.get("indicator_key", "")),
                    "display_name": str(raw_item.get("display_name", "")),
                    "pane": _to_int(raw_item.get("pane", 0)),
                    "render_type": str(raw_item.get("render_type", "")),
                    "value": _to_float_or_text(raw_item.get("value")),
                    "warmup": _to_bool(raw_item.get("warmup", False)),
                    "confirmed": _to_bool(raw_item.get("confirmed", True)),
                    "meta": _json_loads_or_default(raw_item.get("meta_json"), {}),
                }
            )

    indicator_keys = [item["indicator_key"] for item in items if item["indicator_key"]]
    panes = sorted({item["pane"] for item in items})
    render_types = sorted(
        {item["render_type"] for item in items if item["render_type"]}
    )
    value_by_key = {
        item["indicator_key"]: item["value"] for item in items if item["indicator_key"]
    }
    items_by_key = {
        item["indicator_key"]: item for item in items if item["indicator_key"]
    }
    warmup_count = sum(1 for item in items if bool(item.get("warmup", False)))

    timestamp_ns = _to_int(payload.get("timestamp", 0))
    base_message["snapshot"] = {
        "owner_strategy_id": str(payload.get("owner_strategy_id", "")),
        "symbol": _normalize_symbol(payload.get("symbol")),
        "timestamp": timestamp_ns,
        # Milliseconds alongside the nanosecond value, so a snapshot can be fed
        # straight to a chart library. Items share the snapshot timestamp.
        "timestamp_ms": _timestamp_ms(payload, timestamp_ns),
        "indicator_count": _to_int(payload.get("indicator_count", len(items))),
        "confirmed": _to_bool(payload.get("confirmed", "true")),
        "items": items,
        "indicator_keys": indicator_keys,
        "panes": panes,
        "render_types": render_types,
        "value_by_key": value_by_key,
        "items_by_key": items_by_key,
        "warmup_count": warmup_count,
        "has_warmup": warmup_count > 0,
    }
    return base_message


def to_indicator_messages(
    events: Iterable[BacktestStreamEvent],
) -> list[dict[str, Any]]:
    """Convert an event iterable into indicator-only frontend messages."""
    messages: list[dict[str, Any]] = []
    for event in events:
        message = to_indicator_message(event)
        if message is not None:
            messages.append(message)
    return messages
