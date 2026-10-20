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


def _to_text(value: Any) -> str:
    """None/缺失归一成空串, 其余去首尾空白."""
    return str(value or "").strip()


def _to_phase(value: Any) -> str:
    text = str(value or "").strip()
    return text or "run"


def _series_key(raw: Any, indicator_key: str) -> str:
    """取 series_key, 缺失或为空时回落到 indicator_key(兼容 1.4 事件)."""
    text = str(raw.get("series_key") or "").strip() if isinstance(raw, dict) else ""
    return text or indicator_key


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
        point_key = str(payload.get("indicator_key", ""))
        base_message["indicator"] = {
            "owner_strategy_id": str(payload.get("owner_strategy_id", "")),
            "indicator_key": point_key,
            # 图上的 series 名, 与历史接口 series_keys 同规则(1.5);
            # 1.4 事件与第三方 sink 没有它, 回落到 indicator_key。
            "series_key": _series_key(payload, point_key),
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
            # 缺失时默认 run: 1.4 之前的事件与旧协议 sink 都不来自预热回放。
            "phase": _to_phase(payload.get("phase")),
            # candle 分量的组名与分量名(1.6); 其余 render_type 与 1.5 事件为空串。
            "group": _to_text(payload.get("group")),
            "role": _to_text(payload.get("role")),
            "meta": _json_loads_or_default(payload.get("meta_json"), {}),
        }
        return base_message

    raw_items = _json_loads_or_default(payload.get("items_json"), [])
    snapshot_owner = str(payload.get("owner_strategy_id", ""))
    items: list[dict[str, Any]] = []
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            item_key = str(raw_item.get("indicator_key", ""))
            items.append(
                {
                    # 1.5 起 item 自带 owner; 1.4 事件回落到快照的 owner
                    # (一份快照只含一个 owner 的点)。
                    "owner_strategy_id": str(
                        raw_item.get("owner_strategy_id") or snapshot_owner
                    ),
                    "indicator_key": item_key,
                    "series_key": _series_key(raw_item, item_key),
                    "display_name": str(raw_item.get("display_name", "")),
                    "pane": _to_int(raw_item.get("pane", 0)),
                    "render_type": str(raw_item.get("render_type", "")),
                    "value": _to_float_or_text(raw_item.get("value")),
                    "warmup": _to_bool(raw_item.get("warmup", False)),
                    "confirmed": _to_bool(raw_item.get("confirmed", True)),
                    "phase": _to_phase(raw_item.get("phase")),
                    "group": _to_text(raw_item.get("group")),
                    "role": _to_text(raw_item.get("role")),
                    "meta": _json_loads_or_default(raw_item.get("meta_json"), {}),
                }
            )

    keyed_items = [item for item in items if item["indicator_key"]]
    indicator_keys = [item["indicator_key"] for item in keyed_items]
    # 与 indicator_keys 一一对应; 两个 owner 同名时 series_key 不同, 下面两个
    # 映射按它作键才不会互相覆盖(1.5)。
    series_keys = [item["series_key"] for item in keyed_items]
    panes = sorted({item["pane"] for item in items})
    render_types = sorted(
        {item["render_type"] for item in items if item["render_type"]}
    )
    value_by_key = {item["series_key"]: item["value"] for item in keyed_items}
    items_by_key = {item["series_key"]: item for item in keyed_items}
    warmup_count = sum(1 for item in items if bool(item.get("warmup", False)))

    timestamp_ns = _to_int(payload.get("timestamp", 0))
    base_message["snapshot"] = {
        "owner_strategy_id": snapshot_owner,
        "symbol": _normalize_symbol(payload.get("symbol")),
        "timestamp": timestamp_ns,
        # Milliseconds alongside the nanosecond value, so a snapshot can be fed
        # straight to a chart library. Items share the snapshot timestamp.
        "timestamp_ms": _timestamp_ms(payload, timestamp_ns),
        "indicator_count": _to_int(payload.get("indicator_count", len(items))),
        "confirmed": _to_bool(payload.get("confirmed", "true")),
        "phase": _to_phase(payload.get("phase")),
        "items": items,
        "indicator_keys": indicator_keys,
        "series_keys": series_keys,
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
