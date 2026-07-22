import json
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

from .chart import (
    normalize_meta_json as _normalize_meta_json,
)
from .chart import (
    normalize_pane_index,
    normalize_reference_lines,
    normalize_render_type,
    normalize_scale_group,
    timestamp_ms_from_ns,
    timestamp_to_ms_and_ns,
)


@runtime_checkable
class IndicatorSink(Protocol):
    """Public protocol for objects that collect indicator points during a run.

    Pass an implementation to ``run_backtest(indicator_recorder=...)`` to route
    every ``Strategy.record_indicator`` call into your own collector instead of
    the built-in :class:`IndicatorRecorder`. This is the supported extension
    point for chart backends and frontends: implement ``record`` to capture
    points and ``build_payload`` to return the definitions/instances/points
    payload attached to :class:`BacktestResult.indicator_outputs`.

    The built-in :class:`IndicatorRecorder` satisfies this protocol.
    """

    def record(
        self,
        *,
        name: str,
        value: Any,
        symbol: str,
        timestamp: Any,
        owner_strategy_id: str,
        display_name: Optional[str] = ...,
        pane: int = ...,
        render_type: str = ...,
        unit: Optional[str] = ...,
        precision: Optional[int] = ...,
        color: Optional[str] = ...,
        meta: Optional[Dict[str, Any]] = ...,
        reference_lines: Optional[list[Dict[str, Any]]] = ...,
        scale_group: Optional[str] = ...,
        warmup: bool = ...,
    ) -> None:
        """Record one indicator point."""
        ...

    def build_payload(self) -> Dict[str, list[Dict[str, Any]]]:
        """Return the row-oriented definitions/instances/points payload."""
        ...

    def flush_stream_snapshot(self) -> None:
        """Emit any buffered stream snapshot for the current callback cycle."""
        ...

    def set_stream_emitter(
        self,
        stream_emitter: Optional[
            Callable[[str, Optional[str], str, Dict[str, str]], None]
        ],
    ) -> None:
        """Attach or replace the runtime stream emitter."""
        ...


def _normalize_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _normalize_timestamp_ns(timestamp: Any) -> int:
    _, timestamp_ns = timestamp_to_ms_and_ns(timestamp)
    return timestamp_ns


class IndicatorRecorder:
    """Collect normalized indicator metadata and point values during a run."""

    def __init__(
        self,
        stream_emitter: Optional[
            Callable[[str, Optional[str], str, Dict[str, str]], None]
        ] = None,
    ) -> None:
        """Initialize recorder state and optional stream emitter."""
        self._definitions: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        self._points: list[Dict[str, Any]] = []
        self._stream_emitter = stream_emitter
        self._pending_snapshots: Dict[Tuple[str, str, int], list[Dict[str, Any]]] = {}

    @property
    def has_data(self) -> bool:
        """Return whether any indicator point has been recorded."""
        return bool(self._points)

    def set_stream_emitter(
        self,
        stream_emitter: Optional[
            Callable[[str, Optional[str], str, Dict[str, str]], None]
        ],
    ) -> None:
        """Attach or replace the runtime stream emitter."""
        self._stream_emitter = stream_emitter

    def record(
        self,
        *,
        name: str,
        value: Any,
        symbol: str,
        timestamp: Any,
        owner_strategy_id: str,
        display_name: Optional[str] = None,
        pane: int = 0,
        render_type: str = "line",
        unit: Optional[str] = None,
        precision: Optional[int] = None,
        color: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        reference_lines: Optional[list[Dict[str, Any]]] = None,
        scale_group: Optional[str] = None,
        warmup: bool = False,
    ) -> None:
        """Record one indicator definition, instance, and point update."""
        indicator_key = _normalize_text(name)
        if not indicator_key:
            raise ValueError("indicator name cannot be empty")

        symbol_text = _normalize_text(symbol, default="_unknown")
        strategy_id = _normalize_text(owner_strategy_id, default="_default")
        pane_index = normalize_pane_index(pane)
        render_type_text = normalize_render_type(render_type)
        display_name_text = _normalize_text(display_name, default=indicator_key)
        unit_text = _normalize_text(unit)
        color_text = _normalize_text(color)
        meta_json = _normalize_meta_json(meta)
        reference_lines_norm = normalize_reference_lines(reference_lines)
        scale_group_norm = normalize_scale_group(scale_group)
        timestamp_ns = _normalize_timestamp_ns(timestamp)
        timestamp_ms = timestamp_ms_from_ns(timestamp_ns)

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = float("nan")

        definition = self._definitions.get(indicator_key)
        if definition is None:
            self._definitions[indicator_key] = {
                "indicator_key": indicator_key,
                "display_name": display_name_text,
                "pane": pane_index,
                "render_type": render_type_text,
                "unit": unit_text,
                "precision": precision,
                "color": color_text,
                "reference_lines": reference_lines_norm,
                "scale_group": scale_group_norm,
            }
        else:
            if not definition.get("display_name"):
                definition["display_name"] = display_name_text
            if definition.get("pane") is None:
                definition["pane"] = pane_index
            if not definition.get("render_type"):
                definition["render_type"] = render_type_text
            if not definition.get("unit"):
                definition["unit"] = unit_text
            if definition.get("precision") is None and precision is not None:
                definition["precision"] = precision
            if not definition.get("color"):
                definition["color"] = color_text
            if not definition.get("reference_lines") and reference_lines_norm:
                definition["reference_lines"] = reference_lines_norm
            if not definition.get("scale_group") and scale_group_norm:
                definition["scale_group"] = scale_group_norm

        instance_key = (strategy_id, symbol_text, indicator_key, meta_json)
        instance = self._instances.get(instance_key)
        if instance is None:
            instance = {
                "instance_id": f"ins_{len(self._instances) + 1}",
                "owner_strategy_id": strategy_id,
                "symbol": symbol_text,
                "indicator_key": indicator_key,
                "meta_json": meta_json,
            }
            self._instances[instance_key] = instance

        self._points.append(
            {
                "instance_id": instance["instance_id"],
                "owner_strategy_id": strategy_id,
                "symbol": symbol_text,
                "indicator_key": indicator_key,
                "timestamp": timestamp_ns,
                "timestamp_ms": timestamp_ms,
                "value": numeric_value,
                "warmup": bool(warmup),
            }
        )
        if self._stream_emitter is not None:
            self._stream_emitter(
                "indicator_point",
                None if symbol_text == "_unknown" else symbol_text,
                "info",
                {
                    "owner_strategy_id": strategy_id,
                    "indicator_key": indicator_key,
                    "display_name": display_name_text,
                    "pane": str(pane_index),
                    "render_type": render_type_text,
                    "symbol": symbol_text,
                    "timestamp": str(timestamp_ns),
                    "timestamp_ms": str(timestamp_ms),
                    "value": repr(numeric_value),
                    "warmup": str(bool(warmup)).lower(),
                    "scale_group": scale_group_norm,
                    "meta_json": meta_json,
                },
            )
            snapshot_key = (strategy_id, symbol_text, timestamp_ns)
            snapshot_items = self._pending_snapshots.setdefault(snapshot_key, [])
            snapshot_items.append(
                {
                    "indicator_key": indicator_key,
                    "display_name": display_name_text,
                    "pane": pane_index,
                    "render_type": render_type_text,
                    "value": numeric_value,
                    "warmup": bool(warmup),
                    "meta_json": meta_json,
                }
            )

    def flush_stream_snapshot(self) -> None:
        """Emit pending indicator snapshots for the latest callback cycle."""
        if self._stream_emitter is None or not self._pending_snapshots:
            self._pending_snapshots.clear()
            return
        for (strategy_id, symbol_text, timestamp_ns), items in list(
            self._pending_snapshots.items()
        ):
            self._stream_emitter(
                "indicator_snapshot",
                None if symbol_text == "_unknown" else symbol_text,
                "info",
                {
                    "owner_strategy_id": strategy_id,
                    "symbol": symbol_text,
                    "timestamp": str(timestamp_ns),
                    "timestamp_ms": str(timestamp_ms_from_ns(timestamp_ns)),
                    "indicator_count": str(len(items)),
                    "items_json": json.dumps(
                        items,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                },
            )
        self._pending_snapshots.clear()

    def build_payload(self) -> Dict[str, list[Dict[str, Any]]]:
        """Return stable row-oriented outputs for result access and export."""
        definitions = [
            self._definitions[key] for key in sorted(self._definitions.keys(), key=str)
        ]
        instances = sorted(
            self._instances.values(),
            key=lambda row: (
                str(row.get("owner_strategy_id", "")),
                str(row.get("symbol", "")),
                str(row.get("indicator_key", "")),
                str(row.get("instance_id", "")),
            ),
        )
        points = sorted(
            self._points,
            key=lambda row: (
                str(row.get("owner_strategy_id", "")),
                str(row.get("symbol", "")),
                str(row.get("indicator_key", "")),
                int(row.get("timestamp", 0)),
            ),
        )
        return {
            "definitions": definitions,
            "instances": instances,
            "points": points,
        }
