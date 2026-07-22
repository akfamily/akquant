"""Lightweight streaming indicator sink for live sessions.

Unlike the backtest :class:`IndicatorRecorder`, this sink never accumulates
points in memory — a live session is a long-running process and unbounded
accumulation would leak. It only emits ``indicator_point`` / ``indicator_snapshot``
stream events (via an ``on_event`` callback) so a frontend can draw indicators
incrementally, and returns an empty payload for the batch-oriented API.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..backtest import BacktestStreamEvent
from ..chart import (
    normalize_meta_json,
    normalize_pane_index,
    normalize_render_type,
    timestamp_ms_from_ns,
    timestamp_to_ms_and_ns,
)


class StreamingIndicatorSink:
    """An :class:`akquant.IndicatorSink` that streams without accumulating.

    :param on_event: Callback receiving fully-formed indicator stream events.
    :param run_id: Stable run id stamped onto every emitted event.
    """

    def __init__(
        self,
        on_event: Callable[[BacktestStreamEvent], None],
        *,
        run_id: str = "live",
    ) -> None:
        self._on_event = on_event
        self._run_id = str(run_id)
        self._seq = 0
        self._pending: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}

    def _emit(
        self, event_type: str, symbol: Optional[str], payload: Dict[str, str], ts: int
    ) -> None:
        self._seq += 1
        event: BacktestStreamEvent = {
            "run_id": self._run_id,
            "seq": self._seq,
            "ts": ts,
            "event_type": event_type,
            "symbol": None if symbol is None else str(symbol),
            "level": "info",
            "payload": {str(k): str(v) for k, v in payload.items()},
        }
        self._on_event(event)

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
        warmup: bool = False,
    ) -> None:
        """Emit one ``indicator_point`` and buffer it for the next snapshot."""
        indicator_key = str(name or "").strip()
        if not indicator_key:
            raise ValueError("indicator name cannot be empty")
        symbol_text = str(symbol or "").strip() or "_unknown"
        strategy_id = str(owner_strategy_id or "").strip() or "_default"
        pane_index = normalize_pane_index(pane)
        render = normalize_render_type(render_type)
        display = str(display_name or "").strip() or indicator_key
        meta_json = normalize_meta_json(meta)
        _, timestamp_ns = timestamp_to_ms_and_ns(timestamp)
        timestamp_ms = timestamp_ms_from_ns(timestamp_ns)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = float("nan")

        self._emit(
            "indicator_point",
            None if symbol_text == "_unknown" else symbol_text,
            {
                "owner_strategy_id": strategy_id,
                "indicator_key": indicator_key,
                "display_name": display,
                "pane": str(pane_index),
                "render_type": render,
                "symbol": symbol_text,
                "timestamp": str(timestamp_ns),
                "timestamp_ms": str(timestamp_ms),
                "value": repr(numeric_value),
                "warmup": str(bool(warmup)).lower(),
                "meta_json": meta_json,
            },
            timestamp_ns,
        )
        key = (strategy_id, symbol_text, timestamp_ns)
        self._pending.setdefault(key, []).append(
            {
                "indicator_key": indicator_key,
                "display_name": display,
                "pane": pane_index,
                "render_type": render,
                "value": numeric_value,
                "warmup": bool(warmup),
                "meta_json": meta_json,
            }
        )

    def flush_stream_snapshot(self) -> None:
        """Emit an ``indicator_snapshot`` per buffered callback cycle."""
        if not self._pending:
            return
        for (strategy_id, symbol_text, timestamp_ns), items in list(
            self._pending.items()
        ):
            self._emit(
                "indicator_snapshot",
                None if symbol_text == "_unknown" else symbol_text,
                {
                    "owner_strategy_id": strategy_id,
                    "symbol": symbol_text,
                    "timestamp": str(timestamp_ns),
                    "timestamp_ms": str(timestamp_ms_from_ns(timestamp_ns)),
                    "indicator_count": str(len(items)),
                    "items_json": json.dumps(
                        items, ensure_ascii=False, sort_keys=True, default=str
                    ),
                },
                timestamp_ns,
            )
        self._pending.clear()

    def build_payload(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return an empty payload — this sink does not accumulate points."""
        return {"definitions": [], "instances": [], "points": []}

    def set_stream_emitter(
        self,
        stream_emitter: Optional[
            Callable[[str, Optional[str], str, Dict[str, str]], None]
        ],
    ) -> None:
        """Ignore the emitter; this sink emits via ``on_event`` (protocol compat)."""
        return None
