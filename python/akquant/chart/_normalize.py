"""Stateless normalization helpers for chart-facing indicator payloads.

These functions are the single source of truth shared by the core
``IndicatorRecorder`` (pure backtest path) and any frontend chart backend.
Keeping them here guarantees that the same ``record_indicator`` call produces an
identical pane index, render type, and metadata encoding regardless of whether the
run feeds a plain backtest export or a live chart bridge.

Pane semantics are **integer row indices** (``0`` = main price pane,
``1``..``MAX_SUB_PANES`` = stacked sub panes).  Unknown or string pane values
raise ``ValueError`` immediately rather than silently degrading.

``render_type`` is a **closed string enum** (``RENDER_TYPE_CANONICAL``).
Unknown render types also raise ``ValueError`` so that downstream chart renderers
can implement exhaustive branches without encountering undefined values.
"""

import json
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Main pane plus up to four stacked sub panes.
MAX_SUB_PANES = 4

# Closed set of render types a chart renderer must handle. Each value has a
# defined rendering in ``akquant.plot.indicator`` and is documented in the guide.
#   line      -> connected line series
#   area      -> line series with fill to zero
#   bar       -> vertical bars
#   column    -> alias of bar (semantic label for categorical columns)
#   histogram -> alias of bar (semantic label for distribution-style bars)
#   scatter   -> disconnected point markers
#   signal    -> trade-signal markers, drawn on the main price pane
RENDER_TYPE_CANONICAL = (
    "line",
    "area",
    "bar",
    "column",
    "histogram",
    "scatter",
    "signal",
)


def normalize_render_type(render_type: Any = "line") -> str:
    """Normalize a render-type specifier against the canonical enum.

    :param render_type: Render type. Empty/``None`` resolves to ``"line"``.
    :return: A canonical render type from :data:`RENDER_TYPE_CANONICAL`.
    :raises ValueError: If the value is not a recognized render type.
    """
    if render_type is None:
        return "line"
    text = str(render_type).strip().lower()
    if not text:
        return "line"
    if text not in RENDER_TYPE_CANONICAL:
        raise ValueError(
            "render_type must be one of %s" % ", ".join(RENDER_TYPE_CANONICAL)
        )
    return text


def normalize_pane_index(pane: Any = 0) -> int:
    """Normalize a pane specifier into a canonical integer index.

    Panes are plain integer row indices, matching what chart renderers actually
    consume: ``0`` is the main (price) pane and ``1..MAX_SUB_PANES`` are stacked
    sub panes below it. ``None`` resolves to the main pane.

    :param pane: Pane index. Accepts ``int`` or an all-digit ``str``.
    :return: Pane index in ``0..MAX_SUB_PANES``.
    :raises ValueError: If the pane is out of range or not an integer index.
    """
    if pane is None:
        return 0
    if isinstance(pane, bool):
        raise ValueError("pane must be an integer index in 0..%d" % MAX_SUB_PANES)
    if isinstance(pane, str):
        text = pane.strip()
        if not text:
            return 0
        if not (text.lstrip("+").isdigit()):
            raise ValueError(
                "pane must be an integer index in 0..%d (0=main)" % MAX_SUB_PANES
            )
        pane = int(text)
    if isinstance(pane, int):
        if 0 <= pane <= MAX_SUB_PANES:
            return pane
        raise ValueError(
            "pane index supports 0 (main) plus 1..%d sub panes" % MAX_SUB_PANES
        )
    raise ValueError("pane must be an integer index in 0..%d" % MAX_SUB_PANES)


def timestamp_to_ms_and_ns(timestamp: Any) -> Tuple[int, int]:
    """Return ``(milliseconds, nanoseconds)`` for a timestamp-like value.

    Recognizes ``pandas.Timestamp``/datetime-like inputs and integer epoch
    values in millisecond (13-digit), microsecond (16-digit), or nanosecond
    (19-digit) magnitude. Falls back to pandas parsing for strings.

    :param timestamp: Timestamp-like value.
    :return: Tuple of epoch milliseconds and epoch nanoseconds.
    """
    if isinstance(timestamp, bool):
        raise ValueError("timestamp must be a timestamp value")
    if isinstance(timestamp, pd.Timestamp):
        ns = int(timestamp.value)
        return ns // 1_000_000, ns
    if isinstance(timestamp, str):
        text = timestamp.strip()
        if text.isdigit():
            return _ms_ns_from_epoch_int(int(text))
        ns = int(pd.Timestamp(text).value)
        return ns // 1_000_000, ns
    if isinstance(timestamp, float):
        if not timestamp.is_integer():
            ns = int(pd.Timestamp(timestamp).value)
            return ns // 1_000_000, ns
        timestamp = int(timestamp)
    if isinstance(timestamp, int):
        return _ms_ns_from_epoch_int(timestamp)
    if hasattr(timestamp, "to_pydatetime"):
        ns = int(pd.Timestamp(timestamp).value)
        return ns // 1_000_000, ns
    ns = int(pd.Timestamp(timestamp).value)
    return ns // 1_000_000, ns


def _ms_ns_from_epoch_int(value: int) -> Tuple[int, int]:
    """Infer ms/ns from an integer epoch value by its digit magnitude."""
    digits = len(str(abs(value)))
    if digits <= 13:
        return value, value * 1_000_000
    if digits <= 16:
        ms = value // 1_000
        return ms, ms * 1_000_000
    return value // 1_000_000, value


def timestamp_ms_from_ns(timestamp_ns: int) -> int:
    """Convert an epoch-nanosecond integer into epoch milliseconds."""
    return int(timestamp_ns) // 1_000_000


def normalize_meta_json(meta: Optional[Dict[str, Any]]) -> str:
    """Serialize indicator metadata into a stable, human-readable JSON string.

    Uses ``ensure_ascii=False`` so CJK metadata stays readable, and
    ``sort_keys=True`` so the encoding is deterministic (important for building
    stable instance keys).

    :param meta: Optional metadata mapping.
    :return: JSON string, or an empty string when there is no metadata.
    """
    if not meta:
        return ""
    return json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str)
