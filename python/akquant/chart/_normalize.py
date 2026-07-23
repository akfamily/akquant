"""Stateless normalization helpers for chart-facing indicator payloads.

These functions are the single source of truth shared by the core
``IndicatorRecorder`` (pure backtest path) and any frontend chart backend.
Keeping them here guarantees that the same ``record_indicator`` call produces an
identical pane index, render type, and metadata encoding regardless of whether the
run feeds a plain backtest export or a live chart bridge.

Pane semantics are **integer row indices** (``0`` = main price pane,
``1``..``MAX_SUB_PANES`` = stacked sub panes). ``MAX_SUB_PANES`` is a soft,
overridable default (env var ``AKQUANT_MAX_SUB_PANES`` or the ``max_sub_panes``
argument), not a hard ceiling. Unknown or string pane values raise
``ValueError`` immediately rather than silently degrading.

``render_type`` is a **closed string enum** (``RENDER_TYPE_CANONICAL``).
Unknown render types also raise ``ValueError`` so that downstream chart renderers
can implement exhaustive branches without encountering undefined values.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _default_max_sub_panes() -> int:
    """Resolve the default sub-pane cap, honoring an env-var escape hatch.

    A conservative software cap (not a hard product limit): eight stacked sub
    panes already exceed what a single screen reads well, yet multi-factor or
    derivatives workflows occasionally need more. Setting
    ``AKQUANT_MAX_SUB_PANES`` lets such callers lift the cap without forking,
    while a plain misspelled ``pane`` still fails fast against the default.
    """
    raw = os.environ.get("AKQUANT_MAX_SUB_PANES")
    if raw is None:
        return 8
    try:
        value = int(raw)
    except ValueError:
        return 8
    return value if value >= 1 else 8


# Main pane (index 0) plus up to ``MAX_SUB_PANES`` stacked sub panes. This is a
# soft, overridable default rather than a hard ceiling — see
# :func:`_default_max_sub_panes` and the ``max_sub_panes`` parameter below.
MAX_SUB_PANES = _default_max_sub_panes()

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


def normalize_pane_index(pane: Any = 0, max_sub_panes: Optional[int] = None) -> int:
    """Normalize a pane specifier into a canonical integer index.

    Panes are plain integer row indices, matching what chart renderers actually
    consume: ``0`` is the main (price) pane and ``1..max_sub_panes`` are stacked
    sub panes below it. ``None`` resolves to the main pane.

    :param pane: Pane index. Accepts ``int`` or an all-digit ``str``.
    :param max_sub_panes: Optional per-call cap overriding the module default
        :data:`MAX_SUB_PANES`. Lets an advanced caller lift the limit for a
        specific run without mutating global state.
    :return: Pane index in ``0..max_sub_panes``.
    :raises ValueError: If the pane is out of range or not an integer index.
    """
    cap = MAX_SUB_PANES if max_sub_panes is None else max_sub_panes
    if pane is None:
        return 0
    if isinstance(pane, bool):
        raise ValueError("pane must be an integer index in 0..%d" % cap)
    if isinstance(pane, str):
        text = pane.strip()
        if not text:
            return 0
        if not (text.lstrip("+").isdigit()):
            raise ValueError("pane must be an integer index in 0..%d (0=main)" % cap)
        pane = int(text)
    if isinstance(pane, int):
        if 0 <= pane <= cap:
            return pane
        raise ValueError("pane index supports 0 (main) plus 1..%d sub panes" % cap)
    raise ValueError("pane must be an integer index in 0..%d" % cap)


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


def normalize_reference_lines(value: Any = None) -> list[Dict[str, Any]]:
    """Normalize static reference-line metadata attached to an indicator.

    Each line carries a required numeric ``value`` plus optional ``label`` and
    ``color``. Labels keep CJK text intact (rendered later with
    ``ensure_ascii=False``). Non-list inputs, or entries whose ``value`` is
    missing or non-numeric, fail fast rather than degrading silently.

    :param value: A list of ``{"value", "label"?, "color"?}`` mappings.
    :return: Normalized list of reference-line dicts; ``[]`` when empty/None.
    :raises ValueError: If ``value`` is not a list or an entry is malformed.
    """
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("reference_lines must be a list of {value, label?, color?}")
    lines: list[Dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict) or "value" not in entry:
            raise ValueError("each reference line must be a dict with a 'value' key")
        try:
            line_value = float(entry["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError("reference line 'value' must be numeric") from exc
        lines.append(
            {
                "value": line_value,
                "label": str(entry.get("label") or "").strip(),
                "color": str(entry.get("color") or "").strip(),
            }
        )
    return lines


def normalize_scale_group(value: Any = None) -> str:
    """Normalize a shared-scale group label (open-ended semantic tag).

    Unlike pane/render_type this is not a closed enum: the group name is a free
    semantic hint (e.g. ``"percent"``, ``"price"``) that a frontend interprets.

    :param value: Group label or None.
    :return: Stripped label, or ``""`` when empty/None.
    """
    if value is None:
        return ""
    return str(value).strip()
