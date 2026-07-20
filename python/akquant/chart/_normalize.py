"""Stateless normalization helpers for chart-facing indicator payloads.

These functions are the single source of truth shared by the core
``IndicatorRecorder`` (pure backtest path) and any frontend chart backend.
Keeping them here guarantees that the same ``record_indicator`` call produces an
identical pane label, timestamp, and metadata encoding regardless of whether the
run feeds a plain backtest export or a live chart bridge.

Pane semantics are **string labels** (``"main"`` / ``"sub1".."sub4"`` /
``"signal"``), matching the contract locked by the backtest result API and the
indicator stream. This differs from the integer pane index used internally by
the d3kline renderer; the renderer is free to map these labels to indices.
"""

import json
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Main pane plus up to four stacked sub panes.
MAX_SUB_PANES = 4


def normalize_pane_label(pane: Any, default: str = "sub1") -> str:
    """Normalize a pane specifier into a canonical string label.

    Accepts the historical spellings used across the codebase and the frontend:
    ``main`` / ``主图`` / ``0`` -> ``"main"``; ``sub`` / ``sub1`` / ``副图1`` /
    integer ``1`` -> ``"sub1"``; ``signal`` is preserved verbatim. A bare
    ``"sub"`` (the historical default) resolves to ``default`` instead of
    raising, fixing a crash on the frontend bridge path.

    :param pane: Raw pane specifier (str or int).
    :param default: Label used when the value is empty or a bare ``"sub"``.
    :return: Canonical pane label.
    """
    if pane is None:
        return "main"
    if isinstance(pane, bool):
        raise ValueError("pane must be a string label or integer index")
    if isinstance(pane, int):
        return _sub_label_from_index(pane)

    text = str(pane).strip().lower()
    if not text:
        return default
    if text in {"signal"}:
        return "signal"
    if text in {"0", "main", "主图"}:
        return "main"
    if text == "sub":
        return default
    if text.startswith("sub"):
        text = text[3:]
    elif text.startswith("副图"):
        text = text[2:]
    elif text == "副图":
        return default
    if not text:
        return default
    if text.isdigit():
        return _sub_label_from_index(int(text))
    raise ValueError("pane must be main, sub1..sub4, signal, or 0..4")


def _sub_label_from_index(index: int) -> str:
    """Map an integer pane index to a canonical label."""
    if index == 0:
        return "main"
    if 1 <= index <= MAX_SUB_PANES:
        return f"sub{index}"
    raise ValueError(f"pane index supports main plus 1..{MAX_SUB_PANES} sub panes")


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
