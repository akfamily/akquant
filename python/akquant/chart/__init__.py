"""Chart-oriented helpers shared by indicator recording and frontend bridges.

This package hosts the single source of truth for chart-facing normalization
(pane names, millisecond timestamps, JSON-safe values) so that the pure
backtest path and any frontend chart bridge emit identical payloads.
"""

from ._normalize import (
    MAX_SUB_PANES,
    normalize_meta_json,
    normalize_pane_label,
    timestamp_ms_from_ns,
    timestamp_to_ms_and_ns,
)

__all__ = [
    "MAX_SUB_PANES",
    "normalize_meta_json",
    "normalize_pane_label",
    "timestamp_ms_from_ns",
    "timestamp_to_ms_and_ns",
]
