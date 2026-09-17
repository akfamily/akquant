"""Chart-oriented helpers shared by indicator recording and frontend bridges.

This package hosts the single source of truth for chart-facing normalization
(pane names, millisecond timestamps, JSON-safe values) so that the pure
backtest path and any frontend chart bridge emit identical payloads.
"""

from ._normalize import (
    MAX_SUB_PANES,
    RENDER_TYPE_CANONICAL,
    normalize_meta_json,
    normalize_pane_index,
    normalize_reference_lines,
    normalize_render_type,
    normalize_scale_group,
    timestamp_ms_from_ns,
    timestamp_to_ms_and_ns,
)
from .d3kline import (
    D3KLINE_SERIES_TYPE,
    SUB_PANE_HEIGHT,
    to_d3kline_options,
    to_raw_panes,
)

__all__ = [
    "D3KLINE_SERIES_TYPE",
    "MAX_SUB_PANES",
    "RENDER_TYPE_CANONICAL",
    "normalize_meta_json",
    "normalize_pane_index",
    "normalize_reference_lines",
    "normalize_render_type",
    "normalize_scale_group",
    "timestamp_ms_from_ns",
    "SUB_PANE_HEIGHT",
    "timestamp_to_ms_and_ns",
    "to_d3kline_options",
    "to_raw_panes",
]
