"""Unit tests for the shared chart normalization helpers."""

import pandas as pd
import pytest
from akquant.chart import (
    RENDER_TYPE_CANONICAL,
    normalize_meta_json,
    normalize_pane_index,
    normalize_render_type,
    timestamp_ms_from_ns,
    timestamp_to_ms_and_ns,
)


@pytest.mark.parametrize("value", RENDER_TYPE_CANONICAL)
def test_normalize_render_type_accepts_canonical(value: str) -> None:
    """Every canonical render type normalizes to itself."""
    assert normalize_render_type(value) == value


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "line"), ("", "line"), ("  ", "line"), ("LINE", "line"), ("Bar", "bar")],
)
def test_normalize_render_type_defaults_and_casing(raw: object, expected: str) -> None:
    """Empty values default to line and casing is normalized."""
    assert normalize_render_type(raw) == expected


@pytest.mark.parametrize("raw", ["candlestick", "spline", "pie", "unknown"])
def test_normalize_render_type_rejects_unknown(raw: object) -> None:
    """Render types outside the canonical enum raise ValueError (fail-fast)."""
    with pytest.raises(ValueError):
        normalize_render_type(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 0),
        (0, 0),
        ("0", 0),
        (1, 1),
        (4, 4),
        ("2", 2),
        ("", 0),
    ],
)
def test_normalize_pane_index_accepts_integer_specifiers(
    raw: object, expected: int
) -> None:
    """Pane specifiers normalize to an integer row index (0=main)."""
    assert normalize_pane_index(raw) == expected


@pytest.mark.parametrize("raw", [5, 9, -1, "9", "main", "sub1", "副图2", "signal"])
def test_normalize_pane_index_rejects_out_of_range_or_labels(raw: object) -> None:
    """Out-of-range indices and legacy string labels raise ValueError."""
    with pytest.raises(ValueError):
        normalize_pane_index(raw)


def test_normalize_pane_index_rejects_bool() -> None:
    """Booleans are not valid pane specifiers even though they are ints."""
    with pytest.raises(ValueError):
        normalize_pane_index(True)


def test_timestamp_to_ms_and_ns_from_pandas_timestamp() -> None:
    """A pandas Timestamp yields matching ms and ns integers."""
    ts = pd.Timestamp("2024-01-02 10:00:00")
    ms, ns = timestamp_to_ms_and_ns(ts)
    assert ns == int(ts.value)
    assert ms == ns // 1_000_000


@pytest.mark.parametrize(
    ("value", "expected_ms"),
    [
        (1_704_189_600_000, 1_704_189_600_000),  # 13-digit ms
        (1_704_189_600_000_000, 1_704_189_600_000),  # 16-digit us
        (1_704_189_600_000_000_000, 1_704_189_600_000),  # 19-digit ns
    ],
)
def test_timestamp_to_ms_and_ns_infers_epoch_magnitude(
    value: int, expected_ms: int
) -> None:
    """Integer epoch values are disambiguated by digit magnitude."""
    ms, ns = timestamp_to_ms_and_ns(value)
    assert ms == expected_ms
    assert ns == expected_ms * 1_000_000


def test_timestamp_ms_from_ns_round_trip() -> None:
    """Nanosecond-to-millisecond conversion truncates as expected."""
    assert timestamp_ms_from_ns(1_704_189_600_123_456_789) == 1_704_189_600_123


def test_normalize_meta_json_keeps_cjk_readable() -> None:
    """CJK metadata stays human-readable (no ASCII escaping) and is sorted."""
    encoded = normalize_meta_json({"名称": "均线", "period": 20})
    assert "均线" in encoded
    assert encoded == '{"period": 20, "名称": "均线"}'


def test_normalize_meta_json_empty() -> None:
    """Empty or missing metadata serializes to an empty string."""
    assert normalize_meta_json(None) == ""
    assert normalize_meta_json({}) == ""
