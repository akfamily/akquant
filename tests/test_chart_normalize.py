"""Unit tests for the shared chart normalization helpers."""

import pandas as pd
import pytest
from akquant.chart import (
    normalize_meta_json,
    normalize_pane_label,
    timestamp_ms_from_ns,
    timestamp_to_ms_and_ns,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "main"),
        ("main", "main"),
        ("主图", "main"),
        (0, "main"),
        ("0", "main"),
        ("signal", "signal"),
        ("SIGNAL", "signal"),
        ("sub", "sub1"),  # historical default must not crash
        ("sub1", "sub1"),
        ("sub4", "sub4"),
        (1, "sub1"),
        (4, "sub4"),
        ("副图2", "sub2"),
    ],
)
def test_normalize_pane_label_accepts_known_spellings(
    raw: object, expected: str
) -> None:
    """All historical pane spellings map to a canonical label."""
    assert normalize_pane_label(raw) == expected


@pytest.mark.parametrize("raw", ["sub9", "副图9", "unknown", 9])
def test_normalize_pane_label_rejects_out_of_range(raw: object) -> None:
    """Out-of-range or unknown pane specifiers raise ValueError."""
    with pytest.raises(ValueError):
        normalize_pane_label(raw)


def test_normalize_pane_label_rejects_bool() -> None:
    """Booleans are not valid pane specifiers even though they are ints."""
    with pytest.raises(ValueError):
        normalize_pane_label(True)


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
