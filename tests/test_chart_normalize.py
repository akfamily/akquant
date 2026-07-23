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
        (8, 8),
        ("2", 2),
        ("", 0),
    ],
)
def test_normalize_pane_index_accepts_integer_specifiers(
    raw: object, expected: int
) -> None:
    """Pane specifiers normalize to an integer row index (0=main)."""
    assert normalize_pane_index(raw) == expected


@pytest.mark.parametrize("raw", [9, 12, -1, "9", "main", "sub1", "副图2", "signal"])
def test_normalize_pane_index_rejects_out_of_range_or_labels(raw: object) -> None:
    """Out-of-range indices and legacy string labels raise ValueError."""
    with pytest.raises(ValueError):
        normalize_pane_index(raw)


def test_normalize_pane_index_honors_per_call_override() -> None:
    """An explicit max_sub_panes lifts the cap for that call only."""
    assert normalize_pane_index(10, max_sub_panes=16) == 10
    with pytest.raises(ValueError):
        normalize_pane_index(17, max_sub_panes=16)
    # The override does not leak into the default-capped path.
    with pytest.raises(ValueError):
        normalize_pane_index(10)


def test_default_max_sub_panes_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """AKQUANT_MAX_SUB_PANES overrides the default cap; junk falls back to 8."""
    import importlib

    import akquant.chart._normalize as norm

    monkeypatch.setenv("AKQUANT_MAX_SUB_PANES", "12")
    assert norm._default_max_sub_panes() == 12
    monkeypatch.setenv("AKQUANT_MAX_SUB_PANES", "not-a-number")
    assert norm._default_max_sub_panes() == 8
    monkeypatch.setenv("AKQUANT_MAX_SUB_PANES", "0")
    assert norm._default_max_sub_panes() == 8
    monkeypatch.delenv("AKQUANT_MAX_SUB_PANES", raising=False)
    assert norm._default_max_sub_panes() == 8
    importlib.reload(norm)  # restore module-level MAX_SUB_PANES cleanly


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


def test_normalize_reference_lines_normalizes_full_entries() -> None:
    """Reference lines normalize value to float and preserve labels/colors."""
    from akquant.chart import normalize_reference_lines

    result = normalize_reference_lines(
        [
            {"value": 70, "label": "超买", "color": "#ef4444"},
            {"value": "30", "label": "超卖"},
        ]
    )
    assert result == [
        {"value": 70.0, "label": "超买", "color": "#ef4444"},
        {"value": 30.0, "label": "超卖", "color": ""},
    ]


def test_normalize_reference_lines_empty_and_none() -> None:
    """None and empty lists return empty result."""
    from akquant.chart import normalize_reference_lines

    assert normalize_reference_lines(None) == []
    assert normalize_reference_lines([]) == []


def test_normalize_reference_lines_rejects_non_list() -> None:
    """Non-list inputs raise ValueError (fail-fast)."""
    from akquant.chart import normalize_reference_lines

    with pytest.raises(ValueError):
        normalize_reference_lines({"value": 70})


def test_normalize_reference_lines_rejects_missing_value() -> None:
    """Missing 'value' key raises ValueError."""
    from akquant.chart import normalize_reference_lines

    with pytest.raises(ValueError):
        normalize_reference_lines([{"label": "超买"}])


def test_normalize_reference_lines_rejects_non_numeric_value() -> None:
    """Non-numeric 'value' raises ValueError."""
    from akquant.chart import normalize_reference_lines

    with pytest.raises(ValueError):
        normalize_reference_lines([{"value": "high"}])


def test_normalize_scale_group_strips_and_defaults() -> None:
    """Scale group is stripped and None/empty default to empty string."""
    from akquant.chart import normalize_scale_group

    assert normalize_scale_group("  percent ") == "percent"
    assert normalize_scale_group(None) == ""
    assert normalize_scale_group("") == ""
