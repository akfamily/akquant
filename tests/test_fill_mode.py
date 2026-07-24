"""Tests for FillMode classes and core translation."""

import pytest
from akquant.backtest.fill_mode import (
    CurrentClose,
    FillMode,
    NextAverage,
    NextClose,
    NextHighLowMid,
    NextOpen,
    fill_mode_from_core,
)


def test_each_mode_maps_to_expected_core() -> None:
    """Each mode returns its correct core triple."""
    assert NextOpen()._to_core() == ("open", 1, "same_cycle")
    assert NextClose()._to_core() == ("close", 1, "same_cycle")
    assert NextAverage()._to_core() == ("ohlc4", 1, "same_cycle")
    assert NextHighLowMid()._to_core() == ("hl2", 1, "same_cycle")
    assert CurrentClose()._to_core() == ("close", 0, "same_cycle")
    assert CurrentClose(timer_fill_timing="deferred")._to_core() == (
        "close",
        0,
        "next_event",
    )


def test_current_close_default_is_immediate() -> None:
    """CurrentClose defaults to immediate timing."""
    assert CurrentClose().timer_fill_timing == "immediate"


def test_next_open_has_no_timer_param() -> None:
    """NextOpen does not accept timer_fill_timing parameter."""
    with pytest.raises(TypeError):
        NextOpen(timer_fill_timing="deferred")  # type: ignore[call-arg]


def test_mode_is_frozen() -> None:
    """FillMode instances are frozen and immutable."""
    with pytest.raises(Exception):
        CurrentClose().timer_fill_timing = "deferred"  # type: ignore[misc]


def test_roundtrip_from_core() -> None:
    """fill_mode_from_core reconstructs modes correctly."""
    for mode in (
        NextOpen(),
        NextClose(),
        NextAverage(),
        NextHighLowMid(),
        CurrentClose(),
        CurrentClose(timer_fill_timing="deferred"),
    ):
        basis, offset, temporal = mode._to_core()
        assert (
            fill_mode_from_core(basis, offset, temporal)._to_core() == mode._to_core()
        )


def test_base_class_to_core_raises() -> None:
    """FillMode base class raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        FillMode()._to_core()
