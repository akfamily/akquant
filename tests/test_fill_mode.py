"""Tests for FillMode classes and core translation."""

import akquant
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


def test_unrepresentable_core_triple_raises() -> None:
    """fill_mode_from_core raises ValueError on unrepresentable triples."""
    with pytest.raises(ValueError, match="Unrepresentable core triple"):
        fill_mode_from_core("open", 0, "same_cycle")


def test_invalid_timer_fill_timing_raises() -> None:
    """CurrentClose raises ValueError on invalid timer_fill_timing."""
    with pytest.raises(ValueError, match="timer_fill_timing"):
        CurrentClose(timer_fill_timing="Immediate")  # type: ignore[arg-type]


def test_invalid_temporal_in_fill_mode_from_core_raises() -> None:
    """fill_mode_from_core raises ValueError on invalid temporal for close/0."""
    with pytest.raises(ValueError, match="temporal"):
        fill_mode_from_core("close", 0, "bogus")


# --- Task 2: run_backtest(fill_policy=) integration tests ---


def _one_bar_data(symbol: str = "X") -> list:
    import pandas as pd

    ts = pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai").value
    ts2 = pd.Timestamp("2023-01-02 10:01:00", tz="Asia/Shanghai").value
    return [
        akquant.Bar(ts, 10.0, 10.0, 10.0, 10.0, 1000.0, symbol),
        akquant.Bar(ts2, 11.0, 11.0, 11.0, 11.0, 1000.0, symbol),
    ]


def test_run_backtest_accepts_fillmode_object() -> None:
    """run_backtest accepts a FillMode object without raising."""
    akquant.run_backtest(
        data=_one_bar_data(),
        strategy=lambda ctx, bar: None,
        symbols="X",
        fill_policy=NextClose(),
        initial_cash=100000.0,
        show_progress=False,
    )


def test_run_backtest_rejects_legacy_dict() -> None:
    """run_backtest rejects a legacy fill_policy dict with TypeError."""
    with pytest.raises(TypeError):
        akquant.run_backtest(
            data=_one_bar_data(),
            strategy=lambda ctx, bar: None,
            symbols="X",
            fill_policy={"price_basis": "close", "bar_offset": 0},
            initial_cash=100000.0,
            show_progress=False,
        )


def test_fillmode_importable_from_top_level() -> None:
    """FillMode subclasses are importable from the top-level akquant package."""
    assert hasattr(akquant, "NextOpen")
    assert hasattr(akquant, "CurrentClose")
