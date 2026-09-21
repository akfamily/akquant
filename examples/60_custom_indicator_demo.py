#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant custom indicator demo.

This example shows the three ways to declare a private indicator through the
single `self.I()` entry point:
1. Vectorized precompute with `Indicator(name, fn)`.
2. Incremental with a custom `Indicator` subclass (stateful `update`).
3. Both at once in one strategy - no mutually exclusive mode switch.
"""

from collections import deque
from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar, Indicator, Strategy


def make_demo_data() -> pd.DataFrame:
    """Build a small single-symbol synthetic price series."""
    timestamps = pd.date_range("2024-01-01 09:30:00", periods=8, freq="min", tz="UTC")
    closes = [10.0, 10.5, 11.0, 11.8, 12.1, 11.7, 12.4, 12.9]
    records: list[dict[str, object]] = []
    for ts, close in zip(timestamps, closes):
        records.append(
            {
                "timestamp": ts,
                "symbol": "DEMO",
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.3,
                "close": close,
                "volume": 1000.0,
            }
        )
    return pd.DataFrame(records)


class RollingMomentum(Indicator):
    """Simple close[t] - close[t-period+1] momentum indicator."""

    def __init__(self, period: int = 3) -> None:
        """Initialize the rolling momentum indicator."""
        super().__init__(
            f"rolling_momentum_{period}",
            lambda df: df["close"] - df["close"].shift(period - 1),
        )
        self.period = period
        self.buffer: deque[float] = deque(maxlen=period)
        self._current_value = float("nan")

    def update(self, value: float) -> float:
        """Update the rolling momentum using one close value."""
        self.buffer.append(float(value))
        if len(self.buffer) < self.period:
            self._current_value = float("nan")
        else:
            self._current_value = self.buffer[-1] - self.buffer[0]
        return self._current_value

    @property
    def value(self) -> float:
        """Return the latest incremental value."""
        return self._current_value


class PrecomputeCustomIndicatorStrategy(Strategy):
    """Vectorized custom indicator example."""

    mom3: Any

    def on_start(self) -> None:
        """Declare a vectorized indicator - computed once, queried by timestamp."""
        self.mom3 = self.I(
            Indicator("mom3", lambda df: df["close"] - df["close"].shift(2)),
            name="mom3",
        )

    def on_bar(self, bar: Bar) -> None:
        """Print the current precomputed custom indicator value."""
        value = self.mom3[0]
        local_ts = self.format_time(bar.timestamp)
        shown = "n/a" if value is None else f"{value:.2f}"
        print(f"[precompute] {local_ts} | close={bar.close:.2f} | mom3={shown}")


class IncrementalCustomIndicatorStrategy(Strategy):
    """Stateful custom indicator example with bootstrap history."""

    mom3: Any

    def on_start(self) -> None:
        """Declare the incremental custom indicator with bootstrap history."""
        self.mom3 = self.I(
            factory=lambda: RollingMomentum(period=3),
            name="mom3",
            source="close",
            symbols=["DEMO"],
            warmup_bars=3,
        )

    def on_bar(self, bar: Bar) -> None:
        """Print the current incremental custom indicator value."""
        value = self.mom3[0]
        local_ts = self.format_time(bar.timestamp)
        shown = "n/a" if value is None else f"{value:.2f}"
        print(f"[incremental] {local_ts} | close={bar.close:.2f} | mom3={shown}")


class MixedIndicatorStrategy(Strategy):
    """Both kinds side by side - impossible before the mode switch was removed."""

    vec_mom: Any
    inc_sma: Any

    def on_start(self) -> None:
        """Declare one vectorized and one incremental indicator together."""
        self.vec_mom = self.I(
            Indicator("vec_mom", lambda df: df["close"] - df["close"].shift(2)),
            name="vec_mom",
        )
        # Plot metadata makes the framework report this one automatically -
        # no record_indicator call needed in on_bar.
        self.inc_sma = self.I(aq.SMA(3), name="inc_sma", pane=0, color="#3f51b5")

    def on_bar(self, bar: Bar) -> None:
        """Read both indicators, including one-bar lookback."""
        local_ts = self.format_time(bar.timestamp)
        mom = self.vec_mom[0]
        sma_now, sma_prev = self.inc_sma[0], self.inc_sma[1]
        mom_text = "n/a" if mom is None else f"{mom:.2f}"
        sma_text = "n/a" if sma_now is None else f"{sma_now:.2f}"
        trend = "-"
        if sma_now is not None and sma_prev is not None:
            trend = "up" if sma_now > sma_prev else "down"
        print(
            f"[mixed] {local_ts} | close={bar.close:.2f} | "
            f"vec_mom={mom_text} | inc_sma={sma_text} ({trend})"
        )


def run_precompute_demo(data: pd.DataFrame) -> None:
    """Run the vectorized custom indicator demo."""
    print("=== Precompute Custom Indicator Demo ===")
    aq.run_backtest(
        strategy=PrecomputeCustomIndicatorStrategy,
        data=data,
        symbols=["DEMO"],
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
    )


def run_incremental_demo(data: pd.DataFrame) -> None:
    """Run the incremental custom indicator demo with warmup bars."""
    print("\n=== Incremental Custom Indicator Demo ===")
    print("Expected: the first active bar already has a valid mom3 value.")
    aq.run_backtest(
        strategy=IncrementalCustomIndicatorStrategy,
        data=data,
        symbols=["DEMO"],
        start_time=pd.Timestamp("2024-01-01 09:33:00", tz="UTC"),
        end_time=pd.Timestamp("2024-01-01 09:37:00", tz="UTC"),
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
    )


def run_mixed_demo(data: pd.DataFrame) -> None:
    """Run both indicator kinds inside a single strategy."""
    print("\n=== Mixed (Precompute + Incremental) Demo ===")
    result = aq.run_backtest(
        strategy=MixedIndicatorStrategy,
        data=data,
        symbols=["DEMO"],
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
    )
    frame = result.indicator_df()
    print(f"auto-reported indicator points: {len(frame)}")


if __name__ == "__main__":
    demo_data = make_demo_data()
    run_precompute_demo(demo_data)
    run_incremental_demo(demo_data)
    run_mixed_demo(demo_data)
