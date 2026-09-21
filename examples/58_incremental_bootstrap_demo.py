#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant incremental bootstrap demo.

This example shows a hybrid workflow for incremental indicators:
1. Load enough history before `start_time`.
2. Bootstrap an incremental indicator with `warmup_bars`.
3. Continue updating the same indicator from the live event stream.

The recommended multi-symbol pattern is `indicator_factory`, which creates
one isolated indicator instance per symbol.
"""

from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def make_demo_data() -> pd.DataFrame:
    """Build two-symbol synthetic minute bars."""
    timestamps = pd.date_range("2024-01-01 09:30:00", periods=8, freq="min", tz="UTC")
    records: list[dict[str, object]] = []
    base_prices = {"AAPL": 100.0, "MSFT": 200.0}

    for symbol, base in base_prices.items():
        for i, ts in enumerate(timestamps):
            close = base + float(i)
            records.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": close - 0.2,
                    "high": close + 0.3,
                    "low": close - 0.4,
                    "close": close,
                    "volume": 1000.0 + i * 10.0,
                }
            )
    return pd.DataFrame(records)


class IncrementalBootstrapStrategy(Strategy):
    """Show per-symbol incremental indicators with historical bootstrap."""

    sma3: Any

    def __init__(self) -> None:
        """Initialize demo strategy state."""
        super().__init__()
        self.seen_active_bars: dict[str, int] = {}

    def on_start(self) -> None:
        """Register one SMA per symbol and request bootstrap history."""
        self.sma3 = self.I(
            factory=lambda: aq.SMA(3),
            name="sma3",
            source="close",
            symbols=["AAPL", "MSFT"],
            warmup_bars=3,
        )

    def on_bar(self, bar: Bar) -> None:
        """Print the active-stream indicator value for each symbol."""
        self.seen_active_bars[bar.symbol] = self.seen_active_bars.get(bar.symbol, 0) + 1
        sma = self.sma3.value
        local_ts = self.format_time(bar.timestamp)
        print(
            f"{local_ts} | {bar.symbol} | close={bar.close:.2f} | "
            f"sma3={sma:.2f} | active_bar={self.seen_active_bars[bar.symbol]}"
        )


if __name__ == "__main__":
    demo_data = make_demo_data()
    start_time = pd.Timestamp("2024-01-01 09:34:00", tz="UTC")
    end_time = pd.Timestamp("2024-01-01 09:37:00", tz="UTC")

    print("Running AKQuant incremental bootstrap demo...")
    print(
        "Expected behavior: SMA is already warm on the first active bar because "
        "bars before start_time are used for bootstrap."
    )

    result = aq.run_backtest(
        strategy=IncrementalBootstrapStrategy,
        data=demo_data,
        symbols=["AAPL", "MSFT"],
        start_time=start_time,
        end_time=end_time,
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
    )

    print("\n=== Backtest Summary ===")
    print(result)
