# -*- coding: utf-8 -*-
"""
AKQuant + OpenCTP Realtime Paper Trading Example.

This script demonstrates how to integrate CTP real-time market data with AKQuant engine
for paper trading simulation.

Prerequisites:
    pip install openctp-ctp

Architecture:
    - Thread 1 (CTP): Receives market data (Ticks) from CTP Gateway.
    - Thread 2 (Main): Runs AKQuant Engine in realtime mode, executing strategy logic.
"""

import sys
from datetime import datetime
from typing import Any

# Import akquant
from akquant import AssetType, Bar, Instrument, Strategy

# Import LiveRunner from akquant.live
try:
    from akquant.live import LiveRunner
except ImportError as e:
    print(f"Error importing LiveRunner: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# 1. Simple Strategy
# -----------------------------------------------------------------------------
class DemoLiveStrategy(Strategy):
    """Demo strategy for live trading."""

    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
        print("[Strategy] Strategy Initialized")
        self.tick_count = 0

    def on_start(self) -> None:
        """Start the strategy execution."""
        print("[Strategy] Strategy Started")

    def on_bar(self, bar: Bar) -> None:
        """Process a new bar."""
        self.tick_count += 1
        dt = datetime.fromtimestamp(bar.timestamp / 1e9)
        print(
            f"[Strategy] ON_BAR Triggered | Time: {dt} | Symbol: {bar.symbol} | "
            f"Close: {bar.close} | Vol: {bar.volume}"
        )

        # Simple Logic: Buy on first tick, Sell on 10th tick
        pos = self.get_position(bar.symbol)
        print(f"[Strategy] Current Position for {bar.symbol}: {pos}")

        if self.tick_count % 2 == 0:
            print(f"[Strategy] Logic Signal Triggered at tick {self.tick_count}")
            if pos == 0:
                print(f"[Strategy] SIGNAL: BUY 1 {bar.symbol} @ Market")
                self.buy(bar.symbol, 1)
            elif pos > 0:
                print(f"[Strategy] SIGNAL: SELL 1 {bar.symbol} @ Market")
                self.sell(bar.symbol, 1)

    def on_order(self, order: Any) -> None:
        """Handle order status updates."""
        print(
            f"[Strategy] ON_ORDER Callback -> Symbol: {order.symbol} | "
            f"Side: {order.side} | Status: {order.status} | "
            f"Filled: {order.filled_quantity} @ {order.average_filled_price}"
        )

    def on_trade(self, trade: Any) -> None:
        """Handle trade execution updates."""
        print(
            f"[Strategy] ON_TRADE Callback -> Symbol: {trade.symbol} | "
            f"Side: {trade.side} | Price: {trade.price} | Qty: {trade.quantity}"
        )


# -----------------------------------------------------------------------------
# 2. Main Entry
# -----------------------------------------------------------------------------
def main() -> None:
    """Run the main entry point for CTP paper trading."""
    # Configuration
    # CTP Market Data Front Address (SimNow 7x24)
    MD_FRONT = "tcp://182.254.243.31:40011"
    # CTP Trade Front Address (SimNow 7x24)

    # Register Instruments for Shadow Account (Simulation)
    # Required for correct PnL calculation and Order Validation
    print("[Main] Defining Instruments...")

    instruments = [
        # Gold (au)
        Instrument(
            symbol="au2606",
            asset_type=AssetType.Futures,
            multiplier=1000.0,
            margin_ratio=0.1,
            tick_size=0.02,
            lot_size=1,
            option_type=None,
            strike_price=None,
            expiry_date=None,
        ),
        # Rebar (rb)
        Instrument(
            symbol="rb2605",
            asset_type=AssetType.Futures,
            multiplier=10.0,
            margin_ratio=0.1,
            tick_size=1.0,
            lot_size=1,
            option_type=None,
            strike_price=None,
            expiry_date=None,
        ),
        # Silver (ag)
        Instrument(
            symbol="ag2606",
            asset_type=AssetType.Futures,
            multiplier=15.0,
            margin_ratio=0.1,
            tick_size=1.0,
            lot_size=1,
            option_type=None,
            strike_price=None,
            expiry_date=None,
        ),
    ]

    # Create and Run LiveRunner
    # use_aggregator=False means we process every Tick as a Bar (High Frequency)
    runner = LiveRunner(
        strategy_cls=DemoLiveStrategy,
        instruments=instruments,
        md_front=MD_FRONT,
        use_aggregator=False,
    )

    # Run the strategy
    # You can stop manually via Ctrl+C, or set a duration (e.g., duration="1h", "30m")
    runner.run(cash=1_000_000, show_progress=False, duration="1m")


if __name__ == "__main__":
    main()
