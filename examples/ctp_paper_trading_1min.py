# -*- coding: utf-8 -*-
"""AKQuant + OpenCTP Realtime Paper Trading Example (1-Minute Bar Aggregation)."""

import sys
from datetime import datetime
from typing import Any

# Import akquant
from akquant import AssetType, Bar, Instrument, Strategy

# Import LiveRunner
try:
    from akquant.live import LiveRunner
except ImportError as e:
    print(f"Error importing LiveRunner: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Strategy
# -----------------------------------------------------------------------------
class Demo1MinStrategy(Strategy):
    """Demo strategy for 1-minute bar aggregation."""

    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
        self.bar_count = 0

    def on_start(self) -> None:
        """Start the strategy execution."""
        print("[Strategy] 1-Min Strategy Started")

    def on_bar(self, bar: Bar) -> None:
        """Process a new bar."""
        self.bar_count += 1
        dt = datetime.fromtimestamp(bar.timestamp / 1e9)
        print(
            f"[Strategy] ON_BAR (1-Min) | Time: {dt} | Sym: {bar.symbol} | "
            f"C: {bar.close} | V: {bar.volume}"
        )

        pos = self.get_position(bar.symbol)

        # Logic: Trade every bar to test
        if self.bar_count % 1 == 0:
            if pos == 0:
                print(f"[Strategy] BUY Signal {bar.symbol}")
                self.buy(bar.symbol, 1)
            elif pos > 0:
                print(f"[Strategy] SELL Signal {bar.symbol}")
                self.sell(bar.symbol, 1)

    def on_order(self, order: Any) -> None:
        """Handle order status updates."""
        print(
            f"[Strategy] Order Update: {order.symbol} {order.status} "
            f"{order.filled_quantity}@{order.average_filled_price}"
        )

    def on_trade(self, trade: Any) -> None:
        """Handle trade execution updates."""
        print(
            f"[Strategy] Trade Executed: {trade.symbol} {trade.side} "
            f"{trade.quantity}@{trade.price}"
        )


# -----------------------------------------------------------------------------
# Main Entry
# -----------------------------------------------------------------------------
def main() -> None:
    """Run the main entry point for CTP paper trading 1-min example."""
    # SimNow 7x24 Addresses
    MD_FRONT = "tcp://182.254.243.31:40011"

    print("[Main] Defining Instruments...")
    instruments = [
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
    # use_aggregator=True is default, but explicit here for clarity
    runner = LiveRunner(
        strategy_cls=Demo1MinStrategy,
        instruments=instruments,
        md_front=MD_FRONT,
        use_aggregator=True,
    )

    # Run strategy (Ctrl+C to stop, or use duration="2h")
    runner.run(cash=1_000_000, show_progress=False)


if __name__ == "__main__":
    main()
