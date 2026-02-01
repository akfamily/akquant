from datetime import datetime

import akquant
import pandas as pd


class StopOrderStrategy(akquant.Strategy):
    """Test strategy for verifying stop orders and target position."""

    def on_start(self) -> None:
        """Initialize strategy state on start."""
        self.stop_order_placed = False
        self.target_order_placed = False
        print("Strategy started")

    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
        self.stop_order_placed = False
        self.target_order_placed = False

    def on_bar(self, bar: akquant.Bar) -> None:
        """Handle bar data events."""
        dt = datetime.fromtimestamp(bar.timestamp / 1e9)
        print(f"On Bar: {dt}, Close: {bar.close}, High: {bar.high}")

        # Test 1: Stop Market Order
        # Place a Stop Buy order at 110 when price is around 100
        if not self.stop_order_placed:
            print("Placing Stop Market Buy at 110")
            self.buy(bar.symbol, 100, price=None, trigger_price=110.0)
            self.stop_order_placed = True
            return

        # Test 2: Target Position
        # On the last day, try to adjust position to target value
        if dt.day == 3:  # Simple check for 3rd data point
            print("Adjusting target value to 50000")
            # Current price is 120, holding 100 shares (from stop order) = 12000 value
            # Target 50000 -> Need to buy ~38000 / 120 worth
            self.order_target_value(50000, bar.symbol)


def verify_stop_and_target() -> None:
    """Run the verification backtest."""
    # 1. Create Data
    dates = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
        datetime(2023, 1, 4),
    ]

    # Day 1: High 105 (Stop 110 not triggered)
    # Day 2: High 115 (Stop 110 triggered, should fill at Open or Close depending
    # on logic, let's say Close 112)
    # Day 3: Price 120. Test Target Position.
    data = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 108.0, 118.0, 125.0],
            "high": [105.0, 115.0, 122.0, 130.0],
            "low": [95.0, 105.0, 115.0, 120.0],
            "close": [102.0, 112.0, 120.0, 128.0],
            "volume": [10000, 20000, 15000, 10000],
            "symbol": ["TEST"] * 4,
        }
    )

    # 2. Setup Engine
    engine = akquant.Engine()
    engine.use_simple_market(
        0.0001
    )  # Use SimpleMarket to allow 24/7 trading (avoids session checks)
    engine.set_cash(100000.0)

    # Convert DataFrame to Bars
    bars = []
    for _, row in data.iterrows():
        # Timestamp to nanoseconds
        ts = int(row["date"].timestamp() * 1e9)
        bar = akquant.Bar(
            ts,
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
            row["symbol"],
        )
        bars.append(bar)

    engine.add_bars(bars)

    strategy = StopOrderStrategy()

    # 3. Run
    print("Running Backtest...")
    engine.run(strategy, show_progress=False)

    # 4. Analyze Results
    trades = engine.trades
    print(f"Total Trades: {len(trades)}")

    print("Orders:")
    for o in engine.orders:
        print(
            f"Order: {o.id} {o.symbol} {o.side} Type={o.order_type} "
            f"Status={o.status} Qty={o.quantity} Trigger={o.trigger_price}"
        )

    for t in trades:
        print(f"Trade: {t.symbol} {t.side} {t.quantity} @ {t.price} on {t.timestamp}")

    # Validation
    # Trade 1: Stop Order Triggered
    # Trigger price 110. Day 2 High is 115. Triggered.
    # Note: Trade timestamp might be Day 2 timestamp (end of bar)

    trade1 = None
    for t in trades:
        if t.symbol == "TEST" and t.side == akquant.OrderSide.Buy and t.quantity == 100:
            trade1 = t
            break

    assert trade1 is not None, "Stop order should be filled on Day 2"
    print(f"Stop Order Filled: {trade1.price} at {trade1.timestamp}")

    # Trade 2: Target Position
    # Target Value 50000. Price ~120-128.
    # Should result in another Buy order.
    trade2 = None
    for t in trades:
        # Rough check for second trade
        if t.symbol == "TEST" and t.side == akquant.OrderSide.Buy and t.quantity != 100:
            trade2 = t
            break

    assert trade2 is not None, "Target position rebalancing should occur"
    print(f"Target Position Trade: {trade2.quantity} @ {trade2.price}")


if __name__ == "__main__":
    verify_stop_and_target()
