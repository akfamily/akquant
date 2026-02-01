import unittest
from datetime import datetime

from akquant import (
    AssetType,
    Bar,
    Engine,
    ExecutionMode,
    Instrument,
    OrderSide,
    OrderStatus,
    Strategy,
)


class TestTPlusOne(unittest.TestCase):
    """Test T+1 trading rules."""

    def setUp(self) -> None:
        """Set up the test engine."""
        self.engine = Engine()
        self.engine.use_china_market()
        self.engine.set_t_plus_one(True)
        self.engine.set_execution_mode(ExecutionMode.CurrentClose)
        self.engine.set_cash(1_000_000.0)

        # Add a stock instrument
        self.symbol = "000001"
        instr = Instrument(
            symbol=self.symbol,
            asset_type=AssetType.Stock,
            multiplier=1.0,
            margin_ratio=1.0,
            tick_size=0.01,
            option_type=None,
            strike_price=None,
            expiry_date=None,
            lot_size=100.0,
        )
        self.engine.add_instrument(instr)

        # Configure Risk Manager
        self.engine.risk_manager.config.active = True
        self.engine.risk_manager.config.max_position_size = 10000.0

    def test_t_plus_one_mechanics(self) -> None:
        """Test T+1 mechanics."""
        print("\n=== Testing T+1 Mechanics ===")

        # --- Day 1: Buy ---
        print("Day 1: Buying 100 shares...")

        # Create a dummy strategy context (or just manipulate engine directly for
        # testing)
        # We'll simulate bar events manually by calling process_trades via run logic,
        # but here we can just test the internal state transitions if we could access
        # them.
        # Since we are in Python, we have to use the public API.
        # It's easier to write a simple script that mimics a strategy or just
        # injects orders?
        # Engine.run takes a strategy object.
        # Let's write a minimal strategy that places orders.
        pass


# We will implement a simple procedural test using a mock strategy
class T1Strategy(Strategy):
    """Mock strategy for T+1 testing."""

    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
        self.day = 0
        self.order_ids: list[str] = []

    def on_start(self) -> None:
        """Subscribe to symbols on start."""
        self.subscribe("000001")

    def on_bar(self, bar: Bar) -> None:
        """Handle bar events."""
        if self.ctx is None:
            return

        # Day 1 (2023-01-01)
        if self.day == 0:
            # Buy 100 shares
            if bar.close < 1000:  # Always true
                print("[Day 1] Sending Buy Order")
                self.ctx.buy(symbol=bar.symbol, quantity=100, price=bar.close)
                self.day += 1

        # Day 1 (Later) - Try to Sell
        elif self.day == 1:
            # Check position
            pos = self.ctx.get_position(bar.symbol)
            avail = self.ctx.get_available_position(bar.symbol)
            print(f"[Day 1] Position: {pos}, Available: {avail}")

            # Try to sell 100 shares (Should fail)
            print("[Day 1] Sending Sell Order (Should Fail)")
            self.ctx.sell(symbol=bar.symbol, quantity=100, price=bar.close)
            self.day += 1

        # Day 2 (2023-01-02)
        elif self.day == 2:
            # Check position
            pos = self.ctx.get_position(bar.symbol)
            avail = self.ctx.get_available_position(bar.symbol)
            print(f"[Day 2] Position: {pos}, Available: {avail}")

            # Try to sell 100 shares (Should Success)
            if avail >= 100:
                print("[Day 2] Sending Sell Order (Should Success)")
                self.ctx.sell(symbol=bar.symbol, quantity=100, price=bar.close)
            self.day += 1


def run_test() -> None:
    """Run the T+1 test."""
    engine = Engine()
    engine.use_china_market()
    engine.set_t_plus_one(True)
    engine.set_execution_mode(ExecutionMode.CurrentClose)  # Fill immediately at close
    engine.set_cash(100_000.0)

    symbol = "000001"
    instr = Instrument(
        symbol=symbol,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        option_type=None,
        strike_price=None,
        expiry_date=None,
        lot_size=100.0,
    )
    engine.add_instrument(instr)

    # Generate Data: Day 1 and Day 2
    # Day 1: 2023-01-04 (Wed)
    # Day 2: 2023-01-05 (Thu)
    bars = []

    # Day 1 Bar 1: Buy Trigger
    bars.append(
        Bar(
            timestamp=int(datetime(2023, 1, 4, 9, 31).timestamp() * 1e9),
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
            volume=1000.0,
            symbol=symbol,
        )
    )

    # Day 1 Bar 2: Sell Trigger (Same Day)
    bars.append(
        Bar(
            timestamp=int(datetime(2023, 1, 4, 14, 00).timestamp() * 1e9),
            open=10.5,
            high=10.5,
            low=10.5,
            close=10.5,
            volume=1000.0,
            symbol=symbol,
        )
    )

    # Day 2 Bar 1: Sell Trigger (Next Day)
    bars.append(
        Bar(
            timestamp=int(datetime(2023, 1, 5, 9, 31).timestamp() * 1e9),
            open=11.0,
            high=11.0,
            low=11.0,
            close=11.0,
            volume=1000.0,
            symbol=symbol,
        )
    )

    engine.add_bars(bars)

    strategy = T1Strategy()
    engine.run(strategy, show_progress=False)

    print("\n=== Test Results ===")

    # Verify Orders
    print(f"Total Orders: {len(engine.orders)}")
    for order in engine.orders:
        print(f"Order: {order.side} {order.quantity} @ {order.price} -> {order.status}")

    # Verify Day 1 Sell was Rejected
    day1_sells = [
        o
        for o in engine.orders
        if o.side == OrderSide.Sell and o.status == OrderStatus.Rejected
    ]
    if len(day1_sells) == 1:
        print("PASS: Day 1 Sell was correctly Rejected.")
    else:
        print(f"FAIL: Expected 1 Rejected Sell order on Day 1, found {len(day1_sells)}")

    # Verify Day 2 Sell was Filled
    day2_sells = [
        o
        for o in engine.orders
        if o.side == OrderSide.Sell and o.status == OrderStatus.Filled
    ]
    if len(day2_sells) == 1:
        print("PASS: Day 2 Sell was correctly Filled.")
    else:
        print(f"FAIL: Expected 1 Filled Sell order on Day 2, found {len(day2_sells)}")


if __name__ == "__main__":
    run_test()
