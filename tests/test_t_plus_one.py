from datetime import datetime, timedelta, timezone

import akquant
import pandas as pd
from akquant import AssetType, Bar, Engine, ExecutionMode, Instrument, Strategy


class TPlusOneStrategy(Strategy):
    """Strategy to test T+1 rule."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.day_1_buy_filled = False
        self.day_1_sell_rejected = False  # Or just check position
        self.day_2_sell_filled = False
        self.check_points: dict = {}

    def on_bar(self, bar: Bar) -> None:
        """Handle bar event."""
        # Convert timestamp to Shanghai time for check
        tz = timezone(timedelta(hours=8))
        dt = datetime.fromtimestamp(bar.timestamp / 1e9, tz=tz)
        print(f"DEBUG: OnBar {dt} (Day={dt.day}, Hour={dt.hour})")
        day = dt.day
        hour = dt.hour

        # Day 1, 10:00: Buy 100 shares
        # Use day 4 (Wednesday) instead of 1
        if day == 4 and hour == 10:
            pos = self.get_position(bar.symbol)
            print(f"Day 1 10:00 pos: {pos}")
            if pos == 0:
                self.buy(bar.symbol, 100)
                print("Placed buy order")

        # Day 1, 14:00: Check T+1 restriction
        elif day == 4 and hour == 14:
            # We expect the buy order from 10:00 to be filled
            pos = self.get_position(bar.symbol)
            print(f"Day 1 14:00 pos: {pos}")

            # Check orders
            # print(f"Orders: {self.ctx.get_orders(bar.symbol)}")

            if pos == 100:
                self.day_1_buy_filled = True

                # Check available (Should be 0 because T+1)
                avail = self.get_available_position(bar.symbol)
                self.check_points["day_1_avail"] = avail

        # Day 2, 10:00: Sell 100 shares (T+1 unlocked)
        elif day == 5:
            # Check available position again (Should be 100)
            avail = self.get_available_position(bar.symbol)
            print(f"Day 2 10:00 avail: {avail}")
            self.check_points["day_2_avail"] = avail

            if avail == 100:
                self.sell(bar.symbol, 100)
                self.day_2_sell_filled = True

    def on_order(self, order: akquant.Order) -> None:
        """Handle order update."""
        print(
            f"Order Update: {order.status} "
            f"{order.filled_quantity}@{order.average_filled_price}"
        )

    def on_trade(self, trade: akquant.Trade) -> None:
        """Handle trade update."""
        print(f"Trade: {trade.side} {trade.quantity}@{trade.price}")


def test_t_plus_one_mechanics() -> None:
    """Test T+1 trading rules."""
    # 1. Setup Data (3 Bars)
    # Day 1 10:00, Day 1 14:00, Day 2 10:00
    # Use weekdays to avoid market session filtering (Jan 4 2023 is Wed)
    tz = timezone(timedelta(hours=8))
    dates = [
        datetime(2023, 1, 4, 10, 0, tzinfo=tz),
        datetime(2023, 1, 4, 14, 0, tzinfo=tz),
        datetime(2023, 1, 5, 10, 0, tzinfo=tz),
    ]
    data = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 11.0, 12.0],
            "high": [12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0],
            "close": [11.0, 12.0, 13.0],
            "volume": [10000, 10000, 10000],
            "symbol": ["000001"] * 3,
        }
    )

    # 2. Setup Engine
    engine = Engine()
    engine.use_china_market()  # Sets T+1, simple fees
    # engine.set_t_plus_one(True) # Already set by use_china_market
    engine.set_cash(1_000_000.0)

    # Ensure execution mode allows filling on the same day (CurrentClose)
    engine.set_execution_mode(ExecutionMode.CurrentClose)

    # Add instrument
    instr = Instrument(
        symbol="000001",
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100.0,
    )
    engine.add_instrument(instr)

    # Convert to bars
    bars = []
    for _, row in data.iterrows():
        ts = int(row["date"].timestamp() * 1e9)
        bars.append(
            Bar(
                ts,
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
                row["symbol"],
            )
        )

    engine.add_bars(bars)

    # 3. Run Strategy
    strategy = TPlusOneStrategy()
    engine.run(strategy, show_progress=False)

    # 4. Assertions
    # Day 1: Buy filled, but available position should be 0 (T+1 lock)
    assert strategy.day_1_buy_filled, "Day 1 Buy should be filled"
    # Note: We check 'day_1_avail' captured at 14:00 on Day 1
    assert strategy.check_points.get("day_1_avail") == 0, (
        "Day 1 Available position should be 0 (T+1)"
    )

    # Day 2: Available position should unlock, Sell should fill
    assert strategy.check_points.get("day_2_avail") == 100, (
        "Day 2 Available position should be 100"
    )
    assert strategy.day_2_sell_filled, "Day 2 Sell should be filled"
