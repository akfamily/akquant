from datetime import datetime, timedelta, timezone
from typing import Any, cast

import akquant
import pandas as pd
from akquant import (
    AssetType,
    Bar,
    CurrentClose,
    Engine,
    Instrument,
    Strategy,
    run_backtest,
)


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

    # Ensure execution policy allows filling on the same day (close + bar_offset=0)
    if hasattr(engine, "set_fill_mode"):
        cast(Any, engine).set_fill_mode(
            akquant.ExecutionMode.CurrentClose, "same_cycle"
        )

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


def _flat_daily_frame(symbol: str) -> pd.DataFrame:
    """Build a flat 5-day daily frame for ``symbol``."""
    days = [
        pd.Timestamp(f"2023-01-{day:02d} 10:00:00", tz="Asia/Shanghai")
        for day in (3, 4, 5, 6, 9)
    ]
    return pd.DataFrame(
        {
            "date": days,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1e6,
            "symbol": symbol,
        }
    )


def test_next_open_t_plus_one_rebalance_checks_position_at_fill_time() -> None:
    """Issue #391: a next-open sell may target a position unlocked by its fill.

    The sell leg is submitted at the previous close, when T+1 still locks the
    position, but it is matched at the next open *after* settlement has
    unlocked it. Checking availability at submission time wrongly rejected it,
    which starved the paired buy leg of cash.
    """

    class Rotate(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.step = 0

        def on_cross_section(self, trading_date: object, timestamp: int) -> None:
            if self.step == 0:
                self.rebalance_weights({"AAA": 0.99}, liquidate_unmentioned=True)
            elif self.step == 1:
                self.rebalance_weights({"BBB": 0.99}, liquidate_unmentioned=True)
            self.step += 1

    result = run_backtest(
        data={"AAA": _flat_daily_frame("AAA"), "BBB": _flat_daily_frame("BBB")},
        strategy=Rotate,
        symbols=["AAA", "BBB"],
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=100,
        history_depth=2,
        t_plus_one=True,
        show_progress=False,
    )

    orders = result.orders_df.copy()
    orders["symbol"] = orders["symbol"].astype(str).str.upper()
    orders["side"] = orders["side"].astype(str).str.lower()
    orders["status"] = orders["status"].astype(str).str.lower()
    observed = sorted(
        (sym, side.split(".")[-1], status.split(".")[-1])
        for sym, side, status in zip(
            orders["symbol"], orders["side"], orders["status"], strict=True
        )
    )
    assert observed == [
        ("AAA", "buy", "filled"),
        ("AAA", "sell", "filled"),
        ("BBB", "buy", "filled"),
    ]

    final_positions = result.positions.iloc[-1]
    assert float(final_positions.get("AAA", 0.0)) == 0.0
    assert float(final_positions.get("BBB", 0.0)) > 0.0


def test_next_open_cumulative_sells_cannot_oversell_available_position() -> None:
    """Two deferred sells in one slice must not both fill off one position.

    Neither sell is checked at submission (both are delayed), so the fill-time
    projection is the only thing stopping the second one from selling shares
    the first already sold. Exercises that the projection advances per fill.
    """

    class DoubleSell(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.step = 0

        def on_cross_section(self, trading_date: object, timestamp: int) -> None:
            if self.step == 0:
                self.buy("AAA", 100)
            elif self.step == 2:
                # Position is 100 and unlocked; ask for 100 twice in one slice.
                self.sell("AAA", 100)
                self.sell("AAA", 100)
            self.step += 1

    result = run_backtest(
        data={"AAA": _flat_daily_frame("AAA")},
        strategy=DoubleSell,
        symbols=["AAA"],
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=100,
        history_depth=2,
        t_plus_one=True,
        show_progress=False,
    )

    orders = result.orders_df
    sells = orders[orders["side"].astype(str).str.lower().str.contains("sell")]
    statuses = sells["status"].astype(str).str.lower().tolist()
    assert len(statuses) == 2, f"expected two sell orders, got {statuses}"
    filled = [s for s in statuses if "filled" in s and "partially" not in s]
    rejected = [s for s in statuses if "reject" in s]
    assert len(filled) == 1, f"exactly one sell may fill, got {statuses}"
    assert len(rejected) == 1, f"the oversell must be rejected, got {statuses}"

    # The position must not go short.
    final_positions = result.positions.iloc[-1]
    assert float(final_positions.get("AAA", 0.0)) == 0.0


def test_next_open_sell_without_position_is_still_rejected() -> None:
    """Deferring the check must not let a genuinely unbacked sell through.

    Guards the obvious over-correction: if the position is never bought, the
    sell has no availability at fill time either and must still be rejected.
    """

    class SellNaked(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.step = 0
            self.rejects: list[str] = []

        def on_cross_section(self, trading_date: object, timestamp: int) -> None:
            if self.step == 1:
                self.sell("AAA", 100)
            self.step += 1

        def on_reject(self, order: akquant.Order) -> None:
            self.rejects.append(str(order.reject_reason))

    result = run_backtest(
        data={"AAA": _flat_daily_frame("AAA")},
        strategy=SellNaked,
        symbols=["AAA"],
        initial_cash=1_000_000.0,
        lot_size=100,
        history_depth=2,
        t_plus_one=True,
        show_progress=False,
    )

    orders = result.orders_df
    statuses = orders["status"].astype(str).str.lower().tolist()
    assert len(statuses) == 1
    assert "reject" in statuses[0]
    reasons = orders["reject_reason"].astype(str).tolist()
    assert "Insufficient available position" in reasons[0]


def test_same_cycle_sell_of_locked_position_is_rejected_at_submission() -> None:
    """T+0 intraday resale must stay rejected under same-cycle close fills.

    ``bar_offset == 0`` means submission time *is* fill time, so there is no
    settlement in between and no reason to defer the check.
    """

    class BuyThenSell(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.seen = 0

        def on_bar(self, bar: Bar) -> None:
            self.seen += 1
            if self.seen == 1:
                self.buy(bar.symbol, 100)
            elif self.seen == 2:
                # Same trading day: position exists but is T+1 locked.
                self.sell(bar.symbol, 100)

    tz = timezone(timedelta(hours=8))
    dates = [
        datetime(2023, 1, 4, 10, 0, tzinfo=tz),
        datetime(2023, 1, 4, 14, 0, tzinfo=tz),
    ]
    data = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 10.0],
            "high": [10.0, 10.0],
            "low": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [1e6, 1e6],
            "symbol": ["000001"] * 2,
        }
    )

    result = run_backtest(
        data={"000001": data},
        strategy=BuyThenSell,
        symbols=["000001"],
        initial_cash=1_000_000.0,
        lot_size=100,
        t_plus_one=True,
        fill_policy=CurrentClose(),
        show_progress=False,
    )

    orders = result.orders_df
    sells = orders[orders["side"].astype(str).str.lower().str.contains("sell")]
    assert len(sells) == 1
    assert "reject" in str(sells.iloc[0]["status"]).lower()
    reason = str(sells.iloc[0]["reject_reason"])
    assert "Insufficient available position" in reason
    # Must be rejected at submission, not deferred to the fill: with
    # bar_offset == 0 there is no settlement in between, so deferring would
    # only move `on_reject` later for no benefit.
    assert "at execution" not in reason, (
        f"same-cycle sell must keep the submission-time rejection, got: {reason}"
    )
