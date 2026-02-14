from datetime import timedelta
from typing import List, cast

import akquant as aq
import numpy as np
import pandas as pd
from akquant import InstrumentConfig

"""
Multi-Frequency Backtest Demo
=============================

This example demonstrates how to run a strategy with mixed frequencies:
1.  **STOCK_1D**: Daily bars, used for trend filtering (e.g., Moving Average).
2.  **STOCK_1M**: 1-minute bars, used for signal execution.

The strategy logic:
-   **Daily Trend**: Calculate a Simple Moving Average (SMA) on the Daily data.
    -   If Close > SMA, Trend = UP.
    -   If Close < SMA, Trend = DOWN.
-   **Intraday Signal**:
    -   If Trend is UP and Price < Intraday Lower Band -> Buy.
    -   If Trend is DOWN and Price > Intraday Upper Band -> Sell.
"""


def create_dummy_data_1m(symbol: str, start_date: str, days: int) -> pd.DataFrame:
    """Generate 1-minute dummy data for a few days (A-share hours)."""
    # Trading hours: 9:31-11:30 (120 min), 13:01-15:00 (120 min)
    # Total 240 bars per day
    timestamps: List[pd.Timestamp] = []
    base_price = 100.0
    prices: List[float] = []

    print(f"DEBUG: Generating data starting from {start_date}")
    # Ensure start_date is parsed correctly
    current_date = pd.Timestamp(start_date)
    # Make sure it is naive
    if current_date.tz is not None:
        current_date = current_date.tz_localize(None)

    for _ in range(days):
        # Skip weekends if needed, but for simplicity we assume start_date is Monday
        # or just generate consecutive days for demo logic

        # Morning session: 09:31 to 11:30
        # date_range end is inclusive, so we go from 09:31 to 11:30
        rng_am = pd.date_range(
            start=current_date + timedelta(hours=9, minutes=31),
            end=current_date + timedelta(hours=11, minutes=30),
            freq="1min",
        )

        # Afternoon session: 13:01 to 15:00
        rng_pm = pd.date_range(
            start=current_date + timedelta(hours=13, minutes=1),
            end=current_date + timedelta(hours=15, minutes=0),
            freq="1min",
        )

        timestamps.extend(rng_am)
        timestamps.extend(rng_pm)

        current_date += timedelta(days=1)

    # Generate random walk
    n = len(timestamps)
    np.random.seed(42)
    changes = np.random.randn(n) * 0.1  # Small changes for 1min
    p = base_price
    for c in changes:
        p += c
        prices.append(p)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": np.array(prices) + 0.05,
            "low": np.array(prices) - 0.05,
            "close": prices,
            "volume": 1000,
            "symbol": symbol,
        },
        index=timestamps,
    )
    return df


def create_dummy_data_1d(
    symbol: str, start_date: str, days: int, df_1m: pd.DataFrame
) -> pd.DataFrame:
    """Generate Daily dummy data derived from 1-minute data."""
    df_1m_copy = df_1m.copy()
    # Resample by Day
    daily_groups = df_1m_copy.resample("1D")

    daily_data = []
    daily_index: List[pd.Timestamp] = []

    for date_val, group in daily_groups:
        if group.empty:
            continue

        date = cast(pd.Timestamp, date_val)
        # Daily bar timestamp: 15:00 of the day (Asia/Shanghai)
        ts = date + timedelta(hours=15)
        # Ensure ts is naive
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)

        daily_data.append(
            {
                "open": group.iloc[0]["open"],
                "high": group["high"].max(),
                "low": group["low"].min(),
                "close": group.iloc[-1]["close"],
                "volume": group["volume"].sum(),
                "symbol": symbol,
            }
        )
        daily_index.append(ts)

    df = pd.DataFrame(daily_data, index=daily_index)
    return df


class MultiFreqStrategy(aq.Strategy):
    """Strategy using both 1-minute and Daily data."""

    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
        self.daily_trend = 0  # 1 for Up, -1 for Down, 0 for Neutral
        self.ma_window = 3  # Short MA for demo
        self.daily_prices: List[float] = []

        # Track position manually for simple demo (optional, akquant handles it)
        self.has_position = False

    def on_start(self) -> None:
        """Call when strategy starts."""
        print("Strategy Started")

    def on_bar(self, bar: aq.Bar) -> None:
        """
        Handle incoming bars.

        Distinguish behavior based on bar.symbol.
        """
        # Convert UTC nanoseconds to configured timezone (default Asia/Shanghai)
        ts_bj = self.to_local_time(bar.timestamp)

        # Use helper for formatted string (Default: %Y-%m-%d %H:%M:%S)
        ts_str = self.format_time(bar.timestamp)

        if bar.symbol == "000001.SZ_1D":
            print(f"\n{'=' * 60}")
            print(
                f"EVENT: Daily Bar Arrived | {ts_str} (BJ) | "
                f"{bar.symbol} | Close: {bar.close:.2f}"
            )
            self._handle_daily_bar(bar)
            print(f"{'=' * 60}\n")
        elif bar.symbol == "000001.SZ":
            # For demonstration: Print first bar of the day to show alignment
            if ts_bj.hour == 9 and ts_bj.minute == 31:
                print(f"--- Market Open {ts_bj.strftime('%Y-%m-%d')} ---")

            self._handle_minute_bar(bar, ts_bj)

    def _handle_daily_bar(self, bar: aq.Bar) -> None:
        """Process Daily Bar: Update Trend."""
        self.daily_prices.append(bar.close)

        # Keep only needed history
        if len(self.daily_prices) > self.ma_window + 5:
            self.daily_prices.pop(0)

        # Calculate MA
        if len(self.daily_prices) >= self.ma_window:
            ma = sum(self.daily_prices[-self.ma_window :]) / self.ma_window
            prev_trend = self.daily_trend

            # Determine Trend
            if bar.close > ma:
                self.daily_trend = 1
                trend_str = "UP"
            else:
                self.daily_trend = -1
                trend_str = "DOWN"

            print(
                f"[Logic 1D] MA({self.ma_window}): {ma:.2f} vs Close: {bar.close:.2f}"
            )
            print(
                f"[Logic 1D] Trend Update: {prev_trend} -> "
                f"{self.daily_trend} ({trend_str})"
            )
        else:
            print(
                f"[Logic 1D] Collecting history... "
                f"({len(self.daily_prices)}/{self.ma_window})"
            )

    def _handle_minute_bar(self, bar: aq.Bar, ts_bj: pd.Timestamp) -> None:
        """Process Minute Bar: Execute Signals based on Trend."""
        # ctx.get_position returns the float quantity directly
        pos_qty = 0.0
        if self.ctx:
            pos_qty = self.ctx.get_position("000001.SZ")

        # Format time for logging
        ts_str = ts_bj.strftime("%H:%M")

        # Debug: Print first bar of the day to verify timestamp alignment
        # if ts_bj.hour == 9 and ts_bj.minute == 31:
        #    print(f"DEBUG: Bar Time: {ts_bj} (UTC: {ts_utc})")

        if self.daily_trend == 1:  # Uptrend
            # Buy logic
            if pos_qty == 0:
                print(
                    f"  >>> SIGNAL [BUY] at {ts_str} (BJ) | Price: {bar.close:.2f} | "
                    "Reason: Trend UP & No Position"
                )
                self.buy("000001.SZ", 100)

        elif self.daily_trend == -1:  # Downtrend
            # Sell logic
            if pos_qty > 0:
                print(
                    f"  >>> SIGNAL [SELL] at {ts_str} (BJ) | Price: {bar.close:.2f} | "
                    "Reason: Trend DOWN & Have Position"
                )
                self.sell("000001.SZ", pos_qty)

        # If trend is 0, do nothing


if __name__ == "__main__":
    # 1. Prepare Data
    print("Generating dummy data (5 Days) for A-shares...")
    # Generate 5 days of data
    df_1m = create_dummy_data_1m("000001.SZ", "2024-01-01", 5)
    df_1d = create_dummy_data_1d("000001.SZ_1D", "2024-01-01", 5, df_1m)

    print(f"Generated {len(df_1m)} minute bars and {len(df_1d)} daily bars.")

    # Pack data
    # Note: We are simulating trading on '000001.SZ'.
    # '000001.SZ_1D' is just for reference.
    data = {"000001.SZ": df_1m, "000001.SZ_1D": df_1d}

    # 2. Configure Instruments
    # We treat them as separate instruments in configuration,
    # but logically they refer to the same asset.
    # We only trade 000001.SZ, so we configure it.
    stock_1m_config = InstrumentConfig(
        symbol="000001.SZ",
        asset_type="STOCK",
        multiplier=1.0,
    )
    # STOCK_1D config is optional if we don't trade it,
    # but good practice to define it if we wanted to be strict.
    # akquant creates default configs if missing.

    # 3. Run Backtest
    print("\nStarting Multi-Frequency Backtest (A-share Simulated)...")
    result = aq.run_backtest(
        data=data,
        strategy=MultiFreqStrategy,
        instruments_config=[stock_1m_config],
        initial_cash=100_000.0,
        show_progress=True,
    )

    print("\n" + "=" * 50)
    print("Backtest Results")
    print("-" * 50)
    print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
    print("=" * 50)
