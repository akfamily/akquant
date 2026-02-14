"""
AKQuant Intraday (Minute-level) Visualization Demo.

This script demonstrates AKQuant's capability to handle high-frequency
(minute-level) data:
1. Data Acquisition: Fetching 1-minute interval data.
2. Strategy: A simple moving average crossover strategy adapted for intraday.
3. Visualization: Showcasing adaptive X-axis and performance optimizations for
   large datasets.
"""

import akshare as ak
import numpy as np
import pandas as pd
from akquant import (
    Bar,
    Strategy,
    run_backtest,
)


# --------------------------------------------------------------------------------
# 2. Strategy Implementation
# --------------------------------------------------------------------------------
class IntradayMAStrategy(Strategy):
    """
    Intraday Moving Average Crossover Strategy.

    Logic:
    - Buy when Fast MA (5) > Slow MA (15)
    - Sell when Fast MA (5) < Slow MA (15)
    """

    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
        self.fast_window = 5
        self.slow_window = 15
        self.prices: list[float] = []

    def on_start(self) -> None:
        """Initialize strategy state."""
        # Subscribe to 1-minute data
        pass

    def on_bar(self, bar: Bar) -> None:
        """Handle bar updates."""
        symbol = bar.symbol
        self.prices.append(bar.close)

        if len(self.prices) < self.slow_window:
            return

        # Calculate MAs (Simple implementation for demo)
        # In production, use optimized indicators
        fast_ma = np.mean(self.prices[-self.fast_window :])
        slow_ma = np.mean(self.prices[-self.slow_window :])

        current_pos = self.get_position(symbol)

        # Entry Condition
        if current_pos == 0 and fast_ma > slow_ma:
            self.buy(symbol, 1000)  # Buy 1000 shares

        # Exit Condition
        elif current_pos > 0 and fast_ma < slow_ma:
            self.close_position(symbol)


# --------------------------------------------------------------------------------
# 3. Main Execution
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    SYMBOL = "sh600000"

    # Simulate Intraday Data generation
    # Since AKShare minute data might be limited or require specific permissions/tokens
    # for long history, we will generate synthetic minute data based on daily data
    # for demonstration purposes.
    print("Generating synthetic minute data for demonstration...")

    # 1. Get Daily Data first
    daily_df = ak.stock_zh_a_daily(
        symbol=SYMBOL, start_date="20230101", end_date="20230131"
    )

    # 2. Expand to Minute Data (Synthetic)
    # 4 hours trading per day = 240 minutes
    minute_rows = []
    for _, row in daily_df.iterrows():
        date = row["date"]
        open_p = row["open"]
        close_p = row["close"]
        high_p = row["high"]
        low_p = row["low"]
        vol = row["volume"] / 240  # Distribute volume

        # Create a simple intraday pattern: Open -> High -> Low -> Close
        # Interpolate 240 points
        # 09:30 - 11:30 (120 mins), 13:00 - 15:00 (120 mins)

        timestamps = []
        morning_start = pd.Timestamp(f"{date} 09:30:00")
        timestamps.extend([morning_start + pd.Timedelta(minutes=i) for i in range(120)])
        afternoon_start = pd.Timestamp(f"{date} 13:00:00")
        timestamps.extend(
            [afternoon_start + pd.Timedelta(minutes=i) for i in range(120)]
        )

        # Random walk bridge from Open to Close within High/Low bounds
        # Simplified: Linear interpolation for demo
        prices = np.linspace(open_p, close_p, 240)
        # Add some noise
        noise = np.random.normal(0, (high_p - low_p) * 0.05, 240)
        prices += noise

        for ts, price in zip(timestamps, prices):
            minute_rows.append(
                {
                    "timestamp": ts,
                    "symbol": SYMBOL,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": vol,
                }
            )

    minute_df = pd.DataFrame(minute_rows)
    # Ensure column order matches Bar definition if needed, or rely on prepare_dataframe
    # akquant expects 'date' or 'timestamp'
    minute_df.rename(columns={"timestamp": "date"}, inplace=True)

    print(f"Generated {len(minute_df)} minute bars.")

    # 2. Run Backtest
    print("\nRunning Intraday Backtest...")
    result = run_backtest(
        data=minute_df,
        strategy=IntradayMAStrategy,
        symbol=SYMBOL,
        initial_cash=1_000_000.0,
        show_progress=True,
    )

    # 3. Print Metrics
    print("\nPerformance Metrics:")
    metrics = result.metrics
    print(f"  Total Return:      {metrics.total_return_pct:>6.2f}%")
    print(f"  Annualized Return: {metrics.annualized_return:>6.2f}%")
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:>6.2f}")
    print(f"  Max Drawdown:      {metrics.max_drawdown_pct:>6.2f}%")
    print(f"  Win Rate:          {metrics.win_rate:>6.2f}%")
    print(f"  Total Trades:      {len(result.trades_df)}")

    # 4. Visualization
    print("\nGenerating Visualization...")
    report_file = "akquant_report_minute.html"

    result.report(
        title=f"AKQuant Intraday Report - {SYMBOL}", filename=report_file, show=True
    )
    print(f"  - Report saved to: {report_file}")
    print("  - Open this file in your browser to view the report.")

    print("\nIntraday Demo completed successfully!")
