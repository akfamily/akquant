"""
AKQuant Visualization Demo.

This script demonstrates the complete workflow of AKQuant:
1. Data Acquisition: Fetching real market data using AKShare.
2. Strategy Implementation: A simple trend-following strategy.
3. Backtesting: Running the backtest engine.
4. Visualization: Generating professional-grade interactive reports.
"""

import akshare as ak
from akquant import (
    Bar,
    Strategy,
    run_backtest,
)


# --------------------------------------------------------------------------------
# 2. Strategy Implementation
# --------------------------------------------------------------------------------
class MyStrategy(Strategy):
    """
    Simple Trend Following Strategy.

    Logic:
    - Buy when Close > Open (Bullish Bar) and no position.
    - Sell when Close < Open (Bearish Bar) and holding position.
    """

    def on_bar(self, bar: Bar) -> None:
        """Handle new bar data."""
        symbol = bar.symbol
        current_pos = self.get_position(symbol)

        # Entry Condition: Bullish Candle & No Position
        if current_pos == 0 and bar.close > bar.open:
            self.buy(symbol, 100)

        # Exit Condition: Bearish Candle & Holding Position
        elif current_pos > 0 and bar.close < bar.open:
            self.close_position(symbol)


# --------------------------------------------------------------------------------
# 3. Main Execution
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    SYMBOL = "sh600000"
    START_DATE = "20120101"
    END_DATE = "20231231"
    INITIAL_CASH = 100_000.0

    df = ak.stock_zh_a_daily(symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE)
    df["symbol"] = SYMBOL

    # 2. Run Backtest
    print("\nRunning Backtest...")
    result = run_backtest(
        data=df,
        strategy=MyStrategy,
        symbol=SYMBOL,
        initial_cash=INITIAL_CASH,
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
    report_file = "akquant_report.html"

    # Generate Consolidated Report
    # Using the new object-oriented API
    result.report(
        title=f"AKQuant Report - {SYMBOL}",
        filename=report_file,
        show=True,  # Open automatically in browser
    )
    # result.report_quantstats(
    #     benchmark=None, filename=report_file, title="Test Report"
    # )

    print(f"  - Report saved to: {report_file}")
    print("  - Open this file in your browser to view the report.")

    print("\nDemo completed successfully!")
