"""
AKQuant Visualization Demo.

This script demonstrates the complete workflow of AKQuant:
1. Data Acquisition: Fetching real market data using AKShare.
2. Strategy Implementation: A simple trend-following strategy.
3. Backtesting: Running the backtest engine.
4. Visualization: Generating professional-grade interactive reports.
5. Trade Replay: Rendering K-line with buy/sell markers in final report.
"""

import akshare as ak
from akquant import (
    Bar,
    Strategy,
    run_backtest,
)
from akquant.utils import format_metric_value


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
    END_DATE = "20261231"
    INITIAL_CASH = 100_000.0

    df = ak.stock_zh_a_daily(symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE)
    df["symbol"] = SYMBOL

    # from akquant.config import BacktestConfig, StrategyConfig, RiskConfig
    # # 配置风险参数：safety_margin
    # risk_config = RiskConfig(safety_margin=0.0001)
    # strategy_config = StrategyConfig(risk=risk_config)
    # backtest_config = BacktestConfig(
    #     strategy_config=strategy_config,
    #     # start_time="20200131",
    #     # end_time="20260210"
    # )

    # 2. Run Backtest
    print("\nRunning Backtest...")
    result = run_backtest(
        data=df,
        strategy=MyStrategy,
        symbols=SYMBOL,
        initial_cash=INITIAL_CASH,
        show_progress=True,
        # config=backtest_config,
        start_time="20160101",
        end_time="20201231",
    )

    # 3. Print Metrics
    print("\nPerformance Metrics:")
    metrics = result.metrics
    total_return_display = format_metric_value(
        "total_return_pct", metrics.total_return_pct, width=6
    )
    annualized_return_display = format_metric_value(
        "annualized_return", metrics.annualized_return, width=6
    )
    max_drawdown_display = format_metric_value(
        "max_drawdown_pct", metrics.max_drawdown_pct, width=6
    )
    win_rate_display = format_metric_value("win_rate", metrics.win_rate, width=6)
    print(f"  Total Return:      {total_return_display}")
    print(f"  Annualized Return: {annualized_return_display}")
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:>6.2f}")
    print(f"  Max Drawdown:      {max_drawdown_display}")
    print(f"  Win Rate:          {win_rate_display}")
    print(f"  Total Trades:      {len(result.trades_df)}")

    # 4. Visualization
    print("\nGenerating Visualization...")
    report_file = "akquant_report.html"

    # Generate Consolidated Report
    # Using the new object-oriented API
    result.viz.report(
        title=f"AKQuant Report - {SYMBOL}",
        filename=report_file,
        show=True,  # Open automatically in browser
        market_data=df,
        plot_symbol=SYMBOL,
        include_trade_kline=True,
    )
    # result.viz.quantstats(
    #     benchmark=None, filename=report_file, title="Test Report"
    # )

    print(f"  - Report saved to: {report_file}")
    print("  - Report now includes K-line trade replay with buy/sell markers.")
    print("  - Open this file in your browser to view the report.")

    print("\nDemo completed successfully!")
