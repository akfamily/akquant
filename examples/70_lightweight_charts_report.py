"""
AKQuant Lightweight Charts Report Demo.

This script demonstrates the plotly-free visualization path:
1. Data Acquisition: Fetching real market data using AKShare.
2. Strategy Implementation: A simple bullish/bearish bar strategy.
3. Backtesting: Running the backtest engine on multiple symbols.
4. Static Report: Generating a single-file HTML report powered by
   TradingView Lightweight Charts (metrics + equity/drawdown + K-line
   trade review). All reviewed symbols are embedded, so the reviewed
   stock can be hot-switched inside the page via the symbol input box.
5. Interactive Review: Starting a local web server where the reviewed
   stock is switched by typing any resolvable code in the page; codes
   missing from the preloaded data are resolved on demand from a
   data provider.
"""

import akshare as ak
import pandas as pd
from akquant import Bar, Strategy, run_backtest


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

        if current_pos == 0 and bar.close > bar.open:
            self.buy(symbol, 100)
        elif current_pos > 0 and bar.close < bar.open:
            self.close_position(symbol)


def load_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for the given symbols via AKShare."""
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = ak.stock_zh_a_daily(
            symbol="sh" + symbol, start_date="20230101", end_date="20261231"
        )
        df["symbol"] = symbol
        frames[symbol] = df
    return frames


if __name__ == "__main__":
    SYMBOLS = ["600000", "600004", "600006"]
    market_data = load_data(SYMBOLS)

    print("Running Backtest...")
    result = run_backtest(
        data=market_data,
        strategy=MyStrategy,
        symbols=SYMBOLS,
        initial_cash=1_000_000.0,
        show_progress=True,
    )
    print(f"Total Trades: {len(result.trades_df)}")

    # 1) Static single-file report: all symbols embedded, hot-switch in page.
    report_file = result.report_lwc(
        title="AKQuant 策略回测报告 (Lightweight Charts)",
        filename="akquant_lwc_report.html",
        market_data=market_data,
        show=True,
    )
    print(f"Report written to {report_file}")

    # 2) Interactive review server: type any code in the page to switch.
    #    Codes beyond `market_data` are resolved via `data_provider`.
    def data_provider(code: str) -> pd.DataFrame:
        """Resolve a bare 6-digit code into an OHLCV frame via AKShare."""
        prefix = "sh" if code.startswith("6") else "sz"
        df = ak.stock_zh_a_daily(
            symbol=prefix + code, start_date="20230101", end_date="20261231"
        )
        df["symbol"] = code
        return df

    result.serve_review(
        market_data=market_data,
        data_provider=data_provider,
        port=8765,
    )
