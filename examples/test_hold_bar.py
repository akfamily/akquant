import akquant as aq
import pandas as pd
from akquant import Bar, Strategy

pd.set_option("display.max_columns", None)


class TestHoldBarStrategy(Strategy):
    """Test strategy for hold_bar functionality."""

    def on_bar(self, bar: Bar) -> None:
        """Handle bar event."""
        symbol = bar.symbol
        pos = self.get_position(symbol)

        # 1. 第2个Bar买入
        if pos == 0:
            print(f"[{self.format_time(bar.timestamp)}] Buying...")
            self.buy(symbol, 100)

        # 2. 打印持有天数
        hold_bars = self.hold_bar(symbol)
        if pos != 0:
            ts_str = self.format_time(bar.timestamp)
            print(f"[{ts_str}] Holding {symbol}, hold_bar={hold_bars}")

        # 3. 持有 5 个 Bar 后卖出
        if hold_bars >= 5:
            print(f"[{self.format_time(bar.timestamp)}] Selling after 5 bars...")
            self.close_position(symbol)


def generate_data() -> pd.DataFrame:
    """Generate synthetic data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 101.0,
            "volume": 10000,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_data()
    print("Starting backtest...")
    result = aq.run_backtest(
        df, TestHoldBarStrategy, symbol="TEST", cash=10000, show_progress=False
    )
    print(result.trades_df)
