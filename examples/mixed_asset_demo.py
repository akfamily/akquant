import akquant as aq
import numpy as np
import pandas as pd
from akquant import InstrumentConfig

"""
Mixed Asset Backtest Demo
=========================

This example demonstrates how to configure multiple instruments with different
parameters (multiplier, margin ratio, etc.) using `instruments_config`.
"""


def create_dummy_data(
    symbol: str, start_date: str, n_bars: int, price: float = 100.0
) -> pd.DataFrame:
    """Generate simple dummy data for demonstration."""
    dates = pd.date_range(start_date, periods=n_bars, freq="B")
    # Random walk price
    np.random.seed(42)
    changes = np.random.randn(n_bars)
    prices = price + np.cumsum(changes)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 1000,
            "symbol": symbol,
        },
        index=dates,
    )
    return df


class TestStrategy(aq.Strategy):
    """A simple test strategy for mixed asset backtesting."""

    def __init__(self) -> None:
        """Initialize the strategy."""
        self.count = 0

    def on_bar(self, bar: aq.Bar) -> None:
        """Handle new bar events."""
        # Simple logic: Buy both assets on the first few bars
        if self.count < 2:
            print(f"[{bar.timestamp}] Buying {bar.symbol}")
            self.buy(bar.symbol, 1)
        self.count += 1


if __name__ == "__main__":
    # 1. Prepare Data
    print("Generating dummy data...")
    df_stock = create_dummy_data("STOCK_A", "2023-01-01", 100, 100.0)
    df_future = create_dummy_data("FUTURE_B", "2023-01-01", 100, 3500.0)

    data = {"STOCK_A": df_stock, "FUTURE_B": df_future}

    # 2. Configure Instruments
    # Define configuration for Future:
    # - Symbol: FUTURE_B
    # - Type: FUTURES
    # - Multiplier: 300 (e.g. IF Index Future)
    # - Margin Ratio: 0.1 (10% margin)
    future_config = InstrumentConfig(
        symbol="FUTURE_B",
        asset_type="FUTURES",
        multiplier=300.0,
        margin_ratio=0.1,
        tick_size=0.2,
    )

    # Note: STOCK_A will use default settings (Stock, Multiplier 1.0, Margin 1.0)

    # 3. Run Backtest
    print("Running mixed asset backtest...")
    result = aq.run_backtest(
        data=data,
        strategy=TestStrategy,
        instruments_config=[future_config],  # Pass the config list here
        cash=1_000_000.0,
        show_progress=True,
    )

    print("-" * 50)
    print("Backtest finished.")
    print(f"Total Return: {result.metrics.total_return_pct:.2f}%")

    print("\nDaily Positions Head:")
    print(result.daily_positions_df.head())
