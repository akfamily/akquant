import pandas as pd
from akquant import Bar, Strategy, run_backtest


class TestStrategy(Strategy):
    """Strategy for testing trades dataframe."""

    def on_bar(self, bar: Bar) -> None:
        """Execute on every bar."""
        # Simple logic to generate trades
        # Buy on day 1, Sell on day 3
        if self.position.size == 0:
            self.buy(bar.symbol, 100)
        elif self.position.size > 0:
            # Sell 2 days later to ensure duration > 0
            self.sell(bar.symbol, 100)


def test_trades_df() -> None:
    """Test the structure and content of trades_df."""
    data = []
    # Create enough bars to generate trades
    # Day 1: Buy
    # Day 2: Hold
    # Day 3: Sell
    for i in range(5):
        data.append(
            Bar(
                timestamp=pd.Timestamp(f"2023-01-{i + 1:02d} 10:00:00").value,
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=100.0 + i,
                volume=1000,
                symbol="TEST",
            )
        )

    print("Running backtest...")
    result = run_backtest(
        data=data, strategy=TestStrategy, symbol="TEST", show_progress=False
    )

    print("\nTrades DataFrame:")
    print(result.trades_df)

    # Check columns
    expected = [
        "symbol",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "quantity",
        "side",
        "pnl",
        "net_pnl",
        "return_pct",
        "commission",
        "duration_bars",
        "duration",
    ]
    missing = [c for c in expected if c not in result.trades_df.columns]
    assert not missing, f"Missing columns: {missing}"

    if not result.trades_df.empty:
        trade = result.trades_df.iloc[0]
        assert trade["side"] == "Long"
        assert trade["quantity"] == 100.0
        assert pd.notna(trade["duration"])
        # Check duration type
        assert isinstance(trade["duration"], pd.Timedelta)
        print("\nAll expected columns present and duration is Timedelta.")
