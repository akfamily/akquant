import akquant as aq
import pandas as pd
from akquant.backtest import BacktestResult

# We need a BacktestResult to inspect.
# We can run a quick backtest to get one.


def run_quick_backtest() -> BacktestResult:
    """Run a quick backtest to generate results for verification."""

    # Simple strategy
    class SimpleStrategy(aq.Strategy):
        def on_bar(self, bar: aq.Bar) -> None:
            pass

    # Create dummy data
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
            "symbol": "TEST",
        },
        index=dates,
    )

    result = aq.run_backtest(data=df, strategy=SimpleStrategy, show_progress=False)
    return result


if __name__ == "__main__":
    print("Running quick backtest...")
    result = run_quick_backtest()

    print("\n--- Inspecting metrics object ---")
    metrics = result._raw.metrics
    print(f"Type of metrics: {type(metrics)}")

    print(f"dir(metrics): {dir(metrics)}")

    print("\n--- Checking metrics_df ---")
    df = result.metrics_df
    print("metrics_df shape:", df.shape)
    print("metrics_df index:", df.index.tolist())
    print("metrics_df columns:", df.columns.tolist())

    print("\n--- Content ---")
    print(df)

    # Check if we have expected fields
    expected_fields = ["total_return", "sharpe_ratio", "max_drawdown"]
    missing = [f for f in expected_fields if f not in df.index]

    if missing:
        print(f"FAILED: Missing fields: {missing}")
    else:
        print("SUCCESS: Found expected fields.")

    # Check if we have values
    if df.empty:
        print("FAILED: DataFrame is empty")
    else:
        print("SUCCESS: DataFrame has data")
