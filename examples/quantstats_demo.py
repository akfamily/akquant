import os
from typing import cast

import akquant as aq
import pandas as pd
from akquant import Strategy


# Define a simple strategy
class SimpleStrategy(Strategy):
    """Simple Buy and Hold strategy for testing."""

    def on_bar(self, bar: aq.Bar) -> None:
        """Execute buy on the first bar."""
        # Buy on first bar
        if not self.get_position(bar.symbol):
            self.buy(symbol=bar.symbol, quantity=100)
        # Hold forever (Buy and Hold)


# Create synthetic data
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
data = pd.DataFrame(
    {
        "date": dates,
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 100.0 + pd.Series(range(100)) * 0.1,  # Steady uptrend
        "volume": 1000,
        "symbol": "TEST",
    }
)

# Run backtest
print("Running backtest...")
result = aq.run_backtest(
    data=data, strategy=SimpleStrategy, symbol="TEST", initial_cash=10000.0
)

# Verify integration
print("\n--- Verifying QuantStats Integration ---")

# 1. Test to_quantstats()
print("Testing to_quantstats()...")
qs_returns = result.to_quantstats()
print(f"Returns Type: {type(qs_returns)}")
print(f"Returns Head:\n{qs_returns.head()}")
# Use cast to help mypy understand it's a DatetimeIndex
idx = cast(pd.DatetimeIndex, qs_returns.index)
print(f"Timezone info: {idx.tz}")

if qs_returns.empty:
    print("Error: Returns are empty!")
    exit(1)

# 2. Test report_quantstats()
print("\nTesting report_quantstats()...")
report_file = "test_qs_report.html"
try:
    # Use a mock benchmark to avoid network calls if possible, or just None
    # QS usually downloads SPY by default if benchmark is not None.
    # We pass None to skip benchmark download for speed/offline test.
    result.report_quantstats(benchmark=None, filename=report_file, title="Test Report")

    if os.path.exists(report_file):
        print(f"Success! Report generated at {report_file}")
        # Clean up
        os.remove(report_file)
    else:
        print("Error: Report file not found.")
except Exception as e:
    print(f"Error during report generation: {e}")

print("\nVerification Complete.")
