# Quick Start

Welcome to AKQuant! Let's get the simplest strategy running as fast as possible.

## 1. Installation

Open your terminal and run:

```bash
pip install akquant
```

## 2. Minimal Example: Buy and Hold

Copy the code below into `quickstart.py` and run it. This strategy is very simple: **Buy 100 shares on the first day and hold them forever**.

```python
import pandas as pd
from akquant import Strategy, run_backtest

# 1. Define Strategy
class BuyAndHold(Strategy):
    def on_bar(self, bar):
        # If no position, buy 100 shares
        if self.get_position(bar.symbol) == 0:
            print(f"[{bar.timestamp_str}] Buy {bar.symbol}")
            self.buy(bar.symbol, 100)

# 2. Prepare Data (Generating simple mock data here)
dates = pd.date_range("2023-01-01", periods=100)
df = pd.DataFrame({
    "date": dates,
    "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, # Assuming price stays flat
    "volume": 1000,
    "symbol": "DEMO"
})

# 3. Run Backtest
print("Starting backtest...")
result = run_backtest(
    data=df,
    strategy=BuyAndHold,
    cash=10000.0
)

# 4. View Results
print(result)
```

In the output, you will see core metrics like `total_return` and `max_drawdown`.

## 3. Advanced Learning

Too simple? Want to learn how to write real quantitative strategies (like Dual Moving Average, MACD, etc.)?

👉 **Please read [Tutorial: Writing Your First Strategy](tutorial.md)**

This tutorial will cover:

*   How to get historical data (`get_history`)
*   How to calculate technical indicators (MA, RSI)
*   How to implement stop-loss and take-profit
