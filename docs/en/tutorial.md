# Tutorial: Writing Your First Strategy

This tutorial will guide you step-by-step through writing a classic **Dual Moving Average Strategy**. This is often the "Hello World" for quantitative traders.

We will cover the entire process from data preparation and strategy writing to backtest analysis.

## 1. Strategy Logic

The logic of the Dual Moving Average strategy is very intuitive:
*   **Golden Cross**: When the short-term moving average (e.g., 10-day MA) crosses above the long-term moving average (e.g., 30-day MA), the trend is considered upward, so **Buy**.
*   **Death Cross**: When the short-term moving average crosses below the long-term moving average, the trend is considered downward, so **Sell**.

## 2. Complete Code

For your convenience, we have put all the code in one file. You can copy the following code into `my_first_strategy.py` and run it.

```python
import pandas as pd
import numpy as np
from akquant import Strategy, run_backtest

# --------------------------
# Step 1: Prepare Data
# --------------------------
# In a real scenario, you would use pd.read_csv("stock_data.csv")
# Here, we generate some mock data for demonstration purposes
def generate_mock_data():
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    n = len(dates)

    # Generate a random walk
    np.random.seed(42) # Fix random seed for consistent results
    returns = np.random.normal(0.0005, 0.02, n)
    price = 100 * np.cumprod(1 + returns)

    # Construct DataFrame
    df = pd.DataFrame({
        "date": dates,
        "open": price,
        "high": price * 1.01,
        "low": price * 0.99,
        "close": price,
        "volume": 10000,
        "symbol": "AAPL" # Assuming this is Apple stock
    })
    return df

# --------------------------
# Step 2: Write Strategy
# --------------------------
class DualMAStrategy(Strategy):
    """
    Inherits from akquant.Strategy, which is the base class for all strategies.
    """

    # New (Recommended): Declarative setting of history warmup period
    # The framework will automatically handle set_history_depth for you, no need to call manually in on_start
    # Priority: Dynamic property (self.warmup_period) > Class property (warmup_period) > AST inference
    warmup_period = 40 # 30-day MA + 10 safety margin

    def __init__(self, fast_window=10, slow_window=30):
        # Define strategy parameters: fast and slow window sizes
        self.fast_window = fast_window
        self.slow_window = slow_window

        # Advanced tip: Dynamic warmup period
        # If window sizes are passed dynamically, it's recommended to override warmup_period here
        self.warmup_period = slow_window + 10

    def on_start(self):
        """
        Executed once when the strategy starts.
        Tell the system which stocks we want to watch here.
        """
        print("Strategy starting...")
        self.subscribe("AAPL")

        # Method 1 (Recommended): Use warmup_period (supports Class/Instance property / AST inference)
        # Method 2 (Legacy): Manually call self.set_history_depth(self.slow_window + 10)

        # Since we have configured warmup_period, no action is needed here
        # The framework automatically takes max(warmup_period, ast_inferred_value, run_backtest_history_depth)

    def on_bar(self, bar):
        """
        Core logic: Triggered every time a bar closes.
        The bar parameter contains current market data (bar.close, bar.high, etc.).
        """

        # 1. Get historical closing prices
        # get_history returns a numpy array containing the last N days of data
        closes = self.get_history(count=self.slow_window, symbol=bar.symbol, field="close")

        # If not enough data to calculate the long MA (e.g., first few days of backtest), return immediately
        if len(closes) < self.slow_window:
            return

        # 2. Calculate Moving Averages
        # Use numpy to calculate mean
        fast_ma = np.mean(closes[-self.fast_window:]) # Average of the last fast_window data points
        slow_ma = np.mean(closes[-self.slow_window:]) # Average of the last slow_window data points

        # 3. Get Current Position
        # Returns 0 if no position, 1000 if holding 1000 shares
        position = self.get_position(bar.symbol)

        # 4. Trading Signal Judgment
        # Golden Cross: Short-term > Long-term, and currently empty position -> Buy
        if fast_ma > slow_ma and position == 0:
            print(f"[{bar.timestamp_str}] Golden Cross Buy! Price: {bar.close:.2f}")
            self.buy(symbol=bar.symbol, quantity=1000)

        # Death Cross: Short-term < Long-term, and currently holding position -> Sell
        elif fast_ma < slow_ma and position > 0:
            print(f"[{bar.timestamp_str}] Death Cross Sell! Price: {bar.close:.2f}")
            self.sell(symbol=bar.symbol, quantity=position) # Sell all holdings

# --------------------------
# Step 3: Run Backtest
# --------------------------
if __name__ == "__main__":
    # 1. Get Data
    df = generate_mock_data()

    # 2. Run Backtest
    print("Starting backtest...")
    result = run_backtest(
        data=df,
        strategy=DualMAStrategy, # Pass our strategy class
        strategy_params={"fast_window": 10, "slow_window": 30}, # Adjust parameters
        cash=100_000.0,    # Initial capital 100k
        commission=0.0003  # Commission 0.03%
    )

    # 3. Print Results
    print("\n" + "="*30)
    print(result) # Print detailed performance report
    print("="*30)
```

## 3. Code Detailed Analysis

### 3.1 Strategy Structure
Every AKQuant strategy is a Python class inheriting from `Strategy`. You need to focus on three main methods:

*   **`__init__`**: Set strategy parameters (e.g., moving average periods).
*   **`on_start`**: Initialization work.
    *   **Data Warmup (New Recommended)**: Set `warmup_period`.
        *   **Static Setting**: Class attribute `warmup_period = 40`.
        *   **Dynamic Setting**: `self.warmup_period = slow_window + 10` in `__init__`.
        *   **Auto Inference**: If code contains `SMA(30)`, the framework will also try to infer automatically (AST).
    *   **Data Warmup (Legacy Compatible)**: Call `self.set_history_depth(40)`. If not set, `get_history` will raise an error.
*   **`on_bar`**: This is an infinite loop where the system passes every K-line to you in chronological order. You make decisions here: buy or sell?

### 3.2 Getting Data
*   `bar.close`: Closing price of the current K-line.
*   `self.get_history(N, symbol, "close")`: Get an array of closing prices for the past N days. This is the most efficient way to access data.

### 3.3 Placing Orders
*   `self.buy(symbol, quantity)`: Send a buy order. Defaults to a market order (executes at current price).
*   `self.sell(symbol, quantity)`: Send a sell order.

## 4. Frequently Asked Questions (FAQ)

**Q: Why does the program error with `RuntimeError: History tracking is not enabled`?**
A: This is because you called `get_history` but did not set the historical data depth. Please ensure you set the `warmup_period` attribute in the strategy class (recommended), or call `self.set_history_depth()` in `on_start`. The system defaults to not enabling history caching for performance, so it must be explicitly enabled.

**Q: Why does `get_history` return a lot of `NaN`?**
A: This usually happens at the beginning of the backtest. For example, if you set depth=30 but call `get_history(30)` on the 5th day. At this time, there is insufficient data, so the system automatically pads with `NaN` to ensure the returned array length is consistent. It is recommended to check `if len(closes) < N: return` at the beginning of `on_bar`.

**Q: What is `sharpe_ratio` in the backtest results?**
A: Sharpe Ratio, a metric measuring the cost-effectiveness of a strategy. Greater than 1 is usually considered okay, and greater than 2 is considered excellent.

**Q: How do I switch to my own data?**
A: Just replace `data=df` with your own DataFrame. Ensure your DataFrame contains columns: `date`, `open`, `high`, `low`, `close`, `volume`, `symbol`.

## 5. Advanced: Parameter Optimization

In actual development, we often need to find the optimal parameters for a strategy (e.g., is 10/30 better for moving averages, or 5/20?). AKQuant provides built-in grid search tools.

```python
from akquant import run_optimization

# Define parameter grid
param_grid = {
    "fast_window": range(5, 30, 5),   # [5, 10, 15, 20, 25]
    "slow_window": range(20, 60, 10), # [20, 30, 40, 50]
}

# Run optimization
results_df = run_optimization(
    strategy=DualMAStrategy,
    param_grid=param_grid,
    data=df,
    cash=100_000.0,
    commission=0.0003,
    sort_by="total_return", # Sort by return
    ascending=False
)

# Print top 5
print(results_df.head(5))
```

`run_optimization` will automatically utilize multi-core CPUs to accelerate backtesting in parallel and return a DataFrame containing all parameter combinations and corresponding performance metrics.
