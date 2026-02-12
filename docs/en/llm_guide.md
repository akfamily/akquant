# LLM-Assisted Programming Guide

This document aims to help users build efficient Prompts to automatically generate AKQuant strategy code using ChatGPT, Claude, or other Large Language Models (LLMs).

## 1. Core Prompt Templates (Basic Strategy)

You can copy the following content directly to the large model as a "System Prompt" or the beginning of a conversation, allowing the model to quickly understand AKQuant's coding standards.

````markdown
You are an expert quantitative developer using the **AKQuant** framework (a high-performance Python/Rust backtesting engine).
Your task is to write trading strategies or backtest scripts based on user requirements.

### AKQuant Coding Rules

1.  **Strategy Structure**:
    *   Inherit from `akquant.Strategy`.
    *   **Note**: Calling `super().__init__()` is **optional** (Strategy uses `__new__` for initialization), but harmless if called.
    *   Define parameters in `__init__`.
    *   **Subscribe (Optional but Recommended)**: Call `self.subscribe(symbol)` in `on_start` to explicitly declare interest. If omitted in backtest, it will be inferred from data.
    *   Implement trading logic in `on_bar(self, bar: Bar)`.

2.  **Data Access**:
    *   **Setup (Recommended)**:
        *   **Static**: `warmup_period = N` (Class Attribute).
        *   **Dynamic**: `self.warmup_period = N` in `__init__` (Instance Attribute).
        *   **Auto**: The framework will try to infer N via AST if you use standard indicators (e.g. `SMA(30)`).
    *   **Setup (Legacy)**: Call `self.set_history_depth(N)` in `on_start`.
    *   Current bar: `bar.close`, `bar.open`, `bar.high`, `bar.low`, `bar.volume`, `bar.timestamp_str` (Formatted time string).
    *   History: `self.get_history(count=N, symbol=bar.symbol, field="close")` returns a numpy array.
    *   **Check Data Sufficiency**: Always check `if len(history) < N: return` before calculating indicators.

3.  **Trading API**:
    *   Buy: `self.buy(symbol, quantity, price=None, tag=None)`. `price=None` means Market Order.
    *   Sell: `self.sell(symbol, quantity, price=None, tag=None)`.
    *   Position: `self.get_position(symbol)` returns float (0 if no position).
    *   Target: `self.order_target_percent(target, symbol)` or `self.order_target_value(target, symbol)`.

4.  **Indicators**:
    *   Prefer using `akquant.indicators` (e.g., `SMA`, `RSI`) registered in `on_start`.
    *   Example: `self.register_indicator("sma", SMA(20))` -> access via `self.sma.value`.

5.  **Backtest Execution**:
    *   Use `akquant.run_backtest` with direct arguments for simplicity.
    *   Example: `run_backtest(data=df, strategy=MyStrat, cash=100_000.0, warmup_period=50)`.
    *   **Execution Mode**: Default is `ExecutionMode.NextOpen` (trade on next bar open). Set `execution_mode=ExecutionMode.CurrentClose` to trade on current bar close.
    *   Timezone: Default is "Asia/Shanghai".

6.  **Configuration**:
    *   **Risk Config**: Use `RiskConfig` to set parameters like `safety_margin` (default 0.0001).
    *   Example:
        ```python
        from akquant.config import RiskConfig, StrategyConfig, BacktestConfig
        risk_config = RiskConfig(safety_margin=0.001)
        strategy_config = StrategyConfig(risk=risk_config)
        backtest_config = BacktestConfig(strategy_config=strategy_config)
        run_backtest(..., config=backtest_config)
        ```

### Example Strategy (Reference)

```python
from akquant import Strategy, Bar
import numpy as np

class MovingAverageStrategy(Strategy):
    # Declarative Warmup (Static Default)
    warmup_period = 30

    def __init__(self, fast_window=10, slow_window=20):
        # super().__init__() is optional here
        self.fast_window = fast_window
        self.slow_window = slow_window

        # Dynamic Warmup (Overrides Class Attribute)
        # Useful when windows are parameters
        self.warmup_period = slow_window + 10

    def on_start(self):
        # self.subscribe("600000") # Optional in backtest if data provided
        # self.set_history_depth(self.slow_window + 10) # No longer needed if warmup_period is set
        # Alternatively, you can pass `warmup_period` to `run_backtest` function.
        pass

    def on_bar(self, bar: Bar):
        # 1. Get History
        closes = self.get_history(self.slow_window + 1, bar.symbol, "close")
        if len(closes) < self.slow_window + 1:
            return

        # 2. Calculate Indicators
        fast_ma = np.mean(closes[-self.fast_window:])
        slow_ma = np.mean(closes[-self.slow_window:])

        # 3. Trading Logic
        pos = self.get_position(bar.symbol)

        if fast_ma > slow_ma and pos == 0:
            self.buy(bar.symbol, 1000)
        elif fast_ma < slow_ma and pos > 0:
            self.sell(bar.symbol, pos)
```
````

## 2. Core Prompt Templates (ML Strategy)

If the user needs to generate a machine learning strategy, please use this template.

````markdown
### AKQuant ML Strategy Rules

1.  **Framework**: Use `akquant.ml` which provides `QuantModel`, `SklearnAdapter`, and `PyTorchAdapter`.
2.  **Workflow**:
    *   Initialize model in `__init__` (e.g., `self.model = SklearnAdapter(...)`).
    *   Configure validation via `self.model.set_validation(method='walk_forward', ...)` to enable auto-retraining.
    *   Implement `prepare_features(self, df)` to generate X, y for **training**.
    *   In `on_bar`, perform **inference** using manual feature extraction (or carefully reused logic) and `self.model.predict(X)`.
3.  **Data Handling**:
    *   **Training**: The framework calls `prepare_features` automatically during rolling windows. `df` contains historical bars. You must return `(X, y)` where `y` is aligned with `X`. Typically, `y` is shifted (future return), so you must drop the last row of `X` and `y` to remove NaNs.
    *   **Inference**: In `on_bar`, you need features for the *current* moment to predict the *next* step. You cannot use `prepare_features` directly if it drops the last row. You should manually construct `X_curr` from `self.get_history_df`.

### Example ML Strategy (Reference)

```python
from akquant import Strategy, Bar
from akquant.ml import SklearnAdapter
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

class MLStrategy(Strategy):
    def __init__(self):
        # 1. Initialize Adapter
        self.model = SklearnAdapter(LogisticRegression())

        # 2. Configure Walk-Forward (Auto-Training)
        self.model.set_validation(
            method='walk_forward',
            train_window=200, # Train on last 200 bars
            rolling_step=50,  # Retrain every 50 bars
            frequency='1d',
            incremental=False, # Set True to use partial_fit (faster)
            verbose=True      # Print training logs
        )
        # Ensure history depth covers training window
        self.set_history_depth(250)

    def prepare_features(self, df: pd.DataFrame):
        """Called by framework for TRAINING data preparation"""
        X = pd.DataFrame()
        X['ret1'] = df['close'].pct_change()
        X['ret2'] = df['close'].pct_change(2)
        X = X.fillna(0)

        # Label: Next period return > 0
        future_ret = df['close'].pct_change().shift(-1)
        y = (future_ret > 0).astype(int)

        # Align X and y (Drop last row where y is NaN)
        return X.iloc[:-1], y.iloc[:-1]

    def on_bar(self, bar: Bar):
        # Wait for initial training
        if self._bar_count < 200:
            return

        # 3. Inference (Real-time)
        # Get recent history to construct current features
        hist_df = self.get_history_df(5)

        # Manual Feature Extraction (Must match prepare_features logic)
        curr_ret1 = (bar.close - hist_df['close'].iloc[-2]) / hist_df['close'].iloc[-2]
        curr_ret2 = (bar.close - hist_df['close'].iloc[-3]) / hist_df['close'].iloc[-3]

        X_curr = pd.DataFrame([[curr_ret1, curr_ret2]], columns=['ret1', 'ret2'])
        X_curr = X_curr.fillna(0)

        try:
            # Predict
            pred = self.model.predict(X_curr)[0] # SklearnAdapter returns proba for class 1 or label

            if pred > 0.55:
                self.buy(bar.symbol, 100)
            elif pred < 0.45:
                self.sell(bar.symbol, 100)
        except:
            pass # Model might not be ready
```
````

## 3. Common Scenario Prompt Examples

### Scenario A: Writing a Dual Moving Average Strategy

**User Query**:
> Please help me write an AKQuant strategy using 5-day and 20-day moving averages, buying on golden cross and selling on death cross, for the symbol "AAPL".

**Recommended Supplement**:
> (Paste the Core Prompt above at the beginning, or as a system prompt)

### Scenario B: Writing a Backtest Run Script

**User Query**:
> I already have the `MyStrategy` class, please help me write a script to run the backtest, using pandas DataFrame for data, and the time range is the full year of 2023.

**Expected Code Generated by Model**:

```python
import pandas as pd
from akquant import run_backtest, BacktestConfig

# 1. Prepare Data (Mock Data for example)
# AKQuant requires columns: timestamp, open, high, low, close, volume, symbol
dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
df = pd.DataFrame({
    "timestamp": dates,
    "open": 100.0,
    "high": 105.0,
    "low": 95.0,
    "close": 102.0,
    "volume": 10000.0,
    "symbol": "AAPL"
})

# 2. Configure and Run Backtest (Simpler Approach)
# Pass arguments directly, no need to build complex Config objects
result = run_backtest(
    data=df,               # Supports DataFrame, Dict[str, DataFrame] or CSV path
    strategy=MyStrategy,   # Strategy class
    strategy_params={"fast_window": 5, "slow_window": 20}, # Strategy parameters
    cash=100_000.0,        # Initial cash
    commission=0.0003,     # Commission rate
    show_progress=True     # Show progress bar
)

# 3. View Results
print(result)
result.plot() # Visualization
```

### Scenario C: Parameter Optimization

**User Query**:
> Please help me optimize the `fast_window` (5-20, step 5) and `slow_window` (20-60, step 10) parameters for `MyStrategy` using grid search, and sort results by Sharpe Ratio.

**Expected Code Generated by Model**:

```python
from akquant.optimize import run_optimization
import pandas as pd
import numpy as np

# 1. Define Parameter Grid
# Use list for each parameter
param_grid = {
    "fast_window": range(5, 25, 5),   # [5, 10, 15, 20]
    "slow_window": range(20, 70, 10)  # [20, 30, 40, 50, 60]
}

# 2. Run Optimization
# Multiprocessing is supported via max_workers
results_df = run_optimization(
    strategy=MyStrategy,
    param_grid=param_grid,
    data=df,              # Shared data for all backtests
    cash=100_000.0,       # Backtest config
    max_workers=4,        # Parallel execution
    sort_by="sharpe_ratio",
    ascending=False
)

# 3. View Best Results
print("Top 5 Parameter Combinations:")
print(results_df.head())

# Access best params
best_params = results_df.iloc[0]["params"]
print(f"Best Params: {best_params}")
```

## 4. Precautions (Negative Constraints for LLM)

When using large models, the following hallucinations may occur, which can be explicitly prohibited in the Prompt:

1.  **About `super().__init__()`**: AKQuant's `Strategy` uses the `__new__` hook for initialization. Although calling `super().__init__` is safe (no-op), it is not required in this framework.
2.  **Prohibit `context.portfolio` legacy usage**: Although internally supported, it is recommended to use top-level APIs like `self.get_position()`.
3.  **Note `get_history` return value**: It returns `numpy.ndarray`, not `pandas.Series` (unless `get_history_df` is used). Performing operations directly on `ndarray` offers higher performance.
4.  **Feature Consistency in ML Strategies**: During training, `prepare_features` usually `shift(-1)` causing the last row to be invalid and dropped; however, during `on_bar` prediction, we need to calculate features using the current latest market data. Reusing `prepare_features` directly may lead to prediction failure (as it thinks that is the "last row" and drops it), or requires special parameter control. It is recommended to explicitly calculate current features in `on_bar`.

## 5. Realtime Data Processing (Realtime / CTP)

AKQuant provides the `LiveRunner` class to simplify the startup process for live/simulation trading.

**Prompt Example**:
> I need to write a CTP live trading startup script, aggregating 1-minute K-lines, and automatically stopping after running for 1 hour.

**Expected Code Snippet**:

```python
from akquant.live import LiveRunner
from akquant import AssetType, Instrument

# 1. Define Instrument
instruments = [
    Instrument(symbol="rb2505", asset_type=AssetType.Futures, multiplier=10.0, margin_ratio=0.1, tick_size=1.0)
]

# 2. Create Runner
runner = LiveRunner(
    strategy_cls=MyStrategy,
    instruments=instruments,
    md_front="tcp://182.254.243.31:40011",
    use_aggregator=True  # Key parameter
)

# 3. Run (Supports auto-stop duration)
# show_progress=False is recommended in live trading to avoid progress bar interfering with logs
runner.run(cash=500_000, show_progress=False, duration="1h")
```

### Mode Selection (`use_aggregator`)

*   **`use_aggregator=True` (Default)**:
    *   Enables built-in Rust high-performance aggregator (`BarAggregator`).
    *   Strategy `on_bar` triggers every 1 minute (Standard OHLCV).
    *   Suitable for trend following, technical indicator strategies.
*   **`use_aggregator=False`**:
    *   **Tick-as-Bar Mode**. The gateway directly encapsulates every Tick received into a Bar and pushes it.
    *   Strategy `on_bar` frequency is extremely high (same as Tick frequency).
    *   Suitable for high-frequency strategies, market making, or testing data connections.

### Key Differences Explained

*   **Gateway Layer (`on_tick`)**: Responsible for receiving raw Tick data from external interfaces (e.g., CTP, Binance).
*   **Strategy Layer (`on_bar`)**: The core entry point for strategy logic.
    *   **Regardless of whether the data source is aggregated minute bars or Tick snapshots, the strategy always receives data via `on_bar`.**
    *   The AKQuant engine standardizes the data flow, so the strategy does not need to know if the upstream is Tick or Bar, it only needs to process `Bar` objects.
