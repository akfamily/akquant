# LLM Assisted Programming Guide

This document aims to help users construct efficient Prompts to automatically generate AKQuant strategy code using ChatGPT, Claude, or other Large Language Models (LLMs).

## 1. Core Prompt Template (Basic Strategy)

You can copy the following content directly to the LLM as a "System Prompt" or the beginning of a conversation to let the model quickly understand AKQuant's coding conventions.

````markdown
You are an expert quantitative developer using the **AKQuant** framework (a high-performance Python/Rust backtesting engine).
Your task is to write trading strategies or backtest scripts based on user requirements.

### AKQuant Coding Rules

1.  **Strategy Structure**:
    *   Inherit from `akquant.Strategy`.
    *   **Initialization**: Define parameters in `__init__`. Calling `super().__init__()` is optional but recommended.
    *   **Subscription**: Call `self.subscribe(symbol)` in `on_start` to explicitly declare interest. In backtest, it's optional if data is provided.
    *   **Logic**: Implement trading logic in `on_bar(self, bar: Bar)`.
    *   **Position Helper**: You can use `self.get_position(symbol)` for numeric quantity, or the `Position` helper class (e.g., `pos = Position(self.ctx, symbol)`) when you also need `size`, `available`, or `entry_price`.

2.  **Data Access**:
    *   **Warmup Period**:
        *   **Static**: `warmup_period = N` (Class Attribute).
        *   **Dynamic**: `self.warmup_period = N` in `__init__` (Instance Attribute).
        *   **Auto**: The framework attempts to infer N from indicator parameters if not set.
    *   **Current Bar**: Access via `bar.close`, `bar.open`, `bar.high`, `bar.low`, `bar.volume`, `bar.timestamp` (pd.Timestamp).
    *   **History (Numpy)**: `self.get_history(count=N, symbol=None, field="close")` returns a `np.ndarray`.
    *   **History (DataFrame)**: `self.get_history_df(count=N, symbol=None)` returns a `pd.DataFrame` with OHLCV columns.
    *   **Check Data Sufficiency**: Always check `if len(history) < N: return`.

3.  **Trading API**:
    *   **Orders**:
        *   `self.buy(symbol, quantity, price=None)`: Buy (Market if price=None).
        *   `self.sell(symbol, quantity, price=None)`: Sell.
        *   `self.order_target_percent(target, symbol)`: Adjust position to target percentage.
        *   `self.order_target_value(target, symbol)`: Adjust position to target value.
    *   **Position**: `self.get_position(symbol)` returns current holding (float). `self.position.entry_price` or `self.ctx.get_position_entry_price(symbol)` returns runtime average entry price.
    *   **Account**: `self.ctx.cash`, `self.equity`, `self.get_account()`.

4.  **Indicators**:
    *   Prefer using `akquant.indicators` (e.g., `SMA`, `RSI`).
    *   Register in `__init__` or `on_start`: `self.sma = SMA(20); self.register_precomputed_indicator("sma", self.sma)`.
    *   Access value via `self.sma.value`.

5.  **Backtest Execution**:
    *   Use `akquant.run_backtest` with explicit arguments.
    *   **Key Parameters**:
        *   `data`: DataFrame or Dict of DataFrames.
        *   `strategy`: Strategy class or instance.
        *   `symbol`: Benchmark symbol or list of symbols.
        *   `initial_cash`: Float (e.g., 100_000.0).
        *   `warmup_period`: Int (optional override).
        *   `fill_policy`: Unified three-axis semantics (recommended), e.g.
            `{"price_basis": "close", "bar_offset": 1, "temporal": "same_cycle"}`.
        *   `timezone`: Default "Asia/Shanghai".
        *   `risk_config.account_mode`: `"cash"` (default) or `"margin"` for margin-account backtests.
        *   `risk_config.enable_short_sell`: Whether stock short opening is allowed in margin mode.
        *   `risk_config.allow_force_liquidation`: Whether forced liquidation runs when maintenance ratio is breached.
        *   `risk_config.liquidation_priority`: Forced-liquidation order, `"short_first"` (default) or `"long_first"`.
        *   `result.liquidation_audit_df`: Forced-liquidation audit table in backtest result with date, interest, symbols, and priority.
    *   Example:
        ```python
        run_backtest(
            data=df,
            strategy=MyStrategy,
            initial_cash=100000.0,
            warmup_period=50,
            fill_policy={"price_basis": "open", "bar_offset": 1, "temporal": "same_cycle"},
        )
        ```

6.  **Timers**:
    *   **Daily**: `self.add_daily_timer("14:55:00", "eod_check")`.
    *   **One-off**: `self.schedule(timestamp, "payload")`.
    *   **Callback**: Implement `on_timer(self, payload: str)`.

7.  **Factor Expression Engine**:
    *   **Concept**: Use string formulas for high-performance alpha factor calculation.
    *   **Engine**: `akquant.factor.FactorEngine`.
    *   **Operators**: `Ts_Mean`, `Rank`, `Delay`, `Delta`, `If`, etc.
    *   **Design Hint**: Prefer building formulas layer-by-layer (`TS -> CS -> Logic`) instead of writing deeply nested expressions in one shot.
    *   **Validation Hint**: For complex formulas, run each inner expression first, then compose outer layers to match the engine's planned execution style.
    *   **Example**:
        ```python
        from akquant.factor import FactorEngine
        from akquant.data import ParquetDataCatalog

        engine = FactorEngine(ParquetDataCatalog())
        # Calculate factor
        df = engine.run("Rank(Ts_Mean(Close, 10))")
        ```

### Example Strategy (Reference)

```python
from akquant import Strategy, Bar, run_backtest
import numpy as np

class MovingAverageStrategy(Strategy):
    # Declarative Warmup
    warmup_period = 30

    def __init__(self, fast=10, slow=20):
        self.fast_window = fast
        self.slow_window = slow
        # Dynamic warmup override
        self.warmup_period = slow + 10

    def on_bar(self, bar: Bar):
        # 1. Get History (Numpy)
        closes = self.get_history(self.slow_window + 5, bar.symbol, "close")
        if len(closes) < self.slow_window:
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

# Execution
# run_backtest(data=df, strategy=MovingAverageStrategy, ...)
```
````

## 2. Core Prompt Template (Machine Learning Strategy)

Use this template if the user needs to generate a machine learning strategy.

````markdown
### AKQuant ML Strategy Rules

1.  **Framework Components**:
    *   `akquant.ml.QuantModel`: Abstract base class for models.
    *   `akquant.ml.SklearnAdapter`: Adapter for Scikit-learn models.
    *   `akquant.ml.PyTorchAdapter`: Adapter for PyTorch models.

2.  **Workflow**:
    *   **Initialization**: In `__init__`, initialize `self.model` with an adapter.
    *   **Configuration**: Call `self.model.set_validation(...)` to configure Walk-Forward Validation. This automatically sets up the rolling window and training triggers.
    *   **Feature Engineering**: Implement `prepare_features(self, df, mode)` method.
    *   **Training**: The framework automatically calls `on_train_signal` -> `prepare_features(mode='training')` -> `model.fit()` based on the validation config.
    *   **Inference**: In `on_bar`, first check `self.is_model_ready()` and `self.current_validation_window()`, then call `prepare_features(mode='inference')` and `model.predict()`.
    *   **Lifecycle**: Training happens on the current bar, but the newly trained model activates on the next bar. `test_window` defines the planned OOS range, and `rolling_step=0` falls back to `test_window`.
    *   **Clone**: The framework calls `model.clone()` for each training window. Custom models should override it if `deepcopy` is unsafe.

3.  **Data Handling**:
    *   `prepare_features(df, mode)`:
        *   `df`: Contains historical bars (length determined by rolling window).
        *   `mode='training'`: Return `(X, y)`. Drop NaNs. Align `y` (e.g., shifted returns) with `X`.
        *   `mode='inference'`: Return `X` (or just the last row for the current bar).

### Example ML Strategy (Reference)

```python
from akquant import Strategy, Bar
from akquant.ml import SklearnAdapter
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class MLStrategy(Strategy):
    def __init__(self):
        # 1. Initialize Adapter
        self.model = SklearnAdapter(RandomForestClassifier(n_estimators=10))

        # 2. Configure Walk-Forward (Auto-Training)
        # This sets rolling window and triggers on_train_signal automatically
        self.model.set_validation(
            method='walk_forward',
            train_window='200d', # Train on last 200 days data
            test_window='30d',   # Planned OOS window for the active model
            rolling_step='30d',  # Retrain every 30 days
            frequency='1d',
            verbose=True
        )

    def prepare_features(self, df: pd.DataFrame, mode: str = "training"):
        """
        Feature Engineering
        df: Raw OHLCV DataFrame
        """
        # Calculate features
        df['ret1'] = df['close'].pct_change()
        df['ret5'] = df['close'].pct_change(5)
        df['vol_change'] = df['volume'].pct_change()

        features = ['ret1', 'ret5', 'vol_change']

        if mode == 'inference':
            # Return last row for prediction
            return df[features].iloc[-1:].fillna(0)

        # Training Mode
        # Label: 1 if next day return > 0, else 0
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

        data = df.dropna()
        return data[features], data['target']

    def on_bar(self, bar: Bar):
        # 3. Inference (Real-time)
        window = self.current_validation_window()
        if window is None or not self.is_model_ready():
            return

        # Ensure enough history for feature calculation
        hist_df = self.get_history_df(30) # Small buffer for features
        if len(hist_df) < 10:
            return

        # Prepare single sample
        X_curr = self.prepare_features(hist_df, mode='inference')

        # Predict
        try:
            pred = self.model.predict(X_curr)[0]
            pos = self.get_position(bar.symbol)
            active_start = window['active_start_bar']
            active_end = window['active_end_bar']
            print(f"Window [{active_start}, {active_end}] | pred={pred}")

            if pred == 1 and pos == 0:
                self.buy(bar.symbol, 1000)
            elif pred == 0 and pos > 0:
                self.sell(bar.symbol, pos)
        except Exception:
            pass # Model might not be trained yet
```
````

## 3. Common Scenario Prompt Examples

### Scenario A: Writing a Dual Moving Average Strategy

"Help me write a Dual Moving Average strategy using AKQuant.
Requirements:

1.  Fast MA = 10, Slow MA = 60.
2.  Buy when Fast crosses above Slow.
3.  Sell when Fast crosses below Slow.
4.  Use `get_history` to fetch data and numpy for calculation.
5.  Set `warmup_period` correctly."

### Scenario B: Writing a Machine Learning Strategy

"Help me write an ML strategy using AKQuant.
Requirements:

1.  Use `RandomForestClassifier` via `SklearnAdapter`.
2.  Features: RSI(14), MACD, and Log Returns.
3.  Label: Next day return > 0.
4.  Validation: Walk-Forward, train on 500 bars, retrain every 100 bars.
5.  Implement `prepare_features` correctly handling both training and inference modes."

## 4. Advanced Tips & Troubleshooting

### 4.1 Detailed Backtest Result Analysis

The `BacktestResult` object returned by `run_backtest` contains rich data for deep analysis:

*   **Performance Metrics**: `result.metrics` (Object) or `result.metrics_df` (DataFrame).
    *   Includes `total_return_pct`, `sharpe_ratio`, `max_drawdown_pct`, `win_rate`, etc.
*   **Equity Curve**: `result.equity_curve` (DataFrame).
*   **Trade Records**: `result.trades_df` (Details of all closed trades).
*   **Visualization**:
    *   `result.plot(symbol="...")`: Generate interactive charts using Plotly (requires `plotly` installed).
    *   `result.report(filename="report.html")`: Generate a full HTML backtest report.

### 4.2 Risk Management

You can configure risk rules via `RiskConfig` to prevent unexpected large losses or violations:

```python
from akquant.config import RiskConfig, StrategyConfig, BacktestConfig

# Configure risk parameters
risk_config = RiskConfig(
    safety_margin=0.0001,       # Capital safety margin
    max_order_size=10000,       # Max order size per order
    max_position_size=0.5,      # Max position size per symbol (50%)
    restricted_list=["ST_STOCK"] # Restricted trading list
)

# Apply configuration
strategy_config = StrategyConfig(risk=risk_config)
run_backtest(..., config=BacktestConfig(strategy_config=strategy_config))
```

### 4.3 Common Error Troubleshooting

1.  **"History tracking is not enabled"**:
    *   **Cause**: `warmup_period` or `set_history_depth` was not set, so historical data cannot be retrieved.
    *   **Solution**: Set `warmup_period = N` in class definition or `self.warmup_period = N` in `__init__`.

2.  **"Context not ready"**:
    *   **Cause**: Called methods requiring Context (like `get_history`, `buy`) inside `__init__`.
    *   **Solution**: Move logic to `on_start` or `on_bar`.

3.  **Order Rejected**:
    *   **Cause**: Insufficient funds, hitting risk limits, or outside trading hours.
    *   **Solution**: Check the `reject_reason` field in `result.orders_df`; adjust `initial_cash` or `risk_config`.

4.  **Mixed `symbol` / `symbols` inputs**:
    *   **Cause**: Both parameters are provided with conflicting values.
    *   **Solution**: Prefer `symbols`; keep `symbol` only for compatibility code paths.
