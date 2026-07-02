# Parameter Optimization

Parameter optimization is a crucial part of quantitative strategy development. AKQuant provides powerful optimization tools to help you explore how parameters affect strategy performance and evaluate strategy robustness.

Currently, two main optimization modes are supported:

1.  **Grid Search**: `run_grid_search`
2.  **Walk-Forward Optimization**: `run_walk_forward`

---

## 1. Grid Search

Grid Search is an exhaustive parameter optimization method. It iterates through given parameter combinations and runs backtests on the same historical data to find the best-performing parameter set.

### Use Cases
*   Exploring strategy sensitivity to parameters.
*   Determining reasonable parameter ranges.
*   Finding the "theoretical ceiling" of a strategy on historical data.

### Usage

Use the `akquant.run_grid_search` function:

```python
from akquant import run_grid_search, Strategy

# 1. Define Strategy
class MyStrategy(Strategy):
    def __init__(self, ma_period, stop_loss):
        # ...

# 2. Define Parameter Grid
param_grid = {
    "ma_period": [10, 20, 30],
    "stop_loss": [0.01, 0.02, 0.05]
}

# 3. Run Grid Search
results = run_grid_search(
    strategy=MyStrategy,
    param_grid=param_grid,
    data=df,
    sort_by="sharpe_ratio",  # Sort by Sharpe Ratio
    ascending=False          # Descending order
)

print(results.head())
```

### Parameter-Model-Driven Optimization (Recommended)

When you need to expose strategy parameters in Web UI / API workflows, use a **`PARAM_MODEL + param_grid` dual-layer pattern**:

1. `PARAM_MODEL` (from `akquant.params`) handles **single-run validation and schema export**.
2. `param_grid` (native `run_grid_search` input) handles **discrete combination search**.

This keeps the optimization engine stable while enabling frontend auto-generated forms.

```python
from akquant import (
    IntParam,
    ParamModel,
    Strategy,
    get_strategy_param_schema,
    validate_strategy_params,
    run_grid_search,
)


class SmaParams(ParamModel):
    fast_period: int = IntParam(10, ge=2, le=200, title="Fast Window")
    slow_period: int = IntParam(30, ge=3, le=500, title="Slow Window")


class SmaStrategy(Strategy):
    PARAM_MODEL = SmaParams

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period


schema = get_strategy_param_schema(SmaStrategy)
runtime_params = validate_strategy_params(
    SmaStrategy,
    {"fast_period": 12, "slow_period": 36},
)

results = run_grid_search(
    strategy=SmaStrategy,
    data=df,
    param_grid={"fast_period": [5, 10, 15], "slow_period": [20, 30, 60]},
)
```

If a strategy does not declare `PARAM_MODEL`, the adapter falls back to `__init__` signature inference for backward compatibility.

### Multi-Objective Optimization

In real trading, a single metric is often insufficient (e.g., Sharpe Ratio alone might select a survivor that traded only once). AKQuant supports sorting based on multiple metrics:

```python
results = run_grid_search(
    strategy=MyStrategy,
    param_grid=param_grid,
    data=df,
    sort_by=["sharpe_ratio", "total_return"], # Primary: Sharpe, Secondary: Total Return
    ascending=[False, False]                  # Both descending
)
```

### Result Filtering

To avoid "survivorship bias" or filter out parameter combinations that do not meet risk control requirements (e.g., too few trades, excessive drawdown), you can use the `result_filter` callback:

```python
def result_filter(metrics):
    # Filter conditions:
    # 1. Trade count >= 50
    # 2. Sharpe Ratio > 1.0
    # 3. Max Drawdown < 20%
    return (
        metrics.get("closed_trade_count", 0) >= 50 and
        metrics.get("sharpe_ratio", 0) > 1.0 and
        metrics.get("max_drawdown_pct", 1.0) < 0.2
    )

results = run_grid_search(
    ...,
    result_filter=result_filter
)
```

### Core Parameters

*   `strategy`: Strategy class.
*   `param_grid`: Parameter dictionary where keys are parameter names and values are lists of parameter values.
*   `data`: Backtest data.
*   `sort_by`: Metric(s) to sort results. Supports single string or list of strings.
*   `ascending`: Sort direction. Supports boolean or list of booleans.
*   `result_filter`: (Optional) Callback function to filter results based on metrics.
*   `warmup_calc`: (Optional) Callback function for dynamic warmup period calculation.
*   `constraint`: (Optional) Callback function for parameter constraints to filter invalid combinations.
*   `forward_worker_logs`: (Optional) Whether to forward worker-process `self.log()` output to the main process during parallel optimization.
    *   `False` (default): Better throughput, worker logs may not be visible in main process output.
    *   `True`: Main process aggregates worker logs, useful for debugging and teaching.
*   `strict_strategy_params`: (default `True`, injected by `run_grid_search` into `run_backtest`).
    *   Enforces strict validation between `param_grid` keys and strategy constructor signature.
    *   Raises fast on unknown parameters to avoid silent fallback and distorted optimization results.

### Resource Control & Error Handling

When dealing with massive parameter combinations or potential infinite loops/OOM issues, use the following parameters:

```python
results = run_grid_search(
    ...,
    timeout=60.0,           # Max 60s per task, skip if timeout
    max_tasks_per_child=1   # Force restart worker after each task
)
```

*   `timeout`: Timeout for a single backtest task (seconds). If a task exceeds this time, it will be marked as failed and skipped. Useful for preventing infinite loops.
*   `max_tasks_per_child`: Worker process restart frequency. Setting to `1` forces a new process for each task, effectively preventing memory leaks (OOM) and cleaning up timeout threads.

### Windows Parallel Execution Notes (`max_workers > 1`)

On Windows, parallel optimization in `run_grid_search` / `run_walk_forward` uses multiprocessing `spawn`. This means:

*   Strategy classes must be defined in an **importable module**, not directly in `__main__`.
*   Script entry must be guarded by `if __name__ == "__main__":`.
*   This is a Python multiprocessing limitation, **not** a fill-policy semantics issue.

Recommended pattern:

```python
from my_strategy_module import TailTradingStrategy
from akquant import run_grid_search


def main() -> None:
    results = run_grid_search(
        strategy=TailTradingStrategy,
        param_grid=param_grid,
        data=all_data,
        max_workers=4,
    )
    print(results.head())


if __name__ == "__main__":
    main()
```

### Parallel Log Visibility & Warning Rules

When `max_workers > 1`, warning behavior is tied to `forward_worker_logs`:

*   `forward_worker_logs=False`: warns that worker logs may not be visible in the main process.
*   `forward_worker_logs=True` with active main-process logger handlers: log forwarding is enabled and visibility warning is suppressed.
*   `forward_worker_logs=True` without active main-process handlers: warns that forwarding was requested but no handler is available.

It is recommended to configure logging explicitly before launching optimization, especially when you want aggregated worker logs in the main process:

```python
import akquant

akquant.configure_logging(
    akquant.LogConfig(
        profile="optimize",
        level="INFO",
        console=True,
        filename="logs/optimize.log",
        file_level="DEBUG",
        file_json=True,
        file_max_bytes=50_000_000,
        file_backup_count=3,
    )
)
```

Notes:

*   `profile="optimize"` includes process identity in the default format, which helps distinguish worker output.
*   `forward_worker_logs=True` only forwards worker logs back to the main process; it does not create handlers automatically.
*   If you only want quick console logging, `akquant.register_logger(level="INFO")` remains valid.
*   If you want file logs to be easier for external log pipelines to consume, enable `file_json=True`.
*   Once aggregation is enabled, Rust execution-path warnings are also forwarded back to the main process. That includes insufficient-margin rejects, session-close expiry, unknown cancel requests, and `same-cycle` deferrals, all of which are easier to filter through structured fields such as `phase=execution`, `order_id`, and `strategy_id`.

### Persistence & Resume

For scenarios with extremely large parameter sets (e.g., > 10,000 combinations), running on a single machine might take days. AKQuant supports real-time result persistence to SQLite, enabling breakpoint resume.

```python
results = run_grid_search(
    ...,
    db_path="optimization_results.db"  # Specify DB path
)
```

*   `db_path`: Path to SQLite database file.
    *   **Real-time Saving**: Results (params, metrics, duration, error) are written to DB immediately after each task completes.
    *   **Resume**: Upon restart, the program checks the DB. If a parameter combination has already been run (based on JSON matching), it skips it and only runs the remaining tasks.
    *   **Reuse**: You can query the DB directly for analysis without re-running backtests.

### ⚠️ Risk Warning
Grid Search is prone to **Overfitting**. The "optimal parameters" selected might just happen to fit the noise of the historical data and often fail in future live trading. Do not use the optimal results from Grid Search directly for live trading.

---

## 2. Walk-Forward Optimization

Walk-Forward Optimization (WFO) is a validation method closer to real-world scenarios. It simulates the passage of time by slicing data into multiple `[Train | Test]` windows, continuously rolling "In-Sample Optimization" and "Out-of-Sample Validation".

### Use Cases
*   Validating real strategy robustness.
*   Evaluating strategy performance on "unknown" data.
*   Generating parameter paths that adjust dynamically over time.

### Principle
The core idea of WFO is: **Always use past data to determine current parameters.**

1.  **Window 1**:
    *   **Train (In-Sample)**: Use Jan-Mar data for Grid Search to find optimal parameter A.
    *   **Test (Out-of-Sample)**: Use parameter A to backtest on Apr data.
2.  **Window 2**:
    *   **Train**: Roll to Feb-Apr data, re-run Grid Search to find optimal parameter B.
    *   **Test**: Use parameter B to backtest on May data.
3.  **Concatenation**: Concatenate results from all test segments to form the final equity curve.

### Usage

Use the `akquant.run_walk_forward` function:

```python
from akquant import run_walk_forward

# Run WFO
wfo_results = run_walk_forward(
    strategy=MyStrategy,
    param_grid=param_grid,
    data=df,
    train_period=250,      # Train window length (e.g., 250 bars)
    test_period=60,        # Test/Rolling step (e.g., 60 bars)
    metric="sharpe_ratio", # Optimization target
    initial_cash=100_000.0,
    warmup_calc=warmup_calc, # Support dynamic warmup
    constraint=param_constraint, # Support parameter constraints
    max_workers=4,
    forward_worker_logs=True,    # Forward in-sample worker logs
    strict_strategy_params=True, # Keep strict constructor validation
)

# wfo_results contains the concatenated equity curve and parameters used for each segment
print(wfo_results)
```

### Core Parameters

*   `train_period`: Training window length (number of Bars). Longer windows mean more stable parameters; shorter windows adapt faster to changes.
*   `test_period`: Test window length (number of Bars). Usually also the rolling step size.
*   `metric`: The metric used to select optimal parameters on the training set (e.g., `sharpe_ratio`, `total_return`).
*   `kwargs` passthrough rules:
    *   forwarded to `run_grid_search` during in-sample optimization;
    *   forwarded to `run_backtest` during out-of-sample validation;
    *   therefore `forward_worker_logs` and `strict_strategy_params` remain available in WFO.

---

## 3. Advanced Features

### Dynamic Warmup

Different parameter combinations may require different warmup periods (e.g., long-term moving average strategies need more history). You can dynamically specify this via the `warmup_calc` callback:

```python
def warmup_calc(params):
    # Warmup period = Long window + 1
    return params["long_window"] + 1

run_grid_search(..., warmup_calc=warmup_calc)
```

### Parameter Constraint

Some parameter combinations are logically invalid (e.g., short window > long window). You can filter these out in advance using the `constraint` callback to save computational resources:

```python
def param_constraint(params):
    # Only keep combinations where short window < long window
    return params["short_window"] < params["long_window"]

run_grid_search(..., constraint=param_constraint)
```

---

## 4. Grid Search vs Walk-Forward Comparison

| Feature | Grid Search | Walk-Forward Optimization |
| :--- | :--- | :--- |
| **Data Usage** | Uses all data for one-time optimization | Rolling data slices, strict train/test separation |
| **Parameter Result** | **1 Set** Global static parameters | **Multiple Sets** Dynamic parameters |
| **Overfitting Risk** | **Very High** (Looking at the answer to find the solution) | **Low** (Simulating real unknown environment) |
| **Core Purpose** | Explore parameter sensitivity, find "theoretical ceiling" | Validate strategy robustness, evaluate "real-world expectation" |
| **Code Mapping** | `akquant.run_grid_search` | `akquant.run_walk_forward` |
