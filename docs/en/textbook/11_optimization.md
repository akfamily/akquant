# Chapter 11: Parameter Optimization and Robustness Validation

This chapter is currently maintained in Chinese first.

- Chinese chapter: [第 11 章：参数优化与稳健性检验](../../zh/textbook/11_optimization.md)
- Textbook home: [Chinese textbook index](../../zh/textbook/index.md)
- Practice links:
  - Primary example: [examples/textbook/ch11_optimization.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch11_optimization.py)
  - Extended example: [examples/02_parameter_optimization.py](https://github.com/akfamily/akquant/blob/main/examples/02_parameter_optimization.py)
  - Functional twin example: [examples/textbook/ch11_optimization_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch11_optimization_functional.py)
  - Guide: [Optimization Guide](../guide/optimization.md)

Key optimization parameters (new):

- `forward_worker_logs` in `run_grid_search`:
  - `False` (default): better throughput; worker `self.log()` may not be visible in main process output.
  - `True`: forwards worker logs to the main process for debugging and teaching.
- `strict_strategy_params` in `run_backtest` (default `True`):
  - enforces strict constructor parameter validation;
  - fails fast on unknown strategy parameters to avoid silent fallback and misleading optimization results.
- `run_walk_forward` accepts these options via `**kwargs` passthrough:
  - `forward_worker_logs` applies to in-sample optimization (`run_grid_search`);
  - `strict_strategy_params` stays effective in both optimization and OOS validation.

WFO passthrough example:

```python
wfo_results = run_walk_forward(
    strategy=TailTradingStrategy,
    param_grid=param_grid,
    data=all_data,
    train_period=250,
    test_period=60,
    max_workers=4,
    forward_worker_logs=True,
    strict_strategy_params=True,
)
```

Windows note for parallel optimization (`max_workers > 1`):

- Define strategy classes in an importable module, not in `__main__`.
- Guard script entry with `if __name__ == "__main__":`.
- This is caused by Windows multiprocessing `spawn`, not by fill-policy semantics.
