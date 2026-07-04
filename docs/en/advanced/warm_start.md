# Warm Start Guide

AKQuant provides a powerful **Warm Start** workflow that lets you save current backtest state (snapshot) and resume later. This is useful for long-horizon segmented backtests, rolling workflows, and production-like continuation runs.

## 1. What Is Warm Start?

In a traditional event-driven backtest, each run is a cold start, and all state (positions, cash, indicator history) is rebuilt from scratch.

With warm start, you can:

1. **Save (Snapshot)**: Serialize in-memory engine and strategy state (positions, open orders, strategy attributes, indicator state) to disk.
2. **Resume**: Load the snapshot and continue with new data as if execution was not interrupted.

## 2. Basic Usage

### 2.1 Save Snapshot (Phase 1)

At the end of phase 1 (or at any strategy checkpoint), save state with `save_snapshot`.

```python
from akquant.checkpoint import save_snapshot

# Run phase 1
result1 = run_backtest(data=data_phase1, strategy=MyStrategy, ...)

# Save snapshot file
checkpoint_file = "checkpoint_phase1.pkl"
save_snapshot(result1.engine, result1.strategy, checkpoint_file)
print(f"Snapshot saved to {checkpoint_file}")
```

### 2.2 Resume and Continue (Phase 2)

Use `run_warm_start` with the snapshot path and phase-2 data.

Important: snapshots only store dynamic runtime state (`Portfolio`, `Orders`, strategy attributes). Static configuration (`Instrument`, `MarketModel`) is not persisted and must be reconfigured on resume.

```python
import akquant as aq

# Prepare phase-2 data
data_phase2 = ...

config = aq.BacktestConfig(
    strategy_config=aq.StrategyConfig(
        strategy_id="alpha",
        strategies_by_slot={"beta": BetaStrategy},
        strategy_max_order_size={"alpha": 10, "beta": 20},
    )
)

# Resume from snapshot and continue
result2 = aq.run_warm_start(
    checkpoint_path="checkpoint_phase1.pkl",
    data=data_phase2,
    symbols="AAPL",
    commission_rate=0.0003,
    stamp_tax_rate=0.001,
    transfer_fee_rate=0.00001,
    config=config,
    t_plus_one=True,
)
```

## 3. Strategy Adaptation

To support warm start correctly, strategy code should separate initialization from restoration behavior.

### 3.1 Lifecycle Hooks

AKQuant provides two startup hooks:

- `on_start()`: Called for both cold start and warm start.
- `on_resume()`: Called only in warm start, before `on_start()`.

### 3.2 Avoid Overwriting Restored State

A common mistake is unconditional indicator reinitialization in `on_start`, which overwrites restored indicator state.

Incorrect:

```python
def on_start(self):
    self.sma = SMA(30)
    self.subscribe(self.symbol)
```

Correct:

```python
def on_start(self):
    if not self.is_restored:
        self.sma = SMA(30)
    else:
        self.log("Resumed from snapshot. Indicators retained.")

    self.register_precomputed_indicator("sma", self.sma)
    self.subscribe(self.symbol)
```

### 3.3 Indicator Serialization

Built-in indicators (`SMA`, `EMA`, etc.) support pickle serialization. For custom indicators or third-party objects (for example TA-Lib wrappers), ensure they are pickle-compatible or implement state handling explicitly. For the full custom-indicator workflow, see the [Custom Indicator Guide](../guide/custom_indicator.md).

## 4. Notes

1. **Instrument re-registration**: `run_warm_start` auto-registers default instrument info for symbols in new data. If your strategy depends on custom `lot_size` or `multiplier`, verify and override in `on_start`.
2. **MarketModel reset**: Fee settings and trading rules (for example T+1) are not persisted in snapshots. Re-pass them via explicit args or `config.strategy_config` on resume, including `commission_policy` / `commission_rate`, taxes, and transfer-fee settings.
3. **Initial cash display**: `result2.metrics.initial_cash` is adjusted to resumed-phase starting cash, so phase-2 return metrics remain interpretable.
4. **Data continuity**: Keep phase-1 end and phase-2 start continuous to avoid indicator jumps.
5. **`get_history()` continuity**: New snapshots also persist the history buffer, so `get_history()` and `get_history_map()` resume with the phase-1 rolling window intact. In the normal warm-start path you no longer need to manually prepend extra lookback bars.
6. **Runtime config injection**: Use `strategy_runtime_config` in `run_warm_start` to override runtime behavior at resume.
7. **Strategy-level risk state continuity**: Strategy limits, strategy cashflow, daily-loss baseline, drawdown peak, and reduce-only activation state are persisted and restored.
8. **Default timezone**: If `timezone` is not explicitly provided to `run_warm_start`, default is `Asia/Shanghai`.

## 5. Full Example

See [21_warm_start_demo.py](https://github.com/akfamily/akquant/blob/main/examples/21_warm_start_demo.py) for a complete runnable example.

```python
class MyStrategy(Strategy):
    def on_start(self):
        if not self.is_restored:
            self.sma = SMA(10)
        self.register_precomputed_indicator("sma", self.sma)

# ... run phase 1 ...
save_snapshot(engine, strategy, "checkpoint.pkl")

# ... run phase 2 ...
run_warm_start("checkpoint.pkl", data_new, ...)
```

## 6. Further Reading

- API reference for `run_warm_start`: [API Reference](../reference/api.md#akquantrun_warm_start)
- Runtime behavior overrides during resume: [Runtime Config Guide](runtime_config.md)
- Multi-slot continuity and strategy-level risk mapping: [Multi-Strategy Guide](multi_strategy_guide.md)
