# Multi-Strategy Guide

This guide focuses on organizing multi-slot execution under a shared account model, with parameter mapping, rollout steps, acceptance checks, and troubleshooting order.

## 1. When to Use This Guide

- You run `run_backtest(strategy=...)` and need strategy ownership analytics.
- You want multiple strategy slots under the same account.
- You need post-risk actions (`reduce_only`, cooldown) per strategy.
- You use warm-start and need continuity across snapshots.

## 2. Before You Start

- account model: shared account is acceptable
- ownership granularity: strategy-level split is required
- risk actions: reduce-only and cooldown requirements are clear
- runtime mode: warm-start continuity is required or not

## 3. Parameter Mapping

- `StrategyConfig.strategy_id`
- `StrategyConfig.strategies_by_slot`
- `run_backtest(indicators=[...])` / `run_live(indicators=[...])`: a list of draw-only `aq.Indicator` script classes, merged into `strategies_by_slot` internally (key = the class-line `id`, default snake_case class name). Same topology as the previous item; see the custom indicator guide, "Indicator scripts".
- `StrategyConfig.strategy_max_order_value` / `strategy_max_order_size` / `strategy_max_position_size`
- `StrategyConfig.strategy_max_daily_loss` / `strategy_max_drawdown`
- `StrategyConfig.strategy_reduce_only_after_risk`
- `StrategyConfig.strategy_risk_cooldown_bars`

Rules:

- all strategy-level maps must use configured strategy ids as keys
- empty keys, unknown keys, or negative thresholds fail fast

## 4. Recommended Steps

### 4.1 Start with Single-Strategy Ownership

```python
from akquant import BacktestConfig, StrategyConfig, run_backtest

config = BacktestConfig(
    strategy_config=StrategyConfig(
        strategy_id="alpha",
    )
)
```

Checks:

- ownership columns appear in `orders_df` / `trades_df`
- report output remains stable

### 4.2 Add Slots

```python
from akquant import BacktestConfig, StrategyConfig

config = BacktestConfig(
    strategy_config=StrategyConfig(
        strategy_id="alpha",
        strategies_by_slot={"beta": BetaStrategy},
    )
)
```

Checks:

- multiple `owner_strategy_id` values appear
- per-strategy order counts are expected

Callback routing (all slots share one account; matching, risk and cash are not split per slot, only who receives callbacks follows ownership):

- User timers (`schedule` / `schedule_daily` etc.) go only to the slot that registered them.
- Order and trade callbacks (`on_order` / `on_trade` / `on_reject`, including cancels, expiries and bracket/OCO child orders) go only to the slot named by the order's `owner_strategy_id`; if another slot cancels your order, the `Cancelled` callback still goes to you.
- Trading-day lifecycle hooks (`on_pre_open` / `on_before_trading` / `on_after_trading` etc.) and `on_expiry` go to all slots: expiry settlement concerns positions of the shared account, not a single strategy.
- Orders without an `owner_strategy_id` (e.g. orders created on the broker side in live trading) go only to the default slot (the main strategy named by `strategy_id`).

### 4.3 Add Strategy-Level Risk Actions

```python
from akquant import BacktestConfig, StrategyConfig

config = BacktestConfig(
    strategy_config=StrategyConfig(
        strategy_id="alpha",
        strategies_by_slot={"beta": BetaStrategy},
        strategy_max_order_size={"alpha": 10, "beta": 20},
        strategy_reduce_only_after_risk={"alpha": True, "beta": False},
        strategy_risk_cooldown_bars={"alpha": 2, "beta": 0},
    )
)
```

Checks:

- reject reasons are interpretable
- reduce-only and cooldown reasons are visible in `orders_df.reject_reason`

### 4.4 Validate Warm-Start Continuity

- save snapshot and resume via `run_from_checkpoint`
- use the same `config` to keep strategy topology/risk mapping centralized
- verify default strategy id, slot topology, and risk runtime states are preserved

## 5. Minimum Acceptance Matrix

- single strategy without `strategy_id` remains stable
- single strategy with `strategy_id` has correct ownership data
- multi-slot ownership and reject stats are correct
- reduce-only and cooldown behavior works after risk triggers
- warm-start continuity remains intact

## 6. Common Issues

### 6.1 “unknown strategy id”

- Cause: strategy-level risk map keys are not in configured strategy ids.
- Fix: align all map keys with `strategy_id + strategies_by_slot`.

### 6.2 Ownership mismatch after warm-start

- Cause: old snapshot or inconsistent restored default strategy ownership.
- Fix: use snapshots from current version and verify slot topology + ownership fields.

### 6.3 Report vs DataFrame mismatch

1. inspect `orders_df.reject_reason`
2. inspect strategy-level analysis outputs
3. inspect report fallback/data-present branch behavior

## 7. Further Reading

- [Warm Start Guide](warm_start.md)
