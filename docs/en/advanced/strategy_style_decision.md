# Strategy Style Decision Guide

This guide standardizes how teams choose between class-based and function-style strategies to reduce long-term migration and maintenance cost.

## 1. Executive Summary

- Default for production: class-based strategy.
- Default for rapid prototyping: function-style strategy.
- For complex domains (team collaboration, long lifecycle, heavy risk controls): class-based is required.

## 2. Decision Matrix

| Dimension | Class-based | Function-style |
| :--- | :--- | :--- |
| Development speed | Medium | High |
| Maintainability | High | Medium |
| Lifecycle expressiveness | Strong (`on_start/on_stop/...`) | Medium (same callbacks supported) |
| State organization | Strong (OO structure) | Medium (`ctx` convention) |
| Complex risk/modular workflows | Strong | Medium |
| Backtest support | Full | Full |
| run_live support | Full | Full (callable + lifecycle) |
| Multi-slot + risk controls | Full | Supported (validate in small scope first) |

## 3. Selection Rules

- Choose class-based if any of the following is true:
  - maintenance horizon is longer than 3 months
  - multiple developers will maintain one strategy
  - warm-start, complex scheduling, or layered risk controls are required
  - strict regression and test stability are required
- Function-style is acceptable when all are true:
  - goal is fast alpha hypothesis validation
  - logic is small and callback count is low
  - strategy is likely to be rewritten into class-based later

## 4. Recommended Rollout

## 4.1 Phase A: Function-style Discovery

- Validate signals quickly with function-style (1-2 weeks).
- Minimum callbacks: `initialize/on_start/on_bar/on_stop`.
- Freeze parameters and risk thresholds after validation.

## 4.2 Phase B: Migrate to Class-based

- Move `ctx` state into class attributes.
- Refactor reusable logic into private methods.
- Keep callback names aligned to minimize behavior drift.

## 4.3 Phase C: Production Readiness

- Add multi-slot + risk regression tests.
- Rehearse both paper and broker_live paths.
- Standardize alerts, logs, and event persistence rules.

## 5. Minimum Acceptance Checklist

- Lifecycle: `on_start/on_stop` each triggers exactly once.
- Ownership: `owner_strategy_id` is traceable.
- Risk triggers: reject reasons are interpretable and replayable.
- Backtest vs live: key behavioral paths stay aligned.

## 6. Reference Examples

- Functional baseline: [23_functional_callbacks_demo.py](https://github.com/akfamily/akquant/blob/main/examples/23_functional_callbacks_demo.py)
- Live functional entry: [38_live_functional_strategy_demo.py](https://github.com/akfamily/akquant/blob/main/examples/38_live_functional_strategy_demo.py)
- broker_live functional submit: [39_live_broker_submit_order_demo.py](https://github.com/akfamily/akquant/blob/main/examples/39_live_broker_submit_order_demo.py)
- Functional multi-slot + risk: [40_functional_multi_slot_risk_demo.py](https://github.com/akfamily/akquant/blob/main/examples/40_functional_multi_slot_risk_demo.py)
