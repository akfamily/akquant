# Cross-Section Strategy Playbook Checklist

Use this checklist to move a cross-section strategy from "it runs" to "it is reproducible, explainable, and production-ready."

## 1. Design

*   Select trigger model explicitly: prefer `on_timer`; use timestamp-completion only when no stable rebalance time exists.
*   Define signal-to-fill timing clearly, especially for `fill_policy=NextOpen()`.
*   Version your `universe` source and track constituent effective dates.
*   Define position constraints: max symbol weight, sector cap, cash buffer, lot-size rules.

## 2. Data

*   Align timezone, trading calendar, and missing-data policy before scoring.
*   Enforce history window checks and track valid sample coverage each rebalance.
*   Define explicit fallback behavior for suspensions, limit-up/limit-down, and volume anomalies.
*   Freeze data pull parameters and snapshots to keep runs reproducible.

## 3. Execution

*   Use target-position APIs to reduce sell-then-buy drift during rebalance.
*   Add rebalance tolerance bands to avoid turnover spikes from tiny rank changes.
*   Monitor order rejections centrally via `orders_df.reject_reason`.
*   Add next-cycle convergence logic when one cycle cannot fully reach target weights.

## 4. Risk

*   Enable account-level guardrails: `max_account_drawdown`, `max_daily_loss`, `stop_loss_threshold`.
*   Configure strategy-specific limits like `max_position_pct` and `max_order_value`.
*   Define post-trigger behavior: de-risk, pause, or close-only mode.
*   Persist risk-trigger logs for post-mortem analysis and parameter iteration.

## 5. Validation

*   Validate with time-sliced evaluation (rolling windows / regime segments), not only full-sample metrics.
*   Track stability metrics: turnover, concentration, slippage sensitivity, and capacity constraints.
*   Compare execution modes and rebalance frequencies to confirm return source robustness.
*   Save parameter sets and result snapshots for auditability.

## 6. Pre-Live Readiness

*   Require all five domains to pass: trigger, data, execution, risk, validation.
*   Dry-run failure scenarios: missing feed, rejection bursts, delayed triggers, trading-day transitions.
*   Pin runtime config and dependency versions to avoid environment drift.
*   Prepare rollback paths: parameter rollback, strategy disable switch, version rollback.

Related references:

*   [Strategy Guide: Recommended Cross-Section Pattern](strategy.md#34-recommended-cross-section-pattern)
*   [Strategy Guide: Plan B Timestamp Completion](strategy.md#35-cross-section-plan-b-execute-after-collecting-one-timestamp)
