# Custom Broker Production Checklist

Use this checklist to move a custom broker from "works in demo" to "production-ready".

## 1. Authentication and Connectivity

- Separate credentials by environment (paper/sim/live)
- Run startup health checks and expose observable connection state
- Implement heartbeat and reconnect strategy
- Define retry policies for network jitter, auth failures, and broker throttling

## 2. Order Submission and Idempotency

- Use `client_order_id` consistently and keep submissions idempotent
- Return existing `broker_order_id` on duplicate active submission
- Make cancel requests idempotent
- Cover full order lifecycle states: `Submitted/PartiallyFilled/Filled/Cancelled/Rejected`
- If you treat `client_order_id` as an idempotency key: AKQuant generates `{broker}-{session tag}-{sequence}`, where the session tag is regenerated on every `run_live`, so ids never collide across restarts or parallel processes. Do not assume the id is reproducible after a restart.
- Declare `supports_short_sell` truthfully: when it is `False`, AKQuant rejects `side=Sell` + `position_effect=open` locally before the order reaches your gateway.

## 3. Reports and State Sync

- Implement `on_order/on_trade/on_execution_report` callbacks
- Ensure reports carry both `client_order_id` and `broker_order_id`
- Implement recovery sync via `sync_open_orders/sync_today_trades`
- Handle out-of-order and duplicated reports safely

## 4. Risk and Protection

- Add pre-trade checks (order size/value, exposure, frequency limits)
- Add price guards (deviation bands, price-limit checks, tick-size validation)
- Add trading-session guards (reject outside trading session)
- Preserve normalized `reject_reason` for risk rejections

## 5. Reconciliation and Consistency

- Run intraday reconciliation for orders, trades, positions, and cash
- Generate end-of-day reconciliation reports with diff markers
- Alert on critical diffs and support manual intervention workflow
- Define conflict priority between strategy state and broker state

## 6. Monitoring and Alerting

- Track core metrics: place success rate, cancel success rate, report latency, disconnect count
- Track latency buckets: order RTT, report handling latency, strategy callback latency
- Define alert levels: P0 (cannot trade), P1 (state mismatch), P2 (performance degradation)
- Keep structured logs with `client_order_id` traceability

## 7. Testing and Drills

- Unit-test normal, error, reconnect, and duplicate-submission paths
- Replay historical broker event samples
- Stress-test deduplication and callback processing under load
- Drill failures: network outage, delayed reports, partial fills, broker restart

## 8. Release and Rollback

- Roll out gradually with small capital, small symbol set, short windows
- Keep a fast fallback switch to `paper` mode or safe-stop
- Arrange release-window on-call and realtime monitoring
- Keep rollback binaries/config snapshots ready

## 9. Minimum Go-Live Gate

- 100% pass on critical-path test suite
- At least one full trading day without unresolved state diffs
- No unresolved critical alerts
- Rollback plan is tested and executable

## 10. Related Docs

- [Custom Broker Registry](./custom_broker_registry.md)
- [API Reference: gateway registry APIs](../reference/api.md)
- [Example: 35_custom_broker_registry_demo.py](https://github.com/akfamily/akquant/blob/main/examples/35_custom_broker_registry_demo.py)
