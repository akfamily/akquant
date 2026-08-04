# Live Functional Strategy Quickstart

This guide focuses on function-style strategy entry with `run_live`, covering both `paper` and `broker_live` modes.

## 1. When to use this

- You prefer `on_bar(ctx, bar)` style over subclassing `Strategy`.
- You want a fast migration path from function-style backtests to live sessions.
- You need direct `submit_order(...)` in `broker_live`.

## 2. Two runtime modes

### 2.1 paper (simulated matching)

Start with paper mode to verify callback flow:

- Example: [38_live_functional_strategy_demo.py](https://github.com/akfamily/akquant/blob/main/examples/38_live_functional_strategy_demo.py)
- Typical setup:
  - `trading_mode="paper"`
  - `strategy_cls=on_bar`
  - `initialize/on_order/on_trade/context`

### 2.2 broker_live (real broker order routing)

Switch to broker_live after gateway connectivity is verified:

- Example: [39_live_broker_submit_order_demo.py](https://github.com/akfamily/akquant/blob/main/examples/39_live_broker_submit_order_demo.py)
- Audit example: [42_live_broker_event_audit_demo.py](https://github.com/akfamily/akquant/blob/main/examples/42_live_broker_event_audit_demo.py)
- Key points:
  - `trading_mode="broker_live"`
  - call `ctx.submit_order(...)` inside `on_bar`
  - pass explicit `client_order_id` for idempotency tracking
  - default execution semantics is `execution_semantics_mode="strict"` (terminal states are driven by broker order callbacks)
  - optional `on_broker_event` for unified `event_type/owner_strategy_id/payload` persistence

You can pass `execution_semantics_mode` via `gateway_options`:

- `strict` (default, recommended for production): terminal states such as `Cancelled/Rejected/Filled` are confirmed by `OnRtnOrder`; error callbacks cache rejection reasons and merge them into subsequent order callbacks.
- `compatible` (migration mode): allows immediate local terminal-state updates for selected error/cancel paths to preserve legacy behavior.

## 3. Function-style template

```python
def initialize(ctx):
    ctx.sent = False

def on_bar(ctx, bar):
    if not ctx.sent and getattr(ctx, "broker_ready", False):
        ctx.submit_order(
            symbol=bar.symbol,
            side="Buy",
            quantity=1.0,
            client_order_id="demo-1",
            order_type="Market",
        )
        ctx.sent = True

from akquant import run_live

run_live(
    strategy_cls=on_bar,
    initialize=initialize,
    on_order=on_order,
    on_trade=on_trade,
    on_timer=on_timer,
    context={"strategy_name": "demo"},
    instruments=instruments,
    broker="ctp",
    trading_mode="broker_live",
    gateway_options={"execution_semantics_mode": "strict"},
    duration="30s",
    show_progress=False,
)
```

## 4. Common troubleshooting

- `submit_order` not ready yet
  - Cause: the trader gateway has not finished connecting/logging in.
  - Fix: guard with `if getattr(ctx, "broker_ready", False):` before placing
    (readiness is decided by run_live's heartbeat poll; submitting before
    ready raises a clear error).
- `duplicate active client_order_id`
  - Cause: reused active client id.
  - Fix: generate a fresh `client_order_id` for each new order.
- Market data arrives but no trades
  - Cause: trader gateway not connected, risk rejection, or invalid lot/tick constraints.
  - Fix: inspect `on_order` status and rejection reason first.
- Cancel request sent but status remains `Submitted`
  - Cause: strict semantics requires `OnRtnOrder(Cancelled)` to finalize terminal state.
  - Fix: verify trader callback path and broker order-return logs, not only request send success.

It is recommended to enable logging explicitly before troubleshooting live/paper runs:

```python
import akquant

akquant.configure_logging(
    akquant.LogConfig(
        profile="live",
        level="INFO",
        console=True,
        file_json=True,
        filename="logs/live_runner.log",
    )
)
```

This places strategy-side `on_order` / `on_trade` logs and gateway/execution warnings into the same pipeline. It makes rejection, unknown-cancel, and strict-semantics state transition issues easier to trace by fields such as `symbol`, `order_id`, `client_order_id`, and `strategy_id`.

## 5. Suggested rollout

- Step 1: validate callback flow in paper mode.
- Step 2: run broker_live with minimum order size.
- Step 3: add advanced logic after connectivity is stable.

## 6. Offline verification: `broker="replay"`

`replay` is a built-in deterministic market-data replay source, used to verify
your strategy's live data path (whether it receives bar/tick events, whether
all symbols arrive, whether `current_tick` is correct) without a real broker.

```python
from akquant import AssetType, Instrument, run_live
from akquant.akquant import Bar

run_live(
    strategy_cls=MyStrategy,
    instruments=[Instrument(symbol="DEMO_A", asset_type=AssetType.Stock, ...)],
    broker="replay",
    trading_mode="paper",
    gateway_options={"bars": bars},   # list[Bar] / list[Tick] / DataFrame
)
```

Events are pushed in ascending timestamp order, globally interleaved across
symbols. The session ends on its own once the data is exhausted.

**Boundaries**:

- It only provides market data and **does not simulate fills**
  (`trader_gateway=None`), so it cannot be used with
  `trading_mode="broker_live"` — that raises `ValueError`. Matching is handled
  by paper mode's simulated execution backend.
- **Timer semantics are not covered.** Replay data carries historical
  timestamps, while the live engine judges timer expiry against the wall
  clock, so the two timelines inevitably diverge. The behavior of `on_timer` /
  `schedule_daily` in a replay session is not guaranteed; use backtesting to
  verify scheduled tasks.
- A multi-symbol DataFrame needs a `股票代码` column, otherwise it degrades to
  a single symbol. For multi-symbol scenarios, prefer `list[Bar]` directly.
- Self-termination depends on every event's timestamp being positive:
  non-positive timestamps are silently dropped by the engine, so the declared
  event count can never be reached and the session hangs (a common cause is a
  date column with values `pd.to_datetime(errors="coerce")` cannot parse,
  which become `NaT` and then a non-positive timestamp). `build_replay_bundle`
  already validates against this at build time and rejects such data, but if
  your data source isn't fully under your control, it is still worth passing
  `duration` explicitly as a safety net — this is also why
  `examples/38_live_functional_strategy_demo.py` still passes `duration`.
