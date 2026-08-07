# External Signal Ingestion

Turn trading instructions pushed by an external quant signal platform into orders. A signal **is already an instruction**, so this path deliberately **bypasses strategy callbacks** — the strategy does not get to decide again.

Use it when signals are produced elsewhere (factors, timing, portfolio optimisation all live on the platform) and AKQuant only handles intake, auth, risk, execution and audit.

Design rationale and rejected alternatives: [Signal Ingestion RFC](../meta/signal-ingestion-rfc.md) (Chinese).

## Minimal runnable example

```python
from akquant import run_live
from akquant.signal import QueueSignalSource, Signal

source = QueueSignalSource()

# Safe to call from any thread
source.put(Signal(
    signal_id="platform-0001",   # idempotency key, unique on the platform side
    symbol="000001.SZ",
    action="buy",
    quantity=100,
    price=10.5,
))

run_live(
    instruments=[...],
    broker="ctp",
    trading_mode="paper",
    signal_source=source,
)
```

`run_live` owns the source lifecycle: `bind` → `start` (before the engine loop) → `stop` (on exit).

Full runnable example: [`examples/61_signal_platform_webhook.py`](https://github.com/akfamily/akquant/blob/main/examples/61_signal_platform_webhook.py).

## ⚠️ Risk coverage differs by mode

This is an **engine-architecture fact, not a setting**. Know it before going live:

| | `trading_mode="paper"` | `trading_mode="broker_live"` |
|---|---|---|
| Order path | engine event channel | broker channel (`BrokerOrderSubmitter`) |
| `max_order_value` / `max_order_size` / `max_position_size` | ✅ | ✅ |
| `max_daily_loss` / `max_drawdown` / `strategy_risk_budget` | ✅ | ❌ **not enforced** |
| Fills | simulated matcher | real venue |

Why: the engine's `RealtimeExecutionClient` does not forward orders to the broker, so broker_live orders must go through the Python-side broker channel, where only the three strategy-level limits are checked up front.

**Recommendation**: validate the real signal flow and your limits under `paper` first, then switch to `broker_live`, and add account-level circuit breakers (daily loss / drawdown) on the platform or venue side.

## Signal contract

`Signal` is a pydantic model — invalid fields fail immediately rather than reaching the venue:

| Field | Required | Notes |
|---|---|---|
| `signal_id` | ✅ | **Idempotency key**. Re-pushing the same id places at most one order |
| `symbol` | ✅ | Instrument code |
| `action` | ✅ | `"buy"` / `"sell"` (string or `SignalAction`) |
| `quantity` | ✅ | Must be positive |
| `price` | | Omit for a market order |
| `order_type` | | `"Limit"` / `"Market"`; inferred from `price` when omitted |
| `strategy_id` | | Defaults to `_default`; risk limits route on it |
| `timestamp` / `tag` | | Platform timestamp, free-form marker |

## Idempotency

Platforms reconnect, restart and retry. Without an idempotency key every re-push is a **new order**.

`SignalDedup` deduplicates on `signal_id`. Two boundaries worth remembering:

- delivery **raised** (gateway down) → the mark is released, a re-push will be accepted;
- sink **synchronously reported no order** (blocked by pre-trade risk) → the mark is kept; re-pushing only gets rejected again.

The seen-set is bounded (100k LRU by default) and logs a WARNING once eviction starts — a very old `signal_id` could then be treated as new. Silent eviction would mean silent duplicate orders.

## Receipts

`SignalSource.on_result` receives a `SignalResult`:

| `status` | Meaning |
|---|---|
| `accepted` | Delivered (**not** necessarily filled) |
| `duplicate` | Dropped idempotently |
| `rejected` | Rejected by risk or the venue |
| `error` | Processing raised |

**Asynchronous** rejections are reported too: `run_live` wraps `SignalDispatcher.handle_reject` onto each strategy's `on_reject` (wrapping, not replacing yours) and maps back via the `signal_id` carried in `tag`.

## Three sources

- **`QueueSignalSource`** — in-process queue, no dependencies. Also the test harness.
- **`HttpSignalSource`** — webhook built on the standard library, no extra dependencies.
- **`RedisSignalSource`** — Redis Stream consumer group; `pip install 'akquant[signal-redis]'`. Uses `XREADGROUP` (consumer groups + explicit ack) rather than `BLPOP`, which drops a message if you crash mid-processing.

```python
from akquant.signal import HttpSignalSource

source = HttpSignalSource(
    token=os.environ["AKQUANT_SIGNAL_TOKEN"],        # required
    port=8765,
    secret=os.environ.get("AKQUANT_SIGNAL_SECRET"),  # required across hosts
)
```

## 🔒 Security

`HttpSignalSource` is a network entry point that can place real orders, so its constraints are **hard**, not advisory:

| Constraint | Behaviour |
|---|---|
| Auth required | `token` must be a non-empty string or construction raises `ValueError`. `None` is rejected explicitly — otherwise `str(None)` yields the literal `"None"`, a seemingly-authenticated endpoint |
| Localhost by default | `host` defaults to `127.0.0.1`; binding elsewhere **raises**, and needs an explicit `allow_remote=True` |
| Replay protection | With `secret` set, requires `HMAC-SHA256(secret, "{ts}.{body}")` plus a timestamp window (±30s default), combined with `signal_id` idempotency |
| No distinguishing oracle | All auth failures return `401 {"error":"unauthorized"}`; the specific cause only goes to the log |

**AKQuant makes no transport-security promise**: terminate TLS and restrict access in a reverse proxy.

Reuse `akquant.signal.sign` on the client side:

```python
import json, time
from akquant.signal import sign

body = json.dumps(payload).encode()
ts = int(time.time())
headers = {
    "Authorization": f"Bearer {token}",
    "X-Signal-Timestamp": str(ts),
    "X-Signal-Signature": sign(secret, body, ts),
}
```

## Deployment

Mirroring vn.py's trade-off (WebTrader runs as a separate process behind RPC), **process separation is recommended for production**:

```
platform ──HTTPS──> reverse proxy ──> intake process ──Redis Stream──> trading process
```

HTTP failures and load stay behind the process boundary instead of reaching the trading loop.

## Custom sources

Implement the `SignalSource` protocol; subclass `SignalSourceBase` to skip the `bind`/`on_result` boilerplate.

⚠️ **`start()` must not return until its thread is actually running.** `run_live` calls it synchronously before the engine loop; once it returns, the main thread enters the Rust loop and holds the GIL for long stretches, so a not-yet-scheduled thread may never run. This is a measured result, not a theoretical concern.

Also: an injected order only fills when a **subsequent** market event arrives. A signal landing after the last bar stays `New` — matching semantics, not a defect.

## See also

- [Custom Broker Registry](custom_broker_registry.md)
- [Custom Broker Production Checklist](custom_broker_production_checklist.md)
- [Multi-Strategy Guide](multi_strategy_guide.md) — `strategy_id` routing and per-strategy risk
