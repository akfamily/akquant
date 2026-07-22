# Custom Indicator Guide

This page answers one practical question: when built-in indicators are not enough, how do you write, register, and maintain your own indicators in AKQuant?

Typical use cases:

- private factors or strategy-specific signals;
- rapid prototypes built on pandas;
- stateful indicators updated bar by bar in the event stream;
- indicators that must survive warm-start resume.

## Start With The Right Scope

In AKQuant, these are related but different tasks:

| Goal | Recommended path | Typical API |
| :--- | :--- | :--- |
| Add a private signal to a strategy | custom `Indicator` / custom incremental object | register on `Strategy` |
| Compute a full series from a `DataFrame` | `indicator_mode="precompute"` | `register_precomputed_indicator(...)` |
| Maintain state bar by bar | `indicator_mode="incremental"` | `register_incremental_indicator(...)` |
| Add a new name to `akquant.talib` | modify the compatibility layer source | not runtime plugin registration |

If your goal is simply "use my own indicator inside a strategy", you usually do not need to extend `akquant.talib`.

## Path 1: Precomputed Indicators

Use `precompute` mode when the indicator is naturally vectorized over the full `DataFrame`.

### Minimal example

```python
from akquant import Indicator, Strategy


class PrecomputeMomentumStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.indicator_mode = "precompute"
        self.mom10 = Indicator(
            "mom10",
            lambda df: df["close"] - df["close"].shift(10),
        )
        self.register_precomputed_indicator("mom10", self.mom10)

    def on_bar(self, bar):
        value = self.mom10.get_value(bar.symbol, bar.timestamp)
        if value == value and value > 0:
            self.buy(bar.symbol, 100)
```

### Good fit when

- the indicator is naturally vectorized;
- you want to reuse pandas `rolling`, `shift`, or `ewm`;
- development speed matters more than streaming-style updates;
- each symbol has a full history slice available up front.

### Notes

- `Indicator(name, fn, **kwargs)` expects a function returning a `pd.Series`;
- `get_value(symbol, timestamp)` reads from the cached series;
- results are cached per symbol.

## Path 2: Incremental Indicators

Use `incremental` mode when the indicator should evolve inside the event stream.

### Minimal example

```python
from collections import deque

import pandas as pd
from akquant import Indicator, Strategy


class MyMomentum(Indicator):
    def __init__(self, period: int = 10):
        super().__init__("my_momentum", lambda df: df["close"] - df["close"].shift(period))
        self.period = period
        self.buffer: deque[float] = deque(maxlen=period)
        self._current_value = float("nan")

    def update(self, value: float) -> float:
        if pd.isna(value):
            return self._current_value
        self.buffer.append(float(value))
        if len(self.buffer) < self.period:
            self._current_value = float("nan")
        else:
            self._current_value = self.buffer[-1] - self.buffer[0]
        return self._current_value

    @property
    def value(self) -> float:
        return self._current_value


class IncrementalMomentumStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.indicator_mode = "incremental"

    def on_start(self):
        self.register_incremental_indicator(
            "mom10",
            indicator_factory=lambda: MyMomentum(period=10),
            source="close",
            symbols=["AAPL", "MSFT"],
            warmup_bars=10,
        )

    def on_bar(self, bar):
        value = self.mom10.value
        if value == value and value > 0:
            self.buy(bar.symbol, 100)
```

## Why `indicator_factory` is recommended

In a multi-symbol strategy, incremental indicators usually carry internal state. Reusing one instance across multiple symbols can mix state and produce incorrect results.

Recommended:

```python
self.register_incremental_indicator(
    "mom10",
    indicator_factory=lambda: MyMomentum(period=10),
    source="close",
    symbols=["AAPL", "MSFT"],
)
```

Single-instance form, better for quick single-symbol experiments:

```python
self.mom10 = MyMomentum(period=10)
self.register_incremental_indicator("mom10", self.mom10, source="close")
```

## What `source` means

`source` tells the framework which field from the market event should be fed into the indicator. Common choices:

- `source="close"`
- `source="open"`
- `source="high"`
- `source="low"`
- `source="volume"`

If your indicator needs multiple inputs, align your `update(...)` signature with the incremental input mode expected by the framework.

## Using `warmup_bars`

`warmup_bars` bootstraps the incremental indicator with bars before `start_time`.

Use it when:

- you want a valid value on the first active bar;
- your indicator depends on a rolling window;
- you do not want to manually skip the first N bars inside `on_bar`.

Runnable example:

- [58_incremental_bootstrap_demo.py](https://github.com/akfamily/akquant/blob/main/examples/58_incremental_bootstrap_demo.py)
- [60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py)

## Warm Start And Serialization

If the strategy uses `run_from_checkpoint`, your custom indicator must preserve its internal state correctly.

Practical rules:

- simple Python objects are often pickle-compatible already;
- if the indicator stores file handles, sockets, locks, or other non-serializable objects, handle them explicitly;
- implement `__getstate__` and `__setstate__` when needed.

Example:

```python
def __getstate__(self):
    state = self.__dict__.copy()
    return state


def __setstate__(self, state):
    self.__dict__.update(state)
```

See also: [Warm Start Guide](../advanced/warm_start.md).

## Boundary With `akquant.talib`

Many users mix up "custom strategy indicators" and "extending `akquant.talib`". A practical mental model:

- `akquant.talib`: built-in TA-Lib-style compatibility layer;
- custom strategy indicators: strategy-local building blocks registered on `Strategy`;
- new Rust high-performance indicators: source-level extension plus recompilation, not runtime hot-plugging.

If you only need a private signal inside one strategy, prefer a custom strategy indicator instead of extending `akquant.talib`.

## Export Indicators For Frontend Use

If your goal is not only to use a custom indicator inside the strategy, but also to send the indicator output to a web frontend, treat indicator calculation and indicator output as separate concerns:

- keep indicator calculation inside `Strategy` / `Indicator`;
- use `Strategy.record_indicator(...)` to record normalized indicator points;
- after the run, use `BacktestResult.indicator_df(...)` or `export_indicators(...)` for downstream systems.

### Minimal example

```python
from akquant import Bar, Strategy


class IndicatorExportStrategy(Strategy):
    def on_bar(self, bar: Bar) -> None:
        spread = bar.high - bar.low
        self.record_indicator(
            name="intrabar_spread",
            value=spread,
            display_name="Intra Bar Spread",
            pane=1,
            render_type="line",
            precision=4,
            meta={"source": ["high", "low"]},
        )
```

!!! note "About the `pane` value"
    `pane` is an **integer row index**: `0` is the main (price) pane and `1`..`4`
    are sub panes stacked below it. Omitting `pane` defaults to the main pane
    (`0`). Values outside `0..4` raise an error. This matches what chart renderers
    actually consume, and `record_indicator` emits the same integer `pane` on both
    the plain backtest export and the frontend stream bridge paths.

!!! warning "Breaking change since 0.3"
    Earlier versions accepted string panes such as `"main"` / `"sub1"` / `"主图"` /
    `"signal"`; these have been removed. Use integer indices instead. The former
    `pane="signal"` semantics (drawing trade signals) is now expressed with
    `render_type="signal"` rendered on the main pane (`pane=0`).

!!! note "About the `render_type` value"
    `render_type` is a **closed enum** of 7 values, so consumers can implement
    exhaustive rendering branches:

    | Value | Rendering |
    | :--- | :--- |
    | `line` | Connected line (default) |
    | `area` | Line filled to zero |
    | `bar` | Vertical bars |
    | `column` | Alias of `bar` (semantic: categorical columns) |
    | `histogram` | Alias of `bar` (semantic: distribution bars) |
    | `scatter` | Disconnected point markers |
    | `signal` | Trade-signal markers, drawn on the main pane |

    Passing a value outside the enum raises `ValueError` (fail-fast) rather than
    silently degrading to a line.

After the run:

```python
result = ...

# 1) Read inside Python
indicator_df = result.indicator_df(name="intrabar_spread", symbol="AAPL")

# 2) Generate a lightweight local preview
fig = result.plot_indicators(
    name="intrabar_spread",
    symbol="AAPL",
    show=False,
    filename="indicator_preview.html",
)

# 3) Export for frontend or external services
result.export_indicators("indicator_outputs.json", format="json")
result.export_indicators("indicator_outputs", format="parquet")
```

When available, the JSON export also includes a top-level `run_id` so downstream services can correlate offline exports with the streaming event flow.

### Built-in Minimal Visualization

If you only want a quick history preview before wiring a full frontend, use:

- `result.plot_indicators(...)`
- `from akquant.plot import plot_indicators`

This built-in path is intentionally lightweight:

- it keeps `result.plot()` focused on the existing account dashboard;
- it splits subplots by `pane`;
- it reuses `render_type`, with day-one support for common `line` and `bar`;
- it supports filtering by `name`, `symbol`, and `include_warmup`;
- it can write a local HTML file for quick inspection alongside exported JSON.

Example:

```python
fig = result.plot_indicators(
    name="intrabar_spread",
    symbol="AAPL",
    include_warmup=False,
    show=False,
    filename="indicator_preview.html",
    title="Indicator Preview",
)
```

If you need enterprise-grade multi-panel UX, persistence, permissions, or realtime subscriptions, keep those concerns in external systems and let AKQuant stay responsible for the preview plus normalized data production.

### Optional Indicator Section In Reports

If you want the indicator preview embedded into the built-in HTML report instead of a separate figure, enable it explicitly:

```python
result.report(
    filename="akquant_report.html",
    show=False,
    include_indicators=True,
    indicator_name="intrabar_spread",
    indicator_symbol="AAPL",
    indicator_include_warmup=False,
)
```

This path is intentionally constrained:

- it is off by default, so existing `report()` output does not change;
- it is meant for a lightweight indicator section inside the strategy report;
- if no indicator data exists, the report shows an empty-state notice;
- if you need richer interaction or layout control, keep that in external frontend systems.

### Bridging Stream Events To Frontend Messages

If your external service is a WebSocket or SSE gateway, keep the raw payload parsing out of your business code and use:

- `akquant.is_indicator_stream_event(event)`
- `akquant.to_indicator_message(event)`
- `akquant.to_indicator_messages(events)`

Example:

```python
def on_event(event):
    if not aq.is_indicator_stream_event(event):
        return
    message = aq.to_indicator_message(event)
    if message is not None:
        websocket.broadcast_json(message)
```

These helpers are meant to:

- bridge only `indicator_point` and `indicator_snapshot`
- coerce numeric fields into frontend-friendly values
- unpack `meta_json` and `items_json`
- preserve the outer stream semantics such as `run_id`, `seq`, and `ts`

The bridged `snapshot` payload now also includes a few shortcut fields so frontend
code does not have to rescan `items` on every update:

- `indicator_keys`
- `panes`
- `render_types`
- `value_by_key`
- `items_by_key`
- `warmup_count`
- `has_warmup`

The bridge helper also normalizes `_unknown` or empty `symbol` values into `None`,
and accepts already-decoded `dict/list` values for `meta_json` and `items_json`,
which makes gateway-side wrapping easier.

This is not a new transport layer. It is just a normalization layer that turns AKQuant stream events into steadier frontend message objects.

### Injecting a custom collector (IndicatorSink)

To route indicator points into your own collection/forwarding logic (e.g. push
to a broadcast queue, write to a time-series DB), you don't need to monkey-patch
private strategy attributes. Both `run_backtest` and `run_live` accept a public
`indicator_recorder` argument — any object implementing the
`akquant.IndicatorSink` protocol:

```python
from akquant import IndicatorSink, run_backtest


class QueueSink:
    """Push each indicator point into your own queue without accumulating."""

    def record(self, *, name, value, symbol, timestamp, owner_strategy_id, **kwargs):
        my_queue.put((name, symbol, timestamp, value))

    def build_payload(self):
        return {"definitions": [], "instances": [], "points": []}

    def flush_stream_snapshot(self):
        ...

    def set_stream_emitter(self, emitter):
        ...


run_backtest(..., indicator_recorder=QueueSink())
```

`IndicatorSink` is the public extension point for indicator collectors
(analogous to Backtrader's Analyzer/Observer):
- the built-in `IndicatorRecorder` satisfies it (used by default in backtests,
  accumulating points for `build_payload`);
- you can pass your own implementation to route indicator data anywhere without
  touching AKQuant internals.

### Live realtime indicator streaming

Backtest and live indicator streams are **isomorphic**: `run_live` also accepts
`on_event` and `indicator_recorder`, producing the same `indicator_point` /
`indicator_snapshot` events as backtests, so one frontend consumer handles both.

```python
from akquant import run_live

run_live(
    strategy_cls=MyStrategy,
    instruments=instruments,
    broker="ctp",
    trading_mode="broker_live",
    on_event=on_event,   # same event callback as run_backtest
)
```

A live session is a long-running process, so it defaults to a lightweight
streaming sink that **only emits, never accumulates**: it fires stream events
without retaining historical points in memory, avoiding unbounded growth over
long runs. If you pass only `on_event` (no `indicator_recorder`), `run_live`
enables this streaming sink automatically.

### Zero-Dependency Browser Live Preview

If you want a browser-based demo that product or frontend teammates can open
immediately, but you do not want to introduce `fastapi`, `uvicorn`, or
`websockets` yet, use `examples/64_indicator_live_web.py`.

This example does three things:

- consumes stream events with `run_backtest(..., on_event=...)`
- normalizes them with `aq.to_indicator_message(event)`
- serves a tiny `/state` JSON endpoint via built-in `http.server`, then lets the browser poll and draw the `close_echo` line

It now supports two polling modes:

- request `/state` for the recent full snapshot window
- request `/state?since_seq=123` for incremental messages with `seq > 123`, while still returning total counts and the latest cursor

The payload shape now separates common metadata from message bodies:

- shared fields live under `cursor`, `counts`, and `latest_indicator_values`
- full snapshot mode returns message windows under `window.point_messages` and `window.snapshot_messages`
- incremental mode returns only new messages under `delta.point_messages` and `delta.snapshot_messages`

Run it with:

```bash
UV_INDEX_URL=https://pypi.org/simple uv run python examples/64_indicator_live_web.py --open
```

For a quick smoke check, keep the server alive for a shorter window:

```bash
UV_INDEX_URL=https://pypi.org/simple uv run python examples/64_indicator_live_web.py --keep-seconds 1
```

The goal is not to become a full frontend product. The goal is to help you:

- verify that indicator stream data is leaving the backtest correctly
- give frontend code a stable `/state` JSON shape to consume first
- demonstrate live indicator rendering without adding new dependencies

### Current output shape

The first implementation exposes three structured layers:

- indicator definitions, such as `display_name`, `pane`, and `render_type`
- indicator instances grouped by `strategy/symbol/indicator/meta`
- indicator points as the actual time series values

Each indicator point carries both `timestamp` (nanoseconds) and `timestamp_ms` (milliseconds): the former is for nanosecond parsing on the Python side, while the latter can be consumed directly by frontend charting libraries without any unit conversion. Indicator `meta` is serialized with `ensure_ascii=False`, so non-ASCII characters (e.g. CJK) stay readable.

This keeps AKQuant focused on producing stable indicator data instead of coupling the framework to a specific charting library.

### Recommended boundary

Suggested split of responsibilities:

- `AKQuant` owns:
  - indicator calculation
  - indicator recording
  - indicator query
  - indicator export
- external systems own:
  - persistence
  - APIs
  - websocket delivery
  - frontend applications

In short, AKQuant should act as the indicator producer, not the full enterprise frontend platform.

### Recommended examples

- [60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py)
- [61_indicator_visualization_export_demo.py](https://github.com/akfamily/akquant/blob/main/examples/61_indicator_visualization_export_demo.py)
- [62_indicator_streaming_demo.py](https://github.com/akfamily/akquant/blob/main/examples/62_indicator_streaming_demo.py)
- [63_indicator_ws_bridge_demo.py](https://github.com/akfamily/akquant/blob/main/examples/63_indicator_ws_bridge_demo.py)
- [64_indicator_live_web.py](https://github.com/akfamily/akquant/blob/main/examples/64_indicator_live_web.py)

## Common Pitfalls

- Pitfall 1: every custom indicator must inherit from `Indicator`
  - Not always. `Indicator(name, fn)` is enough for many precompute cases.
- Pitfall 2: one incremental instance can be shared safely across symbols
  - Usually no. Prefer `indicator_factory` for production multi-symbol strategies.
- Pitfall 3: `warmup_bars=20` double-consumes the first active bar
  - It does not. Warmup only uses history before the active start boundary.
- Pitfall 4: custom indicators automatically work with warm start
  - Not guaranteed. Verify serialization.
- Pitfall 5: strategy-local indicators and `akquant.talib` extensions are the same thing
  - They solve different problems at different layers.

## Recommendation Matrix

| Goal | Recommended approach |
| :--- | :--- |
| Validate an idea quickly | `Indicator(name, fn)` + `precompute` |
| Single-symbol bar-by-bar state | `incremental` + single instance |
| Multi-symbol production strategy | `incremental` + `indicator_factory` |
| Need valid values on the first active bar | incremental + `warmup_bars` |
| Need resumable state | ensure the indicator is serializable |
| Need maximum performance | consider a Rust implementation later |

## Further Reading

- [Strategy Guide](./strategy.md)
- [Warm Start Guide](../advanced/warm_start.md)
- [AKQuant Indicator Reference](./rust_indicator_reference.md)
- [Indicator Playbook](./talib_indicator_playbook.md)
- [Runnable example: 60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py)
