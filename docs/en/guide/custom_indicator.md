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
| Maintain state bar by bar / tick by tick | an incremental object (has `update()`) | `self.I(aq.EMA(20), ...)` |
| Compute a full series from a `DataFrame` with pandas | `Indicator(name, fn)` | `self.I(Indicator(...), ...)` |
| Add a new function name to `akquant.talib` | modify the Python/Rust compatibility layer source | not runtime registration |

If your goal is simply "use my own indicator inside a strategy", you usually do not need to extend `akquant.talib`.

## The Single Entry Point: `self.I()`

`Strategy.I()` is the one and only way to declare an indicator. Declare it once and the
framework takes care of three things for you: **advancing it on every bar**, **serial
lookback through `ind[0]` / `ind[1]`**, and **reporting plot points on demand**. This
mirrors `ta.*` + `plot()` in TradingView Pine Script.

```python
import akquant as aq
from akquant import Bar, Strategy


class MaCross(Strategy):
    def on_start(self) -> None:
        # One declaration = auto update + auto plot + lookback
        self.ema_fast = self.I(aq.EMA(10), pane=0, color="#e91e63", label="EMA10")
        self.ema_slow = self.I(aq.EMA(30), pane=0, color="#3f51b5")
        self.rsi = self.I(aq.RSI(14), pane=1)

    def on_bar(self, bar: Bar) -> None:
        # [0] is the current bar, [1] the previous one — same as Pine's sma[0] / sma[1]
        if self.ema_fast[1] is None or self.ema_slow[1] is None:
            return
        if self.ema_fast[1] <= self.ema_slow[1] and self.ema_fast[0] > self.ema_slow[0]:
            self.buy(bar.symbol, 100)
        # No record_indicator needed — plot arguments make reporting automatic
```

!!! warning "Breaking change since 0.3"
    The old `indicator_mode` switch together with `register_incremental_indicator(...)` /
    `register_precomputed_indicator(...)` has been **removed**. `self.I()` now dispatches
    on the object you hand it. A welcome side effect: the two kinds of indicator can
    **coexist in the same strategy** — the old mode switch was mutually exclusive.

    | Old | New |
    | :--- | :--- |
    | `self.indicator_mode = "incremental"` + `register_incremental_indicator("x", obj, source="close")` | `self.x = self.I(obj, name="x", source="close")` |
    | `self.indicator_mode = "precompute"` + `register_precomputed_indicator("x", ind)` | `self.x = self.I(ind, name="x")` |
    | `register_incremental_indicator(..., indicator_factory=f)` | `self.I(factory=f, ...)` |

### Where to declare

Both `__init__` and `on_start` work, but **`on_start` is recommended**: `self.params` is
injected *after* `__init__`, so a parameterized indicator (the kind `run_grid_search`
sweeps) can only read its parameter values from `on_start`. Reserve `__init__` for literal
arguments.

`subscribe_bars` is the exception — it **must** be called in `__init__`, because the
subscription table is frozen before the engine starts. `self.I(freq="5min")` then validates
that the requested frequency was actually subscribed.

### Serial lookback

- `ind[0]`: the value on the current bar (`.value` is an alias for it)
- `ind[1]`: the previous bar; `ind[n]`: n bars back
- **Out-of-range access returns `None` instead of raising** — the warmup period is
  out of range by nature, and raising would force every strategy to wrap reads in `try`
- Lookback depth is bounded by `lookback` (default `128`) over a ring buffer: a live
  session is an unbounded stream, and an unbounded buffer would leak

A not-ready value is always `None` (both a native indicator with an unfilled window and a
precomputed indicator whose asof lookup misses are normalized to `None`), so a single
`if value is None` check is enough — no separate `NaN` handling.

## Path 1: Incremental Indicators

Passing any object that exposes `update()` selects the incremental path. AKQuant ships
100+ stateful indicator classes implemented in Rust (`aq.SMA` / `aq.EMA` / `aq.RSI` /
`aq.MACD` / `aq.ATR` / `aq.BollingerBands` and many more) — they are the equivalent of
Pine's `ta.*` and work out of the box.

### Writing your own incremental indicator

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
    def on_start(self):
        self.mom10 = self.I(
            factory=lambda: MyMomentum(period=10),
            name="mom10",
            source="close",
            symbols=["AAPL", "MSFT"],
            warmup_bars=10,
        )

    def on_bar(self, bar):
        value = self.mom10[0]
        if value is not None and value > 0:
            self.buy(bar.symbol, 100)
```

### Why `factory` is recommended

In a multi-symbol strategy, incremental indicators usually carry internal state. Sharing
one instance across symbols would cross the wires, so passing an *instance* (rather than a
`factory`) and then using it across multiple symbols raises an error immediately — state is
never silently shared.

Recommended:

```python
self.mom10 = self.I(factory=lambda: MyMomentum(period=10), source="close")
```

Rather than:

```python
self.mom10 = self.I(MyMomentum(period=10), source="close")  # single symbol only
```

!!! warning "`factory` must be picklable under warm start"
    A checkpoint serializes the strategy instance, and a lambda defined inside `on_start`
    is a local object that cannot be pickled. When you plan to use `run_from_checkpoint`,
    switch to a module-level function plus `functools.partial`:

    ```python
    from functools import partial

    def _make_sma(window: int):
        return aq.SMA(window)

    # inside on_start:
    self.sma = self.I(factory=partial(_make_sma, 20), name="sma")
    ```

### What `source` means

`source` tells the framework which field from the market event should be fed into the
indicator. Common choices:

- `source="close"`
- `source="open"`
- `source="high"`
- `source="low"`
- `source="volume"`

Indicators that need several inputs (ATR and friends) declare the feeding shape with
`input_mode`: `"source"` (single value, the default) / `"hl"` / `"hlc"` / `"ohlc"` /
`"close_volume"`. The framework passes the fields to your `update(...)` in that order.

### Multi-value indicators: `outputs`

For indicators such as MACD that emit several components at once, name the components with
`outputs`. They are split into independent lines (`indicator_key` looks like `macd.dif`)
instead of handing a raw tuple to the frontend:

```python
self.macd = self.I(aq.MACD(12, 26, 9), name="macd", pane=2,
                   outputs=("dif", "dea", "hist"))

# Reading:
self.macd[0]        # the whole tuple (dif, dea, hist)
self.macd.dif[0]    # a single component, which also supports [1] lookback
```

If the number of `outputs` does not match the indicator's actual component count, you get
an error rather than a silent misalignment.

## Path 2: Precomputed Indicators

When the indicator is a better fit for a one-shot computation over the full `DataFrame`,
pass an `Indicator(name, fn)` instance and `self.I()` routes it to the vectorized
precompute path.

### Minimal example

```python
from akquant import Indicator, Strategy


class PrecomputeMomentumStrategy(Strategy):
    def on_start(self):
        self.mom10 = self.I(
            Indicator("mom10", lambda df: df["close"] - df["close"].shift(10)),
            name="mom10",
        )

    def on_bar(self, bar):
        value = self.mom10[0]
        if value is not None and value > 0:
            self.buy(bar.symbol, 100)
```

### Good fit when

- the indicator is naturally vectorized;
- it mostly relies on pandas `rolling` / `shift` / `ewm`;
- backtest development speed matters more than streaming-style updates;
- the full history of each symbol can be prepared up front.

### Limitations

- **Not available in live trading**: a live session has no complete history `DataFrame`;
- **Tick input raises**: precompute needs full OHLC, while a tick only carries a trade price;
- **`freq=` is not supported**: use an incremental indicator for windowed frequencies.
- **No `warmup_bars=`**: there is no incremental state to warm up - the whole series is computed up front.
- **Multi-symbol out of the box**: `Indicator` caches its full series per symbol, so no `factory=` is needed.

## Automatic Plotting

Passing any plot argument (`pane` / `color` / `label` / `render_type` / `reference_lines` /
`scale_group`), or setting `plot=True` explicitly, makes the framework report a plot point
after every update — **no hand-written `record_indicator` inside `on_bar`**.

**Nothing is reported by default**, matching Pine (`ta.sma()` only computes; `plot()` draws).
This is not merely a style choice: the live indicator sink emits one event per point, and
turning every indicator on by default across many symbols would flood the frontend link.

Points that are not ready yet (value `None`) are not reported — the counterpart of Pine's
`na` during warmup.

```python
# Compute only, do not draw
self.atr = self.I(aq.ATR(14), input_mode="hlc")

# Compute and draw
self.rsi = self.I(aq.RSI(14), pane=1, reference_lines=[
    {"value": 70, "label": "Overbought"},
    {"value": 30, "label": "Oversold"},
])

# With no other plot argument, opt in explicitly with plot=True
self.sma = self.I(aq.SMA(20), plot=True)
```

The plot arguments mean exactly what they mean on `record_indicator`; see "Export
Indicators For Frontend Use" below. `record_indicator` remains as the low-level escape
hatch for recording **non-indicator** custom values (position size, signal strength,
risk-control intermediates) that `self.I()` does not cover.

## Realtime values: intrabar on the forming window

TradingView refreshes indicators on the realtime bar with every trade and confirms them at
close (`barstate.isrealtime` / `isconfirmed`). AKQuant offers the same semantics for
window-period indicators: declare `intrabar=True` and on every base bar close the framework
computes a **provisional** value on a **copy** of the indicator from the partial window
snapshot (`ctx.current_window()`) - the real state is never touched. The real `update()`
runs when the window closes.

```python
class S(Strategy):
    def __init__(self):
        super().__init__()
        self.subscribe_bars("5min")

    def on_start(self):
        self.sma5 = self.I(aq.SMA(20), freq="5min", intrabar=True, pane=0)

    def on_bar(self, bar):                 # every 1-minute bar
        v0 = self.sma5[0]                  # provisional value on the forming 5-minute window
        v1 = self.sma5[1]                  # last confirmed 5-minute value
        if not self.sma5.confirmed: ...    # [0] is currently provisional
```

Index semantics match Pine exactly: a provisional value occupies `[0]` and the confirmed
series shifts by one. Stream events mark provisional points with `confirmed=false`, and
**their timestamp is the label the window will close with**; the confirmed point arrives
later with the same `time`, so a frontend overwrites by `(indicator_key, symbol, timestamp)`
(`akquant.lwc.to_lwc_update()` already does). Provisional points **never** enter
`indicator_df()` / `export_indicators()` - those exits hold confirmed values only.

Three limits:

- **Base-period indicators (no `freq=`) peek on every tick**: the framework builds the forming
  bar from the ticks itself (real open/high/low/close, so ATR-style H/L indicators work too);
  ticks **no longer `update()`** such an indicator - only the bar close commits state. The
  provisional timestamp is predicted with the aggregator's end-of-interval stamp
  (`(ts // interval + 1) * interval - 1ns`); if a closing bar disagrees (a source stamping at
  interval start), streaming stops for that symbol with one warning while `[0]` keeps working.
  **With the default `intrabar=False` ticks still `update()` base indicators** - pure-tick
  strategies rely on that and it is unchanged.
- **Provisional points stream only when the base period is known** (backtest:
  `run_backtest(data=[Tick,...], freq=)`; live: the gateway declares `metadata["freq"]`).
  With an unknown base period the window label drifts every base bar and a frontend could
  not overwrite; `[0]` is still computed, nothing is streamed, one warning is logged.
- Peeking relies on copying indicator state: built-in Rust indicators go through
  `akquant.clone_indicator()` (pyo3 does not expose `__copy__` for `#[derive(Clone)]`, so
  `copy` / `deepcopy` / `pickle` all raise `TypeError` on them); user-written Python
  indicators go through `copy.deepcopy`.

Runnable examples: [75_intrabar_indicators.py](https://github.com/akfamily/akquant/blob/main/examples/75_intrabar_indicators.py)
(window period) and [76_tick_intrabar_indicators.py](https://github.com/akfamily/akquant/blob/main/examples/76_tick_intrabar_indicators.py)
(base period, tick-driven).

## Pluggable studies: draw, never trade

TradingView separates indicator scripts from strategy scripts, and a chart can stack any
number of indicator-only scripts. AKQuant's counterpart is `Study`: subclass it, declare
indicators with `self.I()` in `on_start`, and attach it to any backtest or live run with
`studies=[...]`.

```python
import akquant as aq
from akquant import Study, run_backtest


class RsiStudy(Study):
    def on_start(self):
        self.rsi = self.I(aq.RSI(14), pane=1)


class MacdStudy(Study):
    study_id = "macd_view"          # defaults to the snake_case class name: "macd_study"

    def on_start(self):
        self.macd = self.I(aq.MACD(12, 26, 9), pane=2, outputs=("dif", "dea", "hist"))


result = run_backtest(strategy=MyStrategy, data=data, symbols=["600000"],
                      strategy_id="main", studies=[RsiStudy, MacdStudy])
result.indicator_df(owner="rsi_study")    # only this study's points
result.viz.review(data)                   # studies land on the LWC chart too
```

Three things to know:

- **No new pipeline.** Each study is one slot of the multi-strategy topology
  (`strategies_by_slot`); its points carry their own `owner_strategy_id`, window
  subscriptions merge across slots, and `run_live(studies=...)` has identical semantics.
- **Not trading is a hard guarantee.** Every trading API on `Study` (`buy` / `sell` /
  `place_*` / `order_target*` / `rebalance_*` / `cancel_*` ...) raises
  `StudyCannotTradeError`. A stray order inside `on_bar` fails fast inside the engine
  loop instead of silently filling. Subclass `Strategy` if you need to trade.
- **Ids must not collide.** A `study_id` equal to `strategy_id` or an existing slot key
  is a `ValueError`; a non-`Study` class inside `studies=` is a `TypeError` - otherwise
  the no-trading guarantee would be void.

Runnable example: [74_pluggable_studies.py](https://github.com/akfamily/akquant/blob/main/examples/74_pluggable_studies.py).

## Using `warmup_bars`

`warmup_bars` bootstraps the indicator with bars before `start_time`, ahead of the active
event stream.

Use it when:

- you want a valid value on the first active bar;
- your indicator depends on a rolling window, e.g. `period=20`;
- you do not want to manually skip the first N bars inside `on_bar`.

Runnable examples:

- [58_incremental_bootstrap_demo.py](https://github.com/akfamily/akquant/blob/main/examples/58_incremental_bootstrap_demo.py)
- [60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py)
- [72_declarative_indicators.py](https://github.com/akfamily/akquant/blob/main/examples/72_declarative_indicators.py)

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

!!! note "About `symbol` ownership"
    Every indicator point **always belongs to exactly one symbol**, and `symbol`
    decides which instrument's chart the indicator is drawn on.

    When `symbol` is omitted (recommended), it defaults to the symbol of the bar
    or tick currently being processed. In a multi-symbol backtest, a single
    `record_indicator` call therefore records an independent series per symbol —
    which is what you want: every instrument gets its own MA5.

    ```python
    def on_bar(self, bar: Bar) -> None:
        # symbol omitted: defaults to bar.symbol
        # In a multi-symbol backtest each symbol gets its own ma5 series
        self.record_indicator(name="ma5", value=self.ma5.value, pane=0)
    ```

    Pass `symbol` explicitly only when you deliberately want to record under a
    different instrument (for example a reference line on a benchmark or index).
    Note that **every symbol's bar still triggers the call**, so without a guard
    the same timestamp is written multiple times:

    ```python
    def on_bar(self, bar: Bar) -> None:
        # Record once, on the primary symbol's bar only
        if bar.symbol == "000300.SH":
            self.record_indicator(name="bench_ma", value=v, symbol="000300.SH")
    ```

    If neither a current bar/tick nor an explicit `symbol` is available, a
    `ValueError` is raised — the point is never silently attached to a
    placeholder symbol. Consumers can rely on `symbol` being non-empty when
    grouping.

!!! note "About the `pane` value"
    `pane` is an **integer row index**: `0` is the main (price) pane and `1`..`N`
    are sub panes stacked below it. Omitting `pane` defaults to the main pane
    (`0`). The default cap `N` is `8` — a soft, screen-readability guideline
    rather than a hard limit. Multi-factor or derivatives workflows that need
    more sub panes can raise it by setting the `AKQUANT_MAX_SUB_PANES` environment
    variable before the run. A `pane` outside `0..N` raises an error (fail-fast),
    so a mistyped index never silently lands on the wrong pane. This matches what
    chart renderers actually consume, and `record_indicator` emits the same
    integer `pane` on both the plain backtest export and the frontend stream
    bridge paths.

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

!!! note "About `reference_lines` and `scale_group`"
    - `reference_lines`: optional, a list of static reference lines, each item
      `{"value": number, "label": text, "color": color}`; used for overbought/oversold
      lines, a zero axis, or other fixed horizontal lines. For lines that move over
      time, record them as an independent indicator bar by bar.
    - `scale_group`: optional, a semantic group name for sharing a scale (e.g.
      `"percent"`), a pure hint frontends use to detect indicators with the same
      unit; it does not change the row layout decided by `pane`.

RSI example with reference lines and a scale group:

```python
self.record_indicator(
    name="rsi",
    value=rsi_value,
    pane=1,
    reference_lines=[
        {"value": 70, "label": "超买", "color": "#ef4444"},
        {"value": 30, "label": "超卖", "color": "#22c55e"},
    ],
    scale_group="percent",
)
```

After the run:

```python
result = ...

# 1) Read inside Python
indicator_df = result.indicator_df(name="intrabar_spread", symbol="AAPL")

# 2) Generate a lightweight local preview
fig = result.viz.indicators(
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

- `result.viz.indicators(...)`
- `from akquant.plot import plot_indicators`

This built-in path is intentionally lightweight:

- it keeps `result.viz.dashboard()` focused on the existing account dashboard;
- it splits subplots by `pane`;
- it reuses `render_type`, with day-one support for common `line` and `bar`;
- it supports filtering by `name`, `symbol`, and `include_warmup`;
- it can write a local HTML file for quick inspection alongside exported JSON.

Example:

```python
fig = result.viz.indicators(
    name="intrabar_spread",
    symbol="AAPL",
    include_warmup=False,
    show=False,
    filename="indicator_preview.html",
    title="Indicator Preview",
)
```

If you need enterprise-grade multi-panel UX, persistence, permissions, or realtime subscriptions, keep those concerns in external systems and let AKQuant stay responsible for the preview plus normalized data production.

#### Indicators on the LWC review chart

`result.viz.review()` now draws the indicators reported through `self.I()` /
`record_indicator` **by default** (`include_indicators=True`): `pane=0` overlays the
main candlestick pane, `pane=k>=1` lands in the `k+1`-th LWC pane below the volume
pane; `reference_lines` become dashed price lines; indicators without a `color`
cycle through a per-theme palette. All seven `render_type` values are mapped -
`line`/`area`/`bar`/`column`/`histogram`/`scatter` get their own series, `signal`
is rendered as markers on the candles instead of a separate series.

```python
result.viz.review(market_data, filename="review.html")            # with indicators
result.viz.review(market_data, filename="review.html", include_indicators=False)
```

The adapter underneath is `akquant.lwc.to_lwc_indicator_series()` - the mirror of
`akquant.chart.to_d3kline_options()`: one indicator contract, two frontends. Reuse
it directly when building your own LWC page.

### Optional Indicator Section In Reports

If you want the indicator preview embedded into the built-in HTML report instead of a separate figure, enable it explicitly:

```python
result.viz.report(
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
- expose both `timestamp` (nanoseconds) and `timestamp_ms` (milliseconds)

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

`timestamp_ms` is present on both `point` and `snapshot` messages, and matches the
value the same `record_indicator` call produces through `indicator_df()` and
`export_indicators()` — all three exits agree, so the frontend never has to convert
nanoseconds itself. When an event comes from an older version or a third-party
`IndicatorSink` and its payload only carries the nanosecond `timestamp`, the bridge
derives the millisecond value.

!!! tip "Negotiate optional fields via `schema_version`"
    Every message envelope carries `schema_version` (`MAJOR.MINOR`, exposed as
    `akquant.STREAM_SCHEMA_VERSION`). Backward-compatible additions bump MINOR, so a
    frontend can decide whether a field exists instead of probing for it.
    `timestamp_ms` ships on the indicator stream from `1.2` onward; when reading
    older messages, fall back to converting the nanosecond `timestamp` yourself.

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

### Live increments to an LWC chart

If your frontend is lightweight-charts you do not need to unpack the fields of
`to_indicator_message()` yourself: `akquant.lwc.to_lwc_update()` turns a point
message into an increment that `series.update()` consumes as-is
(`{indicator_key, pane, series_type, color, confirmed, point:{time, value}}`).
`confirmed` is passed through - once intrabar provisional points exist they will
share the `time` of the confirmed point, and LWC's `update()` overwrites by `time`,
so the frontend needs no change.

```python
from akquant.lwc import load_lwc_js, to_lwc_update

def on_event(event):
    message = aq.to_indicator_message(event)
    update = to_lwc_update(message) if message else None
    if update is not None:
        push_to_page(update)           # transport is yours: polling / WebSocket / SSE
```

`load_lwc_js()` returns the vendored LWC source for inlining into your own page (no
CDN). The full loop is in `examples/73_lwc_live_indicators.py`:
`run_live(broker="replay")` + a stdlib `http.server` poll endpoint + series created
lazily the first time a key appears. The HTTP/WS transport deliberately stays in the
example layer and out of the core package.


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

A point's `symbol` is guaranteed to be non-empty (see "About `symbol` ownership" above), so consumers can group directly by `symbol` + `indicator_key` without handling an "unowned" case. In a multi-symbol backtest the same indicator name is split into one independent series per symbol — always filter by `symbol` before plotting, otherwise same-named indicators from different instruments are drawn as a single line.

This keeps AKQuant focused on producing stable indicator data instead of coupling the framework to a specific charting library.

### Optional adapter: d3Kline `IndicatorOption`

If your frontend renders with d3Kline, `akquant.chart.to_d3kline_options()` turns
indicator definitions and points into the structure `IndicatorManager.create()`
consumes directly, so you do not hand-write the mapping:

```python
from akquant.chart import to_d3kline_options, to_raw_panes

defs = result.indicator_definitions.to_dict(orient="records")
points_by_key = {
    key: group[["timestamp_ms", "value"]].to_dict(orient="records")
    for key, group in result.indicator_df().groupby("indicator_key")
}
options = to_d3kline_options(defs, points_by_key)  # feed IndicatorManager.create()
panes = to_raw_panes(defs, points_by_key)  # akquant-native shape for other frontends
```

It is an **optional consumer-side adapter**, the same kind of thing as the stream bridge
above; it does not change AKQuant's role as a producer decoupled from any frontend. The
only reason it lives in the core package: both the backtest service and the live service
must emit this exact structure, and the frontend renders both with one code path. Two
copies would drift. This is the single source of truth.

Deliberate rules baked in (understand them before changing):

- **All indicators sharing a pane are packed into one option**, each as a series in
  `dataList`. d3Kline's `IndicatorManager` caps sub-indicators at `MAX_SUB_INDICATORS = 3`
  and **silently drops** the excess; packing by pane means "5 sub indicators across 2 panes"
  costs 2 slots, not 5.
- **The main pane (0) carries no `style.height`** so it never squeezes the candlestick
  area; sub panes default to 100.
- **When an indicator has no color the whole `style` key is omitted**, never
  `{"color": null}`. The frontend merges as `{color: default, ...old, ...s.style}`, so
  `null` would override the default and render the series invisible.
- The seven `render_type` values are downgraded to d3Kline's `line | bar`; a downgraded
  series also carries the original `render_type`, so the frontend can upgrade losslessly
  once it supports area/scatter.
- Series `name` is the `indicator_key` (the frontend merges by name; it must be unique
  within a pane); the human-readable name goes in `label`.

Input is plain `Mapping` (dict), not a specific model. Missing fields take contract
defaults (`pane=0`, `render_type="line"`), and `pane` may arrive as a string.

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
  - Not so. `Indicator(name, fn)` covers the precompute case, and any object with an
    `update()` works for the incremental case.
- Pitfall 2: one incremental instance can be shared safely across symbols
  - It cannot. Passing an instance and using it across symbols raises; production
    multi-symbol strategies use `factory=`.
- Pitfall 3: `warmup_bars=20` double-consumes the first active bar
  - It does not. Warmup only uses history before the active start boundary.
- Pitfall 4: custom indicators automatically work with warm start
  - Not guaranteed. Verify the object is picklable, and that `factory` is not a lambda.
- Pitfall 5: strategy-local indicators and `akquant.talib` extensions are the same thing
  - They solve different problems at different layers.
- Pitfall 6: declaring an indicator draws it automatically
  - It does not. Pass a plot argument or `plot=True`, matching Pine where `ta.*` computes
    and `plot()` draws.

## Recommendation Matrix

| Goal | Recommended approach |
| :--- | :--- |
| Validate an idea quickly | `self.I(Indicator(name, fn))` |
| Single-symbol bar-by-bar state | `self.I(aq.EMA(20))` |
| Multi-symbol production strategy | `self.I(factory=...)` |
| Need valid values on the first active bar | `self.I(..., warmup_bars=N)` |
| Need it drawn on the chart | add `pane=` / `color=`, or `plot=True` |
| Detect a golden/death cross | serial lookback via `ind[0]` / `ind[1]` |
| Need resumable state | keep indicator state serializable; use `partial`, not a lambda, for `factory` |
| Need maximum performance | start from the 100+ built-in Rust indicator classes, then consider writing Rust |

## Further Reading

- [Strategy Guide](./strategy.md)
- [Warm Start Guide](../advanced/warm_start.md)
- [AKQuant Indicator Reference](./rust_indicator_reference.md)
- [Indicator Playbook](./talib_indicator_playbook.md)
- [Runnable example: 60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py)
- [Runnable example: 72_declarative_indicators.py](https://github.com/akfamily/akquant/blob/main/examples/72_declarative_indicators.py)
