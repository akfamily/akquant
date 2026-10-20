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
| Compute a full series from a `DataFrame` with pandas | `PrecomputedIndicator(name, fn)` | `self.I(PrecomputedIndicator(...), ...)` |
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
        # One declaration = auto update + lookback; add plot=True for auto plotting
        self.ema_fast = self.I(aq.EMA(10), pane=0, color="#e91e63", label="EMA10", plot=True)
        self.ema_slow = self.I(aq.EMA(30), pane=0, color="#3f51b5", plot=True)
        self.rsi = self.I(aq.RSI(14), pane=1, plot=True)

    def on_bar(self, bar: Bar) -> None:
        # [0] is the current bar, [1] the previous one — same as Pine's sma[0] / sma[1]
        if self.ema_fast[1] is None or self.ema_slow[1] is None:
            return
        if self.ema_fast[1] <= self.ema_slow[1] and self.ema_fast[0] > self.ema_slow[0]:
            self.buy(bar.symbol, 100)
        # No record_indicator needed — plot=True makes reporting automatic
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
from akquant import PrecomputedIndicator, Strategy


class MyMomentum(PrecomputedIndicator):
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
                   outputs=("dif", "dea", "hist"), plot=True)

# Reading:
self.macd[0]        # the whole tuple (dif, dea, hist)
self.macd.dif[0]    # a single component, which also supports [1] lookback
```

If the number of `outputs` does not match the indicator's actual component count, you get
an error rather than a silent misalignment.

## Path 2: Precomputed Indicators

When the indicator is a better fit for a one-shot computation over the full `DataFrame`,
pass a `PrecomputedIndicator(name, fn)` instance and `self.I()` routes it to the vectorized
precompute path.

### Minimal example

```python
from akquant import PrecomputedIndicator, Strategy


class PrecomputeMomentumStrategy(Strategy):
    def on_start(self):
        self.mom10 = self.I(
            PrecomputedIndicator("mom10", lambda df: df["close"] - df["close"].shift(10)),
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
- **Multi-symbol out of the box**: `PrecomputedIndicator` caches its full series per symbol, so no `factory=` is needed.

## Automatic Plotting

For an indicator that is drawn, the framework reports a plot point after every update —
**no hand-written `record_indicator` inside `on_bar`**. Whether it is drawn depends on two
things only: **which kind of script `self.I` is called in**, and whether `plot=` is written
explicitly:

- Inside an `aq.Strategy`, **nothing is drawn by default**, matching Pine (`ta.sma()` only
  computes; `plot()` draws). This is not merely a style choice: the live indicator sink emits
  one event per point, and turning every indicator on by default across many symbols would
  flood the frontend link.
- Inside an indicator script `aq.Indicator`, **everything is drawn by default** - drawing is
  what an indicator script is for (see "Indicator scripts" below).
- `plot=True` / `plot=False` overrides the default.
- `pane` / `color` / `label` / `render_type` / `reference_lines` / `scale_group` only decide
  *how* to draw and **no longer imply `plot=True`**. A strategy that wrote only `pane=` to get
  a chart must add `plot=True`.

| Where | Call | Drawn | pane |
| :--- | :--- | :--- | :--- |
| `aq.Strategy` | `self.I(x)` | no | 0 |
| `aq.Strategy` | `self.I(x, color="red")` | no | 0 |
| `aq.Strategy` | `self.I(x, pane=1)` | no | 1 |
| `aq.Strategy` | `self.I(x, plot=True)` | yes | 0 |
| `aq.Indicator` (`overlay=False`, default) | `self.I(x)` | yes | 1 |
| `aq.Indicator` (`overlay=False`, default) | `self.I(x, pane=0)` | yes | 0 |
| `aq.Indicator` (`overlay=False`, default) | `self.I(x, plot=False)` | no | 1 |
| `aq.Indicator` (`overlay=True`) | `self.I(x)` | yes | 0 |
| `aq.Indicator` (`overlay=True`) | `self.I(x, pane=2)` | yes | 2 |

Points that are not ready yet (value `None`) are not reported — the counterpart of Pine's
`na` during warmup.

```python
# Compute only, do not draw
self.atr = self.I(aq.ATR(14), input_mode="hlc")

# Compute and draw (a strategy must write plot=True explicitly)
self.rsi = self.I(aq.RSI(14), pane=1, plot=True, reference_lines=[
    {"value": 70, "label": "Overbought"},
    {"value": 30, "label": "Oversold"},
])

# pane= alone does not draw - plot arguments no longer imply plot=True
self.sma = self.I(aq.SMA(20), pane=0)
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
        self.sma5 = self.I(aq.SMA(20), freq="5min", intrabar=True, pane=0, plot=True)

    def on_bar(self, bar):                 # every 1-minute bar
        v0 = self.sma5[0]                  # provisional value on the forming 5-minute window
        v1 = self.sma5[1]                  # last confirmed 5-minute value
        if not self.sma5.confirmed: ...    # [0] is currently provisional
```

Index semantics match Pine exactly: a provisional value occupies `[0]` and the confirmed
series shifts by one. Stream events mark provisional points with `confirmed=false`, and
**their timestamp is the label the window will close with**; the confirmed point arrives
later with the same `time`, so a frontend overwrites by `(owner_strategy_id, indicator_key, symbol, timestamp)`
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

## Indicator scripts (`aq.Indicator`): draw, never trade

TradingView separates `indicator()` scripts from `strategy()` scripts, and a chart can stack
any number of indicator-only scripts. AKQuant's counterpart is `aq.Indicator` (mirroring Pine
v5's `indicator()`): subclass it, declare indicators with `self.I()` in `on_start`, and attach
it to any backtest or live run with `indicators=[...]`.

Script metadata goes on the class line, matching Pine's `indicator(title=..., overlay=...)`:

| Argument | Meaning | When omitted |
| :--- | :--- | :--- |
| `title` | Display name, used for presentation only | the class name |
| `overlay` | `True` draws on the main pane (over the candles), `False` on the script's sub-pane | `False` (sub-pane) |
| `id` | Machine id, i.e. the points' `owner_strategy_id` / slot key | snake_case class name |

```python
import akquant as aq
from akquant import run_backtest


class DualMA(aq.Indicator, title="Dual MA", overlay=True, id="dual_ma"):
    def on_start(self):
        self.fast = self.I(aq.EMA(10), name="ema_fast")    # overlay=True: main pane, pane=0
        self.slow = self.I(aq.EMA(30), name="ema_slow")


class RsiIndicator(aq.Indicator, title="RSI"):            # id defaults to "rsi_indicator"
    def on_start(self):
        self.rsi = self.I(aq.RSI(14), name="rsi")          # overlay=False: sub-pane, pane=1


class MacdIndicator(aq.Indicator, title="MACD", id="macd_view"):
    def on_start(self):
        self.macd = self.I(aq.MACD(12, 26, 9), pane=2, outputs=("dif", "dea", "hist"))


result = run_backtest(strategy=MyStrategy, data=data, symbols=["600000"],
                      strategy_id="main",
                      indicators=[DualMA, RsiIndicator, MacdIndicator])
result.indicator_df(owner="rsi_indicator")   # only this script's points
result.viz.review(data)                      # indicator scripts land on the LWC chart too
```

Things to know:

- **Pane numbers are local to the script.** With `overlay=True`, `self.I` defaults to
  `pane=0`; otherwise to `pane=1`. Use `pane=2`, `pane=3`, ... for more sub-panes. An
  explicit `pane=` always wins.
- **No new pipeline.** Each indicator script is one slot of the multi-strategy topology
  (`strategies_by_slot`); its points carry their own `owner_strategy_id`, window
  subscriptions merge across slots, and `run_live(indicators=...)` has identical semantics.
- **Not trading is a hard guarantee.** Every trading API on `aq.Indicator` (`buy` / `sell` /
  `place_*` / `order_target*` / `rebalance_*` / `cancel_*` ...) raises
  `aq.IndicatorCannotTradeError`. A stray order inside `on_bar` fails fast inside the engine
  loop instead of silently filling. Subclass `aq.Strategy` if you need to trade.
- **Ids must not collide.** A script id equal to `strategy_id` or an existing slot key is a
  `ValueError`; a non-`aq.Indicator` class inside `indicators=` is a `TypeError` - otherwise
  the no-trading guarantee would be void.
- **`aq.Strategy` accepts `title=` / `id=` too**: `class Trend(aq.Strategy, title="Trend",
  id="trend")`. Strategies have no main/sub-pane notion, so `overlay=` raises `TypeError`; a
  class `id` that disagrees with `strategy_id=` passed to `run_backtest/run_live` raises
  `ValueError`. A strategy without `id` keeps the default `strategy_id` `"_default"`.

Runnable example: [74_indicator_scripts.py](https://github.com/akfamily/akquant/blob/main/examples/74_indicator_scripts.py).

## Static inspection: `aq.inspect_strategy`

A platform listing uploaded scripts usually needs to know, before anything runs, whether a
file holds indicators or strategies, main pane or sub-pane, and whether anything is drawn at
all - to group the list and grey out actions that do not apply - without executing user code.
`aq.inspect_strategy(path_or_source)` parses the source with `ast` only (**no import, no
execution**) and returns an `aq.InspectResult` (`to_dict()` is JSON-serializable):

```python
report = aq.inspect_strategy("strategies/dual_ma.py")   # a source string works too
report.to_dict()
```

```python
{
  "classes": [
    {
      "name": "DualMA",
      "kind": "indicator",          # "indicator" | "strategy" | "unknown"
      "title": "Dual MA",
      "id": "dual_ma",
      "overlay": True,              # None when kind != "indicator"
      "draws_indicators": True,     # at least one plot=True, or a dynamic plot
      "indicators": [
        {"name": "ema_fast", "pane": 0, "plot": True,
         "outputs": None, "label": "EMA10", "dynamic": []},
      ],
      "has_dynamic": False
    }
  ],
  "error": None                     # "SyntaxError: ... (line N)" on a syntax error; classes is empty
}
```

Conventions:

- **Plaintext only.** A path is read as UTF-8 (a BOM is fine); encrypted strategy files must
  be decrypted by the caller first. A `.py` path that does not exist raises
  `FileNotFoundError`; every other problem is reported through `error` / `dynamic`, never raised.
- **`dynamic` means "cannot be known statically".** When a value comes from parameters (e.g.
  `pane=self.params["pane"]`), the call sits in a loop or comprehension, uses `factory=` or
  `**kwargs`, lives in a conditional branch or helper method, or calls something unrecognized,
  the field is `None` and its name goes into `dynamic` (loops and `factory=` add `"count"`: the
  number of entries is unknown), and the class's `has_dynamic` is `True`. Show "known after a
  run" rather than guessing.
- **`kind="unknown"`**: the base class comes from another file and cannot be traced to
  `Indicator` / `Strategy` within this one.
- **Both `akquant` and `qmfquant` imports are recognized**: `import akquant as aq`,
  `from akquant import Indicator` (including `as` aliases), and the same forms for the
  platform's repackaged `qmfquant`.
- `pane` / `plot` resolve exactly as at runtime (see the rule table under "Automatic
  Plotting"); both sides share one function.

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
it directly when building your own LWC page. Each series carries `owner_strategy_id`
and `series_key` (see "Indicator identity and series keys" below); `indicator_key`
keeps its original value.

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
- `series_keys` (parallel to `indicator_keys`)
- `panes`
- `render_types`
- `value_by_key` (keyed by `series_key`)
- `items_by_key` (keyed by `series_key`)
- `warmup_count`
- `has_warmup`

A `point` message's `indicator` carries `series_key`, and every `snapshot` item carries
`owner_strategy_id` and `series_key`. `series_key` follows the same rule as the
historical exits (see "Indicator identity and series keys" below): the plain
`indicator_key` for a single owner, `"{owner}:{key}"` when the main strategy and an
indicator script share a key. Frontends can key their lines by `series_key` directly
instead of building a mapping from the definition table first.

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
    `phase` ships from `1.4` onward (on points, snapshots and every snapshot
    item): `"preload"` marks points replayed from `run_live(preload_history=)`,
    everything else is `"run"`; treat a missing field as `"run"`.
    `series_key` (on points and every snapshot item), the item-level
    `owner_strategy_id` and the snapshot's `series_keys` ship from `1.5` onward,
    and `items_by_key` / `value_by_key` are keyed by `series_key` since then. For
    `1.4` messages the bridge falls back to `indicator_key` for `series_key` and to
    the snapshot owner for the item owner, so single-owner keys are unchanged.

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

### Live history preload: one continuous series

A chart usually needs the indicators before the session started *and* the live
ones after it. Do not stitch a backtest onto a live run for that: bars can be
missed or duplicated at the seam, and the backtest's indicator instances are not
the live ones, so recursive indicators such as EMA or MACD re-converge and jump.
Use `run_live(preload_history=...)` instead, so one session and one set of
indicator instances consume the history first and then the live feed:

```python
from akquant import run_live

history = fetch_my_history()      # up to the last closed bar; no built-in data source

run_live(
    strategy_cls=MyStrategy,
    instruments=instruments,
    broker="klinedata",
    preload_history=history,      # DataFrame / {symbol: DataFrame} / list[Bar], same as run_backtest(data=)
    on_event=on_event,
)
```

What the preload does:

- Fills the history buffer, so `get_history` works from the first live bar.
  Without an explicit `set_history_depth`, the depth is lifted to the preload
  depth (max rows per symbol); a larger value you set is never lowered.
- Feeds the `subscribe_bars` window aggregator: windows closed during the
  preload land in the window history too.
- Counts toward `warmup_period`: N preloaded bars count as N bars seen.
- Replays every `self.I()` indicator bar by bar **after** `on_start` (window
  `freq=` indicators included), so parameterised indicators declared in
  `on_start` are covered. With indicator scripts or several slots, the replay
  runs once the main strategy's and every slot's `on_start` have all finished
  (main first, then each slot), so same-named indicators carry their final
  `series_key` from the first preload point. Plot points stream as usual, tagged
  `phase="preload"`; afterwards the same instances continue on live bars with
  `phase="run"`.

What it does not do: drive `on_bar`, match orders or touch the account - zero
order risk. Indicators computed by hand in `on_bar` and reported through
`record_indicator` are not replayed; declare them with `self.I()` if they need
the preload.

Boundaries:

- Rows later than session start are dropped with a warning (no look-ahead).
- A first live bar that is not newer than the symbol's preload tail is warned
  about but **still dispatched** (one duplicate is far cheaper than one lost
  bar). Frontends overwrite by `(owner_strategy_id, indicator_key, symbol, timestamp)`.
- A strategy restored from a checkpoint skips the indicator replay (its state
  was restored); the history buffer is still filled.
- A tick-driven session becomes bar/tick dual-stream once bars are preloaded:
  outside `on_bar` / `on_tick` (e.g. `on_timer`) pass `freq='bar'` or
  `freq='tick'` to `get_history`.

The frontend receives one continuous series and can style or filter on
`phase`. The full loop is in `examples/73_lwc_live_indicators.py`: the first 40
bars are preloaded, the last 20 arrive live, and the chart shows one unbroken
line.

### Live increments to an LWC chart

If your frontend is lightweight-charts you do not need to unpack the fields of
`to_indicator_message()` yourself: `akquant.lwc.to_lwc_update()` turns a point
message into an increment that `series.update()` consumes as-is
(`{owner_strategy_id, indicator_key, series_key, pane, series_type, color, confirmed, phase, point:{time, value}}`).
Find the line by `series_key`: it follows the same rule as the `series_key` of the
historical `to_lwc_indicator_series()` series, so same-named keys from the main
strategy and an indicator script no longer merge into one line.
`confirmed` is passed through - once intrabar provisional points exist they will
share the `time` of the confirmed point, and LWC's `update()` overwrites by `time`,
so the frontend needs no change. `phase` is passed through as well (`"preload"` /
`"run"`, see the previous section).

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

#### Indicator identity and series keys

An indicator definition is identified by `(owner_strategy_id, indicator_key)`: the main
strategy (owner `_default`) and an indicator script (owner = its `id`, by default the
snake_case of the class name) may use the same key and each keeps its own `pane`, `color` and so on. `result.indicator_definitions` has
`owner_strategy_id` as its first column and is sorted by `(owner, key)`; points have
always carried `owner_strategy_id`, so consumers should join definitions and points on
`(owner_strategy_id, indicator_key)`.

Chart series keys are prefixed only on conflict: when an `indicator_key` belongs to a
single owner, the series key is just `indicator_key`, so existing charts and frontends are
unchanged; when several owners use it, the series key becomes `"{owner}:{key}"`, e.g.
`_default:ema` and `sub:ema`. The legend label follows the series key when the user gave
no label (display name equal to `indicator_key`) and stays as written otherwise. The LWC,
d3Kline and plotly outputs all take their keys from
`akquant.indicator_identity.series_keys()`; custom frontends should reuse it too.

The live stream (`on_event` point / snapshot messages and `to_lwc_update()`) carries
`series_key` as well, computed by the same function. The stream has to name each
message as it is emitted, before any complete definition table exists, so the
recorder (`IndicatorRecorder` in backtests, `StreamingIndicatorSink` live) keeps a
session-level registry: when it is attached it registers the existing `plot=True`
declarations of every strategy and indicator script (`self.I` in `__init__`), later
`self.I(plot=True)` declarations (in `on_start`) register immediately, and every
`record_indicator` call registers too. So in both backtests and live runs
(including the `preload_history` replay), declarations in `__init__` and
`on_start` are all registered before the first point is emitted, and same-named
indicators carry their final `series_key` from the first point. A custom `IndicatorSink` may implement the
optional method `declare(owner_strategy_id, indicator_key)` to receive these
registrations; without it, `record()` works as before.

Two known differences from the historical exits:

- If a second owner starts using an existing key mid-run (for example a hand-written
  `record_indicator` with the same name in `on_bar`), every owner of that key switches
  to `"{owner}:{key}"` from that moment on; points already emitted keep the
  unprefixed name and are not rewritten, and one warning is logged per key. Declaring
  same-named indicators with `self.I(plot=True)` in `__init__` / `on_start`, or using a
  different `name`, avoids the switch.
- An indicator declared with `plot=True` that never produces a point (for example, it
  never becomes ready) takes part in the live conflict check but is absent from the
  historical definition table, so the live side may prefix a key the historical side
  does not.

Also, a point's `series_key` is computed when it is emitted, a snapshot item's when
the cycle is flushed: if a conflict appears within one snapshot cycle, a point emitted
earlier in that cycle may carry a different `series_key` from the matching snapshot
item; the snapshot wins.

This keeps AKQuant focused on producing stable indicator data instead of coupling the framework to a specific charting library.

### Optional adapter: d3Kline `IndicatorOption`

If your frontend renders with d3Kline, `akquant.chart.to_d3kline_options()` turns
indicator definitions and points into the structure `IndicatorManager.create()`
consumes directly, so you do not hand-write the mapping:

```python
from akquant.chart import group_points_by_identity, to_d3kline_options, to_raw_panes

defs = result.indicator_definitions.to_dict(orient="records")
# (owner_strategy_id, indicator_key) -> list of points
points_by_identity = group_points_by_identity(
    result.indicator_df().to_dict(orient="records")
)
options = to_d3kline_options(defs, points_by_identity)  # feed IndicatorManager.create()
panes = to_raw_panes(defs, points_by_identity)  # akquant-native shape for other frontends
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
- Series `name` is the series key (the frontend merges by name; it must be unique): the
  plain `indicator_key` when only one owner uses it, `"{owner}:{key}"` on a cross-owner
  conflict; the human-readable name goes in `label`. Points are looked up by
  `(owner_strategy_id, indicator_key)`, and each indicator in `to_raw_panes()` also
  carries `owner_strategy_id` and `series_key`.

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

- Pitfall 1: every custom indicator must inherit from some indicator base class
  - Not so. `PrecomputedIndicator(name, fn)` covers the precompute case, and any object with
    an `update()` works for the incremental case. `aq.Indicator` is the base class of
    draw-only indicator scripts, which is a different thing from writing a custom indicator.
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
  - Not inside an `aq.Strategy`: write `plot=True` (`pane=` / `color=` alone is not enough),
    matching Pine where `ta.*` computes and `plot()` draws. Inside an indicator script
    `aq.Indicator` it is drawn by default; `plot=False` turns it off.

## Recommendation Matrix

| Goal | Recommended approach |
| :--- | :--- |
| Validate an idea quickly | `self.I(PrecomputedIndicator(name, fn))` |
| Single-symbol bar-by-bar state | `self.I(aq.EMA(20))` |
| Multi-symbol production strategy | `self.I(factory=...)` |
| Need valid values on the first active bar | `self.I(..., warmup_bars=N)` |
| Need it drawn on the chart | `plot=True` in a strategy, or write an `aq.Indicator` script |
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
