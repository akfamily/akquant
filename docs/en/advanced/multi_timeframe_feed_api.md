# Multi-Timeframe Strategies: Three Paths

AKQuant currently offers three independent multi-timeframe paths, each covering a
different scenario and none replacing the others:

1. **`subscribe_bars` (recommended)**: engine-native — Rust aggregates windows
   directly from base bars; the same code works for both backtest and live.
2. **`BarGenerator`**: an in-strategy Python runtime helper you feed bar-by-bar.
3. **`feed.resample` / `feed.replay`**: offline data orchestration — the
   multi-frequency feeds are prepared before they are ever handed to
   `run_backtest`.

## Which One to Use

| Path | Fits | Backtest | Live | Feeds `get_history`/indicators | Daily label convention |
| --- | --- | --- | --- | --- | --- |
| `subscribe_bars` (recommended) | Declarative in-strategy multi-timeframe subscription | Yes | Yes | Yes, via `get_history(freq=)` / `self.I(freq=)` | Last base bar of the trading day |
| `BarGenerator` | In-strategy manual aggregation, no engine configuration needed | Yes | Yes | No — only the bar delivered to the callback is available | Next midnight (pandas `resample` convention) |
| `feed.resample` / `feed.replay` | Offline data orchestration before `run_backtest` | Backtest only | No | Requires a synthetic symbol; timestamps are user-defined | User-defined |

## `subscribe_bars` Usage

Declare the subscription inside the strategy's `__init__` (subscriptions are
only accepted there; the table is frozen once the engine starts):

```python
from akquant import Bar, Strategy


class FiveMinuteTrend(Strategy):
    """Print the last 3 closes on every 5-minute window close."""

    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars(
            "5min", session_windows=[("09:30", "11:30"), ("13:00", "15:00")]
        )
        self.count = 0

    def on_start(self) -> None:
        self.set_history_depth(50)
        print(f"base freq self.freq = {self.freq!r}")

    def on_window_bar(self, bar: Bar) -> None:
        self.count += 1
        closes = self.get_history(3, bar.symbol, "close")  # auto-resolves to 5min
        print(
            f"[5min] {self.format_time(bar.timestamp)} "
            f"C={bar.close:.2f} last3={closes.round(2).tolist()}"
        )

    def on_stop(self) -> None:
        print(f"received {self.count} 5-minute window bars")
```

A full runnable example is available at
[examples/71_native_multi_timeframe_live.py](https://github.com/akfamily/akquant/blob/main/examples/71_native_multi_timeframe_live.py)
(live/replay) and
[examples/14_multi_frequency.py](https://github.com/akfamily/akquant/blob/main/examples/14_multi_frequency.py)
(backtest).

Key API:

- `subscribe_bars(freq, callback=None, symbols=None, *, session_windows=None)`:
  declares a subscription. `freq` accepts integer minutes (`"Nmin"`), integer
  hours (`"Nh"`), or daily (`"1d"`). `symbols=None` means all symbols.
- `callback`: when omitted, closed window bars are routed to the strategy's
  `on_window_bar(bar)`; passing a specific function (as in example 14's
  `on_daily`) routes that subscription there instead and `on_window_bar` is
  not triggered for it.
- `on_window_bar(bar)`: the default window callback; `bar.freq` carries the
  window's period label (e.g. `"5min"`, `"1d"`).
- `current_window(symbol, freq)`: peeks at the in-progress, not-yet-closed
  window snapshot for a symbol (callable only from inside a market data
  callback); it never fires a callback.
- `get_history(count, symbol, field, freq=)` / `get_history_multi(...)` /
  `self.I(..., freq=)`: pass an already-subscribed
  window period label as `freq` to read that period's history; inside a
  window callback `freq` can be omitted (it resolves to the current callback's
  period automatically).

## Close Timing: Immediate vs. Delayed

When a window closes depends on whether the **base data frequency**
(`self.freq`) is known:

- **Base frequency known** (backtest passes `run_backtest(freq=...)`, or a
  live/replay market gateway declares `metadata["freq"]`, see example 71): the
  window closes and dispatches in the **same step**, right after the `on_bar`
  call of the base bar that closes it (zero delay).
- **Base frequency unknown** (e.g. a backtest fed a plain bar `DataFrame`
  without `freq=`, or without `list[Tick]` input): the engine cannot tell
  whether the next base bar has already crossed into a new window, so the
  window only closes once the **next base bar that falls into a new window**
  arrives (one bar late), and a one-time `WARNING` is logged at configuration
  time. Example 14 is exactly this case: the daily window only closes once the
  next day's first 09:31 minute bar arrives. `session_windows` only splits
  trading-session boundaries correctly — it does not make a window close
  immediately; immediate closing depends solely on whether the base frequency
  is known.

## Limits

- `freq` only accepts integer minutes (`"Nmin"`), integer hours (`"Nh"`), and
  daily (`"1d"`) — no seconds, weekly, or monthly periods; use
  `feed.resample` for those.
- The window period must be strictly greater than the base data frequency, or
  `subscribe_bars` raises `ValueError` at engine configuration time.
- Multiple subscriptions of the same period with overlapping symbol scopes
  must use identical `session_windows`, or a `ValueError` is raised (the same
  period + same `session_windows` with one all-symbols generic subscription
  plus one symbol-specific subscription is fine — both callbacks fire).
- `extra` fields are not aggregated into window bars.
- Window bars are **never** used by the matching engine for fills — they are
  informational only; the strategy still needs to place orders from
  `on_bar` (or other base market data callbacks).
- The CTP gateway's synthesized bars are stamped at the interval's start by
  default; only enabling both `emit_ticks` and `emit_bars` yields an
  end-of-interval stamp. Without both enabled, multi-timeframe aggregation
  will be off by one bar due to the timestamp convention mismatch — enable
  both `emit_ticks` and `emit_bars` when doing multi-timeframe with CTP.
- At session end (backtest completion, or live/replay session termination),
  the engine flushes any still-open tail window and fires one callback for
  it; orders placed there have no further chance to fill in a backtest. If a
  `save_checkpoint()` taken after that flush is later resumed via
  `run_from_checkpoint()`, the tail interval re-forms as a new window bar with
  the same label (the history series will show two bars with the same label —
  this is known, deliberate behavior, not a bug).

## Migrating Off the Synthetic-Symbol Pattern

Before engine-native multi-timeframe shipped,
[examples/14_multi_frequency.py](https://github.com/akfamily/akquant/blob/main/examples/14_multi_frequency.py)
stitched together a daily series with a "synthetic symbol": it manually
`resample`d the same minute data into daily bars, gave them a fake code like
`000001.SZ_1D`, and had the strategy branch on `bar.symbol` to separate the
two series — plus a manual +15 hour timestamp shift to avoid colliding with
the minute data's timestamps. Now:

```python
self.subscribe_bars(
    "1d",
    callback=self.on_daily,
    session_windows=[("09:30", "11:30"), ("13:00", "15:00")],
)
self.daily_sma = self.I(
    aq.SMA(self.ma_window), name="daily_sma", source="close", freq="1d"
)
```

No more synthetic symbol, no more manual resample/timestamp shifting — the
daily window is aggregated directly from base bars by the Rust engine and
dispatched through the `on_daily` callback. See example 14's module docstring
and git history for the full before/after comparison.

## Offline Data Orchestration: `feed.resample` / `feed.replay`

`feed.resample`/`feed.replay` is an **offline** approach: multi-frequency
feeds are constructed ahead of time at the backtest data layer and then
passed to `run_backtest(data=...)` — essentially multi-feed orchestration
specific to backtesting. **Live trading is not supported.** It fits scenarios
where multi-frequency data must be fully prepared before it reaches the
engine.

### API

```python
resampled = feed.resample(
    freq="15min",
    agg={"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"},
    label="right",
    closed="right",
)

replayed = feed.replay(
    freq="1h",
    align="session",
    day_mode="trading",
    emit_partial=False,
    session_windows=[("09:30", "11:30"), ("13:00", "15:00")],
)
```

### Current Minimal Delivery

- `BasePandasFeedAdapter` supports:
  - `resample(freq, agg=None, label="right", closed="right", emit_partial=True)`
  - `replay(freq, align="session", emit_partial=False, agg=None, label="right", closed="right")`
- Both return adapter objects that can be passed directly to `run_backtest(data=...)`.
- `replay(align="session")` supports per-trading-day partitioning, and drops each partition tail when `emit_partial=False`.
- `session_windows` can further split intraday sessions (for example, before/after lunch break) to avoid cross-session aggregation.
- `align` currently supports:
  - `session`: partition by trading day, with optional `session_windows`.
  - `day`: partition by day without `session_windows`, with configurable `day_mode`.
  - `global`: aggregate on the full timeline without day/session partitioning.
- `day_mode` currently supports:
  - `trading`: partition by local trading day under request timezone.
  - `calendar`: partition by UTC calendar day.

### Semantics

#### resample

- Input: the raw `Bar`/`Tick` stream.
- Output: a bar stream aggregated at the target frequency.
- Default aggregation:
  - `open=first`
  - `high=max`
  - `low=min`
  - `close=last`
  - `volume=sum`
- Boundary: defaults to `label=right, closed=right`.

#### replay

- Input: high-frequency data.
- Output: an event stream replayed at the low-frequency clock.
- `align=session`: aligned to trading-session boundaries.
- `emit_partial=False`: unfinished windows are not emitted.

### Event Alignment Strategy

- Unified timezone: internally UTC, converted to local timezone for display.
- Session-first: day boundaries and lunch breaks follow the market session.
- Gap handling:
  - Price columns follow "no fill for no trade".
  - Volume defaults to 0.

### Consistency Checks

- Compared against pandas `resample` with matching parameters.
- Alignment tolerance is limited to floating-point precision.
- Backtest and live streams share the same aggregation implementation.

## Runtime Aggregation (BarGenerator)

`BarGenerator` is a **runtime** approach: the aggregation logic lives inside
the strategy and is fed bar-by-bar as a stream, with no offline data
orchestration involved. Backtest and live trading call the exact same
`update_bar` — strategy code doesn't need to distinguish which mode it's
running under. Both follow the same clock-alignment semantics under matching
parameters (`label="right"`, `closed="right"`, consistent with pandas
`resample(label="right", closed="right")`), so identical parameters produce
identical results.

The key difference from `subscribe_bars`: bars aggregated by `BarGenerator`
**never** enter the `get_history`/incremental-indicator system — only the bar
delivered to the callback is available. The daily timestamp convention also
differs (`BarGenerator` uses pandas' next-midnight convention, while
`subscribe_bars` uses the last base bar of the trading day).

Minimal usage:

```python
from akquant import BarGenerator, Strategy


class MyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        # Aggregate 1-minute bars into 5-minute bars, callback on close
        self.bg = BarGenerator(self.on_5m, 5, "minute")

    def on_bar(self, bar):
        # Same call for both backtest and live
        self.bg.update_bar(bar)

    def on_5m(self, bar):
        # Received the aggregated 5-minute bar; write your signal logic here
        ...

    def on_stop(self):
        # Force-close the trailing partial window on shutdown, otherwise
        # the last incomplete window is dropped
        self.bg.flush()
```

- `window`/`interval` combine to form the target period (e.g.
  `BarGenerator(cb, 15, "minute")`, `BarGenerator(cb, 1, "hour")`,
  `BarGenerator(cb, 1, "day")`).
- `session_windows` segments aggregation by trading session, avoiding dirty
  bars that splice across boundaries like the lunch break, e.g.
  `session_windows=[("09:30", "11:30"), ("13:00", "15:00")]`; omit it for pure
  clock alignment (no special handling of session boundaries).
- `timezone` specifies the market timezone used for clock/session alignment
  (e.g. `"Asia/Shanghai"`).
- `current(symbol)` peeks at the in-progress, not-yet-closed window snapshot
  for a symbol without triggering a callback — useful for intraday display.
- Multiple symbols aggregate independently of each other.

A full runnable example is available at
[examples/65_bar_generator.py](https://github.com/akfamily/akquant/blob/main/examples/65_bar_generator.py).
