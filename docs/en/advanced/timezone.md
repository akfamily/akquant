# Timezone Handling Guide

AKQuant is a backtesting framework supporting multiple markets and instruments. To ensure precise alignment of time series, the framework's core uses **UTC timestamps (nanoseconds)** uniformly. However, when dealing with specific markets (such as China A-shares), correct timezone settings are crucial for data alignment, trading-day splitting and log readability.

This document guides you on how to correctly handle data timezones, backtest configurations, and time conversions in strategies.

## 1. Core Principles

1.  **Internal UTC**: All interactions between the engine's core (`Rust`) and the Python layer are based on UTC timestamps.
2.  **Input Normalization**: Input DataFrames are converted to UTC storage; tz-naive times are localized with the `timezone` parameter.
3.  **Display-Layer Conversion**: Structured time fields stay in UTC by default; only strategy helpers or your own display code convert them into the configured `timezone`.
4.  **The timezone drives trading sessions**: `timezone` also determines the trading-day split, the end-of-day settlement time and the default session tables (see Section 3).

## 2. Data Preparation and Timezones

When preparing backtest data (DataFrame), you have two options:

### Method A: Use Naive Datetime

If your data has no timezone information (tz-naive), AKQuant localizes it with the timezone given by `run_backtest(timezone=...)` (`Asia/Shanghai` when omitted).

*   **A-share data**: pass it directly, no extra step needed.
*   **US stocks, crypto and other timezones**: pass the matching `timezone` (e.g. `timezone="US/Eastern"`), or convert the data to aware datetimes first.

**Example: Constructing A-share 1-minute bar data (Beijing Time)**

```python
import pandas as pd

# Naive time series, interpreted with `timezone` (Beijing Time by default)
# Example: 2024-01-02 09:31:00 (a Tuesday, a trading day)
rng = pd.date_range(
    start="2024-01-02 09:31",
    end="2024-01-02 11:30",
    freq="1min"
)

# Create DataFrame
# Index must be datetime type
df = pd.DataFrame({
    "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000
}, index=rng)

# df.index.tz is None; it is interpreted with `timezone` (default Asia/Shanghai)
```

A minute bar's timestamp is its **interval end**: the 09:31 bar covers (09:30, 09:31].

### Method B: Use Aware Datetime - **Recommended (non-A-share)**

For data not in Beijing Time (e.g. US stocks, FX), or simply to avoid ambiguity, specify the timezone explicitly. AKQuant recognizes the timezone information and converts it to UTC correctly.

```python
# US stock example (US/Eastern)
rng = pd.date_range(
    start="2024-01-02 09:31",
    periods=100,
    freq="1min",
    tz="US/Eastern"  # Explicitly specify timezone
)

# Or localize an existing naive index
# df.index = df.index.tz_localize("US/Eastern")
```

### Important: Timestamp Setting for Daily Data

Daily bars can be written in two ways:

*   **Date only (local midnight)**: the recommended form. The engine rewrites it to the trading day's close (15:00 for A-shares), the bar interval covers the whole trading day, and no `freq` is needed. If the dates include non-trading days, a warning is logged once and those rows are assigned to the next trading day.
*   **Fixed non-midnight time (e.g. 15:00)**: pass `freq="1d"`. Without it the engine does not know these are daily bars and classifies the session by "1ns before the timestamp" — 15:00 becomes 14:59:59.999…, so the bar is classified as the closing call auction (`ctx.session` and `is_market_open()` change accordingly) and the engine logs a warning once. Under the default fill policy (fill on the next bar) orders fill at the same time as with `freq`.

```python
import pandas as pd

# Form 1: date only, rewritten to the trading day's 15:00 close
daily_index = pd.bdate_range("2024-01-02", periods=5)

# Form 2: daily bars stamped at 15:00, use together with run_backtest(..., freq="1d")
daily_index_1500 = daily_index + pd.Timedelta(hours=15)
```

When daily and minute bars are mixed in one backtest, the daily label is the close time, never earlier than the day's last minute bar, so the logical order is naturally correct.

## 3. Backtest Configuration

Specify the backtest timezone via the `timezone` parameter of `run_backtest` (default `Asia/Shanghai`).

This parameter is used for:

1. Localizing tz-naive input data and naive time bounds such as `start_time` / `end_time`.
2. Splitting trading days, computing the end-of-day settlement time, and choosing the default session tables (see the table below).
3. Strategy local-time helpers such as `self.now`, `self.to_local_time(...)` and `self.format_time(...)`, and the local firing time of timers.
4. Daily performance aggregation such as `daily_returns`, volatility, Sharpe / Sortino / VaR / CVaR, which is computed per **trading day** (night sessions belong to the next trading day).

```python
from akquant.backtest import run_backtest

results = run_backtest(
    data=data_feed,
    strategy=MyStrategy,
    timezone="Asia/Shanghai",
    # ...
)
```

### Timezone and Default Session Tables

| `timezone` | Default session table without `sessions` | Trading days without `trading_calendar` |
| :--- | :--- | :--- |
| China time zones (`Asia/Shanghai`, etc.) | By asset type: stocks/funds `CN_STOCK`, futures `CN_FUTURES_DAY`, options `CN_OPTION`, other assets `ALWAYS_OPEN` | Inferred from weekdays: Monday to Friday are all trading days, holidays included (a warning is logged once) |
| Other time zones (e.g. `US/Eastern`, `UTC`) | `ALWAYS_OPEN` (continuous trading all day) | Every day is a trading day |

When a non-China market needs real trading sessions, pass a custom session table with `sessions=` (format in the [API Reference](../reference/api.md#akquantrun_backtest)).

### Trading Calendar `trading_calendar`

Both `run_backtest` and `run_live` accept `trading_calendar=` (a list of trading days as `date` / string / `Timestamp`). When omitted, China markets infer trading days from weekdays, so holidays are treated as trading days and night sessions and end-of-day settlement around holidays are wrong; the engine logs a warning once per run. Long-running live sessions must pass it; since live_server 0.12.0, klinedata sessions fetch the SSE/SZSE trading calendar from the market-data service automatically.

### Live Timezone

`run_live(timezone=...)` means the same as `run_backtest(timezone=...)`, defaults to `Asia/Shanghai`, and **must match the exchange timezone**. For China markets keep the default rather than the local machine's or server's timezone; otherwise trading days, end-of-day settlement and Day-order expiry are split by the wrong timezone and no longer match the exchange.

### Structured Logging And ISO Fields

After the time-semantics refactor, AKQuant uses:

- `event_time`: UTC nanosecond timestamp
- `event_time_iso`: UTC ISO 8601 string

This means:

- structured logs and JSON output no longer default to local-time strings
- `bar.timestamp_iso`, `tick.timestamp_iso`, `trade.timestamp_iso`, `order.created_at_iso`, and `order.updated_at_iso` are all UTC ISO 8601 values
- if you want human-readable strategy-local output, format it explicitly with `self.format_time(...)`

If you print `timestamp_iso` directly, expect UTC output such as `2025-01-02T07:00:00Z`, not exchange-local or strategy-local wall-clock time.

## 4. Time Processing in Strategies

In the strategy's `on_bar` callback, `bar.timestamp` is an integer (int64) representing the **UTC nanosecond timestamp** of the bar's interval end. If you need to print the current time in logs or make logical decisions based on time (e.g., trade only in the afternoon), you need to convert it to local time. To check whether the market is in continuous trading or which trading day a moment belongs to, prefer `self.is_market_open()`, `self.session_of(symbol)` and `self.trading_day` (see the [Strategy Guide](../guide/strategy.md#trading-session)).

### Method 1: Use Strategy Helper Methods (Recommended)

AKQuant provides convenient helper methods in the `Strategy` base class, automatically converting using the configured timezone.

```python
class MyStrategy(Strategy):
    def on_bar(self, bar: Bar):
        # 1. Get local time object (pd.Timestamp)
        ts = self.to_local_time(bar.timestamp)

        # 2. Directly get formatted string (default "%Y-%m-%d %H:%M:%S")
        ts_str = self.format_time(bar.timestamp)
        self.log(f"Current time: {ts_str}")

        # 3. Current strategy time (now)
        # Returns the local time of the bar/tick being processed
        if self.now:
            print(f"Now: {self.now}")
```

### Method 2: Manual Conversion (Under the Hood)

For lower-level control, convert manually:

```python
import pandas as pd
from akquant.strategy import Strategy
from akquant.akquant import Bar

class MyStrategy(Strategy):
    def on_bar(self, bar: Bar):
        # 1. Convert the UTC nanosecond timestamp to a UTC datetime
        ts_utc = pd.to_datetime(bar.timestamp, unit="ns", utc=True)

        # 2. Convert to Beijing Time (or another target timezone)
        ts_bj = ts_utc.tz_convert("Asia/Shanghai")

        # ...
```

## 5. FAQ

**Q: Why do I see 01:31 instead of 09:31 in `event_time_iso` or `bar.timestamp_iso`?**
A: These fields now always carry UTC time. 09:31 Beijing Time is 01:31 UTC. To display strategy-local time, use `self.format_time(bar.timestamp)` or `self.to_local_time(bar.timestamp)`.

**Q: I am backtesting US stocks; why are the data times wrong?**
A: Tz-naive data is localized with `timezone`. Pass `timezone="US/Eastern"`, or localize the index with `.tz_localize("US/Eastern")` before passing the data in. Note that non-China time zones default to continuous trading all day (`ALWAYS_OPEN`); pass `sessions=` when you need real trading sessions.

**Q: What does `AttributeError: 'float' object has no attribute 'quantity'` mean?**
A: This usually happens when accessing positions. `self.ctx.get_position(symbol)` returns the position quantity (float), not an object; use the return value directly as the quantity. For the average entry price, use `self.position.entry_price` or `self.ctx.get_position_entry_price(symbol)`.

**Q: How are daily and minute bars aligned in a mixed-frequency backtest?**
A: AKQuant is event-driven and processes bars in timestamp order; a timestamp is always the bar's interval end.

*   09:31 -> process the (09:30, 09:31] minute bar
*   ...
*   15:00 -> process the (14:59, 15:00] minute bar
*   15:00 -> process the daily bar (the daily label is the trading day's close)

Date-only daily bars are rewritten to the close automatically; daily bars stamped at a fixed time need `freq="1d"`.
