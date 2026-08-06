# Data Preparation and Loading Guide

Data is the cornerstone of quantitative backtesting. As a high-performance backtesting framework, AKQuant has specific requirements for data format and quality. This document details how to prepare, clean, and load data to ensure smooth backtesting.

## 1. Data Format Standard

AKQuant's core engine (Rust) and Python interface layer primarily interact via `pandas.DataFrame` or `List[Bar]`. The most recommended way is to use **Pandas DataFrame**.

Besides pandas, `run_backtest(data=...)` also accepts **`polars.DataFrame` / `polars.LazyFrame` / `pyarrow.Table`** as first-class inputs (internally coerced onto the pandas data path at no cost), so you do not need to call `.to_pandas()` yourself:

```python
import polars as pl
from akquant import run_backtest

pldf = pl.read_parquet("000001.parquet")
result = run_backtest(data=pldf, strategy=MyStrategy, symbols="000001.SZ")
```

### 1.1 Required Columns

Your DataFrame **must** contain the following columns (column names are case-insensitive but are converted to lowercase internally):

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `date` / `time` / `datetime` | `datetime64[ns]` | Timestamp index. Must be Pandas datetime type. |
| `open` | `float` | Open price |
| `high` | `float` | High price |
| `low` | `float` | Low price |
| `close` | `float` | Close price |
| `volume` | `float` | Trading volume |
| `symbol` | `str` | Ticker symbol (e.g., "000001", "AAPL") |

**Note:**

1.  **Column Standardization**: It is recommended to rename columns to lowercase English (e.g., `open`, `close`) before passing them in.
2.  **Symbol Column**: Even if backtesting a single stock, you must include the `symbol` column so the engine can identify the asset.

### 1.2 Index

*   The DataFrame index can be a default integer index or a `DatetimeIndex`.
*   If using `DatetimeIndex`, AKQuant automatically treats it as the time column.
*   **Sorting**: Data must be sorted by time in **ascending** order (Old -> New).

---

## 2. Data Loading Examples

### 2.1 Loading from CSV

This is the most common method. Assume you have a `data.csv` file.

```python
import pandas as pd
from akquant import run_backtest

# 1. Read CSV
df = pd.read_csv("data.csv")

# 2. Convert Time Column
# Ensure the time column is datetime type, not string
df['date'] = pd.to_datetime(df['date'])

# 3. Ensure Correct Column Names
# Assume CSV columns are "Date", "Open", ...
df.columns = [c.lower() for c in df.columns]

# 4. Add Symbol Column (if not in CSV)
if 'symbol' not in df.columns:
    df['symbol'] = "DEMO_TICKER"

# 5. Sort
df = df.sort_values('date').reset_index(drop=True)

# 6. Pass to Backtest
# result = run_backtest(data=df, ...)
```

### 2.2 Using AKShare (China A-Shares)

[AKShare](https://github.com/akfamily/akshare) is a powerful open-source financial data interface library.

```python
import akshare as ak
import pandas as pd

# 1. Download Data (Example: Forward Adjusted)
# period="daily"; adjust="qfq" (forward adjusted)
df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20200101", end_date="20231231", adjust="qfq")

# 2. Rename Columns (AKShare returns Chinese columns)
df = df.rename(columns={
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume"
})

# 3. Type Conversion
df['date'] = pd.to_datetime(df['date'])
df['symbol'] = "000001"

# 4. Filter Columns
df = df[["date", "open", "high", "low", "close", "volume", "symbol"]]
```

### 2.3 Using yfinance (US Stocks)

```python
import yfinance as yf

# 1. Download Data
df = yf.download("AAPL", start="2020-01-01", end="2023-12-31")

# yfinance returns MultiIndex columns (if multiple tickers) or Capitalized columns
# Simplified here for single stock
df.columns = [c.lower() for c in df.columns]
df.reset_index(inplace=True) # Turn Date index into column
df = df.rename(columns={"date": "date"}) # Ensure it is date

df['symbol'] = "AAPL"
```

### 2.4 Using DataFeedAdapter with Multi-Timeframe Aggregation

If you want a single entry that combines data loading and timeframe transformation, use `DataFeedAdapter` directly:

```python
import akquant as aq

base = aq.CSVFeedAdapter(path_template="/data/{symbol}.csv")

feed_15m = base.resample(freq="15min", emit_partial=False)
feed_1h = base.replay(
    freq="1h",
    align="session",            # session | day | global
    day_mode="trading",         # effective only when align='day': trading | calendar
    emit_partial=False,
    session_windows=[("09:30", "11:30"), ("13:00", "15:00")],  # session only
)

result = aq.run_backtest(
    data=feed_1h,
    strategy=MyStrategy,
    symbols="000001",
    show_progress=False,
)
```

Parameter semantics:

*   `align="session"`: Partition by trading day, optionally with `session_windows`.
*   `align="day"`: Partition by day without `session_windows`; `day_mode` supports `trading/calendar`.
*   `align="global"`: Aggregate on the full timeline without day partitioning.

### 2.5 Using `DataFeed` Directly

If you want explicit control over how data enters the engine, you can work with `DataFeed` directly instead of normalizing everything into a `DataFrame` first:

```python
import akquant as aq

feed = aq.DataFeed.from_csv("/data/000001.csv", "000001.SZ")

result = aq.run_backtest(
    data=feed,
    strategy=MyStrategy,
    symbols="000001.SZ",
    show_progress=False,
)
```

For live scenarios, create a writable live feed:

```python
import akquant as aq

feed = aq.DataFeed.create_live()
feed.add_tick(aq.Tick(...))
```

Notes:

*   `DataFeed.from_csv(...)` is a good fit when you want AKQuant to read a CSV-backed event stream directly.
*   `DataFeed.from_parquet(...)` fits **bounded-memory (out-of-core) streaming backtests over very large datasets** (see 2.6).
*   `add_bar(...)`, `add_bars(...)`, and `add_arrays(...)` fit cases where you already have normalized market objects or arrays on the Python side.
*   If CSV or array inputs contain invalid floating-point values, Rust emits warnings that are forwarded into AKQuant's Python `logging` pipeline instead of failing silently.

### 2.6 Very Large Datasets: Out-of-Core Streaming Backtests

When the data is too large to fit in memory at once (e.g. years of whole-market minute bars), use a **streaming Parquet feed**: data is read from disk in chunks, and the backtest's **peak memory is independent of total data size** (bounded memory).

First, use `write_canonical_parquet` to normalize any source (pandas / polars / pyarrow / `List[Bar]`) into a streamable Parquet (a `timestamp` column of int64 nanoseconds UTC, sorted ascending, zstd-compressed; a `symbol` column enables multi-symbol naturally):

```python
import akquant as aq

# any source -> canonical parquet (multi-symbol, single file globally sorted by time)
aq.write_canonical_parquet(df, "market.parquet")
```

Then feed it to the backtest via `DataFeed.from_parquet`:

```python
import akquant as aq

feed = aq.DataFeed.from_parquet("market.parquet", chunk_size=65536)
result = aq.run_backtest(
    data=feed,
    strategy=MyStrategy,
    symbols=["000001.SZ", "600000.SH"],  # multi-symbol
    show_progress=False,
)
```

Notes:

*   The canonical Parquet must be **sorted ascending by `timestamp`**; `write_canonical_parquet` sorts it for you.
*   `chunk_size` controls how many rows are read at a time (default 65536) — roughly the memory ceiling.
*   Multi-symbol just means one file sorted by time with a `symbol` column; the streaming source emits bars across symbols in time order.
*   In streaming mode **results still accumulate in memory** (equity curve, trades, etc.); the data side is bounded, while engine throughput is a separate optimization dimension.
*   `scripts/stress_out_of_core.py` in the repo measures peak memory empirically.

---

### 2.7 Tick Input

Besides a list of `Bar`, `run_backtest(data=...)` accepts three more shapes:

*   **Bars only**: `data=[Bar, Bar, ...]` (the existing usage).
*   **Ticks only**: `data=[Tick, Tick, ...]`.
*   **Mixed list**: `data=[Bar, Tick, ...]` in any order — AKQuant splits them and sorts each group by timestamp before feeding the engine.

```python
import akquant as aq

# Timestamps must be real nanoseconds. The Bar/Tick constructors multiply any
# timestamp below 1e10 by 1e9, so a small integer like 100 is silently rewritten.
ticks = [
    aq.Tick(timestamp=1704164400_000000000, price=10.00, volume=100, symbol="600000"),
    aq.Tick(timestamp=1704164403_000000000, price=10.02, volume=200, symbol="600000"),
    aq.Tick(timestamp=1704164407_000000000, price=10.01, volume=150, symbol="600000"),
]


class TickStrategy(aq.Strategy):
    def on_start(self):
        self.set_history_depth(5)

    def on_tick(self, tick):
        prices = self.get_history(2, tick.symbol, "close")
        print(tick.symbol, tick.price, prices)


result = aq.run_backtest(
    data=ticks,
    strategy=TickStrategy(),
    symbols=["600000"],
    show_progress=False,
)
```

**What ticks-only mode can and cannot do:**

*   `on_tick` fires; `on_bar` does **not**.
*   `get_history` / `get_history_multi` / `get_history_df` all work and return a **series of trade prices**: a tick is written into the history buffer as a degenerate bar (`open=high=low=close=price`), so `get_history(count, symbol, "close")` gives you the most recent trade prices.
*   Incremental indicators (`indicator_mode="incremental"`) work in **single-value** mode: `source` of `open`/`high`/`low`/`close` all return the trade price, `volume` returns the per-trade volume; `close_volume` mode works too.
*   When an incremental indicator's `input_mode` is `"hl"` / `"hlc"` / `"ohlc"`, a tick's high and low are both the trade price, so ATR, range, and similar H/L-dependent indicators would be permanently 0 on such data. If that symbol **also** has a bar source (mixed input, or `freq` aggregation below), those indicators are driven by the bars and work normally while ticks are silently skipped for them. Only when a symbol has **nothing but ticks for the entire session** does AKQuant raise `StrategyConfigurationError` (a `ValueError` subclass) at session end — that is the case where the result really would be a misleading constant 0, so it fails loudly instead. The exception propagates to the `run_backtest` caller; it is not swallowed into a log.
*   Any input containing a `Tick` (**ticks-only or mixed alike**) combined with a registered **precomputed indicator** (`indicator_mode="precompute"`) raises `ValueError`: normalized tick input flows through the `DataFeed` branch, which does not build the DataFrame precomputed indicators need. The guard keys on "is there a tick", independent of whether bars are present too. Use incremental indicators instead, or aggregate with `freq`.

**Aggregating ticks into bars with `freq`:**

```python
result = aq.run_backtest(
    data=ticks,
    freq="1min",
    strategy=MyStrategy(),
    symbols=["600000"],
    show_progress=False,
)
```

With `freq`, the raw ticks still reach `on_tick` as usual, and the aggregated bars additionally reach `on_bar` — giving you full OHLC semantics and every indicator, including H/L-dependent ones like ATR. Key points:

*   The vocabulary matches `feed_adapter.resample(freq=...)`, but only **whole minutes** are supported (`"1min"` / `"5min"` / `"1h"`). Sub-minute or non-integer periods such as `"30s"` raise `ValueError` and point you at `feed_adapter.resample` — no silent rounding.
*   The adapter declares **per-trade** volume semantics (each `Tick.volume` is that single trade's volume, not a running total), and the aggregator sums every tick's volume in the interval directly into the synthesized bar's `volume`.
*   A synthesized bar is stamped at the **end** of its interval (1 nanosecond before the next interval starts), not at the start. Backtests put synthesized bars and their source ticks into the same feed and then sort by timestamp; with an interval-start stamp the bar would sort **ahead of** the ticks that formed it, and the strategy would read high/low/close values from trades that had not happened yet. Stamping at interval end guarantees the bar is strictly later than all of its source ticks.
*   Trailing ticks that do not complete a full period produce no bar (the aggregator offers no flush).
*   Passing `freq` when `data` contains no `Tick` raises `ValueError` — the parameter would be meaningless.

See [examples/68_backtest_tick_demo.py](https://github.com/akfamily/akquant/blob/main/examples/68_backtest_tick_demo.py) for a runnable comparison of both modes.

---

## 3. Multi-Symbol Data

If you need to backtest multiple stocks simultaneously (e.g., a market-wide selection strategy), there are two ways to pass data:

### Method A: Single DataFrame (Recommended)

Concatenate data for all stocks into one large DataFrame.

```python
# Assume df_a, df_b are data for two stocks
df_all = pd.concat([df_a, df_b])

# Must sort by time! AKQuant is an event-driven engine that pushes data by time flow
df_all = df_all.sort_values(['date', 'symbol'])

# run_backtest(data=df_all, ...)
```

### Method B: Dictionary (Dict of DataFrames)

```python
data_map = {
    "AAPL": df_aapl,
    "MSFT": df_msft
}

# run_backtest(data=data_map, ...)
# The engine internally merges and sorts them automatically
```

---

## 4. Advanced Topics

### 4.1 Warmup Period

When calculating technical indicators (e.g., MA60, MACD), the `warmup_period` mechanism allows the strategy to "digest" a portion of historical data before official trading begins.

*   **Issue**: If a strategy needs to calculate MA60 on the first day but receives data starting exactly from the backtest start date, the first 59 days cannot produce indicator values.
*   **Solution**: Ensure the provided data starts earlier than `start_time`.
*   **Configuration**: Set `warmup_period = 60` in the strategy. The engine will automatically consume the first 60 bars from the data stream solely for updating indicators, without triggering `on_bar` logic.

### 4.2 Fetching History (`get_history`)

In a strategy, you can fetch historical market data for the past N days at any time.

*   `self.get_history(n, symbol, field)`: Returns a `numpy.ndarray` — a **safe snapshot copy** of the Rust rolling buffer (not zero-copy: the underlying store is a mutable ring buffer, so a view would dangle after the next Bar; it is returned as a copy). The window is usually small, so the copy cost is negligible.
*   `self.get_history_multi(n, symbol, fields)` / `self.get_history_df(n, symbol)`: Fetch multiple fields in a single FFI crossing, avoiding the per-field call overhead; behavior is identical to calling `get_history` per field.
*   `self.get_history_df(n, symbol)`: Returns a `pd.DataFrame`, convenient for Pandas calculations.

**Note**: `get_history` fetches data **prior to the current moment**, excluding the current Bar (to avoid look-ahead bias). If you need the current Bar's data for calculation, append it manually.

### 4.3 Timezone

AKQuant internally uses UTC timestamps uniformly.
If your data is in local time (e.g., Beijing Time), please specify `timezone="Asia/Shanghai"` in `run_backtest`.
If you call `ParquetDataCatalog.read(start_time=..., end_time=...)` directly, naive boundary values follow the same `timezone` rule and default to `Asia/Shanghai` when not provided explicitly.
For more details, refer to the [Timezone Handling Guide](../advanced/timezone.md).
