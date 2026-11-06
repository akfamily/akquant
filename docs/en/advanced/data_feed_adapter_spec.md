# Data Feed Adapters (DataFeedAdapter)

A data feed adapter wraps "load data for one symbol and time range" in an object that you pass straight to `run_backtest(data=...)`, so you don't have to assemble one big DataFrame first. The backtest calls its `load()` once per symbol in `symbols`.

Two built-in adapters read CSV/Parquet files from a path template; you can also implement the protocol yourself to read from a database or an API. For resampling and replay (`resample` / `replay`), see the "offline data orchestration" section of [Multi-Timeframe Feed API](multi_timeframe_feed_api.md).

## Minimal example

One CSV file per symbol, with `{symbol}` as the placeholder:

```python
import akquant as aq

feed = aq.CSVFeedAdapter(path_template="/data/{symbol}.csv")

result = aq.run_backtest(
    data=feed,
    strategy=MyStrategy,
    symbols=["600000", "000001"],
    start_time="2024-01-02 10:00",
    end_time="2024-01-02 11:00",
)
```

`run_backtest` calls `feed.load(...)` once for each symbol in `symbols`, passing `start_time` / `end_time` in the request, and merges the returned frames into multi-symbol data. The built-in adapters clip to that range (both ends inclusive).

## Protocol

```python
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class FeedSlice:
    symbol: str
    start_time: pd.Timestamp | None = None
    end_time: pd.Timestamp | None = None
    timezone: str | None = None

class DataFeedAdapter(Protocol):
    name: str
    def load(self, request: FeedSlice) -> pd.DataFrame: ...
```

- `run_backtest` only checks for a callable `load` method; any object that has one is treated as an adapter.
- `FeedSlice.start_time` / `end_time` come from the `run_backtest` arguments of the same name (`None` if not given); `timezone` comes from `run_backtest(timezone=...)` (`None` if not given, treated as `Asia/Shanghai`).
- **Clipping is the adapter's job**: the engine does not filter adapter output by `start_time` / `end_time` again. If a custom adapter ignores those fields, everything it returns is backtested.
- `load()` must return a `pandas.DataFrame`, otherwise `TypeError` is raised; an empty frame skips that symbol.

## Returned data

- **Time**: a `DatetimeIndex`; the built-in adapters promote one of the `date`, `timestamp`, `datetime`, `time` columns to the index. Naive timestamps are interpreted in the run's timezone.
- **Required columns**: `open`, `high`, `low`, `close`, `volume`; a missing column raises `ValueError`.
- **Symbol column**: optional. Without a `symbol` column the whole frame belongs to the requested symbol; with one, rows are grouped by it.
- **Unrequested symbols**: when `symbols` is given explicitly, symbols in the returned data that are not in `symbols` are dropped with a warning, so they cannot bypass the symbol whitelist.

## Built-in adapters

| Class | Reads with | Arguments |
|---|---|---|
| `CSVFeedAdapter` | `pd.read_csv` | `path_template` (containing `{symbol}`), `read_kwargs` (passed to the reader) |
| `ParquetFeedAdapter` | `pd.read_parquet` | same |

Both normalise columns and time, and clip to the `FeedSlice` range. You can call `load()` on its own to inspect the data:

```python
df = feed.load(aq.FeedSlice(symbol="600000"))
```

## Custom adapters

Subclass `BasePandasFeedAdapter` to reuse column normalisation (`normalize`), time clipping with the same timezone rules as the built-ins (`_clip_time_range`), and `resample` / `replay`:

```python
import pandas as pd
import akquant as aq
from akquant.feed_adapter import BasePandasFeedAdapter

class MyDbAdapter(BasePandasFeedAdapter):
    name = "my_db"

    def __init__(self, conn):
        self.conn = conn

    def load(self, request: aq.FeedSlice) -> pd.DataFrame:
        frame = pd.read_sql(
            "SELECT date, open, high, low, close, volume FROM bars WHERE code = %s",
            self.conn,
            params=(request.symbol,),
        )
        data = self.normalize(frame, request.symbol)
        return self._clip_time_range(
            data, request.start_time, request.end_time, request.timezone
        )
```

Subclassing is optional: any object with a `name` attribute and a `load()` following the rules above works.

## Limitations

- **English column names only**: the built-in normalisation does not recognise Chinese column names such as `日期` or `开盘`. For data with Chinese column names (e.g. from AKShare), rename them in `load()`, or pass a DataFrame to `run_backtest` directly (the DataFrame path does recognise them).
- **No caching**: the source is read again on every backtest.
- **Errors propagate as-is**: a missing file raises `FileNotFoundError` without extra hints.
- **Backtest only**: `run_live` does not accept adapters; live market data comes from market gateways.
- **Extra columns are passed through, not used**: numeric columns beyond OHLCV (e.g. `adj_factor`) appear in `bar.extra`, but the engine does not use them for price adjustment; adjust prices in your data source.
