# 数据源适配器（DataFeedAdapter）

数据源适配器把“按标的、按时间段取数据”封装成一个对象，直接传给 `run_backtest(data=...)`，不必先把所有数据整理成一张 DataFrame。回测按 `symbols` 逐个标的调用它的 `load()`。

内置了按文件路径模板读 CSV/Parquet 的两个适配器，也可以自己实现协议对接数据库或接口。重采样、重放（`resample` / `replay`）见[多周期策略](multi_timeframe_feed_api.md)中的“离线数据编排”一节。

## 最小例子

每个标的一个 CSV 文件，路径里用 `{symbol}` 占位：

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

`run_backtest` 对 `symbols` 里的每个标的调用一次 `feed.load(...)`，把 `start_time` / `end_time` 放进请求里传给适配器，再把返回的数据合并成多标的数据。内置适配器会按这个时间段截取（两端都包含）。

## 协议

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

- `run_backtest` 只检查对象有没有可调用的 `load` 方法，有就按适配器处理。
- `FeedSlice.start_time` / `end_time` 来自 `run_backtest` 的同名参数，没传时为 `None`；`timezone` 来自 `run_backtest(timezone=...)`，没传时为 `None`（按 `Asia/Shanghai` 处理）。
- **时间截取由适配器负责**：引擎不会再按 `start_time` / `end_time` 过滤适配器返回的数据，自定义适配器忽略这两个字段时，返回多少就回测多少。
- `load()` 必须返回 `pandas.DataFrame`，否则抛 `TypeError`；返回空表时该标的被跳过。

## 返回数据的格式

- **时间**：`DatetimeIndex`；内置适配器会把 `date`、`timestamp`、`datetime`、`time` 列之一设为索引。不带时区的时间按本次回测的时区解释。
- **必需列**：`open`、`high`、`low`、`close`、`volume`，缺列抛 `ValueError`。
- **标的列**：可选。没有 `symbol` 列时，整张表都算作请求的那个标的；有 `symbol` 列时按它分组。
- **混入未请求的标的**：显式传了 `symbols` 时，返回数据里不在 `symbols` 中的标的会被丢弃并告警，防止它们绕过标的白名单。

## 内置适配器

| 类 | 读取方式 | 参数 |
|---|---|---|
| `CSVFeedAdapter` | `pd.read_csv` | `path_template`（含 `{symbol}`）、`read_kwargs`（透传给读取函数） |
| `ParquetFeedAdapter` | `pd.read_parquet` | 同上 |

两者都会做列与时间规范化，并按 `FeedSlice` 的时间段截取。也可以单独调用 `load()` 检查读出来的数据：

```python
df = feed.load(aq.FeedSlice(symbol="600000"))
```

## 自定义适配器

继承 `BasePandasFeedAdapter` 可以复用列规范化（`normalize`）、按内置适配器同样的时区规则截取时间（`_clip_time_range`）以及 `resample` / `replay`：

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

不继承也可以，只要对象有 `name` 属性和符合上面约定的 `load()`。

## 限制

- **只认英文列名**：内置适配器的规范化不识别 `日期`、`开盘` 等中文列名；AKShare 等返回中文列名的数据，请在 `load()` 里先改名，或改用 DataFrame 直接传给 `run_backtest`（DataFrame 路径识别中文列名）。
- **没有缓存**：每次回测都重新读取数据源。
- **文件不存在等错误原样抛出**：例如找不到文件时是 `FileNotFoundError`，不附加标的或路径之外的提示。
- **只用于回测**：`run_live` 不接收适配器，实盘行情走行情网关。
- **额外列只透传、不参与计算**：OHLCV 之外的数值列（如 `adj_factor`）会出现在 `bar.extra` 里，但引擎不会用它们复权，复权需要在数据源侧完成。
