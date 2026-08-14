# 数据准备与加载指南 (Data Guide)

数据是量化回测的基石。AKQuant 作为一个高性能回测框架，对数据的格式和质量有一定的要求。本文档将详细介绍如何准备、清洗和加载数据，以确保回测的顺利进行。

## 1. 数据格式标准 (Data Format)

AKQuant 的核心引擎（Rust）和 Python 接口层主要通过 `pandas.DataFrame` 或 `List[Bar]` 进行交互。最推荐的方式是使用 **Pandas DataFrame**。

除 pandas 外，`run_backtest(data=...)` 也接受 **`polars.DataFrame` / `polars.LazyFrame` / `pyarrow.Table`** 作为一等输入（内部会零成本转为 pandas 数据路径），无需你手动 `.to_pandas()`：

```python
import polars as pl
from akquant import run_backtest

pldf = pl.read_parquet("000001.parquet")
result = run_backtest(data=pldf, strategy=MyStrategy, symbols="000001.SZ")
```

### 1.1 必需列 (Required Columns)

你的 DataFrame **必须** 包含以下列（列名不区分大小写，但在内部会被转换为小写）：

| 列名 (Column) | 类型 (Type) | 说明 |
| :--- | :--- | :--- |
| `date` / `time` / `datetime` | `datetime64[ns]` | 时间戳索引。必须是 Pandas 的 datetime 类型。 |
| `open` | `float` | 开盘价 |
| `high` | `float` | 最高价 |
| `low` | `float` | 最低价 |
| `close` | `float` | 收盘价 |
| `volume` | `float` | 成交量 |
| `symbol` | `str` | 标的代码 (如 "000001", "AAPL") |

**注意：**

1.  **列名标准化**：建议在传入前将列名统一重命名为英文小写（如 `open`, `close`）。
2.  **Symbol 列**：即使只回测一支股票，也必须包含 `symbol` 列，以便引擎识别数据所属标的。

### 1.2 索引 (Index)

*   DataFrame 的索引可以是默认的整数索引，也可以是 `DatetimeIndex`。
*   如果使用 `DatetimeIndex`，AKQuant 会自动将其作为时间列。
*   **排序**：数据必须按时间**升序**排列（旧 -> 新）。

---

## 2. 数据获取与加载示例

### 2.1 从 CSV 加载

这是最常见的方式。假设你有一个 `data.csv` 文件。

```python
import pandas as pd
from akquant import run_backtest

# 1. 读取 CSV
df = pd.read_csv("data.csv")

# 2. 转换时间列
# 必须确保时间列是 datetime 类型，而不是字符串
df['date'] = pd.to_datetime(df['date'])

# 3. 确保列名正确
# 假设 CSV 列名是 "Date", "Open", ...
df.columns = [c.lower() for c in df.columns]

# 4. 添加 symbol 列 (如果 CSV 中没有)
if 'symbol' not in df.columns:
    df['symbol'] = "DEMO_TICKER"

# 5. 排序
df = df.sort_values('date').reset_index(drop=True)

# 6. 传入回测
# result = run_backtest(data=df, ...)
```

### 2.2 使用 AKShare (A股数据)

[AKShare](https://github.com/akfamily/akshare) 是一个非常强大的开源财经数据接口库。

```python
import akshare as ak
import pandas as pd

# 1. 下载数据 (以前复权为例)
# period="daily" 日线; adjust="qfq" 前复权
df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20200101", end_date="20231231", adjust="qfq")

# 2. 重命名列 (AKShare 返回中文列名)
df = df.rename(columns={
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume"
})

# 3. 类型转换
df['date'] = pd.to_datetime(df['date'])
df['symbol'] = "000001"

# 4. 筛选列
df = df[["date", "open", "high", "low", "close", "volume", "symbol"]]
```

### 2.3 使用 yfinance (美股数据)

```python
import yfinance as yf

# 1. 下载数据
df = yf.download("AAPL", start="2020-01-01", end="2023-12-31")

# yfinance 返回 MultiIndex 列 (如果下载多股) 或大写列名
# 这里简化处理单股情况
df.columns = [c.lower() for c in df.columns]
df.reset_index(inplace=True) # 将 Date 索引变成列
df = df.rename(columns={"date": "date"}) # 确保是 date

df['symbol'] = "AAPL"
```

### 2.4 使用 DataFeedAdapter + 多时间框聚合

如果你希望把“数据加载 + 重采样/重放”封装在同一入口，可以直接使用 `DataFeedAdapter`：

```python
import akquant as aq

base = aq.CSVFeedAdapter(path_template="/data/{symbol}.csv")

feed_15m = base.resample(freq="15min", emit_partial=False)
feed_1h = base.replay(
    freq="1h",
    align="session",            # session | day | global
    day_mode="trading",         # 仅 align='day' 时生效: trading | calendar
    emit_partial=False,
    session_windows=[("09:30", "11:30"), ("13:00", "15:00")],  # 仅 align='session'
)

result = aq.run_backtest(
    data=feed_1h,
    strategy=MyStrategy,
    symbols="000001",
    show_progress=False,
)
```

参数语义：

*   `align="session"`：按交易日分区，可叠加 `session_windows`。
*   `align="day"`：按日分区，不接收 `session_windows`；`day_mode` 支持 `trading/calendar`。
*   `align="global"`：按全局时间轴聚合，不按交易日切段。

### 2.5 直接使用 `DataFeed`

如果你希望显式控制“数据如何进入引擎”，可以直接使用 `DataFeed`，而不必先转成 `DataFrame`：

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

实时场景则可以创建可写入的 live feed：

```python
import akquant as aq

feed = aq.DataFeed.create_live()
feed.add_tick(aq.Tick(...))
```

补充说明：

*   `DataFeed.from_csv(...)` 适合让 AKQuant 直接从 CSV 事件流读取数据。
*   `DataFeed.from_parquet(...)` 适合**超大数据集的有界内存（out-of-core）流式回测**（见 2.6）。
*   `add_bar(...)` / `add_bars(...)` / `add_arrays(...)` 适合你已经在 Python 侧拿到标准化行情对象或数组。
*   如果 CSV 或数组里出现非法浮点值，Rust 侧会发出 warning，并通过 AKQuant 的 Python `logging` 输出，而不是静默吞掉。

### 2.6 超大数据集：out-of-core 流式回测

当数据大到无法一次性放进内存（例如全市场多年分钟线）时，可用 **流式 Parquet 数据源**：数据按块从磁盘读取，回测**峰值内存与数据总量无关**（有界内存）。

第一步，用 `write_canonical_parquet` 把任意来源（pandas / polars / pyarrow / `List[Bar]`）规范化为可流式读取的 Parquet（列 `timestamp` 为纳秒 UTC 整数、按时间升序、zstd 压缩；含 `symbol` 列即天然支持多标的）：

```python
import akquant as aq

# 任意来源 -> 规范 parquet（可多标的，单文件按时间全局排序）
aq.write_canonical_parquet(df, "market.parquet")
```

第二步，用 `DataFeed.from_parquet` 流式喂给回测：

```python
import akquant as aq

feed = aq.DataFeed.from_parquet("market.parquet", chunk_size=65536)
result = aq.run_backtest(
    data=feed,
    strategy=MyStrategy,
    symbols=["000001.SZ", "600000.SH"],  # 多标的
    show_progress=False,
)
```

要点：

*   规范 Parquet 需**按 `timestamp` 升序**；`write_canonical_parquet` 会自动排序。
*   `chunk_size` 控制每次读取的行数（默认 65536），即内存上界的量级。
*   多标的只需在同一个按时间排序、带 `symbol` 列的文件里；流式源会按时间序跨标的产出。
*   流式模式下**结果仍在内存累积**（资金曲线、成交等），若要跑到"数千万根 bar"，数据侧已有界，引擎侧吞吐是另一维度的优化。
*   仓库内 `scripts/stress_out_of_core.py` 提供了峰值内存实测脚本。

### 2.7 Tick 输入

`run_backtest(data=...)` 除了 `Bar` 列表，还接受三种形态：

*   **纯 bar**：`data=[Bar, Bar, ...]`（既有用法）。
*   **纯 tick**：`data=[Tick, Tick, ...]`。
*   **混合列表**：`data=[Bar, Tick, ...]`，`Bar` 与 `Tick` 任意顺序排列，AKQuant 会各自按时间戳升序拆分后再送入引擎。

```python
import akquant as aq

# 时间戳必须是真实纳秒；Bar/Tick 构造器会把 < 1e10 的时间戳乘 1e9，
# 传小整数(如 100)会被静默改写，不要图省事。
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

**纯 tick 模式的能力边界：**

*   触发 `on_tick`，**不**触发 `on_bar`。
*   `get_history` / `get_history_multi` / `get_history_df` / `get_rolling_data` 可用，返回**成交价序列**：tick 有自己独立的一条历史序列（不再与 bar 共用缓冲区），省略 `freq`（纯 tick 场景没有歧义，不要求显式指定）等价于 `freq='tick'` 的默认取数路径，`get_history(count, symbol, "close")` 等价于取最近若干笔成交价。
    该独立序列内部仍按 `open=high=low=close=price` 存储（历史包袱：与 bar 共用同一存储结构），但**只有显式传 `freq='tick'` 时**才会在 Python 层校验 `field`——此时请求 `open`/`high`/`low` 会抛 `ValueError`（tick 没有真正的高低开盘价，可用字段为 `price`/`close`/`volume`）；`get_history_df` / `get_rolling_data` 固定取 OHLCV 五字段，故显式传 `freq='tick'` 调用它们必然报错，纯 tick 场景请省略 `freq` 或改用 `get_history(freq='tick', field='price')`。若该 symbol 同时存在 bar 来源（见下面的 `freq` 聚合或第 3 节的双流场景），省略 `freq` 会因两条序列并存而报错，要求显式传 `freq='tick'` / `freq='bar'`。
*   增量指标（`indicator_mode="incremental"`）的**单值模式**可用：`source` 取 `open`/`high`/`low`/`close` 均返回成交价，取 `volume` 返回单笔量；`close_volume` 模式同样可用。
*   增量指标的 `input_mode` 为 `"hl"` / `"hlc"` / `"ohlc"` 时，tick 的最高/最低价恒等于成交价，ATR、振幅等依赖真实 H/L 的指标在这类数据上只会恒为 0。若该标的**同时**有 bar 来源（混合输入，或配合下面的 `freq` 把 tick 聚合成 bar），这类指标由 bar 驱动正常工作，tick 本身对它们静默跳过；只有当某个标的**全程只有 tick、从未有过任何 bar** 时，AKQuant 才会在会话结束时抛 `StrategyConfigurationError`（`ValueError` 的子类，异常会穿透到 `run_backtest` 的调用方，不会被日志吞掉）——此时才是真正只能拿到恒为 0 的误导结果，故显式报错而非静默给出。
*   输入含任意 `Tick`（**纯 tick 或混合皆然**）+ 已注册的**预计算指标**（`indicator_mode="precompute"`）会抛 `ValueError`：归一后走 `DataFeed` 分支，该分支不构建预计算指标所需的 DataFrame。护栏判据是「是否有 tick」，与是否同时有 bar 无关。需要指标时改用增量指标，或用下面的 `freq` 把 tick 聚合成 bar。

**用 `freq` 把 tick 聚合成 bar：**

```python
result = aq.run_backtest(
    data=ticks,
    freq="1min",
    strategy=MyStrategy(),
    symbols=["600000"],
    show_progress=False,
)
```

传入 `freq` 后，原始 tick 仍照常投递给 `on_tick`，同时把 tick 聚合成 bar 投递给 `on_bar`，从而拿到完整 OHLC 语义与全部指标（含 ATR 等 H/L 类）。要点：

*   词汇与 `feed_adapter.resample(freq=...)` 一致，但**只支持整数分钟**（`"1min"` / `"5min"` / `"1h"`）；`"30s"` 等非整分周期会抛 `ValueError` 并指向 `feed_adapter.resample`，不会静默取整。
*   适配层按**单笔口径**声明 `volume`（即每个 `Tick.volume` 就是这一笔的成交量，不是累计量），聚合器把区间内所有 tick 的 volume 直接求和写入合成 bar 的 `volume`。
*   合成 bar 的时间戳打在**区间结束**（下一区间起点前 1 纳秒），而非区间起点：回测把合成 bar 与源 tick 放进同一个 feed 再按时间戳排序，若用区间起点，bar 会排到形成它的 tick **之前**，策略读到的 high/low/close 会是尚未发生的未来数据；打在区间结束保证 bar 严格晚于其所有源 tick。
*   末尾未满一个周期的 tick 不会产生 bar（聚合器不提供 flush）。
*   `data` 中不含任何 `Tick` 时传 `freq` 会抛 `ValueError`（参数无意义）。

---

## 3. 多标的数据 (Multi-Symbol Data)

如果你需要同时回测多只股票（例如全市场选股策略），有两种方式传入数据：

### 方式 A：单一 DataFrame (推荐)

将所有股票的数据拼接成一个巨大的 DataFrame。

```python
# 假设 df_a, df_b 是两只股票的数据
df_all = pd.concat([df_a, df_b])

# 必须按时间排序！AKQuant 是事件驱动引擎，按时间流推送数据
df_all = df_all.sort_values(['date', 'symbol'])

# run_backtest(data=df_all, ...)
```

### 方式 B：字典 (Dict of DataFrames)

```python
data_map = {
    "AAPL": df_aapl,
    "MSFT": df_msft
}

# run_backtest(data=data_map, ...)
# 引擎内部会自动将其合并并排序
```

---

## 4. 高级话题

### 4.1 预热期数据 (Warmup Period)

在计算技术指标（如 MA60, MACD）时，通过 `warmup_period` 机制，AKQuant 允许策略在正式交易前先“消化”一段历史数据。

*   **问题**：如果策略第一天就要计算 MA60，但只传入了从回测开始日期的数据，前 59 天是无法计算指标的。
*   **解决**：确保传入的数据比 `start_time` (回测开始时间) 更早一些。
*   **配置**：在策略中设置 `warmup_period = 60`，引擎会让**每个标的各自**先积累 60 根 Bar 仅用于更新指标，期间不触发 `on_bar` 交易逻辑。
*   **多标的**：门槛按标的**独立**计算——标的 A 自己攒够 60 根就开始交易，不必等标的 B。因此 `warmup_period` 直接按指标窗口设定（如 `self.warmup_period = self.params.long_window + 1`），**不需要**乘以标的数量。

### 4.2 历史数据获取 (`get_history`)

在策略中，你可以随时获取过去 N 天的行情数据。

*   `self.get_history(count, symbol, field)`: 返回 `numpy.ndarray`，是对 Rust 滚动缓冲的一次**安全快照拷贝**（并非零拷贝——底层为可变环形缓冲，返回视图会在下一根 Bar 后失效，故按拷贝返回）。窗口通常很小，拷贝开销可忽略。
*   `self.get_history_multi(count, symbol, fields)` / `self.get_history_df(count, symbol)`: 一次跨界批量取回多字段，避免逐字段多次调用的边界开销，语义与逐字段 `get_history` 完全一致。
*   `self.get_history_df(count, symbol)`: 返回 `pd.DataFrame`，方便使用 Pandas 计算。

**注意**：`get_history` 获取的是**当前时刻之前**的数据，不包含当前 Bar（为了避免未来函数）。如果需要包含当前 Bar 的数据参与计算，可以手动 append。

### 4.3 时区 (Timezone)

AKQuant 内部统一使用 UTC 时间戳。
如果你的数据是本地时间（如北京时间），请在 `run_backtest` 中指定 `timezone="Asia/Shanghai"`。
如果你直接调用 `ParquetDataCatalog.read(start_time=..., end_time=...)`，传入 naive 时间边界时也会按该 `timezone`（未显式传入时默认为 `Asia/Shanghai`）解释。
更多详情请参考 [时区处理指南](../advanced/timezone.md)。
