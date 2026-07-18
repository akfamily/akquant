# 数据准备与加载指南 (Data Guide)

数据是量化回测的基石。AKQuant 作为一个高性能回测框架，对数据的格式和质量有一定的要求。本文档将详细介绍如何准备、清洗和加载数据，以确保回测的顺利进行。

## 1. 数据格式标准 (Data Format)

AKQuant 的核心引擎（Rust）和 Python 接口层主要通过 `pandas.DataFrame` 或 `List[Bar]` 进行交互。最推荐的方式是使用 **Pandas DataFrame**。

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
*   `add_bar(...)` / `add_bars(...)` / `add_arrays(...)` 适合你已经在 Python 侧拿到标准化行情对象或数组。
*   如果 CSV 或数组里出现非法浮点值，Rust 侧会发出 warning，并通过 AKQuant 的 Python `logging` 输出，而不是静默吞掉。

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
*   **配置**：在策略中设置 `warmup_period = 60`，引擎会自动从数据流的开头截取 60 个 Bar 仅用于更新指标，而不触发 `on_bar` 交易逻辑。

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
