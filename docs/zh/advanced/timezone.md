# 时区处理指南 (Timezone Handling Guide)

AKQuant 是一个支持多市场、多品种的回测框架。为了保证时间序列的精确对齐，框架内部核心统一使用 **UTC 时间戳（纳秒）**。但在处理特定市场（如中国 A 股）时，正确的时区设置对于数据对齐、交易日划分和日志阅读至关重要。

本文档将指导你如何正确处理数据时区、回测配置以及策略中的时间转换。

## 1. 核心原则

1.  **内部统一 UTC**：引擎底层 (`Rust`) 和 Python 层的交互全部基于 UTC 时间戳。
2.  **输入自动转换**：用户传入的 DataFrame 数据，AKQuant 会将其转换为 UTC 存储；不带时区的时间按 `timezone` 参数本地化。
3.  **显示层显式转换**：结构化时间字段默认保持 UTC；只有在策略辅助方法或你自己的显示逻辑里，才会按配置的 `timezone` 转换为本地时间。
4.  **时区决定交易时段**：`timezone` 同时决定交易日划分、日终结算时刻与默认时段表（见第 3 节）。

## 2. 数据准备与时区

在准备回测数据（DataFrame）时，你有两种选择：

### 方式 A：使用无时区时间（Naive Datetime）

如果你的数据没有时区信息（tz-naive），AKQuant 按 `run_backtest(timezone=...)` 指定的时区本地化（未传时为 `Asia/Shanghai`）。

*   **如果是 A 股数据**：直接传入即可，无需额外操作。
*   **如果是美股/加密货币等其他时区数据**：传入对应的 `timezone`（如 `timezone="US/Eastern"`），或先手动转换为带时区的时间（Aware Datetime）。

**示例：构造 A 股 1 分钟线数据 (北京时间)**

```python
import pandas as pd

# 生成无时区的时间序列（按 timezone 解释，默认北京时间）
# 例如：2024-01-02 09:31:00（周二，交易日）
rng = pd.date_range(
    start="2024-01-02 09:31",
    end="2024-01-02 11:30",
    freq="1min"
)

# 创建 DataFrame
# 索引必须是 datetime 类型
df = pd.DataFrame({
    "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000
}, index=rng)

# 此时 df.index.tz 是 None，加载时按 timezone（默认 Asia/Shanghai）解释
```

分钟线的时间戳是 bar 的**区间终点**：09:31 这根 bar 覆盖 (09:30, 09:31]。

### 方式 B：使用带时区时间（Aware Datetime） - **推荐（非 A 股）**

对于非北京时间的数据（如美股、外汇），或者为了避免歧义，建议显式指定时区。AKQuant 会识别时区信息并正确转换为 UTC。

```python
# 美股数据示例 (US/Eastern)
rng = pd.date_range(
    start="2024-01-02 09:31",
    periods=100,
    freq="1min",
    tz="US/Eastern"  # 显式指定时区
)

# 或者对现有 naive 数据进行本地化
df.index = df.index.tz_localize("US/Eastern")
```

### 重要：日线数据的时间戳设置

日线有两种写法：

*   **只有日期（本地 0 点）**：推荐写法。引擎自动把它改写到该交易日的收盘时刻（A 股为 15:00），bar 区间覆盖整个交易日，无需传 `freq`。日期里含非交易日时告警一次，该行会归入下一交易日。
*   **固定非 0 点时刻（如 15:00）**：必须传 `freq="1d"`。否则引擎不知道这是日线，按“时间戳前 1ns”判时段——15:00 即 14:59:59.999…，落在收盘集合竞价，订单排队到下一根 bar 才成交，引擎会告警一次。

```python
import pandas as pd

# 写法一：只有日期，自动改写到交易日 15:00 收盘
daily_index = pd.bdate_range("2024-01-02", periods=5)

# 写法二：打在 15:00 的日线，须配合 run_backtest(..., freq="1d")
daily_index_1500 = daily_index + pd.Timedelta(hours=15)
```

日线与分钟线混合回测时，日线标签就是收盘时刻，不早于当日最后一根分钟线，逻辑顺序自然正确。

## 3. 回测配置

在 `run_backtest` 中通过 `timezone` 参数指定回测时区（默认 `Asia/Shanghai`）。

此参数用于：

1.  本地化不带时区的输入数据，以及 `start_time` / `end_time` 等 naive 时间边界。
2.  划分交易日、计算日终结算时刻，选默认时段表（见下表）。
3.  策略中的 `self.now`、`self.to_local_time(...)`、`self.format_time(...)` 等本地化时间辅助方法，以及定时器（Timer）的本地触发时刻。
4.  绩效统计中的“按日”聚合口径，例如 `daily_returns`、波动率、Sharpe / Sortino / VaR / CVaR 等，按**交易日**统计（夜盘归下一交易日）。

```python
from akquant.backtest import run_backtest

results = run_backtest(
    data=data_feed,
    strategy=MyStrategy,
    timezone="Asia/Shanghai",
    # ...
)
```

### 时区与默认时段表

| `timezone` | 未传 `sessions` 时的默认时段表 | 未传 `trading_calendar` 时的交易日 |
| :--- | :--- | :--- |
| 中国时区（`Asia/Shanghai` 等） | 按资产类型：股票/基金 `CN_STOCK`、期货 `CN_FUTURES_DAY`、期权 `CN_OPTION` | 按工作日推算（不含节假日） |
| 其他时区（如 `US/Eastern`、`UTC`） | `ALWAYS_OPEN`（全天连续竞价） | 每天都算交易日 |

非中国市场需要真实交易时段时，用 `sessions=` 传自定义时段表（写法见 [API 参考](../reference/api.md#akquantrun_backtest)）。

### 交易日历 `trading_calendar`

`run_backtest`、`run_live` 都接受 `trading_calendar=`（交易日列表，`date` / 字符串 / `Timestamp`）。不传时中国市场按工作日推算，节假日前后的夜盘与日终结算会算错。长期运行的实盘必须传；live_server 0.12.0 起，klinedata 会话自动从行情服务取沪深交易日历。

### 实盘时区

`run_live(timezone=...)` 与 `run_backtest(timezone=...)` 同义，默认 `Asia/Shanghai`，**须与交易所时区一致**。接中国市场时保持默认，不要改成本机或服务器所在时区，否则交易日、日终结算与 Day 单过期会按该时区划分，与交易所实际不符。

### 关于日志与结构化时间字段

时间语义重构后，AKQuant 的结构化日志统一采用：

- `event_time`: UTC 纳秒时间戳
- `event_time_iso`: UTC ISO 8601 字符串

这意味着：

- 结构化日志和 JSON 输出不会再默认回写为本地时区字符串
- 如果你需要“北京时间”或“策略时区”的可读日志，请在消息文本里显式调用 `self.format_time(...)`
- `bar.timestamp_iso` / `tick.timestamp_iso` / `trade.timestamp_iso` / `order.created_at_iso` 也都是 UTC ISO 8601，而不是本地时间

## 4. 策略中的时间处理

在策略的 `on_bar` 回调中，`bar.timestamp` 是一个整数（int64），表示 **UTC 纳秒时间戳**，为 bar 的区间终点。如果你需要在日志中打印当前时间，或者根据时间做逻辑判断（如：只在下午交易），需要将其转换为本地时间。判断是否处于连续竞价、当前属于哪个交易日，优先用 `self.is_market_open()`、`self.session_of(symbol)` 与 `self.trading_day`（见[策略指南](../guide/strategy.md#trading-session)）。

### 方式 1：使用 Strategy 辅助方法（推荐）

AKQuant 在 `Strategy` 基类中提供了便捷的辅助方法，自动使用回测配置的时区进行转换。

```python
class MyStrategy(Strategy):
    def on_bar(self, bar: Bar):
        # 1. 获取本地时间对象 (pd.Timestamp)
        ts = self.to_local_time(bar.timestamp)

        # 2. 直接获取格式化字符串 (默认 "%Y-%m-%d %H:%M:%S")
        ts_str = self.format_time(bar.timestamp)
        self.log(f"Current time: {ts_str}")

        # 3. 获取当前策略时间 (now)
        # 自动根据当前处理的 bar/tick 返回本地时间
        if self.now:
             print(f"Now: {self.now}")
```

### 方式 2：手动转换（底层原理）

如果你需要更底层的控制，可以手动进行转换：

```python
import pandas as pd
from akquant.strategy import Strategy
from akquant.akquant import Bar

class MyStrategy(Strategy):
    def on_bar(self, bar: Bar):
        # 1. 将 UTC 纳秒时间戳转换为 UTC datetime
        ts_utc = pd.to_datetime(bar.timestamp, unit="ns", utc=True)

        # 2. 转换为北京时间 (或其他目标时区)
        ts_bj = ts_utc.tz_convert("Asia/Shanghai")

        # ...
```

## 5. 常见问题 (FAQ)

**Q: 为什么我在 `event_time_iso` 或 `bar.timestamp_iso` 里看到的是 01:31 而不是 09:31？**
A: 因为这些字段现在统一表示 UTC 时间。北京时间 09:31 对应 UTC 01:31。若你需要按策略时区展示，请使用 `self.format_time(bar.timestamp)` 或 `self.to_local_time(bar.timestamp)`。

**Q: 我回测美股，为什么数据时间不对？**
A: 不带时区的数据按 `timezone` 本地化。请传 `timezone="US/Eastern"`，或在传入数据前使用 `.tz_localize("US/Eastern")` 处理数据索引。注意非中国时区默认全天连续（`ALWAYS_OPEN`），需要真实交易时段时请传 `sessions=`。

**Q: `AttributeError: 'float' object has no attribute 'quantity'` 是什么？**
A: 这通常是在访问持仓时发生的错误。`self.ctx.get_position(symbol)` 返回的是持仓数量（float），而不是一个对象。请直接使用返回值作为数量。如果你需要持仓均价，请改用 `self.position.entry_price` 或 `self.ctx.get_position_entry_price(symbol)`。

**Q: 混合频率回测时，日线和分钟线怎么对齐？**
A: AKQuant 是事件驱动的，按时间戳顺序处理 Bar，时间戳一律是 bar 的区间终点。

*   09:31 -> 处理 (09:30, 09:31] 的分钟 Bar
*   ...
*   15:00 -> 处理 (14:59, 15:00] 的分钟 Bar
*   15:00 -> 处理日线 Bar（日线标签为交易日收盘时刻）

只有日期的日线会自动改写到收盘时刻；打在固定时刻的日线请传 `freq="1d"`。
