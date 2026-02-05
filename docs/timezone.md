# 时区处理指南 (Timezone Handling Guide)

AKQuant 是一个支持多市场、多品种的回测框架。为了保证时间序列的精确对齐，框架内部核心统一使用 **UTC 时间戳（纳秒）**。但在处理特定市场（如中国 A 股）时，正确的时区设置对于数据对齐和日志阅读至关重要。

本文档将指导你如何正确处理数据时区、回测配置以及策略中的时间转换。

## 1. 核心原则

1.  **内部统一 UTC**：引擎底层 (`Rust`) 和 Python 层的交互全部基于 UTC 时间戳。
2.  **输入自动转换**：用户传入的 DataFrame 数据，AKQuant 会根据配置的 `timezone` 自动将其转换为 UTC 存储。
3.  **输出本地化**：回测结果和日志展示时，会尝试转换回用户配置的 `timezone`（默认为 `Asia/Shanghai`）。

## 2. 数据准备与时区

在准备回测数据（DataFrame）时，你有两种选择：

### 方式 A：使用无时区时间（Naive Datetime） - **推荐**

如果你的数据是本地时间（例如北京时间），且没有时区信息（tz-naive），你只需要在回测配置中指定 `timezone="Asia/Shanghai"`。AKQuant 会自动认为这些数据属于该时区，并进行转换。

**示例：构造 A 股 1 分钟线数据**

```python
import pandas as pd
from datetime import timedelta

# 生成无时区的时间序列（默认为北京时间）
# 例如：2023-01-01 09:31:00
rng = pd.date_range(
    start="2023-01-01 09:31",
    end="2023-01-01 11:30",
    freq="1min"
)

# 创建 DataFrame
# 索引必须是 datetime 类型
df = pd.DataFrame({
    "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000
}, index=rng)

# 此时 df.index.tz 是 None
```

### 方式 B：使用带时区时间（Aware Datetime）

如果你的数据已经带有时区信息（例如从某些 API 获取的数据），AKQuant 会直接将其转换为 UTC。请确保时区信息是正确的。

```python
# 带有时区的时间序列
rng = pd.date_range(
    start="2023-01-01 09:31",
    periods=100,
    freq="1min",
    tz="Asia/Shanghai"  # 显式指定时区
)
```

### 重要：日线数据的时间戳设置

对于日线数据（1D），为了与分钟线数据（1m）在回测中正确对齐（避免未来函数），**强烈建议将日线时间戳设置为当天的收盘时间**（A 股为 15:00）。

*   如果设置为 00:00，可能会导致日线数据在当天的分钟线之前就“出现”，或者在对齐时产生混淆。
*   设置为 15:00 可以确保日线 Bar 在当天的交易结束后才完成。

```python
# 日线数据索引示例
ts_daily = pd.Timestamp("2023-01-01 15:00:00")  # 北京时间 15:00
```

## 3. 回测配置

在 `run_backtest` 或 `BacktestEngine` 初始化时，通过 `timezone` 参数指定回测的默认时区。

```python
from akquant.backtest import run_backtest

results = run_backtest(
    data=data_feed,
    strategy=MyStrategy,
    timezone="Asia/Shanghai",  # 指定为上海时间
    # ...
)
```

## 4. 策略中的时间处理

在策略的 `on_bar` 回调中，`bar.timestamp` 是一个整数（int64），表示 **UTC 纳秒时间戳**。如果你需要在日志中打印当前时间，或者根据时间做逻辑判断（如：只在下午交易），需要将其转换为本地时间。

### 转换示例

```python
import pandas as pd
from akquant.strategy import Strategy
from akquant.akquant import Bar

class MyStrategy(Strategy):
    def on_bar(self, bar: Bar):
        # 1. 将 UTC 纳秒时间戳转换为 UTC datetime
        ts_utc = pd.to_datetime(bar.timestamp, unit="ns", utc=True)

        # 2. 转换为北京时间
        ts_bj = ts_utc.tz_convert("Asia/Shanghai")

        # 3. 格式化输出
        time_str = ts_bj.strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"Current time (Beijing): {time_str}")

        # 4. 基于时间的逻辑判断
        # 例如：只在 14:50 之后平仓
        if ts_bj.hour == 14 and ts_bj.minute >= 50:
            pass
```

## 5. 常见问题 (FAQ)

**Q: 为什么我在日志里看到的时间是 01:31 而不是 09:31？**
A: 这是因为直接打印 `bar.timestamp` 转换出来的默认可能是 UTC 时间（北京时间 09:31 对应 UTC 01:31）。请按照上述“策略中的时间处理”一节，显式调用 `.tz_convert("Asia/Shanghai")`。

**Q: `AttributeError: 'float' object has no attribute 'quantity'` 是什么？**
A: 这通常是在访问持仓时发生的错误。`self.ctx.get_position(symbol)` 返回的是持仓数量（float），而不是一个对象。请直接使用返回值作为数量。

**Q: 混合频率回测时，日线和分钟线怎么对齐？**
A: AKQuant 是事件驱动的，按时间戳顺序处理 Bar。
*   09:31 -> 处理分钟 Bar
*   ...
*   15:00 -> 处理分钟 Bar
*   15:00 -> 处理日线 Bar (假设日线时间戳设为 15:00)
确保日线的时间戳 >= 当天最后一根分钟线的时间戳，即可保证逻辑顺序正确。
