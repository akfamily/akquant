# 策略编写指南

本文档旨在帮助策略开发者快速掌握 Akquant 的策略编写方法。Akquant 提供了两种风格的策略开发接口：**类风格 (Class-based)** 和 **函数风格 (Functional)**。

## 1. 核心概念

在开始编写策略之前，了解以下核心概念非常重要：

*   **Bar**: 市场数据的基本单元，包含开高低收量 (OHLCV) 和时间戳。
*   **Strategy**: 策略逻辑的载体，负责接收市场事件并生成交易指令。
*   **Context (`ctx`)**: 策略上下文，提供账户信息（资金、持仓）、订单管理以及历史数据访问能力。
*   **生命周期**:
    *   `__init__`: Python 对象初始化，适合定义参数和注册指标。
    *   `on_start`: 策略启动时调用，**必须**在此处使用 `self.subscribe()` 订阅数据。
    *   `on_bar`: 每一根 K 线闭合时触发 (核心交易逻辑)。
    *   `on_tick`: 每一个 Tick 到达时触发 (高频/盘口策略)。
    *   `on_timer`: 定时器触发时调用 (需手动注册)。

## 2. 策略风格选择

| 特性 | 类风格 (推荐) | 函数风格 |
| :--- | :--- | :--- |
| **定义方式** | 继承 `akquant.Strategy` | 定义 `initialize` 和 `on_bar` 函数 |
| **适用场景** | 复杂策略、需要维护内部状态、生产环境 | 快速原型验证、迁移 Zipline/Backtrader 策略 |
| **代码结构** | 面向对象，逻辑封装性好 | 脚本化，简单直观 |
| **API 调用** | `self.buy()`, `self.ctx` | `ctx.buy()`, `ctx` 作为参数传递 |

## 3. 编写类风格策略 (Class-based)

这是 Akquant 推荐的策略编写方式，结构清晰，易于扩展。

### 3.1 示例代码

```python
from akquant import Strategy, Bar
from akquant.indicator import SMA

class MyStrategy(Strategy):
    def __init__(self, ma_window=20):
        super().__init__()
        # 定义指标，系统会自动维护其计算
        self.sma = SMA(ma_window)

    def on_start(self):
        # 显式订阅数据
        self.subscribe("600000")

    def on_bar(self, bar: Bar):
        # 1. 获取指标值
        ma_value = self.sma.value

        # 检查指标是否就绪
        if ma_value is None:
            return

        # 2. 交易逻辑
        if bar.close > ma_value:
            self.buy(symbol=bar.symbol, quantity=100)
        elif bar.close < ma_value:
            self.sell(symbol=bar.symbol, quantity=100)
```

### 3.2 关键方法

*   **`on_start(self)`**: 策略启动入口，用于订阅数据 (`self.subscribe`) 和注册定时器。
*   **`on_bar(self, bar: Bar)`**: 核心回调，每个 Bar 到达时触发。
*   **`self.buy(symbol, quantity, price=None)`**: 发送买入指令。不指定价格则为市价单。
*   **`self.sell(symbol, quantity, price=None)`**: 发送卖出指令。
*   **`self.get_position(symbol)`**: 获取指定标的的当前持仓数量。
*   **`self.get_open_orders(symbol)`**: 获取指定标的的未成交订单列表。
*   **`self.get_history(count, symbol=None, field="close")`**: 获取历史数据序列。

## 4. 编写函数风格策略 (Functional)

这种风格与 Zipline 非常相似，适合习惯函数式编程的用户。

### 4.1 示例代码

```python
from akquant import run_backtest
import numpy as np

def initialize(ctx):
    # 初始化策略参数
    ctx.ma_window = 20

def on_bar(ctx, bar):
    # ctx 即为策略上下文，同时代理了 Strategy 方法
    prices = ctx.get_history(count=ctx.ma_window)

    if np.isnan(prices).any():
        return

    ma = prices.mean()

    # 交易逻辑
    if bar.close > ma:
        ctx.buy(symbol=bar.symbol, quantity=100)
    elif bar.close < ma:
        ctx.sell(symbol=bar.symbol, quantity=100)

# 运行回测
run_backtest(
    data=df,
    strategy=on_bar,
    initialize=initialize,
    history_depth=20,  # 关键：启用历史数据维护
    symbol="600000",
    cash=100000
)
```

## 5. 指标计算模式

Akquant 支持两种指标计算模式，开发者应根据性能需求选择。

### 5.1 向量化预计算 (推荐) - `IndicatorSet`

**这是最高效的方式，类似 PyBroker 的设计。**

通过 `IndicatorSet`，你可以利用 Pandas/Numpy/Talib 在回测开始前一次性计算好所有指标，然后在策略中以极低的开销访问。

*   **优点**: 极快 (比在线计算快 10-50 倍)，代码整洁。
*   **使用方法**:

```python
from akquant.indicator import IndicatorSet
import talib

# 1. 定义计算函数 (接收 DataFrame, 返回 Series)
def calculate_rsi(df, timeperiod=14):
    return talib.RSI(df['close'].values, timeperiod=timeperiod)

# 2. 创建 IndicatorSet
indicators = IndicatorSet()
indicators.add("rsi_14", calculate_rsi, timeperiod=14)
# 支持 lambda
indicators.add("ma_20", lambda df: df['close'].rolling(20).mean())

# 3. 在 run_backtest 中传入
run_backtest(
    # ... 其他参数
    indicators=indicators
)

# 4. 在策略中使用 (自动注入到 self.indicators 或 ctx.indicators)
def on_bar(ctx, bar):
    # 通过 bar.timestamp 自动查找当天的指标值
    rsi = ctx.indicators["rsi_14"]
    ma = ctx.indicators["ma_20"]

    if rsi < 30 and bar.close > ma:
        ctx.buy(symbol=bar.symbol, quantity=100)
```

### 5.2 在线计算 (Online)

**推荐用于实盘、复杂逻辑验证或无法预计算的场景。**

这种模式下，在 `on_bar` 中实时获取历史序列计算指标。

*   **优点**: 代码简单，逻辑与实盘完全一致，无未来函数风险。
*   **缺点**: 速度较慢 (受限于 Python 循环和切片开销)。
*   **实现步骤**:
    1.  在 `run_backtest` 中设置 `history_depth` (如 20)。
    2.  在 `on_bar` 中使用 `ctx.get_history(count=20)` 获取 Numpy 数组。
    3.  调用 `talib` 或 `numpy` 函数计算。

*(参考 `examples/zipline_style_backtest.py`)*

## 6. 订单与交易

### 6.1 订单类型

*   **市价单 (Market Order)**: `buy(symbol, quantity)` (不指定价格)。
    *   **ExecutionMode.CurrentClose**: 以当根 Bar 收盘价成交 (Cheat-on-Close)。
    *   **ExecutionMode.NextOpen**: 以次日 Open 价成交 (更真实)。
*   **限价单 (Limit Order)**: `buy(symbol, quantity, price=10.5)`。
    *   **买入**: 当 Low <= Limit Price 时成交，成交价为 min(Open, Limit)。
    *   **卖出**: 当 High >= Limit Price 时成交，成交价为 max(Open, Limit)。
*   **止损单 (Stop Order)**: `stop_buy(symbol, trigger_price=10.5)` / `stop_sell(...)`。
    *   **止损买入**: 当市价(High) >= 触发价(trigger_price) 时触发。
    *   **止损卖出**: 当市价(Low) <= 触发价(trigger_price) 时触发。
    *   **Stop Market**: 若不指定 `price`，触发后提交市价单。
    *   **Stop Limit**: 若指定 `price`，触发后提交限价单。

### 6.2 交易规则

Akquant 引擎内置了中国市场的交易规则支持：

*   **T+1 / T+0**: 引擎根据 `Instrument` 类型自动处理。
    *   **股票 (Stock) / 基金 (Fund)**: 默认 T+1 (今日买入明日可卖)。
    *   **期货 (Futures) / 期权 (Option)**: 默认 T+0 (当日买入当日可卖)。
*   **最小交易单位 (Lot Size)**:
    *   **股票**: 买入必须是 100 股的整数倍 (一手)，卖出无限制 (支持零股)。
    *   **其他**: 默认为 1。
    *   **配置方式**: 可以在 `run_backtest` 中通过 `lot_size` 参数修改。
        ```python
        # 全局修改为 200
        run_backtest(..., lot_size=200)

        # 针对特定标的修改
        run_backtest(..., lot_size={"000001": 100, "HK0700": 200})
        ```
*   **费率 (Fee)**: 支持分别配置佣金 (Commission)、印花税 (Stamp Tax)、过户费 (Transfer Fee) 和最低佣金。
    *   **基金**: 支持独立的佣金费率。
    *   **期权**: 支持按张收费模式。

## 7. 全局配置 (`StrategyConfig`)

`akquant` 提供了一个全局配置对象 `strategy_config` (类似 PyBroker)，用于控制回测行为。虽然 `run_backtest` 封装了大部分常用配置，但在高级场景下可以直接修改它。

```python
from akquant.config import strategy_config
from akquant import ExecutionMode

# 设置执行模式 (默认 CurrentClose)
strategy_config.execution_mode = ExecutionMode.NextOpen

# 设置最大订单比例 (默认 1.0)
strategy_config.max_order_size = 0.5  # 每次最多买入总资金的 50%
```

## 8. 进阶技巧

*   **自定义参数**: 在 `__init__` 或 `initialize` 中定义成员变量。
*   **定时任务**: 使用 `self.schedule(timestamp, payload)` 注册盘中定时事件，适合做收盘前平仓等逻辑。
*   **多标的**: `on_bar` 会按时间顺序依次推送不同标的的 Bar，使用 `bar.symbol` 区分处理。
    *   在 `run_backtest` 中，可以传入 `data={"AAPL": df1, "GOOG": df2}` 字典来支持多标的回测。

## 9. 结果可视化

Akquant 提供了内置的绘图工具，方便分析策略表现。

```python
from akquant.backtest import plot_result

# 运行回测
result = run_backtest(...)

# 绘制权益曲线和回撤
plot_result(result)
```

## 10. 多品种策略示例

对于涉及多种资产类型的策略，您可以在 `run_backtest` 中指定 `asset_type`，或手动添加不同类型的 `Instrument`。

### 10.1 期权策略示例 (Class-based)

```python
from akquant import Strategy, OptionType

class OptionStrategy(Strategy):
    def on_bar(self, bar):
        # 假设这是一个看涨期权 (Call)
        # 如果标的价格上涨，买入期权
        if bar.close > 3.0 and self.ctx.position.size == 0:
            # 买入 10 张期权 (假设 multiplier=10000, 相当于 10万股标的)
            self.buy(symbol=bar.symbol, quantity=10)

        # 止盈
        elif bar.close > 3.5 and self.ctx.position.size > 0:
            self.sell(symbol=bar.symbol, quantity=10)

# 运行回测时指定期权参数
# run_backtest(..., asset_type=AssetType.Option, option_type=OptionType.Call, ...)
```

## 11. 常见问题 (FAQ)

*(待补充)*
