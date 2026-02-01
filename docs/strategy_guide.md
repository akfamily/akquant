# 策略编写指南

本文档旨在帮助策略开发者快速掌握 AKQuant 的策略编写方法。AKQuant 提供了两种风格的策略开发接口：**类风格 (Class-based)** 和 **函数风格 (Functional)**。

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

这是 AKQuant 推荐的策略编写方式，结构清晰，易于扩展。

### 3.1 示例代码

```python
from akquant import Strategy, Bar
import numpy as np

class MyStrategy(Strategy):
    def __init__(self, ma_window=20):
        super().__init__()
        self.ma_window = ma_window

    def on_start(self):
        # 显式订阅数据
        self.subscribe("600000")

    def on_bar(self, bar: Bar):
        # 1. 获取历史数据 (Online 模式)
        # 获取最近 N 个收盘价
        history = self.get_history(count=self.ma_window, symbol=bar.symbol, field="close")

        # 检查数据是否足够
        if len(history) < self.ma_window:
            return

        # 计算均线
        ma_value = np.mean(history)

        # 2. 交易逻辑
        # 获取当前持仓
        pos = self.get_position(bar.symbol)

        if bar.close > ma_value and pos == 0:
            self.buy(symbol=bar.symbol, quantity=100)
        elif bar.close < ma_value and pos > 0:
            self.sell(symbol=bar.symbol, quantity=100)
```

### 3.2 关键方法

*   **`on_start(self)`**: 策略启动入口，用于订阅数据 (`self.subscribe`) 和注册定时器。
*   **`on_bar(self, bar: Bar)`**: 核心回调，每个 Bar 到达时触发。
*   **`self.buy(symbol, quantity, price=None)`**: 发送买入指令。不指定价格则为市价单。
*   **`self.sell(symbol, quantity, price=None)`**: 发送卖出指令。
*   **`self.get_position(symbol)`**: 获取指定标的的当前持仓数量。
*   **`self.get_open_orders(symbol)`**: 获取指定标的的未成交订单列表。
*   **`self.get_history(count, symbol=None, field="close")`**: 获取历史数据序列。底层通过 Zero-Copy View 直接访问 Rust 内存，性能极高。

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

    if len(prices) < ctx.ma_window:
        return

    ma = prices.mean()

    # 获取持仓
    pos = ctx.get_position(bar.symbol)

    # 交易逻辑
    if bar.close > ma and pos == 0:
        ctx.buy(symbol=bar.symbol, quantity=100)
    elif bar.close < ma and pos > 0:
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

AKQuant 支持两种指标计算模式，开发者应根据性能需求选择。

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

# 3. 在策略中注册
class MyVectorizedStrategy(Strategy):
    def __init__(self):
        super().__init__()
        # 注册指标
        # 这里的 Indicator 包装了计算逻辑
        self.register_indicator("rsi_14", Indicator("rsi_14", calculate_rsi, timeperiod=14))

        # 支持 lambda
        self.register_indicator("ma_20", Indicator("ma_20", lambda df: df['close'].rolling(20).mean()))

# 4. 运行回测
run_backtest(
    # ... 其他参数
    strategy=MyVectorizedStrategy
)

# 5. 在策略中使用
def on_bar(self, bar):
    # 可以通过 self.indicators 获取 (需自定义逻辑) 或直接使用预计算值的缓存
    # 目前建议在 __init__ 中保存引用，或者通过 self.get_indicator(name) (若支持)
    pass
```

### 5.2 在线计算 (Online)

**推荐用于实盘、复杂逻辑验证或无法预计算的场景。**

这种模式下，在 `on_bar` 中实时获取历史序列计算指标。

*   **性能优化**: AKQuant 0.1.3+ 采用 Zero-Copy 内存映射 (Numpy View) 技术，从 Rust 端直接暴露历史数据给 Python。这意味着 `get_history` 操作几乎没有内存拷贝开销，相比传统的 Python 列表切片方式有显著性能提升。
*   **优点**: 代码简单，逻辑与实盘完全一致，无未来函数风险。
*   **缺点**: 相比向量化预计算仍有一定 Python 调用开销。
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

AKQuant 引擎内置了中国市场的交易规则支持：

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

### 6.3 目标仓位管理

AKQuant 提供了便捷的 helper 函数，允许策略直接设定目标持仓市值或百分比，引擎会自动计算并发送买卖指令。

*   **`order_target_value(target_value, symbol)`**: 将持仓调整到指定市值。
    *   `target_value`: 目标持仓金额 (正数)。
    *   示例: `self.order_target_value(50000, "AAPL")` (调整 AAPL 持仓至 5万元)。
*   **`order_target_percent(target_percent, symbol)`**: 将持仓调整到指定账户总资产比例。
    *   `target_percent`: 目标比例 (0.0 - 1.0)。
    *   示例: `self.order_target_percent(0.5, "AAPL")` (调整 AAPL 持仓至总资产的 50%)。

**注意**:
*   计算基于当前 Bar/Tick 的价格。
*   自动处理买入和卖出方向。
*   会自动向下取整以符合最小交易单位 (Lot Size)。

## 7. 全局配置 (`StrategyConfig`)

`AKQuant` 提供了一个全局配置对象 `strategy_config` (类似 PyBroker)，用于控制回测行为。虽然 `run_backtest` 封装了大部分常用配置，但在高级场景下可以直接修改它。

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

AKQuant 提供了内置的绘图工具，方便分析策略表现。

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
        pos = self.get_position(bar.symbol)

        # 如果标的价格上涨，买入期权
        if bar.close > 3.0 and pos == 0:
            # 买入 10 张期权 (假设 multiplier=10000, 相当于 10万股标的)
            self.buy(symbol=bar.symbol, quantity=10)

        # 止盈
        elif bar.close > 3.5 and pos > 0:
            self.sell(symbol=bar.symbol, quantity=10)

# 运行回测时指定期权参数
# run_backtest(..., asset_type=AssetType.Option, option_type=OptionType.Call, ...)
```

## 11. 高级特性 (Advanced Features)

### 11.1 流式数据加载 (Streaming Data)

对于 TB 级别的超大历史数据文件（如高频 Tick 数据或全市场多年的分钟线），一次性加载到内存会导致 OOM (Out of Memory)。AKQuant 提供了 `DataFeed.from_csv` 方法支持流式读取。

```python
from akquant import DataFeed, Engine

# 创建流式数据源 (只占用极少内存)
feed = DataFeed.from_csv("large_data.csv", symbol="SH600000")

# 创建引擎
engine = Engine()
engine.add_data(feed)

# 运行回测 (数据将在回测过程中逐行读取)
engine.run(strategy)
```

### 11.2 实时交易 (Live Trading)

AKQuant 0.1.3+ 支持实时模式，可以接收来自外部接口（如 CTP, TWS, IB Gateway）的数据推送，并驱动策略运行。

**关键点:**

1.  使用 `DataFeed.create_live()` 创建实时数据源。
2.  引擎会自动进入 **Live Loop** 模式：
    *   如果没有数据到达，引擎会挂起 (Wait) 而不是空转，节省 CPU。
    *   引擎会严格根据系统时钟 (Wall Clock) 触发定时器 (Timer)。
3.  `DataFeed` 的 `add_bar` / `add_tick` 方法在 Live 模式下是**线程安全**的，可以在回调线程中直接调用。

**示例: 接入模拟 CTP 数据推送**

```python
import akquant
from akquant import DataFeed, Bar, Decimal
import threading
import time

# 1. 创建实时数据源
live_feed = DataFeed.create_live()

# 2. 初始化引擎和策略
engine = akquant.Engine()
engine.add_data(live_feed)
strategy = MyLiveStrategy()

# 3. 模拟 CTP 接收线程
def ctp_receiver(feed):
    while True:
        # 模拟从 CTP 接收数据 (阻塞调用)
        # raw_data = ctp_api.recv()
        time.sleep(1) # 模拟每秒一个 Tick

        # 转换为 AKQuant Bar
        bar = Bar(
            timestamp=time.time_ns(), # 使用当前纳秒时间戳
            symbol="rb2310",
            open=Decimal(3600),
            high=Decimal(3605),
            low=Decimal(3595),
            close=Decimal(3602),
            volume=Decimal(10),
            # ...
        )

        # 推送数据 (线程安全)
        feed.add_bar(bar)
        print(f"Pushed bar: {bar.timestamp}")

# 启动接收线程
t = threading.Thread(target=ctp_receiver, args=(live_feed,))
t.daemon = True
t.start()

# 4. 运行引擎 (阻塞主线程)
# 引擎将持续运行，处理推送过来的数据
engine.run(strategy)
```

## 12. 常见问题 (FAQ)

*(待补充)*
