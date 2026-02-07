# 策略编写指南

本文档旨在帮助策略开发者快速掌握 AKQuant 的策略编写方法。

## 1. 核心概念 (Glossary)

对于量化交易的新手，这里有一些基础术语的解释：

*   **Bar (K线)**: 包含了某一段时间（如1分钟、1天）内的市场行情，主要包含 5 个数据：
    *   **Open**: 开盘价
    *   **High**: 最高价
    *   **Low**: 最低价
    *   **Close**: 收盘价
    *   **Volume**: 成交量
*   **Strategy (策略)**: 你的交易机器人。它的核心工作就是不断地看行情 (on_bar)，然后决定买 (buy) 还是卖 (sell)。
*   **Context (上下文)**: 机器人的“记事本”和“工具箱”。里面记录了当前有多少钱 (cash)、有多少股票 (positions)，也提供了下单的工具。
*   **Position (持仓)**: 你当前持有的股票或期货数量。正数表示多头（买入持有），负数表示空头（借券卖出）。
*   **Backtest (回测)**: 历史模拟。用过去的数据来测试你的策略，看看如果过去这么做，能赚多少钱。

## 2. 策略生命周期

一个策略从开始到结束，会经历以下几个阶段：
    *   `__init__`: Python 对象初始化，适合定义参数。
    *   `on_start`: 策略启动时调用，**必须**在此处使用 `self.subscribe()` 订阅数据，也可在此注册指标。
    *   `on_bar`: 每一根 K 线闭合时触发 (核心交易逻辑)。
    *   `on_tick`: 每一个 Tick 到达时触发 (高频/盘口策略)。
    *   `on_order`: 订单状态变化时触发 (如提交、成交、取消)。
    *   `on_trade`: 收到成交回报时触发。
    *   `on_timer`: 定时器触发时调用 (需手动注册)。
    *   `on_stop`: 策略停止时调用，适合进行资源清理或结果统计 (参考 Backtrader `stop` / Nautilus `on_stop`)。
    *   `on_train_signal`: 滚动训练触发信号 (仅在 ML 模式下触发)。

## 2. 策略风格选择 {: #style-selection }

AKQuant 提供了两种风格的策略开发接口：

| 特性 | 类风格 (推荐) | 函数风格 |
| :--- | :--- | :--- |
| **定义方式** | 继承 `akquant.Strategy` | 定义 `initialize` 和 `on_bar` 函数 |
| **适用场景** | 复杂策略、需要维护内部状态、生产环境 | 快速原型验证、迁移 Zipline/Backtrader 策略 |
| **代码结构** | 面向对象，逻辑封装性好 | 脚本化，简单直观 |
| **API 调用** | `self.buy()`, `self.ctx` | `ctx.buy()`, `ctx` 作为参数传递 |

## 3. 编写类风格策略 (Class-based) {: #class-based }

这是 AKQuant 推荐的策略编写方式，结构清晰，易于扩展。

```python
from akquant import Strategy, Bar
import numpy as np

class MyStrategy(Strategy):
    def __init__(self, ma_window=20):
        # 注意: Strategy 类使用了 __new__ 进行初始化，子类不再需要调用 super().__init__()
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

## 4. 订单与交易详解 (Orders & Execution)

### 4.1 订单生命周期

在 AKQuant 中，订单状态流转如下：

1.  **New**: 订单对象被创建。
2.  **Submitted**: 订单已发送给交易所/仿真撮合引擎。
3.  **Accepted**: (实盘模式) 交易所确认接收订单。
4.  **Filled**: 订单全部成交。
    *   **PartiallyFilled**: 部分成交 (目前状态码统一为 Filled，需通过 `filled_quantity` 判断)。
5.  **Cancelled**: 订单已取消。
6.  **Rejected**: 订单被风控或交易所拒绝 (如资金不足、超出涨跌停)。

### 4.2 常用交易指令

*   **市价单 (Market Order)**:
    ```python
    self.buy(symbol="AAPL", quantity=100) # 市价买入
    self.sell(symbol="AAPL", quantity=100) # 市价卖出
    ```
*   **限价单 (Limit Order)**:
    指定价格成交，只有当市场价格达到或优于指定价格时才成交。
    ```python
    self.buy(symbol="AAPL", quantity=100, price=150.0) # 限价 150 买入
    ```
*   **止损单 (Stop Order)**:
    当市价触及触发价 (`trigger_price`) 时，转化为市价单。
    ```python
    # 当价格跌破 140 时，市价卖出止损
    self.stop_sell(symbol="AAPL", quantity=100, trigger_price=140.0)
    ```
*   **目标仓位 (Target Orders)**:
    自动计算买卖数量，将仓位调整到目标值。
    ```python
    # 调整仓位到总资产的 50%
    self.order_target_percent(target_percent=0.5, symbol="AAPL", price=None)

    # 调整持仓到 1000 股 (如果是 0 则买入 1000，如果是 2000 则卖出 1000)
    self.order_target_value(target_value=1000 * price, symbol="AAPL") # 注意这里 API 暂未直接支持 target_share，可用 value 模拟
    ```

### 4.3 撮合模式

通过 `engine.set_execution_mode(mode)` 设置（或在 `run_backtest` 中传入 `execution_mode` 参数）：

*   **NextOpen (默认)**: 信号在下一个 Bar 的开盘时撮合。这是更严谨的回测方式，符合实盘逻辑（今收盘后挂单，明开盘撮合）。
*   **CurrentClose**: 信号在当前 Bar 收盘时立即撮合。适合利用收盘价进行结算的特殊策略，或者无法获取次日数据的场景。

### 4.4 事件回调 (Event Callbacks) {: #callbacks }

AKQuant 提供了类似 Backtrader 的回调机制，用于追踪订单状态和成交记录。

#### 4.4.1 订单状态回调 (`on_order`)

当订单状态发生变化（如从 `New` 变为 `Submitted`，或变为 `Filled`）时触发。

```python
from akquant import OrderStatus

def on_order(self, order):
    if order.status == OrderStatus.Filled:
        print(f"订单成交: {order.symbol} 方向: {order.side} 数量: {order.filled_quantity}")
    elif order.status == OrderStatus.Cancelled:
        print(f"订单已取消: {order.id}")
```

#### 4.4.2 成交回报回调 (`on_trade`)

当发生真实成交时触发。与 `on_order` 不同，`on_trade` 包含具体的成交价格、数量和手续费信息。

```python
def on_trade(self, trade):
    print(f"成交回报: {trade.symbol} 价格: {trade.price} 数量: {trade.quantity} 手续费: {trade.commission}")
```

## 5. 风险控制 (Risk Management)

AKQuant 内置了 Rust 层面的风控管理器，可以在回测中模拟交易所或券商的风控规则。

```python
from akquant import RiskConfig

# 在 Engine 初始化后设置
risk_config = RiskConfig()
risk_config.active = True
risk_config.max_order_value = 1_000_000.0  # 单笔最大 100万
risk_config.max_position_size = 5000       # 单标的最大持仓 5000 股
risk_config.restricted_list = ["ST股票"]    # 黑名单 (Symbol)

engine.risk_manager.config = risk_config # 应用配置
```

如果订单违反风控规则，`self.buy()` 等函数会返回 `None` 或生成的订单状态直接为 `Rejected`，并会在日志中记录原因。

## 6. 使用高性能指标 (Built-in Indicators) {: #indicatorset }

AKQuant 在 Rust 层内置了常用的技术指标，通过增量计算 (Incremental Calculation) 避免了重复的全量计算，性能极高。

支持的指标: `SMA`, `EMA`, `MACD`, `RSI`, `BollingerBands`, `ATR`.

### 6.1 注册与使用

```python
from akquant import Strategy
from akquant.indicators import SMA, RSI

class IndicatorStrategy(Strategy):
    def on_start(self):
        self.subscribe("AAPL")

        # 注册指标: 自动处理数据更新
        # 参数: period=20
        self.register_indicator("sma20", SMA(20))

        # 注册 RSI
        self.register_indicator("rsi14", RSI(14))

    def on_bar(self, bar):
        # 直接访问指标值
        # 注意: 如果指标尚未准备好 (数据不足)，value 可能为 None
        sma_val = self.sma20.value
        rsi_val = self.rsi14.value

        if sma_val is None or rsi_val is None:
            return

        if bar.close > sma_val and rsi_val < 30:
            self.buy(bar.symbol, 100)
```

## 7. 常用策略模式 (Cookbook)

### 7.1 移动止损 (Trailing Stop)

```python
class TrailingStopStrategy(Strategy):
    def __init__(self):
        self.highest_price = 0.0
        self.trailing_percent = 0.05 # 5% 回撤止损

    def on_bar(self, bar):
        pos = self.get_position(bar.symbol)

        if pos > 0:
            # 更新最高价
            self.highest_price = max(self.highest_price, bar.high)

            # 检查回撤
            drawdown = (self.highest_price - bar.close) / self.highest_price
            if drawdown > self.trailing_percent:
                print(f"触发移动止损: 最高 {self.highest_price}, 当前 {bar.close}")
                self.close_position(bar.symbol)
                self.highest_price = 0.0 # 重置
        else:
            # 开仓逻辑 (示例)
            if bar.close > 100:
                self.buy(bar.symbol, 100)
                self.highest_price = bar.close # 初始化最高价
```

### 7.2 定时平仓 (Intraday Exit)

```python
class IntradayStrategy(Strategy):
    def on_bar(self, bar):
        # 假设 bar.timestamp 是纳秒时间戳
        # 转换为 datetime (需要 import datetime)
        dt = datetime.fromtimestamp(bar.timestamp / 1e9)

        # 每天 14:55 强制平仓
        if dt.hour == 14 and dt.minute >= 55:
            if self.get_position(bar.symbol) != 0:
                self.close_position(bar.symbol)
            return

        # 其他交易逻辑...
```

### 7.3 多品种轮动 {: #multi-asset }

```python
class RotationStrategy(Strategy):
    def on_bar(self, bar):
        # 注意: on_bar 是针对每个 symbol 触发的
        # 如果需要全市场横截面比较，建议在 on_timer 或 收集完所有 bar 后处理
        # 这里展示简单的独立处理
        pass

    def on_timer(self, payload):
        # 假设注册了一个每日定时器
        # 获取所有订阅标的的当前价格
        scores = {}
        for symbol in self.ctx.portfolio.positions.keys(): # 仅示例，实际应遍历关注列表
             hist = self.get_history(20, symbol)
             scores[symbol] = hist[-1] / hist[0] # 20日动量

        # 排序并调仓...
```

## 8. 多资产混合回测配置 (Mixed Asset Configuration)

AKQuant 支持在同一个策略中混合交易股票、期货、期权等多种资产。不同资产通常具有不同的属性（如合约乘数、保证金率、最小变动价位）。

使用 `InstrumentConfig` 可以方便地为每个标的配置这些属性。

### 8.1 配置步骤

1.  **准备数据**: 为每个标的准备数据 (DataFrame 或 CSV)。
2.  **创建配置**: 使用 `InstrumentConfig` 定义非股票资产的参数。
3.  **运行回测**: 将配置传递给 `run_backtest` 的 `instruments_config` 参数。

### 8.2 配置示例

假设我们要回测一个包含 "股票 A" 和 "股指期货 IF" 的投资组合：

```python
from akquant import InstrumentConfig, run_backtest

# 1. 定义期货配置
future_config = InstrumentConfig(
    symbol="IF2301",          # 标的代码
    asset_type="FUTURES",     # 资产类型: STOCK, FUTURES, OPTION
    multiplier=300.0,         # 合约乘数 (1点 300元)
    margin_ratio=0.1,         # 保证金率 (10%)
    tick_size=0.2             # 最小变动价位
)

# 2. 运行回测
# 注意: 未配置的标的 (如 STOCK_A) 将使用默认参数 (股票, 乘数1, 保证金100%)
run_backtest(
    data=data_dict,
    strategy=MyStrategy,
    instruments_config=[future_config], # 传入配置列表
    # ...
)
```

详细代码请参考 [混合资产回测示例](examples.md#mixed-asset)。
