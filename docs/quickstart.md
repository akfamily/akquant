# 快速开始

本指南将带您快速上手 `AKQuant`，从简单的手动数据回测到使用 Pandas DataFrame 数据回测。

## 1. 基础示例 (手动数据)

这个示例展示了如何创建一个简单的策略并使用手动构造的数据进行回测。

```python
import akquant
from akquant import Engine, Strategy, DataFeed, Bar, AssetType, Instrument
from datetime import datetime

# 1. 定义策略
class MyStrategy(Strategy):
    def on_bar(self, bar):
        # 简单的双均线逻辑模拟
        # 实际开发中可以使用 akquant.Indicator 或 talib 计算指标

        # 获取当前持仓
        position = self.get_position(bar.symbol)

        # 简单的价格突破逻辑
        if bar.close > 100.0 and position == 0:
            print(f"[{datetime.fromtimestamp(bar.timestamp / 1e9)}] 价格 {bar.close} > 100, 买入")
            self.buy(symbol=bar.symbol, quantity=100.0) # 使用默认 symbol

        elif bar.close <= 100.0 and position > 0:
            print(f"[{datetime.fromtimestamp(bar.timestamp / 1e9)}] 价格 {bar.close} <= 100, 卖出")
            self.sell(symbol=bar.symbol, quantity=float(position))

# 2. 准备环境
engine = Engine()
# 设置时区偏移 (UTC+8)
engine.set_timezone(28800)
# 或者使用 Python 封装层的 set_timezone_name (如果已 patch)
# engine.set_timezone_name("Asia/Shanghai")

feed = DataFeed()

# 3. 添加合约 (股票)
# Instrument 参数: symbol, asset_type, multiplier, margin_ratio, tick_size, option_type, strike_price, expiry_date, lot_size
engine.add_instrument(Instrument("AAPL", AssetType.Stock, 1.0, 0.01, 0.01, None, None, None, None))

# 4. 添加数据
feed.add_bar(Bar(
    timestamp="2023-01-01",
    open=95.0, high=105.0, low=90.0, close=102.0,
    volume=1000.0, symbol="AAPL"
))
feed.add_bar(Bar(
    timestamp="2023-01-02",
    open=101.0, high=103.0, low=98.0, close=99.0,
    volume=1200.0, symbol="AAPL"
))
engine.add_data(feed)

# 5. 运行回测
print("开始回测...")
summary = engine.run(MyStrategy(), show_progress=True)
print(summary)

# 6. 查看结果
result = engine.get_results()
print(f"Total Return: {result.metrics.total_return_pct:.2f}%")

# 获取 DataFrame 格式结果 (如果有)
# print("Trades:", result.trades)
```

## 2. 进阶示例 (Pandas DataFrame 数据)

这个示例展示了如何结合 `DataLoader` 和便捷函数 `run_backtest` 快速运行基于 Pandas DataFrame 数据的回测。

```python
import pandas as pd
import akquant
from akquant.backtest import run_backtest
from akquant import Strategy
from akquant.config import BacktestConfig

# 1. 准备数据
dates = pd.date_range(start="2023-01-01", end="2023-06-30")
data = {
    "date": dates,
    "open": [10.0 + i * 0.1 for i in range(len(dates))],
    "high": [10.5 + i * 0.1 for i in range(len(dates))],
    "low": [9.5 + i * 0.1 for i in range(len(dates))],
    "close": [10.2 + i * 0.1 for i in range(len(dates))],
    "volume": [1000.0 for _ in range(len(dates))],
    "symbol": ["600000"] * len(dates)
}
df = pd.DataFrame(data)

# 2. 定义策略
class SmaStrategy(Strategy):
    def on_start(self):
        # 显式订阅数据
        self.subscribe("600000")

    def on_bar(self, bar):
        # 简单策略
        # 获取当前持仓
        pos = self.get_position(bar.symbol)

        if pos == 0:
            self.buy(symbol=bar.symbol, quantity=100)
        elif bar.close > bar.open * 1.05: # 涨幅超过 5% 止盈
            self.sell(symbol=bar.symbol, quantity=100)

# 3. 配置回测
config = BacktestConfig(
    start_date="20230101",
    end_date="20230630",
    cash=500_000.0,
    commission=0.0003,
)

# 4. 运行回测
# 直接传入 DataFrame 字典，key 为 symbol
result = run_backtest(
    strategy_cls=SmaStrategy,
    config=config,
    data={"600000": df}
)

# 5. 查看结果
print("Annualized Return:", result.metrics.annualized_return)
result.plot()
```

## 3. 自定义数据源 (手动加载)

如果您希望手动控制数据下载过程，或者使用 `run_backtest` 以外的数据源，可以使用以下方式手动加载数据。

```python
import pandas as pd
import akquant
from akquant import Engine, Strategy, DataFeed, Instrument, AssetType
from akquant.utils import load_bar_from_df

# 1. 加载数据
# 假设我们有一个 DataFrame (例如从 CSV 读取)
# df = pd.read_csv("data.csv")
# 这里我们构造一个简单的示例数据
dates = pd.date_range("20230101", "20231231")
df = pd.DataFrame({
    "date": dates,
    "open": 100.0,
    "high": 105.0,
    "low": 95.0,
    "close": 101.0,
    "volume": 10000
})

# 2. 定义策略
class MyStrategy(Strategy):
    def on_bar(self, bar):
        # 简单的示例逻辑
        pos = self.ctx.position.size

        # 连涨买入，连跌卖出 (仅作演示)
        if bar.close > bar.open and pos == 0:
            self.buy(symbol=bar.symbol, quantity=100)

        elif bar.close < bar.open and pos > 0:
            self.sell(symbol=bar.symbol, quantity=pos)

# 3. 初始化引擎
engine = Engine()

# 4. 注册标的 (A股 T+1 模式)
# 贵州茅台, 股票类型, 乘数1.0, 最小变动价位0.01
inst = Instrument("600519", AssetType.Stock, 1.0, 0.01)
engine.add_instrument(inst)

# 5. 加载数据到引擎
# 使用 load_bar_from_df 转换 DataFrame
# 可选: 指定列映射 (如果 CSV 列名不同)
# column_map = {"Date": "timestamp", "Close": "close", ...}
bars = load_bar_from_df(df, symbol="600519")

feed = DataFeed()
# 批量添加 Bar (比循环添加更快)
feed.add_bars(bars)
engine.add_data(feed)

# 6. 配置资金与费率
engine.set_cash(1_000_000.0)             # 初始资金 100万
engine.set_stock_fee_rules(commission_rate=0.0003, stamp_tax=0.0005, transfer_fee=0.00001, min_commission=5.0)

# 7. 运行回测
print("开始回测...")
result = engine.run(MyStrategy())

# 8. 输出结果
print("-" * 30)
print(f"Total Return: {result.metrics.total_return:.2%}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.2%}")
print("-" * 30)
```

## 4. 多品种回测 (期货/期权)

`AKQuant` 支持股票、基金、期货、期权等多种资产类型的回测，并内置了相应的交易规则（如股票买入 100 股限制、期权按张收费等）。

### 期货回测示例

```python
from akquant import AssetType

# 假设 df 已经加载了期货数据
result = run_backtest(
    data=df,
    strategy=MyFuturesStrategy,
    symbol="IF2306",
    asset_type=AssetType.Futures,
    multiplier=300.0,      # 合约乘数
    margin_ratio=0.12,     # 保证金比率
    commission=0.000023,   # 期货佣金
    # ...
)
```

### 期权回测示例

```python
from akquant import AssetType, OptionType

# 假设 df 已经加载了期权数据
result = run_backtest(
    data=df,
    strategy=MyOptionStrategy,
    symbol="10005555",
    asset_type=AssetType.Option,
    option_type=OptionType.Call,
    strike_price=3.5,
    option_commission=5.0, # 每张 5 元
    multiplier=10000.0,
    # ...
)
```

## 5. 函数式 API (Zipline 风格)

如果你更喜欢函数式编程，或者需要迁移 Zipline 策略，可以使用 `initialize` 和 `on_bar` 函数：

```python
from akquant.backtest import run_backtest
from akquant import DataLoader

# 获取数据
loader = DataLoader()
df = loader.load_akshare("600000", "20230101", "20230630")

def initialize(ctx):
    # 初始化全局变量
    ctx.stop_loss = 0.05

def on_bar(ctx, bar):
    # 获取持仓
    pos = ctx.get_position(bar.symbol)

    if pos == 0:
        ctx.buy(symbol=bar.symbol, quantity=100)
    elif bar.close < bar.open * (1 - ctx.stop_loss):
        ctx.sell(symbol=bar.symbol, quantity=100)

run_backtest(
    data=df,
    strategy=on_bar,
    initialize=initialize,
    symbol="600000",
    cash=1_000_000.0
)
```

## 6. 结果分析

`AKQuant` 提供了方便的 DataFrame 接口用于分析回测结果：

| 属性 | 描述 |
| :--- | :--- |
| `result.metrics_df` | 包含所有绩效指标 (Sharpe, Sortino, Return, etc.) 的 DataFrame |
| `result.trades_df` | 包含每一笔已平仓交易的详细信息 (Entry/Exit, PnL, Commission) |
| `result.daily_positions_df` | 每日持仓快照 |
| `result.equity_curve` | 权益曲线数据列表 |

```python
# 打印绩效概览
print(result.metrics_df.T)

# 分析亏损交易
trades = result.trades_df
losing_trades = trades[trades['net_pnl'] < 0]
print("Average Loss:", losing_trades['net_pnl'].mean())
```

## 下一步

*   查看 [API 参考](api.md) 了解更多细节。
*   浏览 [示例](examples.md) 学习更复杂的策略，包括指标计算和高级订单。
