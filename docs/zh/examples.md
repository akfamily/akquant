# 示例集合

## 1. 基础示例 (Basic Examples)

*   [快速开始 (Quickstart)](quickstart.md): 包含手动数据回测和 AKShare 数据回测的完整流程。
*   [简单的均线策略 (SMA Strategy)](strategy_guide.md#class-based): 展示了如何使用类风格编写策略，并在 `on_bar` 中进行简单的交易逻辑。

## 2. 进阶示例 (Advanced Examples)

*   **Zipline 风格策略**: 展示了如何使用函数式 API (`initialize`, `on_bar`) 编写策略，适合从 Zipline 迁移的用户。
    *   参考 [策略指南](strategy_guide.md#style-selection)。

*   **多品种回测 (Multi-Asset)**:
    *   **期货策略**: 展示期货回测配置（保证金、乘数）。参考 [策略指南](strategy_guide.md#multi-asset)。
    *   **期权策略**: 展示期权回测配置（权利金、按张收费）。参考 [策略指南](strategy_guide.md#multi-asset)。

*   **向量化指标 (Vectorized Indicators)**:
    *   展示如何使用 `IndicatorSet` 预计算指标以提高回测速度。参考 [策略指南](strategy_guide.md#indicatorset)。

## 3. 常用策略示例 (Common Strategies)

以下是一些常用量化策略的实现代码，可以直接在您的项目中使用。

### 3.1 双均线策略 (Dual Moving Average)

经典的趋势跟踪策略，利用长短周期的移动平均线交叉产生买卖信号。本示例使用了 Rust 实现的高性能增量指标 `aq.SMA`。

```python
import akquant as aq

class DualSMAStrategy(aq.Strategy):
    def __init__(self, short_window=5, long_window=20):
        # 使用 Rust 实现的高性能增量 SMA 指标
        self.sma_short = aq.SMA(short_window)
        self.sma_long = aq.SMA(long_window)

    def on_bar(self, bar: aq.Bar):
        # 更新指标
        short_val = self.sma_short.update(bar.close)
        long_val = self.sma_long.update(bar.close)

        # 指标未就绪则返回
        if short_val is None or long_val is None:
            return

        position = self.get_position(bar.symbol)

        # 金叉 (短均线上穿长均线) -> 买入
        if short_val > long_val and position == 0:
            self.buy(bar.symbol, 100)

        # 死叉 (短均线下穿长均线) -> 卖出平仓
        elif short_val < long_val and position > 0:
            self.sell(bar.symbol, 100)
```

### 3.2 RSI 均值回归策略 (RSI Mean Reversion)

利用相对强弱指标 (RSI) 判断超买超卖。本示例展示了如何利用 `get_history_df` 结合 pandas 计算复杂指标。

```python
import akquant as aq
import pandas as pd
import numpy as np

class RSIStrategy(aq.Strategy):
    def __init__(self, period=14, buy_threshold=30, sell_threshold=70):
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        # 设置足够的历史数据回溯以计算 RSI (period + 预热数据)
        self.set_history_depth(period + 20)

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """使用 pandas 计算 RSI."""
        delta = prices.diff()
        # 简单的 RSI 算法
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def on_bar(self, bar: aq.Bar):
        # 获取历史收盘价 DataFrame
        history = self.get_history_df(self.period + 20, bar.symbol)

        # 数据不足时返回
        if len(history) < self.period + 1:
            return

        # 计算 RSI
        rsi_series = self.calculate_rsi(history['close'])
        current_rsi = rsi_series.iloc[-1]

        if np.isnan(current_rsi):
            return

        position = self.get_position(bar.symbol)

        # RSI < 30 (超卖) -> 买入
        if current_rsi < self.buy_threshold and position == 0:
            self.buy(bar.symbol, 100)

        # RSI > 70 (超买) -> 卖出
        elif current_rsi > self.sell_threshold and position > 0:
            self.sell(bar.symbol, 100)
```

### 3.3 布林带策略 (Bollinger Bands)

利用布林带上下轨作为交易信号。本示例展示了如何通过 pandas 计算统计指标。

```python
import akquant as aq
import pandas as pd

class BollingerStrategy(aq.Strategy):
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
        # 设置历史数据回溯
        self.set_history_depth(window + 5)

    def on_bar(self, bar: aq.Bar):
        # 获取历史数据
        history = self.get_history_df(self.window, bar.symbol)
        if len(history) < self.window:
            return

        # 计算布林带
        close_prices = history['close']
        ma = close_prices.mean()
        std = close_prices.std()
        upper_band = ma + self.num_std * std
        lower_band = ma - self.num_std * std

        position = self.get_position(bar.symbol)
        current_price = bar.close

        # 价格跌破下轨 -> 视为超卖反转信号 -> 买入
        if current_price < lower_band and position == 0:
            self.buy(bar.symbol, 100)
        # 价格突破上轨 -> 视为超买反转信号 -> 卖出
        elif current_price > upper_band and position > 0:
            self.sell(bar.symbol, 100)
```

### 3.4 混合资产回测 (Mixed Asset Backtest) {: #mixed-asset }

展示如何在同一个策略中混合交易股票和期货，并使用 `InstrumentConfig` 配置期货参数。

```python
import akquant as aq
from akquant import InstrumentConfig
import pandas as pd
import numpy as np

# 1. 准备数据 (模拟数据)
def create_dummy_data(symbol, start_date, n_bars, price=100.0):
    dates = pd.date_range(start_date, periods=n_bars, freq="B")
    np.random.seed(42)
    changes = np.random.randn(n_bars)
    prices = price + np.cumsum(changes)

    df = pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1, "close": prices,
        "volume": 1000, "symbol": symbol
    }, index=dates)
    return df

class TestStrategy(aq.Strategy):
    def __init__(self):
        self.count = 0

    def on_bar(self, bar: aq.Bar):
        # 简单逻辑: 前两根 Bar 买入
        if self.count < 2:
            print(f"[{bar.timestamp}] Buying {bar.symbol}")
            self.buy(bar.symbol, 1)
        self.count += 1

# 2. 生成数据
df_stock = create_dummy_data("STOCK_A", "2023-01-01", 100, 100.0)
df_future = create_dummy_data("FUTURE_B", "2023-01-01", 100, 3500.0)
data = {"STOCK_A": df_stock, "FUTURE_B": df_future}

# 3. 配置期货参数
future_config = InstrumentConfig(
    symbol="FUTURE_B",
    asset_type="FUTURES",
    multiplier=300, # 股指期货乘数
    margin_ratio=0.1 # 10% 保证金
)

# 4. 运行
run_backtest(
    data=data,
    strategy=TestStrategy,
    instruments_config=[future_config]
)
```

## 4. 复杂订单与风控 (Complex Orders) {: #complex-orders }

虽然 AKQuant 的核心撮合引擎尚未原生内置 OCO (One-Cancels-Other) 或 Bracket Order 订单类型，但你可以通过策略层的回调函数 (`on_trade`, `on_order`) 轻松实现这些高级逻辑。

### 4.1 OCO 与 Bracket Order

Bracket Order 通常包含三个部分：
1.  **Entry Order**: 进场单（如突破买入）。
2.  **Stop Loss**: 止损单（保护性卖出）。
3.  **Take Profit**: 止盈单（获利卖出）。

其中 Stop Loss 和 Take Profit 构成一组 **OCO** 订单：即如果其中一个成交，另一个应立即自动取消。

👉 **[查看完整代码示例](file:///c:/Users/albert/Documents/trae_projects/akquant/examples/complex_orders.py)**

**核心逻辑实现：**

```python
def on_trade(self, trade):
    # 1. 进场单成交 -> 立即挂止损和止盈
    if trade.order_id == self.entry_order_id:
        # 下达止损单 (Stop Market)
        self.stop_loss_id = self.sell(
            trade.symbol, trade.quantity,
            trigger_price=trade.price * 0.98, # 止损价
            price=None # None 表示触发后市价卖出
        )

        # 下达止盈单 (Limit Sell)
        self.take_profit_id = self.sell(
            trade.symbol, trade.quantity,
            price=trade.price * 1.05 # 止盈价
        )

    # 2. 止损成交 -> 取消止盈
    elif trade.order_id == self.stop_loss_id:
        self.cancel_order(self.take_profit_id)

    # 3. 止盈成交 -> 取消止损
    elif trade.order_id == self.take_profit_id:
        self.cancel_order(self.stop_loss_id)
```

!!! tip "参数优化"
    该策略的 `stop_loss_pct` 和 `take_profit_pct` 参数可以通过 `akquant.run_optimization` 进行网格搜索优化。

    ```python
    from akquant import run_optimization
    from examples.complex_orders import BracketStrategy

    param_grid = {
        "stop_loss_pct": [0.01, 0.02, 0.03],
        "take_profit_pct": [0.03, 0.05, 0.08]
    }

    results = run_optimization(BracketStrategy, param_grid, data=df)
    ```

完整代码请参考 [examples/complex_orders.py](file:///examples/complex_orders.py)。

> **注意**: `buy` / `sell` / `stop_buy` / `stop_sell` 方法都会返回唯一的 `order_id` (str)，你可以利用这个 ID 在 `on_trade` 和 `on_order` 回调中精确追踪每个订单的状态。
