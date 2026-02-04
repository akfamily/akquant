# 示例集合

## 1. 基础示例 (Basic Examples)

*   [快速开始 (Quickstart)](quickstart.md): 包含手动数据回测和 AKShare 数据回测的完整流程。
*   [简单的均线策略 (SMA Strategy)](strategy_guide.md#3-编写类风格策略-class-based): 展示了如何使用类风格编写策略，并在 `on_bar` 中进行简单的交易逻辑。

## 2. 进阶示例 (Advanced Examples)

*   **Zipline 风格策略**: 展示了如何使用函数式 API (`initialize`, `on_bar`) 编写策略，适合从 Zipline 迁移的用户。
    *   参考 [策略指南](strategy_guide.md#4-编写函数风格策略-functional)。

*   **多品种回测 (Multi-Asset)**:
    *   **期货策略**: 展示期货回测配置（保证金、乘数）。参考 [快速开始](quickstart.md#4-多品种回测-期货期权)。
    *   **期权策略**: 展示期权回测配置（权利金、按张收费）。参考 [策略指南](strategy_guide.md#10-多品种策略示例)。

*   **向量化指标 (Vectorized Indicators)**:
    *   展示如何使用 `IndicatorSet` 预计算指标以提高回测速度。参考 [策略指南](strategy_guide.md#51-向量化预计算-推荐---indicatorset)。

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

## 4. 更多资源

*   查看 `examples/` 目录下的源代码获取更多实用示例。
*   阅读 [API 文档](api.md) 了解详细接口定义。
