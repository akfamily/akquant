from datetime import datetime

import pandas as pd
from akquant import AssetType, Bar, DataFeed, Engine, Instrument, Strategy


# 1. 定义策略
class MyStrategy(Strategy):
    """Simple strategy for testing."""

    def on_bar(self, bar: Bar) -> None:
        """Handle new bar event."""
        # 简单的双均线逻辑模拟
        # 实际开发中可以使用 akquant.Indicator 或 talib 计算指标

        # 获取当前持仓
        position = self.get_position(bar.symbol)

        # 简单的价格突破逻辑
        if bar.close > 100.0 and position == 0:
            print(
                f"[{datetime.fromtimestamp(bar.timestamp / 1e9)}] "
                f"价格 {bar.close} > 100, 买入"
            )
            self.buy(symbol=bar.symbol, quantity=100.0)  # 使用默认 symbol

        elif bar.close <= 100.0 and position > 0:
            print(
                f"[{datetime.fromtimestamp(bar.timestamp / 1e9)}] "
                f"价格 {bar.close} <= 100, 卖出"
            )
            self.sell(symbol=bar.symbol, quantity=float(position))


# 2. 准备环境
engine = Engine()
# 设置时区偏移 (UTC+8)
engine.set_timezone(28800)

feed = DataFeed()

# 3. 添加合约 (股票)
# Corrected Instrument creation
try:
    # symbol, asset_type, multiplier, margin_ratio, tick_size, option_type,
    # strike_price, expiry_date, lot_size
    inst = Instrument("AAPL", AssetType.Stock, 1.0, 0.01, 0.01, None, None, None, None)
    engine.add_instrument(inst)
except Exception as e:
    print(f"Instrument creation failed: {e}")

# 4. 添加数据
try:
    feed.add_bars(
        [
            Bar(
                timestamp=pd.Timestamp("2023-01-01").value,
                open=95.0,
                high=105.0,
                low=90.0,
                close=102.0,
                volume=1000.0,
                symbol="AAPL",
            )
        ]
    )
    feed.add_bars(
        [
            Bar(
                timestamp=pd.Timestamp("2023-01-02").value,
                open=101.0,
                high=103.0,
                low=98.0,
                close=99.0,
                volume=1200.0,
                symbol="AAPL",
            )
        ]
    )
    engine.add_data(feed)
except Exception as e:
    print(f"Bar creation failed: {e}")

# 5. 运行回测
print("开始回测...")
try:
    # show_progress is required. Returns string summary.
    summary = engine.run(MyStrategy(), show_progress=True)
    print("Summary:", summary)

    # Get results object
    result = engine.get_results()

    # 6. 查看结果
    print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
except Exception as e:
    print(f"Run failed: {e}")
