import numpy as np
import pandas as pd
from akquant import Bar, Strategy
from akquant.backtest import run_backtest


# 1. 定义策略
class SmaStrategy(Strategy):
    """A simple SMA strategy."""

    def on_bar(self, bar: Bar) -> None:
        """Handle bar events."""
        # 简单策略：价格高于均价时买入，否则卖出
        # 注意：实际中建议使用 IndicatorSet 进行向量化计算
        position = self.get_position(bar.symbol)

        if position == 0:
            self.buy(symbol=bar.symbol, quantity=100)
        # 简单的止盈逻辑：如果当前价格 > 买入均价 * 1.1，则卖出
        # 注意：这里简化处理，假设只有一个持仓方向
        elif position > 0:
            # 获取持仓成本需要从 context 的 positions 或者 orders 推算
            # 暂时简化为只要有持仓且价格 > 昨收 * 1.1 (模拟)
            # 或者我们直接用 bar.open 作为参考
            if bar.close > bar.open:  # 简化逻辑
                self.sell(symbol=bar.symbol, quantity=100)


# 2. 生成模拟数据
def generate_mock_data(
    start_date: str = "20230101", end_date: str = "20280630"
) -> pd.DataFrame:
    """Generate mock data for testing."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    # 随机生成价格走势
    np.random.seed(42)
    price = 10.0 + np.cumsum(np.random.randn(n) * 0.1)

    data = pd.DataFrame(
        {
            "open": price + np.random.randn(n) * 0.05,
            "high": price + 0.2,
            "low": price - 0.2,
            "close": price,
            "volume": np.random.randint(1000, 10000, n),
        },
        index=dates,
    )

    return data


# 3. 运行回测
# run_backtest 需要传入 data 参数
# 我们这里使用生成的模拟数据
data = generate_mock_data()

result = run_backtest(
    data=data,
    strategy=SmaStrategy,
    symbol="600000",  # 浦发银行
    cash=500_000.0,  # 初始资金
    commission=0.0003,  # 万三佣金
    stamp_tax=0.0005,  # 印花税 (A股默认千一，可覆盖)
)

# 4. 查看结果
print(f"Total Return: {result.metrics.total_return:.2%}")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.2%}")
print(f"Total Trades (Closed): {len(result.trades)}")
result.daily_positions_df
# Print Daily Positions Sample
if hasattr(result, "daily_positions_df"):
    print("\nDaily Positions DataFrame (Last 5 days):")
    print(result.daily_positions_df.tail(5))
else:
    print("daily_positions_df not available")

# Print Metrics DataFrame
if hasattr(result, "metrics_df"):
    print("\nMetrics DataFrame:")
    # Transpose for better readability when printing a single row with many columns
    print(result.metrics_df.T)
else:
    print("metrics_df not available")

# Print Trades DataFrame
if hasattr(result, "trades_df"):
    print("\nTrades DataFrame (First 5):")
    print(result.trades_df.head(5))
else:
    print("trades_df not available")

# Check if equity_curve is list or Series
