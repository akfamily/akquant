import akquant as aq
import akshare as ak
import pandas as pd
from akquant import Bar, Strategy

# 1. 准备数据
# 使用 akshare 获取 A 股历史数据 (需安装: pip install akshare)
df = ak.stock_zh_a_daily(symbol="sh600000", start_date="20230101", end_date="20231231")


class MyStrategy(Strategy):
    """Simple demo strategy."""

    def on_bar(self, bar: Bar) -> None:
        """Execute on every bar."""
        # 简单策略示例:
        # 当收盘价 > 开盘价 (阳线) -> 买入
        # 当收盘价 < 开盘价 (阴线) -> 卖出

        # 获取当前持仓
        current_pos = self.get_position(bar.symbol)

        if current_pos == 0 and bar.close > bar.open:
            self.buy(bar.symbol, 100)
            # print(f"[{bar.timestamp_str}] Buy 100 at {bar.close:.2f}")

        elif current_pos > 0 and bar.close < bar.open:
            self.close_position(bar.symbol)
            # print(f"[{bar.timestamp_str}] Sell 100 at {bar.close:.2f}")


# 运行回测
result = aq.run_backtest(data=df, strategy=MyStrategy, symbol="sh600000")


pd.set_option("display.max_columns", None)  # 显示所有行
# 打印回测结果
print("\n=== Backtest Result ===")
print(result.positions)
print(result.positions_df)
result
# 绩效指标 (DataFrame)
result.metrics_df
# 交易明细 (DataFrame) - 包含 symbol, entry_time, pnl, return_pct 等
result.trades_df
# 订单记录 (DataFrame) - 包含 order_id, status, filled_quantity, avg_price 等
result.orders_df
# 持仓记录 (DataFrame) - 包含 date, symbol, long_shares, equity, market_value 等
result.positions_df
result.positions
