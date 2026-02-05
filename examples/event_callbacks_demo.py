"""
AKQuant Event Callbacks Demo.

===========================

这个示例演示了如何使用 AKQuant 的事件回调功能：
1. `on_order`: 监听订单状态变化（New, Submitted, Filled, Cancelled 等）。
2. `on_trade`: 监听成交回报。

Author: AKQuant Team
"""

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, OrderStatus, Strategy


# 1. 创建模拟数据
def generate_mock_data(days: int = 100) -> pd.DataFrame:
    """生成简单的模拟行情数据."""
    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")

    # 简单的正弦波趋势 + 随机噪声
    x = np.linspace(0, 4 * np.pi, days)
    prices = 100 + 10 * np.sin(x) + np.random.normal(0, 1, days)

    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": "MOCK_STOCK",
            "open": prices,
            "high": prices + 2,
            "low": prices - 2,
            "close": prices,
            "volume": 100000,
        }
    )
    return df


# 2. 定义策略
class EventCallbacksStrategy(Strategy):
    """示例策略，演示事件回调功能."""

    def on_start(self) -> None:
        """策略启动回调."""
        print("\n=== 策略启动 ===")
        # 注意：on_start 阶段 self.ctx 尚未初始化，无法访问资金等信息
        self.subscribe("MOCK_STOCK")
        self.order_count = 0

    def on_bar(self, bar: Bar) -> None:
        """K线数据回调."""
        # 简单的交易逻辑：
        # - 没有持仓时买入
        # - 有持仓且盈利超过 2% 时卖出

        position = self.get_position(bar.symbol)

        if position == 0:
            # 每隔 10 天买入一次，避免频繁交易
            if self.order_count % 10 == 0:
                dt = pd.to_datetime(bar.timestamp, unit="ns")
                print(f"\n[Strategy] 触发买入信号 @ {dt} 价格: {bar.close:.2f}")
                # 使用限价单买入，价格比当前收盘价略高，保证成交
                self.buy(bar.symbol, 100, price=bar.close * 1.01)
            self.order_count += 1

        elif position > 0:
            # 持仓检查
            if self.ctx:
                # 简化逻辑，实际应从 Trade 记录获取
                _ = self.ctx.positions.get(bar.symbol, 0)

            # 这里简单演示，直接在持有 5 天后卖出
            if self.order_count % 10 == 5:
                dt = pd.to_datetime(bar.timestamp, unit="ns")
                print(f"\n[Strategy] 触发卖出信号 @ {dt} 价格: {bar.close:.2f}")
                self.sell(bar.symbol, 100)  # 市价卖出
            self.order_count += 1

    def on_order(self, order: Any) -> None:
        """订单状态变化回调."""
        emoji = "❓"
        if order.status == OrderStatus.New:
            emoji = "🆕"
        elif order.status == OrderStatus.Submitted:
            emoji = "📨"
        elif order.status == OrderStatus.Filled:
            emoji = "✅"
        elif order.status == OrderStatus.Cancelled:
            emoji = "❌"
        elif order.status == OrderStatus.Rejected:
            emoji = "🚫"
        elif order.status == OrderStatus.Expired:
            emoji = "⏰"

        # order.status.name 可能不可用，直接打印 order.status
        print(
            f"[Callback] on_order {emoji} | "
            f"ID: {order.id[:8]}... | "
            f"Symbol: {order.symbol} | "
            f"Side: {order.side} | "
            f"Status: {order.status} | "
            f"Qty: {order.filled_quantity}/{order.quantity}"
        )

    def on_trade(self, trade: Any) -> None:
        """成交回报回调."""
        print(
            f"[Callback] on_trade 💰 | "
            f"Time: {trade.timestamp} | "
            f"Symbol: {trade.symbol} | "
            f"Side: {trade.side} | "
            f"Price: {trade.price:.2f} | "
            f"Qty: {trade.quantity} | "
            f"Comm: {trade.commission:.2f}"
        )


# 3. 运行回测
def main() -> None:
    """运行回测主函数."""
    print("生成数据...")
    df = generate_mock_data()

    print("开始回测...")
    results = aq.run_backtest(
        strategy=EventCallbacksStrategy,
        data=df,
        symbol="MOCK_STOCK",
        cash=100_000,
        commission=0.0003,  # 万三佣金
    )

    print("\n=== 回测结束 ===")
    print("最终绩效指标:")
    print(results.metrics_df)


if __name__ == "__main__":
    main()
