from typing import Any, Optional

import numpy as np
import pandas as pd
from akquant import Bar, Order, OrderStatus, Strategy, Trade, run_backtest


class BracketStrategy(Strategy):
    """
    Bracket Order 示例策略.

    演示如何手动实现 '进场 + 止损 + 止盈' 的 Bracket 逻辑:
    1. 当价格突破上轨时，市价买入 (Entry).
    2. 成交后，立即下达:
       - 止损单 (Stop Loss): 价格下跌 x% 卖出.
       - 止盈单 (Take Profit): 价格上涨 y% 卖出.
    3. 当止损或止盈任意一个成交时，取消另一个 (OCO 逻辑).
    """

    def __init__(
        self,
        period: int = 20,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs: Any,
    ) -> None:
        """
        初始化策略.

        :param period: 周期参数
        :param stop_loss_pct: 止损百分比
        :param take_profit_pct: 止盈百分比
        :param kwargs: 其他参数
        """
        self.period = period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # 记录订单 ID，用于管理 OCO 逻辑
        self.entry_order_id: Optional[str] = None
        self.stop_loss_order_id: Optional[str] = None
        self.take_profit_order_id: Optional[str] = None

        # 状态标记
        self.has_position = False

    def on_bar(self, bar: Bar) -> None:
        """
        K线回调.

        :param bar: 最新 K 线数据
        """
        # 1. 如果已有持仓或正在等待进场，跳过
        if self.has_position or self.entry_order_id:
            return

        # 2. 简单的演示逻辑: 只要没有持仓就买入
        # 注意: 实际策略应有更复杂的进场条件

        # 假设当前没有持仓，直接市价买入
        print(f"[{bar.timestamp_str}] 进场买入: {bar.symbol} @ {bar.close}")
        self.entry_order_id = self.buy(bar.symbol, 100)  # 返回订单 ID

    def on_trade(self, trade: Trade) -> None:
        """
        成交回调.

        在这里处理 Bracket 逻辑: 进场成交后，挂止损和止盈.
        """
        print(
            f"[{trade.timestamp}] 成交确认: {trade.side} {trade.quantity} "
            f"@ {trade.price} (ID: {trade.order_id})"
        )

        # 1. 进场单成交 -> 下达止损止盈
        if trade.order_id == self.entry_order_id:
            self.has_position = True
            self.entry_order_id = None  # 清除 ID，防止重复处理

            entry_price = trade.price
            stop_price = entry_price * (1 - self.stop_loss_pct)
            limit_price = entry_price * (1 + self.take_profit_pct)

            print(
                f"  => 部署 Bracket: StopLoss={stop_price:.2f}, "
                f"TakeProfit={limit_price:.2f}"
            )

            # 发送止损单 (Stop Market)
            # sell() 方法支持 trigger_price 参数来实现止损
            self.stop_loss_order_id = self.sell(
                trade.symbol,
                trade.quantity,
                trigger_price=stop_price,  # 触发价格
                price=None,  # None 表示触发后转为市价单 (Stop Market)
            )

            # 发送止盈单 (Limit Sell)
            self.take_profit_order_id = self.sell(
                trade.symbol,
                trade.quantity,
                price=limit_price,  # 限价单
            )

        # 2. OCO 逻辑: 止损成交 -> 取消止盈
        elif trade.order_id == self.stop_loss_order_id:
            print("  => 止损触发，取消止盈单")
            self.has_position = False
            self.stop_loss_order_id = None
            if self.take_profit_order_id:
                self.cancel_order(self.take_profit_order_id)
                self.take_profit_order_id = None

        # 3. OCO 逻辑: 止盈成交 -> 取消止损
        elif trade.order_id == self.take_profit_order_id:
            print("  => 止盈触发，取消止损单")
            self.has_position = False
            self.take_profit_order_id = None
            if self.stop_loss_order_id:
                self.cancel_order(self.stop_loss_order_id)
                self.stop_loss_order_id = None

    def on_order(self, order: Order) -> None:
        """
        订单状态回调.

        用于处理订单被拒绝或取消的情况.
        """
        if order.status == OrderStatus.Cancelled:
            print(f"[{order.id}] 订单已取消")
        elif order.status == OrderStatus.Rejected:
            print(f"[{order.id}] 订单被拒绝")


# --- 数据生成与回测运行 ---
def run_example() -> None:
    """运行示例."""
    # 1. 生成模拟数据 (正弦波 + 噪声)
    # 制造一个先涨后跌的形态，触发止盈或止损
    dates = pd.date_range("2024-01-01", periods=200, freq="1min")
    x = np.linspace(0, 10, 200)
    # 价格从 100 开始，波动向上
    prices = 100 + 5 * np.sin(x) + np.random.normal(0, 0.2, 200)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 1000,
            "symbol": "TEST_STOCK",
        },
        index=dates,
    )

    # 2. 运行回测
    print("开始 Bracket Order 策略回测...")
    run_backtest(
        data={"TEST_STOCK": df}, strategy=BracketStrategy, initial_capital=100000.0
    )


if __name__ == "__main__":
    run_example()
