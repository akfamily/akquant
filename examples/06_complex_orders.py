from typing import Any

import numpy as np
import pandas as pd
from akquant import Bar, Order, OrderStatus, Strategy, Trade, run_backtest


class BracketStrategy(Strategy):
    """Bracket Order 示例策略."""

    def __init__(
        self,
        period: int = 20,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs: Any,
    ) -> None:
        """初始化策略."""
        self.period = period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.entry_order_id = ""
        self.has_position = False

    def on_bar(self, bar: Bar) -> None:
        """K线回调."""
        current_position = self.get_position(bar.symbol)
        self.has_position = current_position > 0
        if self.has_position or self.entry_order_id:
            return

        stop_price = bar.close * (1 - self.stop_loss_pct)
        take_profit_price = bar.close * (1 + self.take_profit_pct)
        print(
            f"[{bar.timestamp_iso}] 提交 Bracket 进场: {bar.symbol} @ {bar.close:.2f}, "
            f"stop={stop_price:.2f}, take={take_profit_price:.2f}"
        )
        self.entry_order_id = self.place_bracket(
            symbol=bar.symbol,
            quantity=100,
            entry_price=None,
            stop_trigger_price=stop_price,
            take_profit_price=take_profit_price,
            entry_tag="bracket_entry",
            stop_tag="bracket_stop",
            take_profit_tag="bracket_take",
        )

    def on_trade(self, trade: Trade) -> None:
        """成交回调."""
        print(
            f"[{trade.timestamp}] 成交确认: {trade.side} {trade.quantity} "
            f"@ {trade.price} (ID: {trade.order_id})"
        )
        if trade.order_id == self.entry_order_id:
            self.has_position = True
            self.entry_order_id = ""

    def on_order(self, order: Order) -> None:
        """订单状态回调."""
        if order.id == self.entry_order_id and order.status in (
            OrderStatus.Cancelled,
            OrderStatus.Rejected,
        ):
            self.entry_order_id = ""
        if order.status == OrderStatus.Cancelled:
            print(f"[{order.id}] 订单已取消")
        elif order.status == OrderStatus.Rejected:
            print(f"[{order.id}] 订单被拒绝")


def run_example() -> None:
    """运行示例."""
    dates = pd.date_range("2024-01-01", periods=200, freq="1min")
    x = np.linspace(0, 10, 200)
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

    print("开始 Bracket Order 策略回测...")
    run_backtest(
        data={"TEST_STOCK": df}, strategy=BracketStrategy, initial_cash=100000.0
    )


if __name__ == "__main__":
    run_example()
