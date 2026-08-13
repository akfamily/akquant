import numpy as np
import pandas as pd
from akquant import Bar, FloatParam, Order, OrderStatus, Strategy, Trade, run_backtest


class TrailingOrderStrategy(Strategy):
    """Trailing Stop / StopLimit 示例策略."""

    trail_offset = FloatParam(1.5, ge=0.01, title="跟踪止损偏移")

    def on_start(self) -> None:
        """初始化订单跟踪状态与最新收盘价缓存."""
        self.entry_order_id = ""
        self.trailing_order_id = ""
        self.last_close: dict[str, float] = {}

    def on_bar(self, bar: Bar) -> None:
        """K线回调."""
        self.last_close[bar.symbol] = float(bar.close)
        if (
            self.get_position(bar.symbol) == 0
            and not self.entry_order_id
            and not self.trailing_order_id
        ):
            entry_receipt = self.buy(bar.symbol, 100, tag="trail-entry")
            self.entry_order_id = str(getattr(entry_receipt, "primary", entry_receipt))
            print(
                f"[{bar.timestamp_iso}] 提交进场单: {bar.symbol}, close={bar.close:.2f}"
            )

    def on_trade(self, trade: Trade) -> None:
        """成交回调."""
        print(
            f"[{trade.timestamp}] 成交: {trade.side} {trade.quantity} "
            f"@ {trade.price:.2f}, oid={trade.order_id}"
        )
        if trade.order_id == self.entry_order_id:
            self.entry_order_id = ""
            ref_price = self.last_close.get(trade.symbol, float(trade.price))
            trailing_receipt = self.place_trailing_stop(
                symbol=trade.symbol,
                quantity=float(trade.quantity),
                trail_offset=self.params.trail_offset,
                side="Sell",
                trail_reference_price=ref_price,
                tag="trail-stop",
            )
            self.trailing_order_id = str(
                getattr(trailing_receipt, "primary", trailing_receipt)
            )
            print(
                f"[{trade.timestamp}] 提交 trailing stop: symbol={trade.symbol}, "
                f"offset={self.params.trail_offset:.2f}, ref={ref_price:.2f}"
            )
        elif trade.order_id == self.trailing_order_id:
            self.trailing_order_id = ""

    def on_order(self, order: Order) -> None:
        """订单状态回调."""
        if order.id in (
            self.entry_order_id,
            self.trailing_order_id,
        ) and order.status in (OrderStatus.Cancelled, OrderStatus.Rejected):
            if order.id == self.entry_order_id:
                self.entry_order_id = ""
            if order.id == self.trailing_order_id:
                self.trailing_order_id = ""
        if order.status == OrderStatus.Cancelled:
            print(f"[{order.id}] 订单已取消")
        elif order.status == OrderStatus.Rejected:
            print(f"[{order.id}] 订单被拒绝")


def run_example() -> None:
    """运行示例."""
    dates = pd.date_range("2024-01-01 09:30:00", periods=220, freq="1min")
    up = np.linspace(100, 112, 140)
    down = np.linspace(112, 96, 80)
    prices = np.concatenate([up, down]) + np.random.normal(0, 0.15, 220)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.4,
            "low": prices - 0.4,
            "close": prices,
            "volume": 1000,
            "symbol": "TEST_STOCK",
        },
        index=dates,
    )
    print("开始 Trailing Order 策略回测...")
    run_backtest(
        data={"TEST_STOCK": df},
        strategy=TrailingOrderStrategy,
        initial_cash=100000.0,
    )


if __name__ == "__main__":
    run_example()
