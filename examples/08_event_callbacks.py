"""Comprehensive event callback demo for class-style strategies."""

from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar, OrderStatus, Strategy


def build_intraday_data() -> list[Bar]:
    """Build deterministic intraday bars so timer callbacks can also fire."""
    rows = [
        ("2024-01-02 09:30:00", 10.00),
        ("2024-01-02 14:55:00", 10.10),
        ("2024-01-02 15:00:00", 10.05),
        ("2024-01-03 09:30:00", 10.20),
        ("2024-01-03 14:55:00", 10.35),
        ("2024-01-03 15:00:00", 10.30),
    ]
    bars: list[Bar] = []
    for dt_str, close in rows:
        ts = pd.Timestamp(dt_str, tz="Asia/Shanghai").value
        bars.append(
            Bar(
                timestamp=ts,
                open=close,
                high=close + 0.10,
                low=close - 0.10,
                close=close,
                volume=1000.0,
                symbol="MOCK_STOCK",
            )
        )
    return bars


class EventCallbacksStrategy(Strategy):
    """Single example covering the most common event callbacks."""

    def __init__(self) -> None:
        """Initialize counters used by the demo output."""
        self.submitted_reject_order = False
        self.submitted_entry_order = False
        self.submitted_exit_order = False
        self.portfolio_updates = 0
        self.timer_hits = 0
        self.trade_hits = 0
        self.reject_hits = 0

    def on_start(self) -> None:
        """Register subscriptions and timers."""
        print("\n=== 策略启动 ===")
        self.subscribe("MOCK_STOCK")
        self.schedule_daily("14:55:00", "close_check")

    def on_bar(self, bar: Bar) -> None:
        """Submit one rejected order and one valid order using bar events."""
        dt_str = str(
            pd.Timestamp(bar.timestamp, unit="ns", tz="UTC").tz_convert("Asia/Shanghai")
        )
        print(f"[Callback] on_bar | {dt_str} | close={bar.close:.2f}")

        if not self.submitted_reject_order:
            self.submitted_reject_order = True
            print("[Strategy] 提交一笔超大买单，故意触发 on_reject")
            self.buy(bar.symbol, 1000)
            return

        if not self.submitted_entry_order and "2024-01-03 09:30:00" in dt_str:
            self.submitted_entry_order = True
            print("[Strategy] 提交一笔有效买单，观察 on_order / on_trade")
            self.buy(bar.symbol, 10)

    def on_timer(self, payload: str) -> None:
        """Use timer as a second event source."""
        self.timer_hits += 1
        position = self.get_position("MOCK_STOCK")
        print(f"[Callback] on_timer | payload={payload} | position={position}")
        if payload == "close_check" and position > 0 and not self.submitted_exit_order:
            self.submitted_exit_order = True
            print("[Strategy] timer 触发卖出，观察同一示例中的定时调仓路径")
            self.sell("MOCK_STOCK", position)

    def on_order(self, order: Any) -> None:
        """Print all order state transitions."""
        status_to_icon = {
            getattr(OrderStatus, "New", None): "[NEW]",
            getattr(OrderStatus, "Submitted", None): "[SUBMIT]",
            getattr(OrderStatus, "Accepted", None): "[ACCEPT]",
            getattr(OrderStatus, "PartiallyFilled", None): "[PART]",
            getattr(OrderStatus, "Filled", None): "[FILLED]",
            getattr(OrderStatus, "Cancelled", None): "[CANCEL]",
            getattr(OrderStatus, "Rejected", None): "[REJECT]",
            getattr(OrderStatus, "Expired", None): "[EXPIRE]",
        }
        icon = status_to_icon.get(order.status, "[ORDER]")
        print(
            f"[Callback] on_order {icon} | "
            f"id={order.id[:8]} | symbol={order.symbol} | side={order.side} | "
            f"status={order.status} | filled={order.filled_quantity}/{order.quantity}"
        )

    def on_trade(self, trade: Any) -> None:
        """Print trade reports."""
        self.trade_hits += 1
        print(
            f"[Callback] on_trade | symbol={trade.symbol} | side={trade.side} | "
            f"price={trade.price:.2f} | qty={trade.quantity} | "
            f"comm={trade.commission:.4f}"
        )

    def on_reject(self, order: Any) -> None:
        """Observe rejected orders separately from generic order updates."""
        self.reject_hits += 1
        print(
            f"[Callback] on_reject | id={order.id[:8]} | "
            f"status={order.status} | qty={order.quantity}"
        )

    def on_portfolio_update(self, snapshot: dict[str, Any]) -> None:
        """Observe cash/equity updates."""
        self.portfolio_updates += 1
        print(
            "[Callback] on_portfolio_update | "
            f"cash={snapshot['cash']:.2f} | equity={snapshot['equity']:.2f}"
        )

    def on_stop(self) -> None:
        """Print a small summary so users know what was triggered."""
        print("\n=== 回调汇总 ===")
        print(f"portfolio_updates={self.portfolio_updates}")
        print(f"timer_hits={self.timer_hits}")
        print(f"trade_hits={self.trade_hits}")
        print(f"reject_hits={self.reject_hits}")


def main() -> None:
    """Run the callback demo."""
    print("准备确定性日内数据...")
    bars = build_intraday_data()

    print("开始回测...")
    results = aq.run_backtest(
        strategy=EventCallbacksStrategy,
        data=bars,
        symbols="MOCK_STOCK",
        initial_cash=1000.0,
        commission_rate=0.0003,
        fill_policy=aq.CurrentClose(),
        show_progress=False,
    )

    print("\n=== 回测结束 ===")
    print("最终绩效指标:")
    print(results.metrics_df)


if __name__ == "__main__":
    main()
