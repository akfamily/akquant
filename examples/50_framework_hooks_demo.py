"""Framework hooks demo: 交易日生命周期钩子与订单/组合回调的先后顺序.

A 股默认时段表下, 每个交易日依次触发:

- ``on_pre_open``: 首个时段(09:15 开盘集合竞价)开始前 1ns;
- ``on_before_trading``: 连续竞价开始(09:30);
- ``on_bar`` / ``on_cross_section``: 行情到达(bar 时间戳为区间终点);
- ``on_after_trading``: 收盘(15:00)后的日终结算之后。
"""

from typing import Any

import pandas as pd
from akquant import Bar, CurrentClose, Strategy, run_backtest


class FrameworkHooksDemoStrategy(Strategy):
    """Demonstrate framework-level hooks that wrap bar/timer callbacks."""

    def __init__(self) -> None:
        """Initialize demo state."""
        self.submitted_reject_order = False
        self.event_log: list[str] = []

    def on_start(self) -> None:
        """Register the demo symbol and announce strategy startup."""
        self.subscribe("HOOKS_DEMO")
        self.log("strategy started")

    def on_pre_open(self, event: dict[str, Any]) -> None:
        """Observe the pre-open hook (1ns before the first session of the day)."""
        trading_date = event["trading_date"]
        self.event_log.append(f"pre_open:{trading_date}")
        self.log(
            f"on_pre_open date={trading_date} "
            f"time={self.format_time(int(event['timestamp']), '%H:%M:%S.%f')}"
        )

    def on_before_trading(self, trading_date: object, timestamp: int) -> None:
        """Observe the start of continuous trading (09:30 for A shares)."""
        self.event_log.append(f"before:{trading_date}")
        self.log(
            f"on_before_trading date={trading_date} "
            f"time={self.format_time(timestamp)} "
            f"market_open={self.is_market_open('HOOKS_DEMO')}"
        )

    def on_cross_section(self, trading_date: object, timestamp: int) -> None:
        """Show the once-per-day cross-section hook (first complete bar slice)."""
        self.event_log.append(f"cross_section:{trading_date}")
        self.log(
            f"on_cross_section date={trading_date} time={self.format_time(timestamp)}"
        )

    def on_after_trading(self, trading_date: object, timestamp: int) -> None:
        """Observe the trading-day boundary after end-of-day settlement."""
        self.event_log.append(f"after:{trading_date}")
        self.log(
            f"on_after_trading date={trading_date} "
            f"time={self.format_time(timestamp, '%H:%M:%S.%f')}"
        )

    def on_portfolio_update(self, snapshot: dict[str, object]) -> None:
        """Print incremental portfolio snapshot updates."""
        self.event_log.append("portfolio")
        self.log(
            f"on_portfolio_update cash={snapshot['cash']} equity={snapshot['equity']}"
        )

    def on_reject(self, order: object) -> None:
        """Record rejected orders separately from generic order updates."""
        order_id = getattr(order, "id", "<unknown>")
        self.event_log.append(f"reject:{order_id}")
        self.log(f"on_reject order_id={order_id}")

    def on_order(self, order: object) -> None:
        """Log every order-state transition observed by the strategy."""
        order_id = getattr(order, "id", "<unknown>")
        status = getattr(order, "status", "<unknown>")
        self.event_log.append(f"order:{order_id}:{status}")
        self.log(f"on_order order_id={order_id} status={status}")

    def on_bar(self, bar: Bar) -> None:
        """Submit one oversize order so the framework hooks become visible."""
        self.event_log.append(f"bar:{bar.symbol}")
        self.log(
            f"on_bar symbol={bar.symbol} close={bar.close} "
            f"trading_day={self.trading_day}"
        )

        # Intentionally oversize one order so risk checks reject it and on_reject fires.
        if not self.submitted_reject_order:
            self.submitted_reject_order = True
            self.buy(bar.symbol, quantity=1000)

    def on_stop(self) -> None:
        """Print the final framework-hook event sequence."""
        print("\n=== framework hooks summary ===")
        for event in self.event_log:
            print(event)


def build_data() -> list[Bar]:
    """Build a tiny deterministic bar series for boundary-hook playback.

    Bar timestamps are interval ends. The first bar of each day sits at 09:31
    (covering 09:30-09:31, the first continuous-session minute); a bar stamped
    09:30 would cover 09:29-09:30, i.e. the hold after the opening auction, and
    arrive before ``on_before_trading`` (which fires when continuous trading
    starts at 09:30). The 15:00 bar covers 14:59-15:00 including the closing
    auction.
    """
    rows = [
        ("2023-01-03 09:31:00", 10.0),
        ("2023-01-03 15:00:00", 10.5),
        ("2023-01-04 09:31:00", 10.8),
        ("2023-01-04 15:00:00", 10.6),
    ]
    bars: list[Bar] = []
    for dt_str, close in rows:
        ts = pd.Timestamp(dt_str, tz="Asia/Shanghai").value
        bars.append(
            Bar(
                timestamp=ts,
                open=close,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=1000.0,
                symbol="HOOKS_DEMO",
            )
        )
    return bars


def main() -> None:
    """Run the framework-hook callback demo."""
    strategy = FrameworkHooksDemoStrategy()
    run_backtest(
        data=build_data(),
        strategy=strategy,
        symbols=["HOOKS_DEMO"],
        initial_cash=1000.0,
        lot_size=1,
        show_progress=False,
        fill_policy=CurrentClose(),
        strategy_max_position_size={"_default": 10},
    )


if __name__ == "__main__":
    main()
