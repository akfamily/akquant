import pandas as pd
from akquant import Bar, Strategy, run_backtest


class FrameworkHooksDemoStrategy(Strategy):
    """Demonstrate framework-level hooks that wrap bar/timer callbacks."""

    def __init__(self) -> None:
        """Initialize demo state and enable precise day-boundary hooks."""
        self.submitted_reject_order = False
        self.event_log: list[str] = []

        # Enable boundary timers so day hooks are fired at the exact session edges.
        self.enable_precise_day_boundary_hooks = True

    def on_start(self) -> None:
        """Register the demo symbol and announce strategy startup."""
        self.subscribe("HOOKS_DEMO")
        self.log("strategy started")

    def on_before_trading(self, trading_date: object, timestamp: int) -> None:
        """Observe the trading-day boundary before the normal session opens."""
        self.event_log.append(f"before:{trading_date}")
        self.log(
            f"on_before_trading date={trading_date} time={pd.Timestamp(timestamp)}"
        )

    def on_cross_section(self, trading_date: object, timestamp: int) -> None:
        """Show the once-per-day cross-section hook (first complete bar slice)."""
        self.event_log.append(f"cross_section:{trading_date}")
        self.log(f"on_cross_section date={trading_date} time={pd.Timestamp(timestamp)}")

    def on_after_trading(self, trading_date: object, timestamp: int) -> None:
        """Observe the trading-day boundary after the normal session closes."""
        self.event_log.append(f"after:{trading_date}")
        self.log(f"on_after_trading date={trading_date} time={pd.Timestamp(timestamp)}")

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
        self.log(f"on_bar symbol={bar.symbol} close={bar.close}")

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
    """Build a tiny deterministic bar series for boundary-hook playback."""
    rows = [
        ("2023-01-03 09:30:00", 10.0),
        ("2023-01-03 15:00:00", 10.5),
        ("2023-01-04 09:30:00", 10.8),
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
        fill_policy={"price_basis": "close", "bar_offset": 0, "temporal": "same_cycle"},
        strategy_max_position_size={"_default": 10},
    )


if __name__ == "__main__":
    main()
