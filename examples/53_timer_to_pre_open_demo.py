import pandas as pd
from akquant import Bar, Strategy, run_backtest


class TimerToPreOpenStrategy(Strategy):
    """Prepare late in the previous day and execute on next-day pre-open."""

    def __init__(self) -> None:
        """Initialize staged state."""
        self.next_day_plan: str | None = None
        self.events: list[str] = []

    def on_start(self) -> None:
        """Subscribe the demo symbol and register a daily preparation timer."""
        self.subscribe("TIMER_PREOPEN")
        self.schedule_daily("15:00:00", "prepare_next_day")

    def on_timer(self, payload: str) -> None:
        """Prepare the next-day plan using end-of-day information."""
        if payload != "prepare_next_day":
            return
        now = self.now
        if now is None:
            return
        latest_close = float(self.close)
        self.next_day_plan = "buy" if latest_close >= 10.5 else "hold"
        self.events.append(f"timer:{now.date()}:{self.next_day_plan}")
        self.log(
            "on_timer "
            f"date={now.date()} close={latest_close} "
            f"next_day_plan={self.next_day_plan}"
        )

    def on_pre_open(self, event: dict[str, object]) -> None:
        """Execute the plan that was staged on the previous day."""
        trading_date = event["trading_date"]
        self.events.append(f"pre_open:{trading_date}:{self.next_day_plan}")
        self.log(f"on_pre_open date={trading_date} staged_plan={self.next_day_plan}")
        if self.next_day_plan == "buy":
            self.buy("TIMER_PREOPEN", quantity=1)
        self.next_day_plan = None

    def on_bar(self, bar: Bar) -> None:
        """Log bar arrival after pre-open processing."""
        self.events.append(f"bar:{self.format_time(bar.timestamp)}:{bar.close}")

    def on_trade(self, trade: object) -> None:
        """Show fills produced by next-day pre-open execution."""
        self.log(
            "on_trade "
            f"price={getattr(trade, 'price', '<unknown>')} "
            f"qty={getattr(trade, 'quantity', '<unknown>')}"
        )

    def on_stop(self) -> None:
        """Print the staged callback order."""
        print("\n=== timer to pre_open summary ===")
        for event in self.events:
            print(event)


def build_data() -> list[Bar]:
    """Build daily bars so the timer can prepare the next-day plan."""
    rows = [
        ("2023-01-03 09:30:00", 10.0, 10.6),
        ("2023-01-04 09:30:00", 10.7, 10.2),
        ("2023-01-05 09:30:00", 10.1, 10.0),
    ]
    bars: list[Bar] = []
    for dt_str, open_price, close_price in rows:
        ts = pd.Timestamp(dt_str, tz="Asia/Shanghai").value
        bars.append(
            Bar(
                timestamp=ts,
                open=open_price,
                high=max(open_price, close_price) + 0.2,
                low=min(open_price, close_price) - 0.2,
                close=close_price,
                volume=1000.0,
                symbol="TIMER_PREOPEN",
            )
        )
    return bars


def main() -> None:
    """Run the timer-to-pre-open demo."""
    run_backtest(
        data=build_data(),
        strategy=TimerToPreOpenStrategy(),
        symbols=["TIMER_PREOPEN"],
        initial_cash=10000.0,
        lot_size=1,
        show_progress=False,
    )


if __name__ == "__main__":
    main()
