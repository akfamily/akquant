import pandas as pd
from akquant import Bar, Strategy, run_backtest


class PreOpenDemoStrategy(Strategy):
    """Show how on_pre_open models "decide before open, fill on this open"."""

    def __init__(self) -> None:
        """Initialize demo state."""
        self.submitted_dates: set[object] = set()
        self.events: list[str] = []

    def on_start(self) -> None:
        """Subscribe the demo symbol."""
        self.subscribe("PREOPEN_DEMO")

    def on_pre_open(self, event: dict[str, object]) -> None:
        """Place one default market order before the session open."""
        trading_date = event["trading_date"]
        self.events.append(f"pre_open:{trading_date}")
        self.log(
            "on_pre_open "
            f"trading_date={trading_date} expected_open_at={event['expected_open_at']}"
        )
        if trading_date in self.submitted_dates:
            return
        self.submitted_dates.add(trading_date)
        self.buy("PREOPEN_DEMO", quantity=1)

    def on_bar(self, bar: Bar) -> None:
        """Print bar arrival after the pre-open hook."""
        self.events.append(f"bar:{self.format_time(bar.timestamp)}:{bar.open}")
        self.log(
            "on_bar "
            f"ts={self.format_time(bar.timestamp)} open={bar.open} close={bar.close}"
        )

    def on_order(self, order: object) -> None:
        """Log order transitions."""
        self.log(
            "on_order "
            f"id={getattr(order, 'id', '<unknown>')} "
            f"status={getattr(order, 'status', '<unknown>')}"
        )

    def on_trade(self, trade: object) -> None:
        """Log fills to show the open-price execution."""
        self.log(
            "on_trade "
            f"price={getattr(trade, 'price', '<unknown>')} "
            f"qty={getattr(trade, 'quantity', '<unknown>')}"
        )

    def on_stop(self) -> None:
        """Print the simplified callback order."""
        print("\n=== pre_open summary ===")
        for event in self.events:
            print(event)


def build_data() -> list[Bar]:
    """Build two daily open bars for the demo."""
    rows = [
        ("2023-01-03 09:30:00", 10.0, 10.4),
        ("2023-01-04 09:30:00", 10.8, 11.0),
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
                symbol="PREOPEN_DEMO",
            )
        )
    return bars


def main() -> None:
    """Run the pre-open callback demo."""
    run_backtest(
        data=build_data(),
        strategy=PreOpenDemoStrategy(),
        symbols=["PREOPEN_DEMO"],
        initial_cash=10000.0,
        lot_size=1,
        # 日线 bar 打在 09:30（非 0 点）时须声明 freq="1d"，让每根 bar 覆盖整个
        # 交易日；否则 09:30 落在开盘集合竞价后的 AuctionHold，盘前订单当日不成交。
        freq="1d",
        show_progress=False,
    )


if __name__ == "__main__":
    main()
