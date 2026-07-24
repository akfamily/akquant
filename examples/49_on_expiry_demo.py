import pandas as pd
from akquant import (
    BacktestConfig,
    Bar,
    CurrentClose,
    InstrumentConfig,
    Strategy,
    StrategyConfig,
    run_backtest,
)
from akquant.backtest import BacktestStreamEvent


class ExpiryCallbackStrategy(Strategy):
    """Buy once and observe on_expiry after futures settlement."""

    def __init__(self) -> None:
        """Initialize one-shot order state and captured expiry events."""
        self.ordered = False
        self.expiry_events: list[dict[str, object]] = []

    def on_bar(self, bar: Bar) -> None:
        """Submit one futures order before expiry."""
        if not self.ordered:
            self.buy(symbol=bar.symbol, quantity=1)
            self.ordered = True

    def on_expiry(self, event: dict[str, object]) -> None:
        """Print expiry payload after portfolio state has been updated."""
        self.expiry_events.append(dict(event))
        print(
            "on_expiry",
            {
                "symbol": event["symbol"],
                "expiry_date": event["expiry_date"],
                "quantity_closed": event["quantity_closed"],
                "cash_flow": event["cash_flow"],
                "settlement_type": event["settlement_type"],
                "position_after": self.get_position(str(event["symbol"])),
            },
        )


def _build_data() -> list[Bar]:
    rows = [
        ("2026-01-30 15:00:00", 100.0),
        ("2026-01-31 15:00:00", 110.0),
        ("2026-02-01 15:00:00", 109.0),
    ]
    bars: list[Bar] = []
    for dt_str, close in rows:
        ts = pd.Timestamp(dt_str, tz="Asia/Shanghai").value
        bars.append(
            Bar(
                timestamp=ts,
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1000.0,
                symbol="FUT_EXP_DEMO",
            )
        )
    return bars


def run_example() -> None:
    """Run the expiry callback demo and also inspect stream events."""
    strategy = ExpiryCallbackStrategy()
    stream_events: list[BacktestStreamEvent] = []
    result = run_backtest(
        data=_build_data(),
        strategy=strategy,
        symbols=["FUT_EXP_DEMO"],
        lot_size=1,
        show_progress=False,
        fill_policy=CurrentClose(),
        on_event=stream_events.append,
        config=BacktestConfig(
            strategy_config=StrategyConfig(),
            instruments_config=[
                InstrumentConfig(
                    symbol="FUT_EXP_DEMO",
                    asset_type="FUTURES",
                    multiplier=10.0,
                    margin_ratio=0.1,
                    tick_size=0.2,
                    expiry_date=20260131,
                    settlement_type="settlement_price",
                    settlement_price=108.0,
                )
            ],
        ),
    )

    expiry_stream_events = [
        event for event in stream_events if event.get("event_type") == "expiry"
    ]
    print("expiry_event_count", len(expiry_stream_events))
    if expiry_stream_events:
        print("stream_expiry_payload", expiry_stream_events[-1]["payload"])
    print("final_positions", result.positions)


if __name__ == "__main__":
    run_example()
