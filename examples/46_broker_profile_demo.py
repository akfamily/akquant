import pandas as pd
from akquant import Bar, CurrentClose, Strategy, run_backtest


class BrokerProfileDemoStrategy(Strategy):
    """Print effective broker profile fields at startup."""

    def __init__(self) -> None:
        """Initialize one-shot submit state."""
        self.submitted = False

    def on_start(self) -> None:
        """Display resolved fee and lot parameters."""
        print(f"strategy_commission_rate={self.commission_rate}")
        print(f"strategy_stamp_tax_rate={self.stamp_tax_rate}")
        print(f"strategy_transfer_fee_rate={self.transfer_fee_rate}")
        print(f"strategy_min_commission={self.min_commission}")
        print(f"strategy_lot_size={self.lot_size}")

    def on_bar(self, bar: Bar) -> None:
        """Submit one buy order on the first bar."""
        if not self.submitted:
            self.buy(symbol=bar.symbol, quantity=self.lot_size)
            self.submitted = True


def _build_data() -> list[Bar]:
    closes = [10.0, 10.2, 10.1, 10.4, 10.3]
    rows: list[Bar] = []
    for i, close in enumerate(closes):
        ts = pd.Timestamp(f"2024-01-0{i + 1} 15:00:00", tz="UTC").value
        rows.append(
            Bar(
                timestamp=ts,
                open=close,
                high=close + 0.1,
                low=close - 0.1,
                close=close,
                volume=10000.0,
                symbol="PROFILE",
            )
        )
    return rows


def run_example() -> None:
    """Run broker profile demo backtest."""
    result = run_backtest(
        data=_build_data(),
        strategy=BrokerProfileDemoStrategy,
        symbols="PROFILE",
        fill_policy=CurrentClose(),
        broker_profile="cn_stock_t1_low_fee",
        show_progress=False,
    )
    print(f"orders={len(result.orders_df)}")
    print("done_broker_profile_demo")


if __name__ == "__main__":
    run_example()
