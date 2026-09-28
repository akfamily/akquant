import pandas as pd
from akquant import (
    BacktestConfig,
    Bar,
    InstrumentConfig,
    Strategy,
    StrategyConfig,
    run_backtest,
)
from akquant.config import RiskConfig


# 1. Define Strategy
class OptionExpiryStrategy(Strategy):
    """Strategy to test Option Expiry and Settlement."""

    def on_start(self) -> None:
        """Initialize strategy."""
        print("Strategy Started")
        # Subscribe to both Option and Underlying (though backtest data is
        # passed directly)
        pass

    def on_bar(self, bar: Bar) -> None:
        """Handle new bar events."""
        if self._bar_count < 5:
            print(f"on_bar: {bar.symbol} {bar.timestamp_iso}")

        # Buy Option on the first bar
        if self.get_position("CALL_OPT") == 0 and bar.symbol == "CALL_OPT":
            print(f"[{bar.timestamp_iso}] Attempting to buy 1 CALL_OPT")
            self.buy(symbol="CALL_OPT", quantity=1.0)


def build_data() -> dict[str, pd.DataFrame]:
    """Build option and underlying minute-bar data."""
    dates = pd.date_range("2023-12-01", "2023-12-02", freq="1min")
    data_opt = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 6.0,
            "high": 6.0,
            "low": 6.0,
            "close": 6.0,
            "volume": 100,
            "symbol": "CALL_OPT",
        }
    )
    data_ul = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 105.0,
            "high": 105.0,
            "low": 105.0,
            "close": 105.0,
            "volume": 1000,
            "symbol": "UL",
        }
    )
    return {"CALL_OPT": data_opt, "UL": data_ul}


def build_config() -> BacktestConfig:
    """Build backtest configuration for option expiry validation."""
    risk_config = RiskConfig()
    risk_config.safety_margin = 0.0001
    return BacktestConfig(
        strategy_config=StrategyConfig(
            initial_cash=100_000.0,
            risk=risk_config,
        ),
        instruments_config=[
            InstrumentConfig(
                symbol="CALL_OPT",
                asset_type="OPTION",
                multiplier=100.0,
                margin_ratio=0.1,
                tick_size=0.01,
                option_type="CALL",
                strike_price=100.0,
                expiry_date=20231201,
                underlying_symbol="UL",
                settlement_type="cash",
            ),
            InstrumentConfig(
                symbol="UL",
                asset_type="STOCK",
                multiplier=1.0,
                margin_ratio=1.0,
                tick_size=0.01,
            ),
        ],
    )


def main() -> None:
    """Run the option expiry example."""
    print("Running Option Backtest...")
    result = run_backtest(
        data=build_data(),
        strategy=OptionExpiryStrategy,
        config=build_config(),
        commission_rate=0.0,
        show_progress=False,
    )

    print("\n--- Results ---")
    print("Orders:")
    for order in result.orders:
        print(
            f"ID: {order.id}, Symbol: {order.symbol}, Status: {order.status}, "
            f"Reason: {order.reject_reason}"
        )

    print(f"Final Cash: {result.metrics.end_market_value:.2f}")
    final_val = result.metrics.end_market_value
    # 内在价值 (105-100)×100=500, 权利金 600,
    # 买入费用 6.6(佣金5+经手1.3+结算0.3), 行权结算费 0.6
    if abs(final_val - 99892.8) < 0.01:
        print("SUCCESS: Option Settlement Verified!")
    else:
        print(f"FAILURE: Expected ~99892.8, got {final_val}")

    print(result.trades_df)


if __name__ == "__main__":
    main()
