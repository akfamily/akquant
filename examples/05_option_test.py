import pandas as pd
from akquant import (
    AssetType,
    BacktestConfig,
    Bar,
    Instrument,
    OptionType,
    SettlementType,
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
            print(f"on_bar: {bar.symbol} {bar.timestamp_str}")

        # Buy Option on the first bar
        if self.get_position("CALL_OPT") == 0 and bar.symbol == "CALL_OPT":
            print(f"[{bar.timestamp_str}] Attempting to buy 1 CALL_OPT")
            self.buy(symbol="CALL_OPT", quantity=1.0)


# 2. Prepare Data
# Underlying: Stock "UL"
# Option: "CALL_OPT", Underlying="UL", Strike=100, Expiry=20231201
# We need data for 12-02 to trigger day close (and expiry) for 12-01
dates = pd.date_range("2023-12-01", "2023-12-02", freq="1min")
# We need data for both option and underlying
# Option Price: Intrinsic Value + Time Value. Let's say 6.0 (Strike 100, Spot 105)
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
# Underlying Price: 105.0
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

# Combine data (akquant supports dict of dfs or single df)
# We pass dict to ensure symbols are correctly mapped
data_dict = {"CALL_OPT": data_opt, "UL": data_ul}

# 3. Define Instruments
# Note: Underlying symbol must match the one in data
opt_instr = Instrument(
    symbol="CALL_OPT",
    asset_type=AssetType.Option,
    multiplier=100.0,  # 1 contract = 100 shares
    margin_ratio=0.1,
    tick_size=0.01,
    option_type=OptionType.Call,
    strike_price=100.0,
    expiry_date=20231201,  # YYYYMMDD
    underlying_symbol="UL",
    settlement_type=SettlementType.Cash,
)

ul_instr = Instrument(
    symbol="UL",
    asset_type=AssetType.Stock,
    multiplier=1.0,
    margin_ratio=1.0,
    tick_size=0.01,
)

# 4. Run Backtest
risk_config = RiskConfig()
risk_config.safety_margin = 0.0001
config = BacktestConfig(strategy_config=StrategyConfig(risk=risk_config))

print("Running Option Backtest...")
result = run_backtest(
    data=data_dict,
    strategy=OptionExpiryStrategy,
    instruments=[opt_instr, ul_instr],
    cash=100_000.0,
    commission=0.0,  # Zero commission to check math easily
    config=config,
)

# 5. Verify Results
print("\n--- Results ---")

print("Orders:")
for order in result.orders:
    print(
        f"ID: {order.id}, Symbol: {order.symbol}, Status: {order.status}, "
        f"Reason: {order.reject_reason}"
    )

print(f"Final Cash: {result.metrics.end_market_value:.2f}")
# Expected:
# Initial Cash: 100,000
# Buy 1 Contract (Price 6.0, Mult 100) -> Cost = 600.0
# Cash after buy = 99,400
# Expiry:
# Spot = 105.0, Strike = 100.0 -> Payoff = 5.0
# Cash Settlement = 5.0 * 100 (Mult) * 1 (Qty) = 500.0
# Final Cash = 99,400 + 500 = 99,900
# Net PnL = -100 (Bought at 6.0/600, Settled at 5.0/500)

final_val = result.metrics.end_market_value
if 99899.0 <= final_val <= 99901.0:
    print("SUCCESS: Option Settlement Verified!")
else:
    print(f"FAILURE: Expected ~99900, got {final_val}")

print(result.trades_df)
