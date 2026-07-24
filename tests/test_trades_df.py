import pandas as pd
from akquant import (
    BacktestConfig,
    Bar,
    ChinaFuturesConfig,
    ChinaFuturesInstrumentTemplateConfig,
    CurrentClose,
    InstrumentConfig,
    Strategy,
    StrategyConfig,
    run_backtest,
)


class TradesTestStrategy(Strategy):
    """Strategy for testing trades dataframe."""

    def on_bar(self, bar: Bar) -> None:
        """Execute on every bar."""
        # Simple logic to generate trades
        # Buy on day 1, Sell on day 3
        if self.position.size == 0:
            self.buy(bar.symbol, 100)
        elif self.position.size > 0:
            # Sell 2 days later to ensure duration > 0
            self.sell(bar.symbol, 100)


class FuturesTradesTestStrategy(Strategy):
    """Strategy for testing futures trades dataframe PnL."""

    def __init__(self) -> None:
        """Initialize strategy state."""
        self.entered = False

    def on_bar(self, bar: Bar) -> None:
        """Open once and close on the following bar."""
        position = self.get_position(bar.symbol)
        if not self.entered:
            self.buy(bar.symbol, 1)
            self.entered = True
        elif position > 0:
            self.close_position(bar.symbol)


def test_trades_df() -> None:
    """Test the structure and content of trades_df."""
    data = []
    # Create enough bars to generate trades
    # Day 1: Buy
    # Day 2: Hold
    # Day 3: Sell
    for i in range(5):
        data.append(
            Bar(
                timestamp=pd.Timestamp(f"2023-01-{i + 1:02d} 10:00:00").value,
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=100.0 + i,
                volume=1000,
                symbol="TEST",
            )
        )

    print("Running backtest...")
    result = run_backtest(
        data=data, strategy=TradesTestStrategy, symbols="TEST", show_progress=False
    )

    print("\nTrades DataFrame:")
    print(result.trades_df)

    # Check columns
    expected = [
        "symbol",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "quantity",
        "side",
        "pnl",
        "net_pnl",
        "return_pct",
        "commission",
        "duration_bars",
        "duration",
    ]
    missing = [c for c in expected if c not in result.trades_df.columns]
    assert not missing, f"Missing columns: {missing}"

    if not result.trades_df.empty:
        trade = result.trades_df.iloc[0]
        assert trade["side"] == "Long"
        assert trade["quantity"] == 100.0
        assert pd.notna(trade["duration"])
        # Check duration type
        assert isinstance(trade["duration"], pd.Timedelta)
        print("\nAll expected columns present and duration is Timedelta.")


def test_futures_trades_df_respects_contract_multiplier() -> None:
    """Futures trades_df PnL should include the instrument multiplier."""
    bars = [
        Bar(
            timestamp=pd.Timestamp("2023-01-01 09:00:00", tz="Asia/Shanghai").value,
            open=4000.0,
            high=4000.0,
            low=4000.0,
            close=4000.0,
            volume=1000,
            symbol="RB2310",
        ),
        Bar(
            timestamp=pd.Timestamp("2023-01-01 09:05:00", tz="Asia/Shanghai").value,
            open=4010.0,
            high=4010.0,
            low=4010.0,
            close=4010.0,
            volume=1000,
            symbol="RB2310",
        ),
    ]

    result = run_backtest(
        data=bars,
        strategy=FuturesTradesTestStrategy,
        symbols="RB2310",
        show_progress=False,
        fill_policy=CurrentClose(),
        config=BacktestConfig(
            strategy_config=StrategyConfig(
                initial_cash=500000.0,
                commission_rate=0.0,
            ),
            instruments_config=[
                InstrumentConfig(
                    symbol="RB2310",
                    asset_type="FUTURES",
                    multiplier=10.0,
                    margin_ratio=0.1,
                )
            ],
            china_futures=ChinaFuturesConfig(
                enforce_sessions=False,
                instrument_templates_by_symbol_prefix=[
                    ChinaFuturesInstrumentTemplateConfig(
                        symbol_prefix="RB",
                        multiplier=10.0,
                        margin_ratio=0.1,
                        tick_size=1.0,
                        lot_size=1.0,
                        commission_rate=0.0,
                        enforce_tick_size=False,
                        enforce_lot_size=True,
                    )
                ],
            ),
        ),
    )

    assert len(result.trades_df) == 1
    trade = result.trades_df.iloc[0]
    assert trade["symbol"] == "RB2310"
    assert trade["quantity"] == 1.0
    assert trade["pnl"] == 100.0
    assert trade["net_pnl"] == 100.0
    assert trade["return_pct"] == 0.25
