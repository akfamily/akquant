from datetime import datetime

from akquant import AssetType, Bar, Engine, ExecutionMode, Instrument, Strategy


class PnLStrategy(Strategy):
    """Strategy for PnL verification."""

    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()
        self.step = 0

    def on_start(self) -> None:
        """Subscribe to symbols on start."""
        self.subscribe("000001")

    def on_bar(self, bar: Bar) -> None:
        """Handle bar events."""
        if self.ctx is None:
            return

        # Step 0: Buy
        if self.step == 0:
            print(f"[Step 0] Buying 10,000 shares @ {bar.close}")
            self.buy(symbol=bar.symbol, quantity=10000, price=bar.close)
            self.step += 1

        # Step 1: Sell (Next Day)
        elif self.step == 1:
            pos = self.ctx.get_position(bar.symbol)
            avail = self.ctx.get_available_position(bar.symbol)
            print(
                f"[Step 1] Selling 10,000 shares @ {bar.close}. "
                f"Pos: {pos}, Avail: {avail}"
            )
            self.sell(symbol=bar.symbol, quantity=10000, price=bar.close)
            self.step += 1


def run_test() -> None:
    """Run the PnL verification test."""
    engine = Engine()
    # Uses defaults: Comm 0.0003, Stamp 0.0005, Transfer 0.00001, Min 5.0
    engine.use_china_market()
    engine.set_execution_mode(ExecutionMode.CurrentClose)
    engine.set_cash(1_000_000.0)

    symbol = "000001"
    instr = Instrument(
        symbol=symbol,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        option_type=None,
        strike_price=None,
        expiry_date=None,
        lot_size=100.0,
    )
    engine.add_instrument(instr)

    # Data
    bars = []

    # Day 1: Buy @ 10.0
    bars.append(
        Bar(
            timestamp=int(datetime(2023, 1, 4, 15, 00).timestamp() * 1e9),
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
            volume=100000.0,
            symbol=symbol,
        )
    )

    # Day 2: Sell @ 11.0
    bars.append(
        Bar(
            timestamp=int(datetime(2023, 1, 5, 15, 00).timestamp() * 1e9),
            open=11.0,
            high=11.0,
            low=11.0,
            close=11.0,
            volume=100000.0,
            symbol=symbol,
        )
    )

    engine.add_bars(bars)

    strategy = PnLStrategy()
    engine.run(strategy, show_progress=False)

    print("\n=== PnL Verification ===")

    # Expected Calculations
    # Buy 10,000 @ 10.0 = 100,000
    # Comm: 100,000 * 0.0003 = 30.0
    # Transfer: 100,000 * 0.00001 = 1.0
    # Buy Cost: 31.0

    # Sell 10,000 @ 11.0 = 110,000
    # Comm: 110,000 * 0.0003 = 33.0
    # Stamp: 110,000 * 0.0005 = 55.0
    # Transfer: 110,000 * 0.00001 = 1.1
    # Sell Cost: 89.1

    # Total PnL = 10,000 - 31.0 - 89.1 = 9879.9
    # Final Cash = 1,000,000 + 9879.9 = 1,009,879.9

    results = engine.get_results()
    final_cash = engine.portfolio.cash

    # Calculate absolute PnL from market values
    initial_mv = results.metrics.initial_market_value
    end_mv = results.metrics.end_market_value
    absolute_pnl = end_mv - initial_mv

    print(f"Final Cash: {final_cash:.4f}")
    print(f"Absolute PnL (End MV - Init MV): {absolute_pnl:.4f}")
    print(f"Total Return %: {results.metrics.total_return:.4%}")

    expected_pnl = 9879.9

    if abs(absolute_pnl - expected_pnl) < 0.01:
        print(f"PASS: PnL matches expected {expected_pnl}")
    else:
        print(f"FAIL: PnL {absolute_pnl} != expected {expected_pnl}")
        diff = absolute_pnl - expected_pnl
        print(f"Diff: {diff}")


if __name__ == "__main__":
    run_test()
