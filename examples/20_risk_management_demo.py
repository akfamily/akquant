# -*- coding: utf-8 -*-
"""
Example 20: Pre-trade Risk Management Demo.

This example demonstrates how to use the built-in Risk Management module
to reject orders that violate pre-defined risk rules.

We will:
1. Fetch A-share data using AKShare.
2. Configure risk rules:
   - Max Position Size (per symbol)
   - Sector Concentration Limit
   - Max Leverage Limit
3. Run a strategy that aggressively buys to trigger these rules.
"""

import akquant  # Import main package
from akquant import AssetType, Bar, Instrument, Strategy
from akquant.utils import fetch_akshare_symbol, load_bar_from_df


class AggressiveBuyerStrategy(Strategy):
    """A strategy that tries to buy as much as possible to test risk limits."""

    def on_bar(self, bar: Bar) -> None:
        """
        Handle bar data.

        :param bar: Bar data
        """
        # Buy about 3% of initial cash on every bar for every symbol.
        # The first few buys fill; repeated buys soon hit the position limit
        # (10% per symbol). The sector limit (Consumer 15%) is checked too.

        price = bar.close
        if price <= 0:
            return

        # Round down to board lots (100 shares), at least one lot
        qty = max(100, int(30_000 / price / 100) * 100)

        # Log attempt
        self.log(f"Attempting to buy {qty} shares of {bar.symbol} at {price:.2f}...")

        # Send Order
        self.buy(bar.symbol, qty)


def main() -> None:
    """Run the risk management demo."""
    initial_cash = 1_000_000.0

    # 2. Define Symbols and Sectors
    # 600519: Kweichow Moutai (Consumer)
    # 000858: Wuliangye (Consumer)
    # 601318: Ping An Insurance (Financial)
    symbols = ["600519", "000858", "601318"]
    sector_map = {"600519": "Consumer", "000858": "Consumer", "601318": "Financial"}

    # 3. Add Instruments and Data
    # Date-only daily bars are passed as a list of Bars (not a DataFeed), so
    # run_backtest moves them to the session close (15:00) of each trading day
    # and orders can fill. A DataFeed is used as-is: its bars would stay at
    # 00:00, which is outside trading hours.
    all_bars: list[Bar] = []
    instruments = []

    for symbol in symbols:
        # Add Instrument
        instr = Instrument(symbol, AssetType.Stock)
        instruments.append(instr)

        # Fetch Data (Recent 1 month)
        start_date = "20231001"
        end_date = "20231101"

        # Use new helper function to fetch AKShare data
        # Handles market prefixes and column renaming automatically
        try:
            df = fetch_akshare_symbol(symbol, start_date, end_date)

            # Convert to Bars
            # load_bar_from_df now automatically fixes timestamp scaling issues
            bars = load_bar_from_df(df, symbol=symbol)

            all_bars.extend(bars)
        except Exception as e:
            print(f"Skipping {symbol}: {e}")

    # 4. Run Backtest with Risk Config
    print("\nRunning Backtest with Risk Rules...")

    # Run backtest directly with risk parameters
    result = akquant.run_backtest(
        strategy=AggressiveBuyerStrategy(),
        data=all_bars,
        symbols=symbols,
        instruments=instruments,
        initial_cash=initial_cash,
        risk_config={
            "max_position_pct": 0.10,
            "sector_concentration": (0.15, sector_map),
        },
        show_progress=True,
    )

    print("- Rule Applied: Max Position 10%")
    print("- Rule Applied: Sector 'Consumer' Max 15%")

    # 5. Analyze Results
    print("\nAnalysis:")

    # Use result.orders_df directly
    orders_df = result.orders_df

    if not orders_df.empty:
        # Check for Rejected status
        # Note: OrderStatus string representation is lowercase "rejected"
        rejected = orders_df[orders_df["status"].astype(str).str.lower() == "rejected"]

        filled = orders_df[orders_df["status"].astype(str).str.lower() == "filled"]

        print(f"Total Orders: {len(orders_df)}")
        print(f"Filled Orders: {len(filled)}")
        print(f"Rejected Orders: {len(rejected)}")

        if not rejected.empty:
            print("\nSample Rejections:")
            for idx, row in rejected.head(5).iterrows():
                print(
                    f"- {row['symbol']} ({row['created_at']}): {row['reject_reason']}"
                )
    else:
        print("No orders generated.")


if __name__ == "__main__":
    main()
