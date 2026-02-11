import pandas as pd
from akquant import Bar, Strategy, run_backtest


class TestStrategy(Strategy):
    """Strategy for testing orders dataframe."""

    def on_bar(self, bar: Bar) -> None:
        """Execute on every bar."""
        # Day 1: Close=90 -> Buy Limit 100
        # Day 2: Close=93 -> Buy Limit 100
        # ...
        if bar.close < 100:
            self.buy(bar.symbol, 100, price=100.0)
        elif bar.close > 110:
            self.sell(bar.symbol, 100)


def test_orders_df() -> None:
    """Test the structure and content of orders_df."""
    data = []
    for i in range(10):
        data.append(
            Bar(
                timestamp=pd.Timestamp(f"2023-01-{i + 1:02d} 10:00:00").value,
                open=90.0 + i * 3,
                high=95.0 + i * 3,
                low=85.0 + i * 3,
                close=90.0 + i * 3,
                volume=1000,
                symbol="TEST",
            )
        )

    print("Running backtest...")
    result = run_backtest(
        data=data, strategy=TestStrategy, symbol="TEST", show_progress=False
    )

    print("\nOrders DataFrame:")
    print(result.orders_df)

    # Check columns
    expected = [
        "id",
        "side",
        "symbol",
        "created_at",
        "quantity",
        "limit_price",
        "avg_price",
        "commission",
        "status",
        "order_type",
        "stop_price",
        "filled_quantity",
        "time_in_force",
    ]
    missing = [c for c in expected if c not in result.orders_df.columns]
    assert not missing, f"Missing columns: {missing}"

    print("\nAll expected columns present.")
    print("Date column dtype:", result.orders_df["created_at"].dtype)

    # Check first order (Limit Buy)
    first_order = result.orders_df.iloc[0]
    print("\nFirst Order:")
    print(first_order)

    assert first_order["symbol"] == "TEST"
    assert first_order["quantity"] == 100.0
    assert first_order["commission"] > 0, "Fees should be greater than 0"
    assert first_order["order_type"] == "limit"
    assert first_order["filled_quantity"] == 100.0

    # Verify fees logic roughly (commission + transfer fee)
    # Buy at 93.0. Value 9300. Commission 5 (min). Transfer 0.093. Total 5.093.
    # Allow some float precision margin
    if first_order["avg_price"] == 93.0 and first_order["side"] == "buy":
        assert abs(first_order["commission"] - 5.093) < 0.001

    # Check a market sell order
    sell_orders = result.orders_df[result.orders_df["side"] == "sell"]
    if not sell_orders.empty:
        sell_order = sell_orders.iloc[0]
        assert sell_order["order_type"] == "market"
        assert pd.isna(sell_order["limit_price"])
