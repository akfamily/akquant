import pandas as pd
from akquant import Bar, CurrentClose, OrderStatus, Strategy, run_backtest


class PartialFillOpenOrderStrategy(Strategy):
    """Strategy for verifying partial-fill status semantics."""

    def __init__(self) -> None:
        """Initialize state."""
        super().__init__()
        self.bar_count = 0
        self.observed_partial = False
        self.open_order_counts: list[int] = []

    def on_bar(self, bar: Bar) -> None:
        """Place one oversized order and track open-order status."""
        self.bar_count += 1
        open_orders = self.get_open_orders(bar.symbol)
        self.open_order_counts.append(len(open_orders))

        if any(o.status == OrderStatus.PartiallyFilled for o in open_orders):
            self.observed_partial = True

        if self.bar_count == 1:
            self.buy(bar.symbol, 100)


def test_partial_filled_status_and_open_orders_visibility() -> None:
    """Partial fills should expose PartiallyFilled status and remain open."""
    data = []
    for i in range(3):
        data.append(
            Bar(
                timestamp=pd.Timestamp(f"2023-01-0{i + 1} 10:00:00").value,
                open=10.0,
                high=10.0,
                low=10.0,
                close=10.0,
                volume=100.0,
                symbol="TEST",
            )
        )

    result = run_backtest(
        data=data,
        strategy=PartialFillOpenOrderStrategy,
        symbols="TEST",
        initial_cash=100000.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        volume_limit_pct=0.1,
        show_progress=False,
    )

    strategy = result.strategy
    assert strategy is not None
    assert strategy.open_order_counts[0] == 0
    assert strategy.observed_partial
    assert strategy.open_order_counts[-1] >= 1

    orders_df = result.orders_df
    assert not orders_df.empty
    order = orders_df.iloc[0]
    assert order["status"] == "partiallyfilled"
    assert 0.0 < order["filled_quantity"] < order["quantity"]


class CancelOpenOrderStrategy(Strategy):
    """Strategy for verifying cancelled orders disappear from open orders."""

    def __init__(self) -> None:
        """Initialize counters for open-order visibility checks."""
        super().__init__()
        self.bar_count = 0
        self.open_order_counts: list[int] = []
        self.open_order_statuses: list[list[OrderStatus]] = []
        self.open_order_counts_after_cancel: list[int] = []

    def on_bar(self, bar: Bar) -> None:
        """Submit then cancel a resting limit order across consecutive bars."""
        self.bar_count += 1
        open_orders = self.get_open_orders(bar.symbol)
        self.open_order_counts.append(len(open_orders))
        self.open_order_statuses.append([o.status for o in open_orders])

        if self.bar_count == 1:
            self.buy(bar.symbol, 100, price=9.0)
        elif self.bar_count == 2:
            self.cancel_all_orders(bar.symbol)
            self.open_order_counts_after_cancel.append(
                len(self.get_open_orders(bar.symbol))
            )


def test_cancelled_order_is_removed_from_open_orders_and_results() -> None:
    """Cancelled pending order should disappear and end as cancelled."""
    data = []
    for i in range(4):
        data.append(
            Bar(
                timestamp=pd.Timestamp(f"2023-01-0{i + 1} 10:00:00").value,
                open=10.0,
                high=10.0,
                low=10.0,
                close=10.0,
                volume=100.0,
                symbol="TEST",
            )
        )

    result = run_backtest(
        data=data,
        strategy=CancelOpenOrderStrategy,
        symbols="TEST",
        initial_cash=100000.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        lot_size=1,
        show_progress=False,
    )

    strategy = result.strategy
    assert strategy is not None
    assert strategy.open_order_counts[:4] == [0, 1, 0, 0]
    assert strategy.open_order_counts_after_cancel == [0]
    assert strategy.open_order_statuses[1] == [OrderStatus.New]
    assert strategy.open_order_statuses[2] == []

    orders_df = result.orders_df
    assert len(orders_df) == 1
    order = orders_df.iloc[0]
    assert order["status"] == "cancelled"
    assert order["filled_quantity"] == 0.0


class TimerCancelOpenOrderStrategy(Strategy):
    """Strategy for verifying timer-driven cancel_all_orders behavior."""

    def __init__(self) -> None:
        """Initialize timer and bar-side open-order counters."""
        super().__init__()
        self.timer_open_before_cancel: list[int] = []
        self.timer_open_after_cancel: list[int] = []
        self.bar_open_counts: list[int] = []

    def on_start(self) -> None:
        """Schedule a prepare-next-day style timer between two bars."""
        self.schedule(
            pd.Timestamp("2023-01-02 10:00:01", tz="Asia/Shanghai"),
            "prepare_next_day",
        )

    def on_bar(self, bar: Bar) -> None:
        """Place a resting limit order and track bar-side open-order counts."""
        self.bar_open_counts.append(len(self.get_open_orders(bar.symbol)))
        if len(self.bar_open_counts) == 1:
            self.buy(bar.symbol, 100, price=9.0)

    def on_timer(self, payload: str) -> None:
        """Cancel all open orders and verify immediate timer-side visibility."""
        if payload != "prepare_next_day" or self.current_bar is None:
            return
        self.timer_open_before_cancel.append(
            len(self.get_open_orders(self.current_bar.symbol))
        )
        self.cancel_all_orders(self.current_bar.symbol)
        self.timer_open_after_cancel.append(
            len(self.get_open_orders(self.current_bar.symbol))
        )


def test_timer_prepare_next_day_cancel_updates_open_orders_immediately() -> None:
    """Timer-driven cancel should hide open orders immediately and persist."""
    data = []
    for i in range(4):
        data.append(
            Bar(
                timestamp=pd.Timestamp(f"2023-01-0{i + 1} 10:00:00").value,
                open=10.0,
                high=10.0,
                low=10.0,
                close=10.0,
                volume=100.0,
                symbol="TEST",
            )
        )

    result = run_backtest(
        data=data,
        strategy=TimerCancelOpenOrderStrategy,
        symbols="TEST",
        initial_cash=100000.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        lot_size=1,
        show_progress=False,
    )

    strategy = result.strategy
    assert strategy is not None
    assert strategy.bar_open_counts[0] == 0
    assert strategy.bar_open_counts[1:] == [0, 0, 0]
    assert strategy.timer_open_before_cancel == [1]
    assert strategy.timer_open_after_cancel == [0]

    orders_df = result.orders_df
    assert len(orders_df) == 1
    order = orders_df.iloc[0]
    assert order["status"] == "cancelled"
    assert order["filled_quantity"] == 0.0
