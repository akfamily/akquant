from collections import defaultdict
from typing import Any, DefaultDict

import pandas as pd
import pytest
from akquant import Bar, CurrentClose, Strategy, run_backtest


class ExposureCapacityStrategy(Strategy):
    """Create deterministic orders for exposure and capacity checks."""

    def __init__(self) -> None:
        """Initialize strategy state."""
        super().__init__()
        self.step = 0

    def on_bar(self, bar: Bar) -> None:
        """Place deterministic orders for exposure and capacity tests."""
        self.step += 1
        if self.step == 1:
            self.buy(symbol=bar.symbol, quantity=100, tag="entry")
        elif self.step == 3:
            self.sell(symbol=bar.symbol, quantity=40, tag="reduce")
        elif self.step == 5:
            self.close_position(symbol=bar.symbol)


class FlatAtEndStrategy(Strategy):
    """Open on the first bar and fully close well before the final bar."""

    def __init__(self) -> None:
        """Initialize strategy state."""
        super().__init__()
        self.step = 0

    def on_bar(self, bar: Bar) -> None:
        """Enter once, then flatten so the run ends with no position."""
        self.step += 1
        if self.step == 1:
            self.buy(symbol=bar.symbol, quantity=100, tag="entry")
        elif self.step == 3:
            self.close_position(symbol=bar.symbol)


class HoldToEndStrategy(Strategy):
    """Open on the first bar and hold the position to the end."""

    def __init__(self) -> None:
        """Initialize strategy state."""
        super().__init__()
        self.step = 0

    def on_bar(self, bar: Bar) -> None:
        """Enter once and never close."""
        self.step += 1
        if self.step == 1:
            self.buy(symbol=bar.symbol, quantity=100, tag="entry")


class AttributionBySymbolStrategy(Strategy):
    """Create multi-symbol tagged trades for attribution checks."""

    def __init__(self) -> None:
        """Initialize per-symbol step counters."""
        super().__init__()
        self.steps: DefaultDict[str, int] = defaultdict(int)

    def on_bar(self, bar: Bar) -> None:
        """Open then close positions with symbol-specific tags."""
        self.steps[bar.symbol] += 1
        step = self.steps[bar.symbol]
        if step == 1:
            self.buy(symbol=bar.symbol, quantity=50, tag=f"entry_{bar.symbol}")
        elif step == 3 and self.get_position(bar.symbol) > 0:
            self.sell(
                symbol=bar.symbol,
                quantity=self.get_position(bar.symbol),
                tag=f"exit_{bar.symbol}",
            )


def _build_single_symbol_data() -> list[Bar]:
    data: list[Bar] = []
    closes = [10.0, 10.5, 11.0, 10.8, 11.2, 11.4]
    for i, close in enumerate(closes):
        data.append(
            Bar(
                timestamp=pd.Timestamp(f"2023-01-{i + 1:02d} 10:00:00").value,
                open=close,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=5000.0,
                symbol="TEST",
            )
        )
    return data


def _build_multi_symbol_data() -> dict[str, pd.DataFrame]:
    timestamps = [
        pd.Timestamp("2023-01-01 10:00:00"),
        pd.Timestamp("2023-01-02 10:00:00"),
        pd.Timestamp("2023-01-03 10:00:00"),
        pd.Timestamp("2023-01-04 10:00:00"),
    ]
    rows_aaa = []
    rows_bbb = []
    closes_aaa = [10.0, 11.0, 12.0, 13.0]
    closes_bbb = [10.0, 9.5, 9.0, 8.5]
    for ts, c1, c2 in zip(timestamps, closes_aaa, closes_bbb):
        rows_aaa.append(
            {
                "date": ts,
                "open": c1,
                "high": c1,
                "low": c1,
                "close": c1,
                "volume": 20000.0,
                "symbol": "AAA",
            }
        )
        rows_bbb.append(
            {
                "date": ts,
                "open": c2,
                "high": c2,
                "low": c2,
                "close": c2,
                "volume": 20000.0,
                "symbol": "BBB",
            }
        )
    return {"AAA": pd.DataFrame(rows_aaa), "BBB": pd.DataFrame(rows_bbb)}


def test_exposure_df_and_capacity_df_basic_properties() -> None:
    """Exposure and capacity outputs should have stable core invariants."""
    result = run_backtest(
        data=_build_single_symbol_data(),
        strategy=ExposureCapacityStrategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
    )

    exposure = result.exposure_df()
    assert not exposure.empty
    assert {
        "date",
        "equity",
        "long_exposure",
        "short_exposure",
        "net_exposure",
        "gross_exposure",
        "net_exposure_pct",
        "gross_exposure_pct",
        "leverage",
    }.issubset(exposure.columns)
    assert (exposure["gross_exposure"] >= exposure["net_exposure"].abs()).all()
    assert (exposure["leverage"] >= 0.0).all()

    capacity = result.capacity_df()
    assert not capacity.empty
    assert {
        "date",
        "order_count",
        "filled_order_count",
        "ordered_quantity",
        "filled_quantity",
        "ordered_value",
        "filled_value",
        "fill_rate_qty",
        "fill_rate_value",
        "equity",
        "turnover",
    }.issubset(capacity.columns)
    assert float(capacity["order_count"].sum()) >= 3.0
    assert (
        (capacity["fill_rate_qty"] >= 0.0) & (capacity["fill_rate_qty"] <= 1.0)
    ).all()


def test_attribution_df_keeps_total_pnl_consistent() -> None:
    """Attribution aggregation should conserve total strategy PnL."""
    result = run_backtest(
        data=_build_multi_symbol_data(),
        strategy=AttributionBySymbolStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=300000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy=CurrentClose(),
        history_depth=2,
        show_progress=False,
    )

    attribution_symbol = result.attribution_df(by="symbol")
    assert not attribution_symbol.empty
    assert set(attribution_symbol["group"]) == {"AAA", "BBB"}
    total_net_pnl = float(result.trades_df["net_pnl"].sum())
    assert float(attribution_symbol["total_pnl"].sum()) == pytest.approx(
        total_net_pnl, rel=1e-9
    )

    attribution_tag = result.attribution_df(by="tag")
    assert not attribution_tag.empty
    if total_net_pnl != 0.0:
        assert float(attribution_tag["contribution_pct"].sum()) == pytest.approx(
            1.0, rel=1e-9
        )


def _run_single_symbol(strategy: type[Strategy]) -> Any:
    """Run a single-symbol backtest through the Python result wrapper."""
    return run_backtest(
        data=_build_single_symbol_data(),
        strategy=strategy,
        symbols="TEST",
        initial_cash=200000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
    )


def test_open_position_count_reflects_final_snapshot() -> None:
    """`open_position_count` must track the final snapshot, not the last held one.

    `positions_df` only carries rows for symbols actually held, so its latest
    date lags the backtest end once everything is closed out. Reading the count
    from there reported a stale open position in `backtest_report.html`.
    """
    flat = _run_single_symbol(FlatAtEndStrategy)
    assert float(flat.metrics_df.loc["open_position_count", "value"]) == pytest.approx(
        0.0
    )
    # The last held snapshot predates the backtest end, which is what used to
    # leak through as a phantom open position.
    assert flat.positions_df["date"].max() < flat.positions.index[-1]
    assert float(flat.positions.iloc[-1].abs().sum()) == pytest.approx(0.0)

    held = _run_single_symbol(HoldToEndStrategy)
    assert float(held.metrics_df.loc["open_position_count", "value"]) == pytest.approx(
        1.0
    )
