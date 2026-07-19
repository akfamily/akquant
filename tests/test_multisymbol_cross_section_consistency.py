from collections import Counter, defaultdict
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest


def _build_symbol_df(
    symbol: str, timestamps: list[pd.Timestamp], closes: list[float]
) -> pd.DataFrame:
    rows = []
    for ts, close in zip(timestamps, closes):
        rows.append(
            {
                "date": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 10000.0,
                "symbol": symbol,
            }
        )
    return pd.DataFrame(rows)


class BucketCrossSectionStrategy(Strategy):
    """Cross-section strategy that rebalances after timestamp completion."""

    def __init__(self, symbols: list[str], lookback: int = 2) -> None:
        """Initialize strategy state."""
        super().__init__()
        self.symbols = symbols
        self.lookback = lookback
        self.warmup_period = 0
        self.seen_by_ts: dict[int, set[str]] = defaultdict(set)
        self.complete_timestamps: list[int] = []
        self.selected_symbols: list[str] = []

    def on_bar(self, bar: Bar) -> None:
        """Collect symbols by timestamp and run cross-section logic once."""
        bucket = self.seen_by_ts[bar.timestamp]
        bucket.add(bar.symbol)
        if len(bucket) < len(self.symbols):
            return

        self.complete_timestamps.append(bar.timestamp)

        scores: dict[str, float] = {}
        for symbol in self.symbols:
            closes = self.get_history(count=self.lookback, symbol=symbol, field="close")
            if len(closes) < self.lookback:
                return
            scores[symbol] = float(closes[-1] / closes[0] - 1.0)

        best_symbol = max(scores, key=lambda symbol: scores[symbol])
        self.selected_symbols.append(best_symbol)

        for symbol in self.symbols:
            if symbol != best_symbol and self.get_position(symbol) > 0:
                self.close_position(symbol)
        self.order_target_percent(target_percent=0.8, symbol=best_symbol)


def _run_multisymbol_bucket(
    data_map: dict[str, pd.DataFrame], symbols: list[str]
) -> Any:
    return run_backtest(
        data=data_map,
        strategy=BucketCrossSectionStrategy,
        symbols=symbols,
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        history_depth=2,
        show_progress=False,
    )


def test_multisymbol_bucket_rebalances_once_per_complete_timestamp() -> None:
    """Rebalance exactly once for each complete timestamp slice."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-05 10:00:00", tz="Asia/Shanghai"),
    ]

    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 11.0, 12.0, 13.0]),
        "CCC": _build_symbol_df("CCC", timestamps, [10.0, 9.0, 8.0, 7.0]),
    }
    symbols = ["AAA", "BBB", "CCC"]

    result = _run_multisymbol_bucket(data_map, symbols)
    strategy = result.strategy

    assert len(strategy.complete_timestamps) == len(timestamps)
    assert all(len(v) == len(symbols) for v in strategy.seen_by_ts.values())
    assert strategy.selected_symbols == ["AAA", "BBB", "BBB", "BBB"]


def test_multisymbol_bucket_is_invariant_to_data_dict_order() -> None:
    """Keep cross-section outputs stable across dict insertion orders."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-05 10:00:00", tz="Asia/Shanghai"),
    ]

    map_a = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 11.0, 12.0, 13.0]),
        "CCC": _build_symbol_df("CCC", timestamps, [10.0, 9.0, 8.0, 7.0]),
    }
    map_b = {
        "CCC": _build_symbol_df("CCC", timestamps, [10.0, 9.0, 8.0, 7.0]),
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 11.0, 12.0, 13.0]),
    }
    symbols = ["AAA", "BBB", "CCC"]

    result_a = _run_multisymbol_bucket(map_a, symbols)
    result_b = _run_multisymbol_bucket(map_b, symbols)

    strategy_a = result_a.strategy
    strategy_b = result_b.strategy

    assert strategy_a.complete_timestamps == strategy_b.complete_timestamps
    assert strategy_a.selected_symbols == strategy_b.selected_symbols


def test_multisymbol_bucket_skips_incomplete_timestamp() -> None:
    """Skip timestamps that are missing at least one symbol."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-05 10:00:00", tz="Asia/Shanghai"),
    ]

    ccc_df = _build_symbol_df("CCC", timestamps, [10.0, 9.0, 8.0, 7.0]).drop(index=[1])

    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 11.0, 12.0, 13.0]),
        "CCC": ccc_df,
    }
    symbols = ["AAA", "BBB", "CCC"]

    result = _run_multisymbol_bucket(data_map, symbols)
    strategy = result.strategy

    assert len(strategy.complete_timestamps) == 3
    assert strategy.selected_symbols == ["AAA", "BBB", "BBB"]


def test_multisymbol_bucket_keeps_single_winner_position() -> None:
    """Keep at most one long position after each timestamp rebalance."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-05 10:00:00", tz="Asia/Shanghai"),
    ]

    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 11.0, 12.0, 13.0]),
        "CCC": _build_symbol_df("CCC", timestamps, [10.0, 9.0, 8.0, 7.0]),
    }
    symbols = ["AAA", "BBB", "CCC"]

    result = _run_multisymbol_bucket(data_map, symbols)
    positions = result.positions
    positive_counts = (positions > 0).sum(axis=1)

    assert (positive_counts <= 1).all()


class TargetWeightsStrategy(Strategy):
    """Strategy for validating multi-symbol target weight rebalancing."""

    def __init__(self, symbols: list[str]) -> None:
        """Initialize state for staged rotation rebalancing."""
        super().__init__()
        self.symbols = symbols
        self.pending: dict[int, set[str]] = defaultdict(set)
        self.rebalance_count = 0

    def on_bar(self, bar: Bar) -> None:
        """Rebalance once all symbols of current timestamp are collected."""
        bucket = self.pending[bar.timestamp]
        bucket.add(bar.symbol)
        if len(bucket) < len(self.symbols):
            return
        self.pending.pop(bar.timestamp, None)

        if self.rebalance_count == 0:
            self.rebalance_weights(
                {"AAA": 0.9},
                liquidate_unmentioned=True,
                rebalance_tolerance=0.0,
            )
        elif self.rebalance_count in (1, 2):
            self.rebalance_weights(
                {"BBB": 0.9},
                liquidate_unmentioned=True,
                rebalance_tolerance=0.0,
            )
        self.rebalance_count += 1


class TargetWeightsSplitStrategy(Strategy):
    """Strategy for validating split target weights."""

    def __init__(self, symbols: list[str]) -> None:
        """Initialize state for one-shot split allocation."""
        super().__init__()
        self.symbols = symbols
        self.pending: dict[int, set[str]] = defaultdict(set)
        self.rebalanced = False

    def on_bar(self, bar: Bar) -> None:
        """Apply target split after timestamp bucket is complete."""
        bucket = self.pending[bar.timestamp]
        bucket.add(bar.symbol)
        if len(bucket) < len(self.symbols):
            return
        self.pending.pop(bar.timestamp, None)

        if not self.rebalanced:
            self.rebalance_weights(
                {"AAA": 0.6, "BBB": 0.3},
                liquidate_unmentioned=True,
                rebalance_tolerance=0.0,
            )
            self.rebalanced = True


class TargetWeightsSellThenBuyStrategy(Strategy):
    """Strategy for validating same-cycle sell-then-buy cash release."""

    def __init__(self, symbols: list[str]) -> None:
        """Initialize staged rotation state."""
        super().__init__()
        self.symbols = symbols
        self.pending: dict[int, set[str]] = defaultdict(set)
        self.rebalance_count = 0
        self.order_event_ids: list[str] = []
        self.trade_order_ids: list[str] = []

    def on_bar(self, bar: Bar) -> None:
        """Rotate from AAA to BBB on the second completed timestamp."""
        bucket = self.pending[bar.timestamp]
        bucket.add(bar.symbol)
        if len(bucket) < len(self.symbols):
            return
        self.pending.pop(bar.timestamp, None)

        if self.rebalance_count == 0:
            self.rebalance_weights(
                {"AAA": 0.9},
                liquidate_unmentioned=True,
                rebalance_tolerance=0.0,
            )
        elif self.rebalance_count == 1:
            self.rebalance_weights(
                {"BBB": 0.9},
                liquidate_unmentioned=True,
                rebalance_tolerance=0.0,
            )
        self.rebalance_count += 1

    def on_order(self, order: Any) -> None:
        """Record order ids observed through callbacks."""
        self.order_event_ids.append(str(order.id))

    def on_trade(self, trade: Any) -> None:
        """Record trade order ids observed through callbacks."""
        self.trade_order_ids.append(str(trade.order_id))


class ExplicitQuantityRejectStrategy(Strategy):
    """Strategy for validating explicit buy quantity rejects."""

    def __init__(self) -> None:
        """Initialize single-submit state."""
        super().__init__()
        self.submitted = False

    def on_bar(self, bar: Bar) -> None:
        """Submit one oversized explicit buy order."""
        if self.submitted:
            return
        self.buy(symbol=bar.symbol, quantity=20_000.0)
        self.submitted = True


def test_rebalance_weights_rotation_liquidates_unmentioned_symbols() -> None:
    """Target weights should support one-call rotation and clear old holdings."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-05 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-06 10:00:00", tz="Asia/Shanghai"),
    ]
    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 10.0, 10.0, 10.0, 10.0]),
    }

    result = run_backtest(
        data=data_map,
        strategy=TargetWeightsStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        show_progress=False,
    )

    final_positions = result.positions.iloc[-1]
    assert float(final_positions.get("AAA", 0.0)) == 0.0
    assert float(final_positions.get("BBB", 0.0)) > 0.0


def test_rebalance_weights_uses_sell_proceeds_before_same_cycle_buy() -> None:
    """Rotation should use pending sell proceeds instead of shrinking the buy leg."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
    ]
    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 10.0, 10.0]),
    }

    result = run_backtest(
        data=data_map,
        strategy=TargetWeightsSellThenBuyStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        show_progress=False,
    )

    orders_df = result.orders_df.sort_values("created_at").reset_index(drop=True)
    assert len(orders_df) == 3
    assert sorted(orders_df["symbol"].astype(str).tolist()) == ["AAA", "AAA", "BBB"]
    assert sorted(orders_df["side"].astype(str).tolist()) == ["buy", "buy", "sell"]
    assert sorted(orders_df["quantity"].astype(float).tolist()) == [
        9000.0,
        9000.0,
        9000.0,
    ]
    assert set(orders_df["status"].astype(str).str.lower()) == {"filled"}
    aaa_buy = orders_df[
        (orders_df["symbol"] == "AAA") & (orders_df["side"] == "buy")
    ].iloc[0]
    aaa_sell = orders_df[
        (orders_df["symbol"] == "AAA") & (orders_df["side"] == "sell")
    ].iloc[0]
    bbb_buy = orders_df[
        (orders_df["symbol"] == "BBB") & (orders_df["side"] == "buy")
    ].iloc[0]
    assert aaa_buy["updated_at"] == timestamps[0]
    assert aaa_sell["updated_at"] == timestamps[1]
    assert bbb_buy["updated_at"] == timestamps[1]
    final_positions = result.positions.iloc[-1]
    assert float(final_positions.get("AAA", 0.0)) == 0.0
    assert float(final_positions.get("BBB", 0.0)) == 9000.0


def test_rebalance_positions_supports_same_cycle_long_short_rotation() -> None:
    """Advanced target positions should support one-call long-short rotation."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
    ]
    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 10.0, 10.0]),
    }

    result = run_backtest(
        data=data_map,
        strategy=TargetPositionsLongShortRotationStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        risk_config={"account_mode": "margin", "enable_short_sell": True},
        show_progress=False,
    )

    orders_df = result.orders_df.sort_values("created_at").reset_index(drop=True)
    assert set(orders_df["status"].astype(str).str.lower()) == {"filled"}
    final_positions = result.positions.iloc[-1]
    assert float(final_positions.get("AAA", 0.0)) == -100.0
    assert float(final_positions.get("BBB", 0.0)) == 50.0


def test_rebalance_weights_split_allocation_is_close_to_target() -> None:
    """Target weights should allocate multi-symbol positions by portfolio ratio."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
    ]
    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 10.0]),
    }

    result = run_backtest(
        data=data_map,
        strategy=TargetWeightsSplitStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        show_progress=False,
    )

    final_positions = result.positions.iloc[-1]
    aaa_value = float(final_positions.get("AAA", 0.0)) * 10.0
    bbb_value = float(final_positions.get("BBB", 0.0)) * 10.0
    total_value = float(result.equity_curve.iloc[-1])

    assert abs(aaa_value / total_value - 0.6) < 0.02
    assert abs(bbb_value / total_value - 0.3) < 0.02


def test_explicit_buy_quantity_is_rejected_without_resizing() -> None:
    """Explicit oversized buy quantity should reject and preserve the requested size."""
    timestamps = [pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai")]
    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0]),
    }

    result = run_backtest(
        data=data_map,
        strategy=ExplicitQuantityRejectStrategy,
        symbols=["AAA"],
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        show_progress=False,
    )

    orders_df = result.orders_df
    assert len(orders_df) == 1
    row = orders_df.iloc[0]
    assert str(row["symbol"]) == "AAA"
    assert float(row["quantity"]) == 20_000.0
    assert float(row["filled_quantity"]) == 0.0
    assert str(row["status"]).lower() == "rejected"
    assert "insufficient" in str(row["reject_reason"]).lower()


def _build_validation_strategy() -> Strategy:
    """Build strategy with minimal mock context for validation tests."""
    strategy = TargetWeightsSplitStrategy(symbols=["AAA", "BBB"])
    strategy.ctx = MagicMock()
    strategy.ctx.positions = {}
    strategy.ctx.cash = 100000.0
    strategy.ctx.get_position.return_value = 0.0
    return strategy


def test_rebalance_weights_rejects_negative_tolerance() -> None:
    """Reject negative rebalance tolerance values."""
    strategy = _build_validation_strategy()
    with pytest.raises(ValueError, match="rebalance_tolerance must be >= 0"):
        strategy.rebalance_weights({"AAA": 0.5}, rebalance_tolerance=-0.01)


def test_rebalance_weights_rejects_weight_sum_over_one_without_leverage() -> None:
    """Reject total weights above one when leverage is disabled."""
    strategy = _build_validation_strategy()
    with pytest.raises(ValueError, match="exceeds 1.0"):
        strategy.rebalance_weights({"AAA": 0.7, "BBB": 0.4})


def test_rebalance_weights_rejects_negative_weight() -> None:
    """Reject negative target weight input."""
    strategy = _build_validation_strategy()
    with pytest.raises(ValueError, match="must be >= 0"):
        strategy.rebalance_weights({"AAA": -0.1})


def test_rebalance_weights_rejects_empty_symbol() -> None:
    """Reject empty symbol key in target weights."""
    strategy = _build_validation_strategy()
    with pytest.raises(ValueError, match="must be non-empty"):
        strategy.rebalance_weights({"": 0.2})


class SellThenBuySameCycleStrategy(Strategy):
    """Validate same-cycle rotation releases sell proceeds before buy sizing."""

    def __init__(self) -> None:
        """Initialize rotation state."""
        super().__init__()
        self.step = 0

    def on_bar(self, bar: Bar) -> None:
        """Rotate from AAA to BBB on the second complete timestamp."""
        if bar.symbol != "BBB":
            return
        if self.step == 0:
            self.order_target_percent(symbol="AAA", target_percent=0.95)
        elif self.step == 1:
            self.order_target_percent(symbol="AAA", target_percent=0.0)
            self.order_target_percent(symbol="BBB", target_percent=0.95)
        self.step += 1


class DailyRebalanceAfterBarSellThenBuySameCycleStrategy(Strategy):
    """Validate after-bar rebalance callbacks observe terminal same-cycle fills."""

    def __init__(self) -> None:
        """Initialize staged rebalance state and callback logs."""
        super().__init__()
        self.step = 0
        self.order_events: list[tuple[str, str, str]] = []
        self.trade_events: list[tuple[str, str]] = []

    def on_cross_section(self, trading_date: object, timestamp: int) -> None:
        """Rotate from AAA to BBB during the after-bar rebalance hook."""
        _ = (trading_date, timestamp)
        if self.step == 1:
            self.order_target_percent(symbol="AAA", target_percent=0.95)
        elif self.step == 2:
            self.order_target_percent(symbol="AAA", target_percent=0.0)
            self.order_target_percent(symbol="BBB", target_percent=0.95)
        self.step += 1

    def on_order(self, order: Any) -> None:
        """Record observable order callbacks."""
        self.order_events.append(
            (
                str(order.symbol),
                str(order.side).split(".")[-1].lower(),
                str(order.status).split(".")[-1].lower(),
            )
        )

    def on_trade(self, trade: Any) -> None:
        """Record observable trade callbacks."""
        self.trade_events.append(
            (
                str(trade.symbol),
                str(trade.side).split(".")[-1].lower(),
            )
        )


class DailyRebalanceAfterBarCrossSymbolCarryoverStrategy(Strategy):
    """Cover after-bar rebalance with retained positions and price drift."""

    def __init__(self) -> None:
        """Initialize staged rebalance state and reject capture."""
        super().__init__()
        self.step = 0
        self.rejected_reasons: list[str] = []

    def on_cross_section(self, trading_date: object, timestamp: int) -> None:
        """Rotate targets across days while retaining one drifting position."""
        _ = (trading_date, timestamp)
        if self.step == 0:
            self.rebalance_weights(
                {"AAA": 0.49, "BBB": 0.49},
                liquidate_unmentioned=True,
            )
        elif self.step == 1:
            self.rebalance_weights(
                {"BBB": 0.49, "CCC": 0.49},
                liquidate_unmentioned=True,
            )
        elif self.step == 2:
            self.rebalance_weights(
                {"BBB": 0.90},
                liquidate_unmentioned=True,
            )
        self.step += 1

    def on_reject(self, order: Any) -> None:
        """Capture unexpected reject reasons for assertion."""
        self.rejected_reasons.append(str(order.reject_reason))


class TargetPositionsLongShortRotationStrategy(Strategy):
    """Exercise multi-symbol signed target positions through one advanced API call."""

    def __init__(self) -> None:
        """Initialize the staged rotation state."""
        self.step = 0

    def on_bar(self, bar: Bar) -> None:
        """Submit staged target-position transitions on the BBB bar."""
        if bar.symbol != "BBB":
            return
        if self.step == 0:
            self.rebalance_positions({"AAA": 100.0}, liquidate_unmentioned=True)
        elif self.step == 1:
            self.rebalance_positions(
                {"AAA": -100.0, "BBB": 50.0},
                liquidate_unmentioned=True,
            )
        self.step += 1


def test_same_cycle_sell_then_buy_uses_post_sell_cash_for_sizing() -> None:
    """Current-close same-cycle rotation should size buy with post-sell cash."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
    ]
    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 10.0, 10.0]),
    }

    result = run_backtest(
        data=data_map,
        strategy=SellThenBuySameCycleStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        show_progress=False,
    )

    orders_df = result.orders_df.sort_values("created_at").reset_index(drop=True)
    assert not any(
        str(status).lower() == "rejected" for status in orders_df["status"].tolist()
    )
    aaa_buy = orders_df[
        (orders_df["symbol"] == "AAA") & (orders_df["side"] == "buy")
    ].iloc[0]
    aaa_sell = orders_df[
        (orders_df["symbol"] == "AAA") & (orders_df["side"] == "sell")
    ].iloc[0]
    bbb_buys = orders_df[(orders_df["symbol"] == "BBB") & (orders_df["side"] == "buy")]
    assert not bbb_buys.empty
    assert set(bbb_buys["status"].astype(str).str.lower()) == {"filled"}
    assert aaa_buy["updated_at"] == timestamps[0]
    assert aaa_sell["updated_at"] == timestamps[1]
    assert bbb_buys.iloc[0]["updated_at"] == timestamps[1]
    assert float(bbb_buys.iloc[0]["filled_quantity"]) >= 499.0
    final_positions = result.positions.iloc[-1]
    assert float(final_positions.get("AAA", 0.0)) == 0.0
    assert float(final_positions.get("BBB", 0.0)) >= 499.0


def test_cross_section_same_cycle_terminal_fill_emits_callbacks() -> None:
    """Last-timestamp same-cycle fills should still reach order/trade callbacks."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
    ]
    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 10.0, 10.0]),
    }

    strategy = DailyRebalanceAfterBarSellThenBuySameCycleStrategy()
    result = run_backtest(
        data=data_map,
        strategy=strategy,
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        show_progress=False,
    )

    orders_df = result.orders_df.sort_values("created_at").reset_index(drop=True)
    bbb_buys = orders_df[(orders_df["symbol"] == "BBB") & (orders_df["side"] == "buy")]
    assert not bbb_buys.empty
    assert set(bbb_buys["status"].astype(str).str.lower()) == {"filled"}
    assert ("BBB", "buy", "filled") in strategy.order_events
    assert ("BBB", "buy") in strategy.trade_events


def test_after_bar_same_cycle_fill_preserves_all_order_callbacks() -> None:
    """Immediate same-step fills should preserve the full on_order lifecycle."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
    ]
    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 10.0, 10.0]),
    }

    strategy = DailyRebalanceAfterBarSellThenBuySameCycleStrategy()
    run_backtest(
        data=data_map,
        strategy=strategy,
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        show_progress=False,
    )

    assert Counter(strategy.order_events) == Counter(
        [
            ("AAA", "buy", "new"),
            ("AAA", "buy", "filled"),
            ("AAA", "sell", "new"),
            ("AAA", "sell", "filled"),
            ("BBB", "buy", "new"),
            ("BBB", "buy", "filled"),
        ]
    )
    assert Counter(strategy.trade_events) == Counter(
        [
            ("AAA", "buy"),
            ("AAA", "sell"),
            ("BBB", "buy"),
        ]
    )


def test_after_bar_same_cycle_fill_preserves_result_views() -> None:
    """After-bar same-cycle rotation should keep result views complete."""
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
    ]
    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 10.0, 10.0]),
    }

    result = run_backtest(
        data=data_map,
        strategy=DailyRebalanceAfterBarSellThenBuySameCycleStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        show_progress=False,
    )

    orders_df = result.orders_df.sort_values("created_at").reset_index(drop=True)
    assert len(orders_df) == 3
    assert sorted(orders_df["symbol"].astype(str).tolist()) == ["AAA", "AAA", "BBB"]
    assert sorted(orders_df["side"].astype(str).tolist()) == ["buy", "buy", "sell"]
    assert set(orders_df["status"].astype(str).str.lower()) == {"filled"}
    aaa_buy = orders_df[
        (orders_df["symbol"] == "AAA") & (orders_df["side"] == "buy")
    ].iloc[0]
    aaa_sell = orders_df[
        (orders_df["symbol"] == "AAA") & (orders_df["side"] == "sell")
    ].iloc[0]
    bbb_buy = orders_df[
        (orders_df["symbol"] == "BBB") & (orders_df["side"] == "buy")
    ].iloc[0]
    assert aaa_buy["updated_at"] == timestamps[1]
    assert aaa_sell["updated_at"] == timestamps[2]
    assert bbb_buy["updated_at"] == timestamps[2]
    assert float(aaa_buy["filled_quantity"]) == 9500.0
    assert float(aaa_sell["filled_quantity"]) == 9500.0
    assert float(bbb_buy["filled_quantity"]) == 9500.0

    executions_df = result.executions_df.sort_values("timestamp").reset_index(drop=True)
    assert len(executions_df) == 3
    assert sorted(executions_df["symbol"].astype(str).tolist()) == ["AAA", "AAA", "BBB"]
    assert sorted(executions_df["side"].astype(str).str.lower().tolist()) == [
        "buy",
        "buy",
        "sell",
    ]
    assert (
        float(
            executions_df[
                (executions_df["symbol"] == "AAA")
                & (executions_df["side"].astype(str).str.lower() == "buy")
            ].iloc[0]["quantity"]
        )
        == 9500.0
    )
    assert (
        float(
            executions_df[
                (executions_df["symbol"] == "AAA")
                & (executions_df["side"].astype(str).str.lower() == "sell")
            ].iloc[0]["quantity"]
        )
        == 9500.0
    )
    assert (
        float(
            executions_df[
                (executions_df["symbol"] == "BBB")
                & (executions_df["side"].astype(str).str.lower() == "buy")
            ].iloc[0]["quantity"]
        )
        == 9500.0
    )

    # trades_df stores closed round trips, so only the closed AAA leg should appear.
    trades_df = result.trades_df.sort_values("entry_time").reset_index(drop=True)
    assert len(trades_df) == 1
    assert str(trades_df.iloc[0]["symbol"]) == "AAA"
    assert str(trades_df.iloc[0]["side"]) == "Long"
    assert float(trades_df.iloc[0]["quantity"]) == 9500.0
    assert trades_df.iloc[0]["entry_time"] == timestamps[1]
    assert trades_df.iloc[0]["exit_time"] == timestamps[2]


def test_after_bar_uses_complete_timestamp_prices_for_cross_symbol_targets() -> None:
    """After-bar rebalance should wait for a complete slice before sizing."""
    timestamps = [
        pd.Timestamp("2022-12-30 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-05 10:00:00", tz="Asia/Shanghai"),
    ]
    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [0.98, 0.99, 1.0, 1.0, 1.0, 1.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [1.0, 0.95, 1.0, 1.0, 2.0, 2.0]),
        "CCC": _build_symbol_df("CCC", timestamps, [1.02, 0.98, 1.0, 1.0, 1.0, 1.0]),
    }

    strategy = DailyRebalanceAfterBarCrossSymbolCarryoverStrategy()
    result = run_backtest(
        data=data_map,
        strategy=strategy,
        symbols=["AAA", "BBB", "CCC"],
        initial_cash=1_000_000.0,
        commission_rate=0.00005,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=5.0,
        lot_size=100,
        history_depth=2,
        start_time="2023-01-02",
        fill_policy={"price_basis": "close", "bar_offset": 0, "temporal": "same_cycle"},
        show_progress=False,
    )

    orders_df = result.orders_df.sort_values("created_at").reset_index(drop=True)
    assert strategy.rejected_reasons == []
    assert not any(
        str(status).lower() == "rejected" for status in orders_df["status"].tolist()
    )
    final_positions = result.positions.iloc[-1]
    assert float(final_positions.get("AAA", 0.0)) == 0.0
    assert float(final_positions.get("CCC", 0.0)) == 0.0
    assert float(final_positions.get("BBB", 0.0)) == 670400.0
