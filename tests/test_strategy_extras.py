import datetime as dt
import logging
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional, cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from akquant import (
    BacktestConfig,
    InstrumentConfig,
    LogConfig,
    StrategyConfig,
    configure_logging,
    load_checkpoint,
    register_logger,
    register_strategy_loader,
    run_backtest,
    run_from_checkpoint,
    save_checkpoint,
)
from akquant.akquant import (
    Bar,
    ExecutionMode,
    OrderStatus,
    PositionEffect,
    StrategyContext,
    Tick,
    TimeInForce,
)
from akquant.backtest import FunctionalStrategy
from akquant.backtest.engine import (
    _build_trading_day_metadata,
    _prime_framework_boundary_timers,
    _prime_framework_pre_open_timers,
)
from akquant.backtest.fill_mode import (
    CurrentClose,
    FillMode,
    NextClose,
    NextOpen,
)
from akquant.config import RiskConfig
from akquant.gateway.order_receipt import OrderReceipt
from akquant.strategy import Strategy, StrategyRuntimeConfig
from akquant.strategy_framework_hooks import (
    collect_boundary_timer_entries,
    collect_pre_open_timer_entries,
    dispatch_pre_open_timer,
)


@pytest.fixture(autouse=True)
def _reset_logger_console_state() -> Iterator[None]:
    try:
        yield
    finally:
        register_logger(console=False, level="INFO")


class MyStrategy(Strategy):
    """Test strategy."""

    def on_bar(self, bar: Bar) -> None:
        """Handle bar event."""
        self.log(f"Bar {self.symbol} Close: {self.close}")

    def on_tick(self, tick: Tick) -> None:
        """Handle tick event."""
        self.log(f"Tick {self.symbol} Price: {self.close}")


class OrderLoggingStrategy(Strategy):
    """Strategy used to verify order/trade log context extraction."""

    def on_order(self, order: Any) -> None:
        """Emit a log from the order callback."""
        self.log(f"order:{order.id}")

    def on_trade(self, trade: Any) -> None:
        """Emit a log from the trade callback."""
        self.log(f"trade:{trade.order_id}")


def test_strategy_logging(caplog: Any) -> None:
    """Test logging."""
    strategy = MyStrategy()

    # Mock context
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0

    # Mock Bar
    ts = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        symbol="AAPL",
    )

    with caplog.at_level(logging.INFO, logger="akquant"):
        strategy._on_bar_event(bar, ctx)

    assert "Bar AAPL Close: 102.0" in caplog.text
    assert "[" in caplog.text and "]" in caplog.text
    assert any(record.name == "akquant.strategy" for record in caplog.records)


def test_strategy_logging_includes_structured_context(caplog: Any) -> None:
    """Strategy logs should carry structured metadata for later formatting/export."""
    strategy = MyStrategy()
    cast(Any, strategy)._owner_strategy_id = "alpha"
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0

    ts = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        symbol="AAPL",
    )

    with caplog.at_level(logging.INFO, logger="akquant"):
        strategy._on_bar_event(bar, ctx)

    strategy_record = next(
        record for record in caplog.records if record.name == "akquant.strategy"
    )
    assert strategy_record.strategy_id == "alpha"
    assert strategy_record.slot == "alpha"
    assert strategy_record.symbol == "AAPL"
    assert strategy_record.phase == "strategy"
    assert strategy_record.event_time_iso == "2023-01-01T01:30:00Z"
    assert strategy_record.event_time == ts


def test_strategy_logging_live_profile_renders_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Live profile should render structured context fields in human-readable logs."""
    configure_logging(LogConfig(profile="live", console=True, level="INFO"))
    strategy = MyStrategy()
    cast(Any, strategy)._owner_strategy_id = "alpha"
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0

    ts = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        symbol="AAPL",
    )
    strategy._on_bar_event(bar, ctx)

    captured = capsys.readouterr()
    assert "akquant.strategy" in captured.out
    assert "strategy_id=alpha" in captured.out
    assert "symbol=AAPL" in captured.out
    assert "event_time_iso=2023-01-01T01:30:00Z" in captured.out


def test_strategy_logging_includes_order_context_during_callbacks(caplog: Any) -> None:
    """Order/trade callbacks should populate structured order logging fields."""
    strategy = OrderLoggingStrategy()
    cast(Any, strategy)._owner_strategy_id = "alpha"
    ctx = _build_ctx_with_order_and_trade(
        order_id="order-1",
        client_order_id="coid-1",
        symbol="AAPL",
    )

    ts = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        symbol="AAPL",
    )

    with caplog.at_level(logging.INFO, logger="akquant"):
        strategy._on_bar_event(bar, ctx)

    order_record = next(
        record for record in caplog.records if "order:order-1" in record.getMessage()
    )
    trade_record = next(
        record for record in caplog.records if "trade:order-1" in record.getMessage()
    )
    assert order_record.phase == "order"
    assert order_record.strategy_id == "alpha"
    assert order_record.slot == "alpha"
    assert order_record.symbol == "AAPL"
    assert order_record.order_id == "order-1"
    assert order_record.client_order_id == "coid-1"
    assert trade_record.phase == "trade"
    assert trade_record.strategy_id == "alpha"
    assert trade_record.slot == "alpha"
    assert trade_record.symbol == "AAPL"
    assert trade_record.order_id == "order-1"
    assert trade_record.client_order_id == "coid-1"


def test_strategy_logging_live_profile_renders_order_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Live profile should render order-related structured fields from callbacks."""
    configure_logging(LogConfig(profile="live", console=True, level="INFO"))
    strategy = OrderLoggingStrategy()
    cast(Any, strategy)._owner_strategy_id = "alpha"
    ctx = _build_ctx_with_order_and_trade(
        order_id="order-live-1",
        client_order_id="coid-live-1",
        symbol="AAPL",
    )

    ts = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        symbol="AAPL",
    )
    strategy._on_bar_event(bar, ctx)

    captured = capsys.readouterr()
    assert "order_id=order-live-1" in captured.out
    assert "client_order_id=coid-live-1" in captured.out
    assert "phase=order" in captured.out
    assert "phase=trade" in captured.out


def test_strategy_properties() -> None:
    """Test properties."""
    strategy = MyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0

    # Test Bar Properties
    ts = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        symbol="AAPL",
    )
    strategy._on_bar_event(bar, ctx)

    assert strategy.symbol == "AAPL"
    assert strategy.close == 102.0
    assert strategy.open == 100.0
    assert strategy.high == 105.0
    assert strategy.low == 95.0
    assert strategy.volume == 1000.0

    # Test Tick Properties
    ts_tick = pd.Timestamp("2023-01-01 09:30:01", tz="Asia/Shanghai").value
    tick = Tick(timestamp=ts_tick, price=103.0, volume=500.0, symbol="GOOG")
    strategy._on_tick_event(tick, ctx)

    assert strategy.symbol == "GOOG"
    assert strategy.close == 103.0  # close maps to price in tick
    assert strategy.volume == 500.0
    # Open/High/Low should be 0.0 or handle gracefully?
    # Current implementation returns 0.0 if not current_bar
    assert strategy.open == 0.0
    assert strategy.high == 0.0
    assert strategy.low == 0.0


class HistoryMapStrategy(Strategy):
    """Strategy for get_history_map tests."""

    def __init__(self) -> None:
        """Initialize captured calls."""
        self.calls: list[tuple[int, str | None, str]] = []

    def get_history(
        self,
        count: int,
        symbol: str | None = None,
        field: str = "close",
        freq: Optional[str] = None,
    ) -> np.ndarray:
        """Return deterministic history and record invocation."""
        del freq
        self.calls.append((count, symbol, field))
        return np.array([1.0, 2.0, 3.0], dtype=float)


def test_get_history_map_normalizes_symbols_and_batches() -> None:
    """get_history_map should normalize symbols and call get_history per symbol."""
    strategy = HistoryMapStrategy()

    history_map = strategy.get_history_map(
        count=3,
        symbols=["BBB", "AAA", "AAA"],
        field="close",
    )

    assert set(history_map.keys()) == {"AAA", "BBB"}
    assert strategy.calls == [(3, "AAA", "close"), (3, "BBB", "close")]


def test_get_history_map_supports_single_symbol() -> None:
    """get_history_map should support single symbol inputs."""
    strategy = HistoryMapStrategy()

    history_map = strategy.get_history_map(count=2, symbols="AAA", field="open")

    assert list(history_map.keys()) == ["AAA"]
    assert strategy.calls == [(2, "AAA", "open")]


class TopNRebalanceStrategy(Strategy):
    """Strategy for rebalance_to_topn tests."""

    def __init__(self) -> None:
        """Initialize call snapshots."""
        self.calls: list[dict[str, Any]] = []

    def rebalance_weights(
        self,
        target_weights: dict[str, float],
        price_map: dict[str, float] | None = None,
        liquidate_unmentioned: bool = False,
        allow_leverage: bool = False,
        rebalance_tolerance: float = 0.0,
        **kwargs: Any,
    ) -> list[str]:
        """Capture target weights invocation."""
        self.calls.append(
            {
                "target_weights": target_weights,
                "price_map": price_map,
                "liquidate_unmentioned": liquidate_unmentioned,
                "allow_leverage": allow_leverage,
                "rebalance_tolerance": rebalance_tolerance,
                "kwargs": kwargs,
            }
        )
        return []


def test_rebalance_to_topn_equal_weight_selection() -> None:
    """rebalance_to_topn should select top symbols and place equal weights."""
    strategy = TopNRebalanceStrategy()

    selected = strategy.rebalance_to_topn(
        scores={"AAA": 0.03, "BBB": 0.08, "CCC": 0.05},
        top_n=2,
    )

    assert selected == ["BBB", "CCC"]
    assert len(strategy.calls) == 1
    assert strategy.calls[0]["target_weights"] == {"BBB": 0.5, "CCC": 0.5}
    assert strategy.calls[0]["liquidate_unmentioned"] is True


def test_rebalance_to_topn_filters_non_positive_when_long_only() -> None:
    """rebalance_to_topn should clear positions when no positive scores remain."""
    strategy = TopNRebalanceStrategy()

    selected = strategy.rebalance_to_topn(
        scores={"AAA": -0.02, "BBB": -0.01},
        top_n=2,
    )

    assert selected == []
    assert len(strategy.calls) == 1
    assert strategy.calls[0]["target_weights"] == {}
    assert strategy.calls[0]["liquidate_unmentioned"] is True


def test_rebalance_to_topn_rejects_invalid_topn() -> None:
    """rebalance_to_topn should validate top_n."""
    strategy = TopNRebalanceStrategy()

    with pytest.raises(ValueError, match="top_n must be greater than 0"):
        strategy.rebalance_to_topn(scores={"AAA": 0.1}, top_n=0)


def test_rebalance_to_topn_supports_score_weight_mode() -> None:
    """rebalance_to_topn should support score-normalized weights."""
    strategy = TopNRebalanceStrategy()

    selected = strategy.rebalance_to_topn(
        scores={"AAA": 0.1, "BBB": 0.3, "CCC": 0.05},
        top_n=2,
        weight_mode="score",
    )

    assert selected == ["BBB", "AAA"]
    assert len(strategy.calls) == 1
    assert strategy.calls[0]["target_weights"]["BBB"] == pytest.approx(0.75)
    assert strategy.calls[0]["target_weights"]["AAA"] == pytest.approx(0.25)


def test_rebalance_to_topn_breaks_score_ties_by_symbol() -> None:
    """rebalance_to_topn should use symbol order as deterministic tie-breaker."""
    strategy = TopNRebalanceStrategy()

    selected = strategy.rebalance_to_topn(
        scores={"BBB": 0.1, "AAA": 0.1, "CCC": 0.05},
        top_n=2,
    )

    assert selected == ["AAA", "BBB"]
    assert strategy.calls[0]["target_weights"] == {"AAA": 0.5, "BBB": 0.5}


def test_rebalance_to_topn_rejects_invalid_weight_mode() -> None:
    """rebalance_to_topn should validate weight_mode."""
    strategy = TopNRebalanceStrategy()

    with pytest.raises(ValueError, match="weight_mode must be one of: equal, score"):
        strategy.rebalance_to_topn(
            scores={"AAA": 0.1, "BBB": 0.2},
            top_n=1,
            weight_mode="invalid",  # type: ignore[arg-type]
        )


class StartCounterStrategy(Strategy):
    """Strategy for lifecycle callback counting."""

    def __init__(self) -> None:
        """Initialize counters."""
        self.start_calls = 0
        self.resume_calls = 0

    def on_start(self) -> None:
        """Count start callbacks."""
        self.start_calls += 1

    def on_resume(self) -> None:
        """Count resume callbacks."""
        self.resume_calls += 1


class WarmStartSequenceStrategy(Strategy):
    """Strategy for warm start callback ordering tests."""

    def __init__(self) -> None:
        """Initialize callback sequence container."""
        self.events: list[str] = []

    def on_resume(self) -> None:
        """Record resume callback."""
        self.events.append("on_resume")

    def on_start(self) -> None:
        """Record start callback."""
        self.events.append("on_start")


def test_on_start_internal_idempotent() -> None:
    """Start callback should run once when internal start is called repeatedly."""
    strategy = StartCounterStrategy()
    strategy._on_start_internal()
    strategy._on_start_internal()
    assert strategy.start_calls == 1
    assert strategy.resume_calls == 0


def test_on_resume_runs_once_before_on_start() -> None:
    """Resume callback should run once before start in restored mode."""
    strategy = StartCounterStrategy()
    strategy._is_restored = True
    strategy._on_start_internal()
    strategy._on_start_internal()
    assert strategy.resume_calls == 1
    assert strategy.start_calls == 1


def test_warm_start_callback_sequence() -> None:
    """Warm start should call on_resume before on_start once."""
    strategy = WarmStartSequenceStrategy()
    strategy._is_restored = True
    strategy._on_start_internal()
    strategy._on_start_internal()
    assert strategy.events == ["on_resume", "on_start"]


class EventCounterStrategy(Strategy):
    """Strategy for event callback counting."""

    def __init__(self) -> None:
        """Initialize counters."""
        self.trade_count = 0
        self.tick_count = 0
        self.timer_count = 0

    def on_bar(self, bar: Bar) -> None:
        """Ignore bar events."""
        return

    def on_tick(self, tick: Tick) -> None:
        """Count tick callbacks."""
        self.tick_count += 1

    def on_timer(self, payload: str) -> None:
        """Count timer callbacks."""
        self.timer_count += 1

    def on_trade(self, trade: Any) -> None:
        """Count trade callbacks."""
        self.trade_count += 1


class TradeDedupeLimitStrategy(Strategy):
    """Strategy for trade de-duplication cache limit tests."""

    trade_dedupe_cache_size = 2

    def __init__(self) -> None:
        """Initialize trade callbacks counter."""
        self.trade_count = 0

    def on_trade(self, trade: Any) -> None:
        """Count trade callbacks."""
        self.trade_count += 1


def test_trade_callback_not_duplicated_on_bar() -> None:
    """Trade callback should be triggered once per bar event."""
    strategy = EventCounterStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = [SimpleNamespace(order_id="o1")]

    ts = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        symbol="AAPL",
    )
    strategy._on_bar_event(bar, ctx)
    assert strategy.trade_count == 1


def test_tick_and_timer_process_order_events() -> None:
    """Tick and timer events should process trade callbacks."""
    strategy = EventCounterStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = [SimpleNamespace(order_id="o1")]

    ts_tick = pd.Timestamp("2023-01-01 09:30:01", tz="Asia/Shanghai").value
    tick = Tick(timestamp=ts_tick, price=103.0, volume=500.0, symbol="GOOG")
    strategy._on_tick_event(tick, ctx)

    assert strategy.tick_count == 1
    assert strategy.trade_count == 1

    ctx.recent_trades = [SimpleNamespace(order_id="o2")]
    strategy._on_timer_event("rebalance", ctx)
    assert strategy.timer_count == 1
    assert strategy.trade_count == 2


def test_trade_callback_not_replayed_across_events() -> None:
    """Repeated recent_trades entries should not replay on_trade callback."""
    strategy = EventCounterStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = [SimpleNamespace(order_id="o1")]

    ts_tick = pd.Timestamp("2023-01-01 09:30:01", tz="Asia/Shanghai").value
    tick = Tick(timestamp=ts_tick, price=103.0, volume=500.0, symbol="GOOG")
    strategy._on_tick_event(tick, ctx)
    assert strategy.trade_count == 1

    strategy._on_timer_event("rebalance", ctx)
    assert strategy.trade_count == 1
    assert strategy.timer_count == 1


def test_trade_dedupe_cache_limit_eviction_allows_replay() -> None:
    """Dedupe cache should evict old keys once limit reached."""
    strategy = TradeDedupeLimitStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []

    ctx.recent_trades = [SimpleNamespace(order_id="o1")]
    ts_tick = pd.Timestamp("2023-01-01 09:30:01", tz="Asia/Shanghai").value
    tick = Tick(timestamp=ts_tick, price=103.0, volume=500.0, symbol="GOOG")
    strategy._on_tick_event(tick, ctx)
    assert strategy.trade_count == 1

    ctx.recent_trades = [SimpleNamespace(order_id="o2")]
    strategy._on_timer_event("t2", ctx)
    assert strategy.trade_count == 2

    ctx.recent_trades = [SimpleNamespace(order_id="o3")]
    strategy._on_timer_event("t3", ctx)
    assert strategy.trade_count == 3

    ctx.recent_trades = [SimpleNamespace(order_id="o1")]
    strategy._on_timer_event("t4", ctx)
    assert strategy.trade_count == 4


class ClosedTradesSnapshotStrategy(Strategy):
    """Strategy for validating get_trades snapshot refresh."""

    def __init__(self) -> None:
        """Initialize observation state."""
        self.entered = False
        self.exited = False

    def on_bar(self, bar: Bar) -> None:
        """Run one round-trip order flow."""
        position = self.get_position(bar.symbol)
        if position == 0 and not self.entered:
            self.buy(bar.symbol, 100)
            self.entered = True
            return
        if position > 0 and not self.exited:
            self.close_position(bar.symbol)
            self.exited = True


def test_get_trades_refreshes_during_backtest() -> None:
    """get_trades should reflect latest closed trades snapshot in strategy context."""
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=6, freq="D", tz="UTC"),
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "high": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
            "close": [10.2, 11.2, 12.2, 13.2, 14.2, 15.2],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        }
    )
    result = run_backtest(
        data=bars,
        strategy=ClosedTradesSnapshotStrategy,
        symbols="AAPL",
        initial_cash=100000.0,
        show_progress=False,
    )
    strategy = cast(ClosedTradesSnapshotStrategy, result.strategy)
    assert strategy is not None
    assert len(result.trades) >= 1
    assert len(strategy.get_trades()) >= 1


class RejectEventBacktestStrategy(Strategy):
    """Strategy used to validate on_reject callback in run_backtest."""

    def __init__(self) -> None:
        """Initialize reject capture state."""
        self.reject_events: list[tuple[str, str]] = []

    def on_bar(self, bar: Bar) -> None:
        """Submit oversized orders on first two bars to trigger risk rejects."""
        if self._bar_count == 1:
            self.sell(symbol=bar.symbol, quantity=100)
        elif self._bar_count == 2:
            self.buy(symbol=bar.symbol, quantity=10000)
        elif self._bar_count == 3:
            self.buy(symbol=bar.symbol, quantity=10)

    def on_reject(self, order: Any) -> None:
        """Capture rejected order id and reason."""
        self.reject_events.append((str(order.id), str(order.reject_reason)))


def test_run_backtest_emits_on_reject_for_risk_rejected_orders() -> None:
    """run_backtest should trigger on_reject for risk rejected orders."""
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [100.5, 101.5, 102.5, 103.5, 104.5],
            "low": [99.5, 100.5, 101.5, 102.5, 103.5],
            "close": [100.0, 100.2, 100.3, 100.4, 100.5],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "symbol": ["STOCK", "STOCK", "STOCK", "STOCK", "STOCK"],
        }
    )
    config = BacktestConfig(
        strategy_config=StrategyConfig(
            risk=RiskConfig(safety_margin=0.0001, max_order_value=5000.0)
        )
    )

    result = run_backtest(
        data=bars,
        strategy=RejectEventBacktestStrategy,
        symbols=["STOCK"],
        initial_cash=10000.0,
        config=config,
        show_progress=False,
    )
    strategy = cast(RejectEventBacktestStrategy, result.strategy)
    assert strategy is not None
    assert len(strategy.reject_events) == 2
    assert all("Risk:" in reason for _, reason in strategy.reject_events)

    rejected_df = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "rejected"
    ]
    assert len(rejected_df) == 2


class RejectOnceBacktestStrategy(Strategy):
    """Strategy used to validate single-fire semantics of on_reject."""

    def __init__(self) -> None:
        """Initialize captured callbacks."""
        self.reject_order_ids: list[str] = []

    def on_bar(self, bar: Bar) -> None:
        """Submit only one oversized order so one order id is rejected."""
        if self._bar_count == 1:
            self.buy(symbol=bar.symbol, quantity=10000)

    def on_reject(self, order: Any) -> None:
        """Capture rejected order id."""
        self.reject_order_ids.append(str(order.id))


def test_run_backtest_on_reject_fires_once_per_order_id() -> None:
    """on_reject should fire once for the same rejected order id."""
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-02-01", periods=6, freq="D"),
            "open": [100.0, 100.2, 100.3, 100.4, 100.5, 100.6],
            "high": [100.5, 100.7, 100.8, 100.9, 101.0, 101.1],
            "low": [99.5, 99.7, 99.8, 99.9, 100.0, 100.1],
            "close": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "symbol": ["STOCK", "STOCK", "STOCK", "STOCK", "STOCK", "STOCK"],
        }
    )
    config = BacktestConfig(
        strategy_config=StrategyConfig(
            risk=RiskConfig(safety_margin=0.0001, max_order_value=5000.0)
        )
    )

    result = run_backtest(
        data=bars,
        strategy=RejectOnceBacktestStrategy,
        symbols=["STOCK"],
        initial_cash=10000.0,
        config=config,
        show_progress=False,
    )
    strategy = cast(RejectOnceBacktestStrategy, result.strategy)
    assert strategy is not None
    assert len(strategy.reject_order_ids) == 1
    assert len(set(strategy.reject_order_ids)) == 1

    rejected_df = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "rejected"
    ]
    assert len(rejected_df) == 1


class RejectLoggingBacktestStrategy(Strategy):
    """Strategy used to verify reject callback logging context."""

    def on_bar(self, bar: Bar) -> None:
        """Submit one oversized order to trigger a reject callback."""
        if self._bar_count == 1:
            self.buy(symbol=bar.symbol, quantity=10000)

    def on_reject(self, order: Any) -> None:
        """Emit a strategy log from the reject callback."""
        self.log(f"reject:{order.id}")


def test_run_backtest_reject_logging_includes_order_context(caplog: Any) -> None:
    """Reject callback logs should include structured order metadata."""
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-02-01", periods=6, freq="D"),
            "open": [100.0, 100.2, 100.3, 100.4, 100.5, 100.6],
            "high": [100.5, 100.7, 100.8, 100.9, 101.0, 101.1],
            "low": [99.5, 99.7, 99.8, 99.9, 100.0, 100.1],
            "close": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "symbol": ["STOCK", "STOCK", "STOCK", "STOCK", "STOCK", "STOCK"],
        }
    )
    config = BacktestConfig(
        strategy_config=StrategyConfig(
            risk=RiskConfig(safety_margin=0.0001, max_order_value=5000.0)
        )
    )

    with caplog.at_level(logging.INFO, logger="akquant"):
        _ = run_backtest(
            data=bars,
            strategy=RejectLoggingBacktestStrategy,
            symbols=["STOCK"],
            initial_cash=10000.0,
            config=config,
            show_progress=False,
        )

    reject_record = next(
        record for record in caplog.records if "reject:" in record.getMessage()
    )
    assert reject_record.name == "akquant.strategy"
    assert reject_record.phase == "order"
    assert reject_record.symbol == "STOCK"
    assert reject_record.order_id


class SequenceStrategy(Strategy):
    """Strategy for callback sequence assertions."""

    def __init__(self) -> None:
        """Initialize callback record list."""
        self.events: list[str] = []

    def on_bar(self, bar: Bar) -> None:
        """Record bar callback."""
        self.events.append("on_bar")

    def on_tick(self, tick: Tick) -> None:
        """Record tick callback."""
        self.events.append("on_tick")

    def on_timer(self, payload: str) -> None:
        """Record timer callback."""
        self.events.append(f"on_timer:{payload}")

    def on_order(self, order: Any) -> None:
        """Record order callback."""
        self.events.append(f"on_order:{order.id}")

    def on_trade(self, trade: Any) -> None:
        """Record trade callback."""
        self.events.append(f"on_trade:{trade.order_id}")


def _build_ctx_with_order_and_trade(
    order_id: str = "o1",
    *,
    client_order_id: str = "coid-1",
    symbol: str = "AAPL",
) -> MagicMock:
    """Build a mocked strategy context with one active order and one trade."""
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.active_orders = [
        SimpleNamespace(
            id=order_id,
            status="Submitted",
            filled_quantity=0.0,
            symbol=symbol,
            client_order_id=client_order_id,
        )
    ]
    ctx.recent_trades = [
        SimpleNamespace(
            order_id=order_id,
            symbol=symbol,
            client_order_id=client_order_id,
        )
    ]
    return ctx


def test_functional_strategy_supports_extended_callbacks() -> None:
    """FunctionalStrategy should delegate tick/order/trade/timer callbacks."""
    events: list[str] = []

    def initialize(ctx: Any) -> None:
        events.append("initialize")

    def on_bar(ctx: Any, bar: Bar) -> None:
        events.append("bar")

    def on_tick(ctx: Any, tick: Tick) -> None:
        events.append("tick")

    def on_order(ctx: Any, order: Any) -> None:
        events.append(f"order:{order.id}")

    def on_trade(ctx: Any, trade: Any) -> None:
        events.append(f"trade:{trade.order_id}")

    def on_timer(ctx: Any, payload: str) -> None:
        events.append(f"timer:{payload}")

    strategy = FunctionalStrategy(
        initialize=initialize,
        on_bar=on_bar,
        on_tick=on_tick,
        on_order=on_order,
        on_trade=on_trade,
        on_timer=on_timer,
    )

    ts_bar = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts_bar,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        symbol="AAPL",
    )
    strategy._on_bar_event(bar, _build_ctx_with_order_and_trade("bar_order"))

    ts_tick = pd.Timestamp("2023-01-01 09:30:01", tz="Asia/Shanghai").value
    tick = Tick(timestamp=ts_tick, price=103.0, volume=500.0, symbol="AAPL")
    strategy._on_tick_event(tick, _build_ctx_with_order_and_trade("tick_order"))
    strategy._on_timer_event(
        "rebalance", _build_ctx_with_order_and_trade("timer_order")
    )

    assert events[0] == "initialize"
    assert "order:bar_order" in events
    assert "trade:bar_order" in events
    assert "bar" in events
    assert "order:tick_order" in events
    assert "trade:tick_order" in events
    assert "tick" in events
    assert "order:timer_order" in events
    assert "trade:timer_order" in events
    assert "timer:rebalance" in events


def test_functional_strategy_callback_sequence_contract() -> None:
    """Function-style callbacks should follow framework callback ordering."""
    events: list[str] = []

    def on_bar(ctx: Any, bar: Bar) -> None:
        events.append("bar")

    def on_tick(ctx: Any, tick: Tick) -> None:
        events.append("tick")

    def on_order(ctx: Any, order: Any) -> None:
        events.append(f"order:{order.id}")

    def on_trade(ctx: Any, trade: Any) -> None:
        events.append(f"trade:{trade.order_id}")

    def on_timer(ctx: Any, payload: str) -> None:
        events.append(f"timer:{payload}")

    strategy = FunctionalStrategy(
        initialize=None,
        on_bar=on_bar,
        on_tick=on_tick,
        on_order=on_order,
        on_trade=on_trade,
        on_timer=on_timer,
    )

    ts_bar = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts_bar,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        symbol="AAPL",
    )
    strategy._on_bar_event(bar, _build_ctx_with_order_and_trade("bar_order"))

    ts_tick = pd.Timestamp("2023-01-01 09:30:01", tz="Asia/Shanghai").value
    tick = Tick(timestamp=ts_tick, price=103.0, volume=500.0, symbol="AAPL")
    strategy._on_tick_event(tick, _build_ctx_with_order_and_trade("tick_order"))
    strategy._on_timer_event(
        "rebalance", _build_ctx_with_order_and_trade("timer_order")
    )

    assert events == [
        "order:bar_order",
        "trade:bar_order",
        "bar",
        "order:tick_order",
        "trade:tick_order",
        "tick",
        "order:timer_order",
        "trade:timer_order",
        "timer:rebalance",
    ]


def test_callback_sequence_on_bar() -> None:
    """Order/trade callbacks should run before bar callback."""
    strategy = SequenceStrategy()
    ctx = _build_ctx_with_order_and_trade("bar_order")
    ts = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        symbol="AAPL",
    )

    strategy._on_bar_event(bar, ctx)
    assert strategy.events == ["on_order:bar_order", "on_trade:bar_order", "on_bar"]


def test_callback_sequence_on_tick() -> None:
    """Order/trade callbacks should run before tick callback."""
    strategy = SequenceStrategy()
    ctx = _build_ctx_with_order_and_trade("tick_order")
    ts_tick = pd.Timestamp("2023-01-01 09:30:01", tz="Asia/Shanghai").value
    tick = Tick(timestamp=ts_tick, price=103.0, volume=500.0, symbol="GOOG")

    strategy._on_tick_event(tick, ctx)
    assert strategy.events == ["on_order:tick_order", "on_trade:tick_order", "on_tick"]


def test_callback_sequence_on_timer() -> None:
    """Order/trade callbacks should run before timer callback."""
    strategy = SequenceStrategy()
    ctx = _build_ctx_with_order_and_trade("timer_order")

    strategy._on_timer_event("rebalance", ctx)
    assert strategy.events == [
        "on_order:timer_order",
        "on_trade:timer_order",
        "on_timer:rebalance",
    ]


def test_now_uses_context_time_for_timer_event() -> None:
    """Now should use context current_time during timer callbacks."""
    strategy = SequenceStrategy()

    bar_ctx = _build_ctx_with_order_and_trade("bar_order")
    bar_ts = pd.Timestamp("2023-01-01 09:59:00", tz="Asia/Shanghai").value
    bar_ctx.current_time = bar_ts
    bar = Bar(
        timestamp=bar_ts,
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        volume=100.0,
        symbol="AAPL",
    )
    strategy._on_bar_event(bar, bar_ctx)

    timer_ctx = _build_ctx_with_order_and_trade("timer_order")
    timer_ts = pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai").value
    timer_ctx.current_time = timer_ts
    strategy._on_timer_event("rebalance", timer_ctx)

    assert strategy.now == pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai")


class WarmStartE2EStrategy(Strategy):
    """Strategy for warm start end-to-end test."""

    def __init__(self) -> None:
        """Initialize strategy state."""
        self.events: list[str] = []
        self.bar_seen = 0

    def on_resume(self) -> None:
        """Record resume callback."""
        self.events.append("on_resume")

    def on_start(self) -> None:
        """Record start callback."""
        self.events.append("on_start")

    def on_bar(self, bar: Bar) -> None:
        """Record bar callback and mutate state."""
        self.bar_seen += 1
        self.events.append(f"on_bar:{self.bar_seen}")


class WarmStartHistoryContinuityStrategy(Strategy):
    """Strategy for warm-start history-buffer continuity tests."""

    def __init__(self) -> None:
        """Track close-history snapshots across bars."""
        self.set_history_depth(3)
        self.samples: list[tuple[int, np.ndarray]] = []

    def on_bar(self, bar: Bar) -> None:
        """Capture the rolling close window seen on each bar."""
        self.samples.append(
            (bar.timestamp, self.get_history(count=3, field="close").copy())
        )


class WarmStartInstrumentSnapshotStrategy(Strategy):
    """Strategy for warm-start instrument snapshot regression tests."""

    def __init__(self) -> None:
        """Initialize snapshot capture state."""
        self.captured_counts: list[int] = []
        self.phase1_snapshot: dict[str, Any] | None = None
        self.phase2_snapshot: dict[str, Any] | None = None

    def on_start(self) -> None:
        """Capture available instrument snapshots on cold and warm starts."""
        current_count = len(self.get_instruments())
        self.captured_counts.append(current_count)
        if self.is_restored:
            phase2 = self.get_instrument("OPT2402C")
            self.phase2_snapshot = {
                "asset_type": phase2.asset_type,
                "multiplier": phase2.multiplier,
                "option_type": phase2.option_type,
                "option_margin_model": phase2.option_margin_model,
                "expiry_date": phase2.expiry_date,
                "underlying_symbol": phase2.underlying_symbol,
                "all_count": current_count,
            }
        else:
            phase1 = self.get_instrument("BASE")
            self.phase1_snapshot = {
                "asset_type": phase1.asset_type,
                "all_count": current_count,
            }

    def on_bar(self, bar: Bar) -> None:
        """No-op bar handler for lifecycle completeness."""
        return


class RuntimeConfigWarmStartStrategy(Strategy):
    """Strategy for warm start runtime config injection test."""

    def __init__(self) -> None:
        """Initialize records."""
        self.errors: list[str] = []
        self.bar_seen = 0

    def on_bar(self, bar: Bar) -> None:
        """Raise only after restored to test warm-start injection."""
        self.bar_seen += 1
        if self.is_restored:
            raise ValueError("warm_boom")

    def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
        """Record callback source."""
        self.errors.append(source)


class RuntimeConfigWarmConflictStrategy(Strategy):
    """Strategy for warm-start runtime config conflict behavior tests."""

    def __init__(self) -> None:
        """Initialize with strict runtime config."""
        self.errors: list[str] = []
        self.runtime_config = StrategyRuntimeConfig(error_mode="raise")

    def on_bar(self, bar: Bar) -> None:
        """Raise only after restored."""
        if self.is_restored:
            raise ValueError("warm_conflict_boom")

    def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
        """Record callback source."""
        self.errors.append(source)


class WarmStartRiskStateStrategy(Strategy):
    """Strategy for risk-state warm-start persistence tests."""

    def __init__(self) -> None:
        """Initialize deterministic step counter."""
        self.step = 0

    def on_bar(self, bar: Bar) -> None:
        """Submit buy sequence across pre/post snapshot phases."""
        self.buy(symbol=bar.symbol, quantity=10)
        self.step += 1


def _functional_slot_id(ctx: Any, default: str = "_default") -> str:
    """Resolve functional strategy slot identity across lifecycle stages."""
    return str(
        getattr(ctx, "_owner_strategy_id", None)
        or getattr(ctx, "strategy_id", None)
        or default
    )


def _functional_warm_initialize(ctx: Any) -> None:
    """Initialize state for function-style warm start tests."""
    ctx.events = []
    ctx.processed_closes = []
    ctx.start_count = 0
    ctx.resume_count = 0


def _functional_warm_on_start(ctx: Any) -> None:
    """Record cold/warm starts for function-style tests."""
    ctx.start_count += 1
    slot_id = _functional_slot_id(ctx)
    ctx.events.append(f"{slot_id}:start:restored={int(bool(ctx.is_restored))}")


def _functional_warm_on_resume(ctx: Any) -> None:
    """Record warm-start-only callback events for function-style tests."""
    ctx.resume_count += 1
    slot_id = _functional_slot_id(ctx)
    ctx.events.append(
        f"{slot_id}:resume:bars={len(ctx.processed_closes)}:starts={ctx.start_count}"
    )


def _functional_warm_on_bar(ctx: Any, bar: Bar) -> None:
    """Accumulate state for primary function-style warm start test."""
    slot_id = _functional_slot_id(ctx, "alpha")
    ctx.processed_closes.append(float(bar.close))
    ctx.events.append(f"{slot_id}:bar:{bar.close:.2f}")


def _functional_warm_alpha_on_bar(ctx: Any, bar: Bar) -> None:
    """Accumulate state for primary slot in multi-slot warm start test."""
    slot_id = _functional_slot_id(ctx, "alpha")
    ctx.processed_closes.append(float(bar.close))
    ctx.events.append(f"{slot_id}:bar:{bar.close:.2f}")


def _functional_warm_beta_on_bar(ctx: Any, bar: Bar) -> None:
    """Accumulate state for secondary slot in multi-slot warm start test."""
    slot_id = _functional_slot_id(ctx, "beta")
    ctx.processed_closes.append(float(bar.close))
    ctx.events.append(f"{slot_id}:bar:{bar.close:.2f}")


class DeferredDailyRebalanceStrategy(Strategy):
    """Regression strategy for after-bar rebalance timing determinism."""

    def __init__(
        self,
        all_symbols: list[str],
        date2symbols: dict[Any, set[str]],
        capture: dict[str, Any],
    ) -> None:
        """Initialize symbol universe, rebalance map, and capture sink."""
        super().__init__()
        self.all_symbols = list(all_symbols)
        self.date2symbols = date2symbols
        self.capture = capture
        self.set_history_depth(2)

    def on_start(self) -> None:
        """Subscribe to all configured symbols."""
        for symbol in self.all_symbols:
            self.subscribe(symbol)

    def on_cross_section(self, trading_date: Any, timestamp: int) -> None:
        """Select the strongest two-bar momentum symbol on the target day."""
        if trading_date not in self.date2symbols:
            return
        symbols = self.date2symbols[trading_date]
        history_map = self.get_history_map(count=2, symbols=symbols, field="close")
        scores: dict[str, float] = {}
        for symbol, closes in history_map.items():
            if np.isnan(closes[0]) or closes[0] == 0:
                continue
            scores[symbol] = float((closes[-1] - closes[0]) / closes[0])
        selected = self.rebalance_to_topn(scores=scores, top_n=1)
        self.capture.setdefault("events", []).append(
            (trading_date, timestamp, tuple(selected))
        )


def _make_order_sensitive_multisymbol_bars(day3_order: list[str]) -> list[Bar]:
    """Create bars whose day-3 leader depends on when rebalance is evaluated."""
    schedule = [
        ("2023-01-01 15:00:00", {"AAA": 10.0, "BBB": 10.0}, ["AAA", "BBB"]),
        ("2023-01-02 15:00:00", {"AAA": 10.0, "BBB": 10.0}, ["AAA", "BBB"]),
        ("2023-01-03 15:00:00", {"AAA": 20.0, "BBB": 11.0}, day3_order),
        ("2023-01-04 15:00:00", {"AAA": 10.0, "BBB": 30.0}, ["AAA", "BBB"]),
    ]
    bars: list[Bar] = []
    for timestamp_text, closes, order in schedule:
        ts = pd.Timestamp(timestamp_text, tz="Asia/Shanghai").value
        for symbol in order:
            close = closes[symbol]
            bars.append(
                Bar(
                    timestamp=ts,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=1000.0,
                    symbol=symbol,
                )
            )
    return bars


def test_run_backtest_cross_section_same_timestamp_order_insensitive() -> None:
    """After-bar rebalance should see a complete same-timestamp cross section."""
    target_day = pd.Timestamp("2023-01-03", tz="Asia/Shanghai").date()
    date2symbols = {target_day: {"AAA", "BBB"}}
    first_capture: dict[str, Any] = {}
    second_capture: dict[str, Any] = {}

    first = run_backtest(
        data=_make_order_sensitive_multisymbol_bars(["AAA", "BBB"]),
        strategy=DeferredDailyRebalanceStrategy(
            all_symbols=["AAA", "BBB"],
            date2symbols=date2symbols,
            capture=first_capture,
        ),
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        show_progress=False,
    )
    second = run_backtest(
        data=_make_order_sensitive_multisymbol_bars(["BBB", "AAA"]),
        strategy=DeferredDailyRebalanceStrategy(
            all_symbols=["BBB", "AAA"],
            date2symbols=date2symbols,
            capture=second_capture,
        ),
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        show_progress=False,
    )

    assert first_capture["events"][-1][0] == target_day
    assert second_capture["events"][-1][0] == target_day
    assert first_capture["events"][-1][2] == ("AAA",)
    assert second_capture["events"][-1][2] == ("AAA",)
    assert first.metrics.total_return == pytest.approx(second.metrics.total_return)


def _make_bars(
    start: str,
    periods: int,
    symbol: str = "TEST",
    start_price: float = 100.0,
) -> list[Bar]:
    """Create deterministic bar list."""
    bars: list[Bar] = []
    idx = pd.date_range(start=start, periods=periods, freq="D")
    for i, ts in enumerate(idx):
        price = start_price + float(i)
        bars.append(
            Bar(
                timestamp=ts.value,
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.5,
                volume=1000.0 + i,
                symbol=symbol,
            )
        )
    return bars


def test_run_backtest_accepts_strategy_source_python_plain(tmp_path: Path) -> None:
    """run_backtest should load strategy class from python source file."""
    strategy_file = tmp_path / "strategy_plain.py"
    strategy_file.write_text(
        "\n".join(
            [
                "from akquant.strategy import Strategy",
                "",
                "class Strategy(Strategy):",
                "    def __init__(self):",
                "        self.calls = 0",
                "",
                "    def on_bar(self, bar):",
                "        self.calls += 1",
            ]
        ),
        encoding="utf-8",
    )
    bars = _make_bars("2023-01-01", 3, symbol="PLAIN")
    result = run_backtest(
        data=bars,
        strategy_source=str(strategy_file),
        strategy_loader="python_plain",
        symbols="PLAIN",
        show_progress=False,
    )
    strategy = result.strategy
    assert strategy is not None
    assert getattr(strategy, "calls", 0) == 3


def test_run_backtest_rebuilds_console_handler_for_imported_strategy(
    tmp_path: Path,
) -> None:
    """run_backtest should restore visible logger handler after NullHandler fallback."""
    strategy_file = tmp_path / "strategy_plain_logging.py"
    strategy_file.write_text(
        "\n".join(
            [
                "from akquant.strategy import Strategy",
                "",
                "class Strategy(Strategy):",
                "    def on_bar(self, bar):",
                "        self.log('import_loader_log_visible')",
            ]
        ),
        encoding="utf-8",
    )
    logger = logging.getLogger("akquant")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    register_logger(console=True, level="INFO")
    logger.handlers = [logging.NullHandler()]
    try:
        bars = _make_bars("2023-01-01", 1, symbol="PLAIN_LOG")
        run_backtest(
            data=bars,
            strategy_source=str(strategy_file),
            strategy_loader="python_plain",
            symbols="PLAIN_LOG",
            show_progress=False,
        )
        has_console = any(
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.NullHandler)
            for handler in logger.handlers
        )
        assert has_console
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)


def test_run_backtest_imported_strategy_log_visible_in_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Imported strategy self.log output should be visible in captured stdout."""
    strategy_file = tmp_path / "strategy_plain_logging_stdout.py"
    strategy_file.write_text(
        "\n".join(
            [
                "from akquant.strategy import Strategy",
                "",
                "class Strategy(Strategy):",
                "    def on_bar(self, bar):",
                "        self.log('import_loader_stdout_visible')",
            ]
        ),
        encoding="utf-8",
    )
    logger = logging.getLogger("akquant")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    register_logger(console=True, level="INFO")
    logger.handlers = [logging.NullHandler()]
    try:
        bars = _make_bars("2023-01-01", 1, symbol="PLAIN_STDOUT")
        run_backtest(
            data=bars,
            strategy_source=str(strategy_file),
            strategy_loader="python_plain",
            symbols="PLAIN_STDOUT",
            show_progress=False,
        )
        captured = capsys.readouterr()
        assert "import_loader_stdout_visible" in captured.out
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)


def test_run_backtest_accepts_strategy_source_encrypted_external(
    tmp_path: Path,
) -> None:
    """run_backtest should load strategy via encrypted_external loader hook."""
    bars = _make_bars("2023-01-01", 2, symbol="ENC")
    strategy_file = tmp_path / "strategy_encrypted.mock"
    strategy_file.write_bytes(b"cipher")

    class EncryptedLoadedStrategy(Strategy):
        def __init__(self) -> None:
            self.calls = 0

        def on_bar(self, bar: Bar) -> None:
            self.calls += 1

    def _mock_decrypt_loader(source: Any, options: dict[str, Any]) -> type[Strategy]:
        _ = source
        _ = options
        return EncryptedLoadedStrategy

    result = run_backtest(
        data=bars,
        strategy_source=str(strategy_file),
        strategy_loader="encrypted_external",
        strategy_loader_options={"decrypt_and_load": _mock_decrypt_loader},
        symbols="ENC",
        show_progress=False,
    )
    strategy = result.strategy
    assert strategy is not None
    assert getattr(strategy, "calls", 0) == 2


def test_run_backtest_rejects_unknown_strategy_loader_name() -> None:
    """Unknown strategy loader should fail fast."""
    bars = _make_bars("2023-01-01", 1, symbol="BAD_LOADER")
    with pytest.raises(ValueError, match="unknown strategy_loader"):
        run_backtest(
            data=bars,
            strategy_source="missing.py",
            strategy_loader="not_exist",
            symbols="BAD_LOADER",
            show_progress=False,
        )


def test_run_backtest_rejects_invalid_strategy_loader_options_type() -> None:
    """Invalid strategy_loader_options type should fail fast."""
    bars = _make_bars("2023-01-01", 1, symbol="BAD_OPT")
    with pytest.raises(TypeError, match="strategy_loader_options"):
        run_backtest(
            data=bars,
            strategy_source="missing.py",
            strategy_loader_options=cast(Any, "bad"),
            symbols="BAD_OPT",
            show_progress=False,
        )


def test_run_backtest_rejects_invalid_strategy_loader_type() -> None:
    """Invalid strategy_loader type should fail fast."""
    bars = _make_bars("2023-01-01", 1, symbol="BAD_LOADER_TYPE")
    with pytest.raises(TypeError, match="strategy_loader must be str"):
        run_backtest(
            data=bars,
            strategy_source="missing.py",
            strategy_loader=cast(Any, 123),
            symbols="BAD_LOADER_TYPE",
            show_progress=False,
        )


def test_run_backtest_rejects_python_plain_with_bytes_source() -> None:
    """python_plain loader should reject bytes source."""
    bars = _make_bars("2023-01-01", 1, symbol="BAD_PLAIN_BYTES")
    with pytest.raises(TypeError, match="python_plain loader"):
        run_backtest(
            data=bars,
            strategy_source=b"cipher",
            strategy_loader="python_plain",
            symbols="BAD_PLAIN_BYTES",
            show_progress=False,
        )


def test_run_backtest_rejects_encrypted_loader_without_callback() -> None:
    """encrypted_external loader should require decrypt callback."""
    bars = _make_bars("2023-01-01", 1, symbol="BAD_ENC_OPT")
    with pytest.raises(ValueError, match="decrypt_and_load"):
        run_backtest(
            data=bars,
            strategy_source="encrypted.mock",
            strategy_loader="encrypted_external",
            symbols="BAD_ENC_OPT",
            show_progress=False,
        )


def test_run_backtest_python_plain_supports_strategy_attr_selection(
    tmp_path: Path,
) -> None:
    """python_plain loader should support selecting strategy by strategy_attr."""
    strategy_file = tmp_path / "strategy_multi.py"
    strategy_file.write_text(
        "\n".join(
            [
                "from akquant.strategy import Strategy",
                "",
                "class Alpha(Strategy):",
                "    def __init__(self):",
                "        self.calls = 0",
                "",
                "    def on_bar(self, bar):",
                "        self.calls += 1",
                "",
                "class Beta(Strategy):",
                "    def __init__(self):",
                "        self.calls = 0",
                "",
                "    def on_bar(self, bar):",
                "        self.calls += 10",
            ]
        ),
        encoding="utf-8",
    )
    bars = _make_bars("2023-01-01", 2, symbol="ATTR")
    result = run_backtest(
        data=bars,
        strategy_source=str(strategy_file),
        strategy_loader="python_plain",
        strategy_loader_options={"strategy_attr": "Beta"},
        symbols="ATTR",
        show_progress=False,
    )
    strategy = result.strategy
    assert strategy is not None
    assert getattr(strategy, "calls", 0) == 20


def test_run_backtest_rejects_python_plain_with_multiple_classes_without_attr(
    tmp_path: Path,
) -> None:
    """python_plain loader should fail on multiple Strategy subclasses without hint."""
    strategy_file = tmp_path / "strategy_multi_no_attr.py"
    strategy_file.write_text(
        "\n".join(
            [
                "from akquant.strategy import Strategy",
                "",
                "class Alpha(Strategy):",
                "    def on_bar(self, bar):",
                "        _ = bar",
                "",
                "class Beta(Strategy):",
                "    def on_bar(self, bar):",
                "        _ = bar",
            ]
        ),
        encoding="utf-8",
    )
    bars = _make_bars("2023-01-01", 1, symbol="MULTI")
    with pytest.raises(ValueError, match="multiple Strategy subclasses found"):
        run_backtest(
            data=bars,
            strategy_source=str(strategy_file),
            strategy_loader="python_plain",
            symbols="MULTI",
            show_progress=False,
        )


def test_run_backtest_rejects_missing_strategy_and_strategy_source() -> None:
    """run_backtest should fail when neither strategy nor strategy_source is given."""
    bars = _make_bars("2023-01-01", 1, symbol="NO_STRATEGY")
    with pytest.raises(ValueError, match="Strategy must be provided"):
        run_backtest(
            data=bars,
            strategy=None,
            strategy_source=None,
            symbols="NO_STRATEGY",
            show_progress=False,
        )


def test_run_backtest_accepts_registered_custom_strategy_loader() -> None:
    """Custom registered loader should be supported."""
    bars = _make_bars("2023-01-01", 2, symbol="CUSTOM_LOADER")

    class CustomLoadedStrategy(Strategy):
        def __init__(self) -> None:
            self.calls = 0

        def on_bar(self, bar: Bar) -> None:
            self.calls += 1

    loader_name = "test_custom_loader_for_source"

    def _loader(source: Any, options: dict[str, Any]) -> type[Strategy]:
        _ = source
        _ = options
        return CustomLoadedStrategy

    register_strategy_loader(loader_name, _loader)
    result = run_backtest(
        data=bars,
        strategy_source=b"mock",
        strategy_loader=loader_name,
        symbols="CUSTOM_LOADER",
        show_progress=False,
    )
    strategy = result.strategy
    assert strategy is not None
    assert getattr(strategy, "calls", 0) == 2


def test_run_backtest_loads_strategy_source_from_config(tmp_path: Path) -> None:
    """Backtest config should provide strategy_source and loader settings."""
    strategy_file = tmp_path / "strategy_from_config.py"
    strategy_file.write_text(
        "\n".join(
            [
                "from akquant.strategy import Strategy",
                "",
                "class Strategy(Strategy):",
                "    def __init__(self):",
                "        self.calls = 0",
                "",
                "    def on_bar(self, bar):",
                "        self.calls += 1",
            ]
        ),
        encoding="utf-8",
    )
    bars = _make_bars("2023-01-01", 2, symbol="CFG")
    cfg = BacktestConfig(
        strategy_config=StrategyConfig(
            strategy_source=str(strategy_file),
            strategy_loader="python_plain",
        ),
        show_progress=False,
    )
    result = run_backtest(
        data=bars,
        strategy=None,
        symbols="CFG",
        config=cfg,
    )
    strategy = result.strategy
    assert strategy is not None
    assert getattr(strategy, "calls", 0) == 2


def test_run_backtest_prefers_explicit_strategy_over_strategy_source(
    tmp_path: Path,
) -> None:
    """Explicit strategy argument should win over strategy_source settings."""
    strategy_file = tmp_path / "ignored_source.py"
    strategy_file.write_text(
        "\n".join(
            [
                "from akquant.strategy import Strategy",
                "",
                "class Strategy(Strategy):",
                "    def __init__(self):",
                "        self.calls = 0",
                "",
                "    def on_bar(self, bar):",
                "        self.calls += 100",
            ]
        ),
        encoding="utf-8",
    )
    bars = _make_bars("2023-01-01", 2, symbol="PRIO")

    class ExplicitPriorityStrategy(Strategy):
        def __init__(self) -> None:
            self.calls = 0

        def on_bar(self, bar: Bar) -> None:
            self.calls += 1

    result = run_backtest(
        data=bars,
        strategy=ExplicitPriorityStrategy,
        strategy_source=str(strategy_file),
        strategy_loader="python_plain",
        symbols="PRIO",
        show_progress=False,
    )
    strategy = result.strategy
    assert strategy is not None
    assert getattr(strategy, "calls", 0) == 2


def test_run_warm_start_end_to_end_lifecycle(tmp_path: Path) -> None:
    """Warm start should preserve state and callback lifecycle ordering."""
    checkpoint = tmp_path / "snapshot.pkl"
    phase1 = _make_bars("2023-01-01", 4)
    phase2 = _make_bars("2023-01-05", 3, start_price=104.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        initial_cash=100000.0,
        show_progress=False,
    )

    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        show_progress=False,
    )

    strategy = result2.strategy
    assert strategy is not None
    assert strategy.bar_seen == 7
    assert strategy.events[0] == "on_start"
    assert "on_resume" in strategy.events
    resume_idx = strategy.events.index("on_resume")
    assert strategy.events[resume_idx + 1] == "on_start"
    assert strategy.events[-1] == "on_bar:7"
    assert result2.metrics.initial_market_value == result1.metrics.end_market_value


def test_run_warm_start_preserves_get_history_window(tmp_path: Path) -> None:
    """Warm start should restore get_history windows without extra lookback input."""
    checkpoint = tmp_path / "snapshot_history.pkl"
    phase1 = _make_bars("2023-01-01", 4)
    phase2 = _make_bars("2023-01-05", 3, start_price=104.0)

    full_result = run_backtest(
        data=phase1 + phase2,
        strategy=WarmStartHistoryContinuityStrategy,
        symbols="TEST",
        initial_cash=100000.0,
        show_progress=False,
    )
    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartHistoryContinuityStrategy,
        symbols="TEST",
        initial_cash=100000.0,
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]
    warm_result = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        show_progress=False,
    )

    full_strategy = full_result.strategy
    warm_strategy = warm_result.strategy
    assert full_strategy is not None
    assert warm_strategy is not None

    full_samples = {timestamp: history for timestamp, history in full_strategy.samples}
    warm_samples = {timestamp: history for timestamp, history in warm_strategy.samples}
    assert list(sorted(warm_samples)) == list(sorted(full_samples))
    for timestamp, full_history in full_samples.items():
        np.testing.assert_allclose(
            warm_samples[timestamp],
            full_history,
            equal_nan=True,
        )


def test_run_warm_start_warns_for_legacy_checkpoint_without_history_feature_marker(
    tmp_path: Path,
) -> None:
    """Warm start should warn when a checkpoint lacks the history-buffer marker."""
    checkpoint = tmp_path / "snapshot_legacy_history.pkl"
    phase1 = _make_bars("2023-01-01", 4)
    phase2 = _make_bars("2023-01-05", 2, start_price=104.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartHistoryContinuityStrategy,
        symbols="TEST",
        initial_cash=100000.0,
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with checkpoint.open("rb") as fh:
        snapshot = pickle.load(fh)
    snapshot.pop("snapshot_features", None)
    with checkpoint.open("wb") as fh:
        pickle.dump(snapshot, fh)

    with pytest.warns(RuntimeWarning, match="history buffer snapshot"):
        result2 = run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
        )
    assert result2.strategy is not None


def test_load_checkpoint_warns_for_pre_tick_split_snapshot(tmp_path: Path) -> None:
    """load_checkpoint should warn when 'history_tick_split' marker is absent.

    Checkpoints saved before the tick/bar history-buffer split share the old
    'data' container for both bar and tick history. Restoring one of those
    into the current Rust HistoryBuffer loads 'data' as bar history and
    leaves 'tick_data' empty (serde default) -- if the original strategy had
    tick history, a resumed run would silently end up with a truncated tick
    window, or hit the dual-stream ambiguity error once new ticks arrive.
    load_checkpoint must surface this loudly instead of staying silent.
    """
    checkpoint = tmp_path / "snapshot_pre_tick_split.pkl"
    phase1 = _make_bars("2023-01-01", 4)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartHistoryContinuityStrategy,
        symbols="TEST",
        initial_cash=100000.0,
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    # Simulate a checkpoint written by a version that already had the
    # history-buffer-snapshot feature but predates the tick/bar split: keep
    # 'history_buffer_snapshot' but drop the newer 'history_tick_split' flag.
    with checkpoint.open("rb") as fh:
        snapshot = pickle.load(fh)
    assert snapshot["snapshot_features"]["history_tick_split"] is True
    snapshot["snapshot_features"].pop("history_tick_split", None)
    with checkpoint.open("wb") as fh:
        pickle.dump(snapshot, fh)

    with pytest.warns(RuntimeWarning, match="tick/bar history-buffer split"):
        engine, strategy = load_checkpoint(str(checkpoint))
    assert strategy is not None
    assert engine is not None

    # A checkpoint carrying both markers must stay silent.
    with checkpoint.open("rb") as fh:
        current_snapshot = pickle.load(fh)
    current_snapshot["snapshot_features"]["history_tick_split"] = True
    with checkpoint.open("wb") as fh:
        pickle.dump(current_snapshot, fh)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_checkpoint(str(checkpoint))


def test_run_warm_start_respects_instruments_config_and_merges_snapshots(
    tmp_path: Path,
) -> None:
    """Warm start should rebuild option instruments without dropping old ones."""
    checkpoint = tmp_path / "snapshot_instruments.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="BASE")
    phase2 = _make_bars("2023-02-01", 2, symbol="OPT2402C", start_price=12.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartInstrumentSnapshotStrategy,
        symbols="BASE",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="OPT2402C",
        show_progress=False,
        config=BacktestConfig(
            strategy_config=StrategyConfig(),
            instruments_config=[
                InstrumentConfig(
                    symbol="OPT2402C",
                    asset_type="OPTION",
                    multiplier=100.0,
                    option_type="CALL",
                    option_margin_model="CHINA_SINGLE_LEG",
                    expiry_date=dt.date(2024, 2, 28),
                    underlying_symbol="510050.SH",
                )
            ],
        ),
    )

    strategy = result2.strategy
    assert strategy is not None
    assert strategy.phase1_snapshot == {"asset_type": "STOCK", "all_count": 1}
    assert strategy.phase2_snapshot == {
        "asset_type": "OPTION",
        "multiplier": 100.0,
        "option_type": "CALL",
        "option_margin_model": "CHINA_SINGLE_LEG",
        "expiry_date": 20240228,
        "underlying_symbol": "510050.SH",
        "all_count": 2,
    }
    assert strategy.captured_counts == [1, 2]


class _WarmStartBuyOnceStrategy(Strategy):
    """Buy exactly one lot on the first phase-2 (restored) bar.

    Phase 1 stays flat so the checkpoint has no pending order to pickle.
    """

    def __init__(self) -> None:
        self.has_bought = False

    def on_bar(self, bar: Bar) -> None:
        if self.is_restored and not self.has_bought:
            self.buy(symbol=bar.symbol, quantity=1.0, tag="warm-slippage")
            self.has_bought = True


def _warm_start_phase2_fill_price(
    tmp_path: Path, slippage: Any, filename: str
) -> float:
    """Run cold->warm with a given global slippage and return phase-2 fill price."""
    checkpoint = tmp_path / filename
    phase1 = _make_bars("2023-01-01", 2, symbol="SLIP")
    phase2 = _make_bars("2023-02-01", 3, symbol="SLIP", start_price=200.0)

    result1 = run_backtest(
        data=phase1,
        strategy=_WarmStartBuyOnceStrategy,
        symbols="SLIP",
        initial_cash=1_000_000.0,
        show_progress=False,
        config=BacktestConfig(strategy_config=StrategyConfig(slippage=slippage)),
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="SLIP",
        show_progress=False,
        # 关键：只通过 config.strategy_config 传滑点，不显式传 slippage 入参
        config=BacktestConfig(strategy_config=StrategyConfig(slippage=slippage)),
    )
    filled = result2.orders_df[
        result2.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    assert len(filled) >= 1, "warm start phase-2 order did not fill"
    return float(filled.iloc[0]["avg_price"])


def test_run_warm_start_inherits_global_slippage_from_config(tmp_path: Path) -> None:
    """Global slippage in config.strategy_config must apply in warm start (issue #282).

    热启动此前完全不应用全局滑点，二阶段按原价撮合。此用例对比有/无滑点两次
    热启动的成交价，锁定继承路径。
    """
    price_with_slippage = _warm_start_phase2_fill_price(
        tmp_path, {"type": "percent", "value": 0.1}, "snapshot_slip_on.pkl"
    )
    price_no_slippage = _warm_start_phase2_fill_price(
        tmp_path, 0.0, "snapshot_slip_off.pkl"
    )
    # 10% 买入滑点必须抬高成交价；未继承时两者相等（旧 bug）
    assert price_with_slippage > price_no_slippage
    assert price_with_slippage == pytest.approx(price_no_slippage * 1.1)


def test_run_warm_start_inherits_config_from_snapshot(tmp_path: Path) -> None:
    """save_checkpoint(config=result) 持久化配置，热启动零显式入参自动继承 (issue #282).

    这是配置继承的根治路径：用户只在完整回测里配一次滑点，分阶段热启动无需
    重复传参即可继承。
    """
    checkpoint = tmp_path / "snapshot_cfg.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="SLIP")
    phase2 = _make_bars("2023-02-01", 3, symbol="SLIP", start_price=200.0)
    slippage = {"type": "percent", "value": 0.1}

    result1 = run_backtest(
        data=phase1,
        strategy=_WarmStartBuyOnceStrategy,
        symbols="SLIP",
        initial_cash=1_000_000.0,
        show_progress=False,
        config=BacktestConfig(strategy_config=StrategyConfig(slippage=slippage)),
    )
    # 关键：把 result 作为 config 传入，持久化已解析配置
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint), config=result1)  # type: ignore[arg-type]

    # 热启动完全不传 slippage / config
    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="SLIP",
        show_progress=False,
    )
    filled = result2.orders_df[
        result2.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    assert len(filled) >= 1
    price = float(filled.iloc[0]["avg_price"])
    # 首根 restored bar 下单，次根 bar open=201.0 撮合，10% 买入滑点 -> 221.1
    assert price == pytest.approx(201.0 * 1.1)


def test_run_warm_start_without_snapshot_config_is_backward_compatible(
    tmp_path: Path,
) -> None:
    """旧快照 (不传 config) 无 backtest_config，热启动降级不报错 (issue #282)."""
    checkpoint = tmp_path / "snapshot_legacy.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="SLIP")
    phase2 = _make_bars("2023-02-01", 3, symbol="SLIP", start_price=200.0)

    result1 = run_backtest(
        data=phase1,
        strategy=_WarmStartBuyOnceStrategy,
        symbols="SLIP",
        initial_cash=1_000_000.0,
        show_progress=False,
        config=BacktestConfig(
            strategy_config=StrategyConfig(slippage={"type": "percent", "value": 0.1})
        ),
    )
    # 不传 config，模拟旧快照
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    # 无快照配置、无显式入参 -> 无滑点，按原价撮合，且不抛错
    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="SLIP",
        show_progress=False,
    )
    filled = result2.orders_df[
        result2.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    assert len(filled) >= 1
    assert float(filled.iloc[0]["avg_price"]) == pytest.approx(201.0)


def test_run_warm_start_explicit_arg_overrides_snapshot_config(
    tmp_path: Path,
) -> None:
    """显式入参优先级高于快照配置 (issue #282)."""
    checkpoint = tmp_path / "snapshot_override.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="SLIP")
    phase2 = _make_bars("2023-02-01", 3, symbol="SLIP", start_price=200.0)

    result1 = run_backtest(
        data=phase1,
        strategy=_WarmStartBuyOnceStrategy,
        symbols="SLIP",
        initial_cash=1_000_000.0,
        show_progress=False,
        config=BacktestConfig(
            strategy_config=StrategyConfig(slippage={"type": "percent", "value": 0.1})
        ),
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint), config=result1)  # type: ignore[arg-type]

    # 显式传 5% 覆盖快照里的 10%
    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="SLIP",
        show_progress=False,
        slippage={"type": "percent", "value": 0.05},
    )
    filled = result2.orders_df[
        result2.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    assert len(filled) >= 1
    assert float(filled.iloc[0]["avg_price"]) == pytest.approx(201.0 * 1.05)


def test_run_warm_start_functional_lifecycle(tmp_path: Path) -> None:
    """Functional warm start should preserve callbacks, order, and state."""
    checkpoint = tmp_path / "snapshot_functional.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="FUNC")
    phase2 = _make_bars("2023-01-03", 2, symbol="FUNC", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=_functional_warm_on_bar,
        initialize=_functional_warm_initialize,
        on_start=_functional_warm_on_start,
        on_resume=_functional_warm_on_resume,
        symbols="FUNC",
        initial_cash=100000.0,
        show_progress=False,
    )

    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="FUNC",
        show_progress=False,
    )

    strategy = result2.strategy
    assert strategy is not None
    assert strategy.events == [
        "_default:start:restored=0",
        "_default:bar:100.50",
        "_default:bar:101.50",
        "_default:resume:bars=2:starts=1",
        "_default:start:restored=1",
        "_default:bar:102.50",
        "_default:bar:103.50",
    ]
    assert strategy.processed_closes == [100.5, 101.5, 102.5, 103.5]
    assert strategy.start_count == 2
    assert strategy.resume_count == 1


def test_run_warm_start_restores_functional_slot_strategies(tmp_path: Path) -> None:
    """Functional multi-slot warm start should restore slot strategy instances."""
    checkpoint = tmp_path / "snapshot_functional_slots.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="FUNC_SLOT")
    phase2 = _make_bars("2023-01-03", 2, symbol="FUNC_SLOT", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=_functional_warm_alpha_on_bar,
        initialize=_functional_warm_initialize,
        on_start=_functional_warm_on_start,
        on_resume=_functional_warm_on_resume,
        symbols="FUNC_SLOT",
        initial_cash=100000.0,
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": _functional_warm_beta_on_bar},
    )

    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="FUNC_SLOT",
        show_progress=False,
    )

    alpha_strategy = result2.strategy
    assert alpha_strategy is not None
    slot_strategies = getattr(alpha_strategy, "_slot_strategies", {})
    beta_strategy = slot_strategies.get("beta")
    assert beta_strategy is not None
    assert alpha_strategy.events == [
        "alpha:start:restored=0",
        "alpha:bar:100.50",
        "alpha:bar:101.50",
        "alpha:resume:bars=2:starts=1",
        "alpha:start:restored=1",
        "alpha:bar:102.50",
        "alpha:bar:103.50",
    ]
    assert beta_strategy.events == [
        "beta:start:restored=0",
        "beta:bar:100.50",
        "beta:bar:101.50",
        "beta:resume:bars=2:starts=1",
        "beta:start:restored=1",
        "beta:bar:102.50",
        "beta:bar:103.50",
    ]
    assert alpha_strategy.processed_closes == [100.5, 101.5, 102.5, 103.5]
    assert beta_strategy.processed_closes == [100.5, 101.5, 102.5, 103.5]
    assert alpha_strategy.start_count == 2
    assert beta_strategy.start_count == 2
    assert alpha_strategy.resume_count == 1
    assert beta_strategy.resume_count == 1
    assert sorted(slot_strategies.keys()) == ["beta"]


def test_run_warm_start_accepts_symbols_alias(tmp_path: Path) -> None:
    """run_warm_start should accept symbols as the primary symbol argument."""
    checkpoint = tmp_path / "snapshot_symbols_alias.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="TEST")
    phase2 = _make_bars("2023-01-03", 2, symbol="TEST", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        show_progress=False,
    )
    strategy = result2.strategy
    assert strategy is not None
    assert strategy.bar_seen == 4


def test_run_warm_start_rejects_legacy_symbol_keyword_alias(
    tmp_path: Path,
) -> None:
    """run_warm_start should reject removed symbol keyword alias."""
    checkpoint = tmp_path / "snapshot_symbols_warn.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="TEST")
    phase2 = _make_bars("2023-01-03", 2, symbol="TEST", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="no longer accepts `symbol`"):
        run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbol="TEST",
            show_progress=False,
        )


def test_run_warm_start_rejects_conflicting_symbol_and_symbols(tmp_path: Path) -> None:
    """run_warm_start should reject conflicting symbol and symbols inputs."""
    checkpoint = tmp_path / "snapshot_symbols_conflict.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="AAA")
    phase2 = _make_bars("2023-01-03", 1, symbol="AAA", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="AAA",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="no longer accepts `symbol`"):
        run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbol="AAA",
            symbols=["BBB"],
            show_progress=False,
        )


def test_run_warm_start_accepts_strategy_runtime_config(tmp_path: Path) -> None:
    """run_warm_start should inject runtime config into restored strategy."""
    checkpoint = tmp_path / "snapshot_runtime_config.pkl"
    phase1 = _make_bars("2023-01-01", 2)
    phase2 = _make_bars("2023-01-03", 2, start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=RuntimeConfigWarmStartStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        show_progress=False,
        strategy_runtime_config={"error_mode": "continue"},
    )

    strategy = result2.strategy
    assert strategy is not None
    assert strategy.errors == ["on_bar", "on_bar"]


def test_run_warm_start_exposes_resolved_execution_policy(tmp_path: Path) -> None:
    """run_warm_start should expose resolved execution policy when overridden."""
    checkpoint = tmp_path / "snapshot_policy_meta.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="TEST")
    phase2 = _make_bars("2023-01-03", 2, symbol="TEST", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        show_progress=False,
        fill_policy=NextClose(),
    )

    policy = result2.resolved_execution_policy
    assert policy is not None
    assert policy["price_basis"] == "close"
    assert int(policy["bar_offset"]) == 1
    # NextClose collapses close+1+next_event → same_cycle
    # (temporal inert for non-close+0 modes)
    assert policy["temporal"] == "same_cycle"
    assert policy["source"] == "fill_policy"


def test_run_warm_start_rejects_legacy_execution_overrides_without_fill_policy(
    tmp_path: Path,
) -> None:
    """run_warm_start should reject legacy execution overrides."""
    checkpoint = tmp_path / "snapshot_policy_warn.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="TEST")
    phase2 = _make_bars("2023-01-03", 2, symbol="TEST", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="run_from_checkpoint no longer accepts",
    ):
        legacy_kwargs: dict[str, Any] = {"execution_mode": "current_close"}
        _ = run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            **legacy_kwargs,
        )


def test_run_warm_start_rejects_legacy_timer_override_without_fill_policy(
    tmp_path: Path,
) -> None:
    """run_warm_start should reject legacy timer policy override."""
    checkpoint = tmp_path / "snapshot_timer_warn.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="TEST")
    phase2 = _make_bars("2023-01-03", 2, symbol="TEST", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="run_from_checkpoint no longer accepts",
    ):
        legacy_kwargs: dict[str, Any] = {"timer_execution_policy": "next_event"}
        _ = run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            **legacy_kwargs,
        )


def test_run_warm_start_rejects_legacy_execution_overrides_when_compat_disabled(
    tmp_path: Path,
) -> None:
    """Reject legacy warm_start execution overrides."""
    checkpoint = tmp_path / "snapshot_legacy_off.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="TEST")
    phase2 = _make_bars("2023-01-03", 2, symbol="TEST", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="run_from_checkpoint no longer accepts",
    ):
        legacy_kwargs: dict[str, Any] = {"execution_mode": "current_close"}
        _ = run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            **legacy_kwargs,
        )


def test_run_warm_start_rejects_non_bool_legacy_execution_policy_compat(
    tmp_path: Path,
) -> None:
    """legacy_execution_policy_compat should be removed in warm_start."""
    checkpoint = tmp_path / "snapshot_legacy_type.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="TEST")
    phase2 = _make_bars("2023-01-03", 2, symbol="TEST", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(
        TypeError, match="legacy_execution_policy_compat is no longer supported"
    ):
        compat_kwargs: dict[str, Any] = {"legacy_execution_policy_compat": "false"}
        _ = run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            **compat_kwargs,
        )


def test_run_warm_start_rejects_legacy_by_env_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy env var should not re-enable removed warm_start legacy overrides."""
    monkeypatch.setenv("AKQ_LEGACY_EXECUTION_POLICY_COMPAT", "false")
    checkpoint = tmp_path / "snapshot_legacy_env_off.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="TEST")
    phase2 = _make_bars("2023-01-03", 2, symbol="TEST", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="run_from_checkpoint no longer accepts",
    ):
        legacy_kwargs: dict[str, Any] = {"execution_mode": "current_close"}
        _ = run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            **legacy_kwargs,
        )


def test_run_warm_start_rejects_invalid_legacy_env_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid legacy compat env value should not affect fill_policy path."""
    checkpoint = tmp_path / "snapshot_legacy_env_bad.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="TEST")
    phase2 = _make_bars("2023-01-03", 2, symbol="TEST", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]
    monkeypatch.setenv("AKQ_LEGACY_EXECUTION_POLICY_COMPAT", "bad")
    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        fill_policy=CurrentClose(),
        show_progress=False,
    )
    assert result2.resolved_execution_policy is not None
    assert result2.resolved_execution_policy["source"] == "fill_policy"


def test_run_warm_start_explicit_fill_policy_dict_raises(tmp_path: Path) -> None:
    """Explicit fill_policy dict override is hard-cut (symmetric with run_backtest)."""
    checkpoint = tmp_path / "snapshot_dict_reject.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="TEST")
    phase2 = _make_bars("2023-01-03", 2, symbol="TEST", start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="no longer accepts a dict"):
        _ = run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            fill_policy={
                "price_basis": "close",
                "bar_offset": 0,
                "temporal": "same_cycle",
            },
        )


def test_run_warm_start_explicit_compat_overrides_env_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removed compat flag should fail in warm_start even when env is set."""
    checkpoint = tmp_path / "snapshot_legacy_env_override.pkl"
    phase1 = _make_bars("2023-01-01", 2, symbol="TEST")
    phase2 = _make_bars("2023-01-03", 2, symbol="TEST", start_price=102.0)
    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]
    monkeypatch.setenv("AKQ_LEGACY_EXECUTION_POLICY_COMPAT", "false")
    with pytest.raises(
        ValueError,
        match="run_from_checkpoint no longer accepts",
    ):
        legacy_kwargs: dict[str, Any] = {"execution_mode": "current_close"}
        _ = run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            **legacy_kwargs,
        )
    with pytest.raises(
        TypeError, match="legacy_execution_policy_compat is no longer supported"
    ):
        compat_kwargs: dict[str, Any] = {"legacy_execution_policy_compat": True}
        _ = run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            fill_policy=CurrentClose(),
            **compat_kwargs,
        )


def test_run_warm_start_restores_strategy_risk_state(tmp_path: Path) -> None:
    """Warm start should preserve strategy-level risk state across snapshots."""
    checkpoint = tmp_path / "snapshot_risk_state.pkl"
    phase1 = _make_bars("2023-01-01", 1)
    phase2 = _make_bars("2023-01-02", 1, start_price=101.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartRiskStateStrategy,
        symbols="TEST",
        initial_cash=100000.0,
        fill_policy=CurrentClose(),
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": WarmStartRiskStateStrategy},
        strategy_max_order_size={"alpha": 5.0, "beta": 20.0},
        strategy_risk_cooldown_bars={"alpha": 2},
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        fill_policy=CurrentClose(),
        show_progress=False,
    )
    engine = result2.engine
    assert engine is not None
    if not hasattr(engine, "get_default_strategy_id") or (
        not hasattr(engine, "get_strategy_slots")
        and not hasattr(engine, "get_strategy_slot_ids")
    ):
        pytest.skip("Engine binary does not expose strategy slot methods")
    assert cast(Any, engine).get_default_strategy_id() == "alpha"
    slot_getter = (
        cast(Any, engine).get_strategy_slot_ids
        if hasattr(engine, "get_strategy_slot_ids")
        else cast(Any, engine).get_strategy_slots
    )
    assert set(slot_getter()) == {"alpha", "beta"}
    orders_df = result2.orders_df
    phase2_rows = orders_df[
        orders_df["created_at"].dt.strftime("%Y-%m-%d") == "2023-01-02"
    ]
    reject_reasons = phase2_rows["reject_reason"].fillna("").astype(str).tolist()
    assert any("cooldown" in reason for reason in reject_reasons), reject_reasons


def test_run_warm_start_accepts_multi_slot_risk_overrides(tmp_path: Path) -> None:
    """run_warm_start should apply slot risk maps with explicit topology args."""
    checkpoint = tmp_path / "snapshot_risk_override.pkl"
    phase1 = _make_bars("2023-01-01", 1)
    phase2 = _make_bars("2023-01-02", 1, start_price=101.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartRiskStateStrategy,
        symbols="TEST",
        initial_cash=100000.0,
        fill_policy=CurrentClose(),
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": WarmStartRiskStateStrategy},
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        fill_policy=CurrentClose(),
        show_progress=False,
        strategy_id="alpha",
        strategies_by_slot={"beta": WarmStartRiskStateStrategy},
        strategy_max_order_size={"alpha": 5.0, "beta": 20.0},
    )
    phase2_rows = result2.orders_df[
        result2.orders_df["created_at"].dt.strftime("%Y-%m-%d") == "2023-01-02"
    ]
    alpha_rows = phase2_rows[phase2_rows["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = phase2_rows[phase2_rows["owner_strategy_id"].astype(str) == "beta"]
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    beta_reject_reasons = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    assert any("order quantity" in reason for reason in alpha_reject_reasons)
    assert not any("order quantity" in reason for reason in beta_reject_reasons)


def test_run_warm_start_accepts_multi_slot_risk_from_config(tmp_path: Path) -> None:
    """run_warm_start should accept strategy slot risk settings from config."""
    checkpoint = tmp_path / "snapshot_risk_from_config.pkl"
    phase1 = _make_bars("2023-01-01", 1)
    phase2 = _make_bars("2023-01-02", 1, start_price=101.0)
    config = BacktestConfig(
        strategy_config=StrategyConfig(
            initial_cash=100000.0,
            strategy_id="alpha",
            strategies_by_slot={"beta": WarmStartRiskStateStrategy},
            strategy_max_order_size={"alpha": 5.0, "beta": 20.0},
        )
    )

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartRiskStateStrategy,
        symbols="TEST",
        fill_policy=CurrentClose(),
        show_progress=False,
        config=config,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        fill_policy=CurrentClose(),
        show_progress=False,
        config=config,
    )
    phase2_rows = result2.orders_df[
        result2.orders_df["created_at"].dt.strftime("%Y-%m-%d") == "2023-01-02"
    ]
    alpha_rows = phase2_rows[phase2_rows["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = phase2_rows[phase2_rows["owner_strategy_id"].astype(str) == "beta"]
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    beta_reject_reasons = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    assert any("order quantity" in reason for reason in alpha_reject_reasons)
    assert not any("order quantity" in reason for reason in beta_reject_reasons)


def test_run_warm_start_explicit_slot_risk_overrides_config(tmp_path: Path) -> None:
    """Explicit warm-start slot risk args should override config values."""
    checkpoint = tmp_path / "snapshot_risk_from_config_override.pkl"
    phase1 = _make_bars("2023-01-01", 1)
    phase2 = _make_bars("2023-01-02", 1, start_price=101.0)
    config = BacktestConfig(
        strategy_config=StrategyConfig(
            initial_cash=100000.0,
            strategy_id="alpha",
            strategies_by_slot={"beta": WarmStartRiskStateStrategy},
            strategy_max_order_size={"alpha": 5.0, "beta": 20.0},
        )
    )

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartRiskStateStrategy,
        symbols="TEST",
        fill_policy=CurrentClose(),
        show_progress=False,
        config=config,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        fill_policy=CurrentClose(),
        show_progress=False,
        config=config,
        strategy_max_order_size={"alpha": 20.0, "beta": 5.0},
    )
    phase2_rows = result2.orders_df[
        result2.orders_df["created_at"].dt.strftime("%Y-%m-%d") == "2023-01-02"
    ]
    alpha_rows = phase2_rows[phase2_rows["owner_strategy_id"].astype(str) == "alpha"]
    beta_rows = phase2_rows[phase2_rows["owner_strategy_id"].astype(str) == "beta"]
    alpha_reject_reasons = alpha_rows["reject_reason"].fillna("").astype(str).tolist()
    beta_reject_reasons = beta_rows["reject_reason"].fillna("").astype(str).tolist()
    assert not any("order quantity" in reason for reason in alpha_reject_reasons)
    assert any("order quantity" in reason for reason in beta_reject_reasons)


def test_run_warm_start_accepts_fee_rate_names_and_default_timezone(
    tmp_path: Path,
) -> None:
    """run_warm_start should accept *_rate fee names and keep Asia/Shanghai default."""
    checkpoint = tmp_path / "snapshot_fee_rate_alias.pkl"
    phase1 = _make_bars("2023-01-01", 2)
    phase2 = _make_bars("2023-01-03", 2, start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        show_progress=False,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        transfer_fee_rate=0.00001,
        min_commission=0.0,
    )
    equity_index = cast(pd.DatetimeIndex, result2.equity_curve.index)
    assert equity_index.tz is not None
    assert str(equity_index.tz) == "Asia/Shanghai"


def test_run_warm_start_rejects_unknown_broker_profile(tmp_path: Path) -> None:
    """run_warm_start should validate broker_profile names."""
    checkpoint = tmp_path / "snapshot_unknown_profile.pkl"
    phase1 = _make_bars("2023-01-01", 2)
    phase2 = _make_bars("2023-01-03", 2, start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartE2EStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown broker_profile"):
        run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            broker_profile="does_not_exist",
        )


def test_run_warm_start_runtime_config_override_true_by_default(
    tmp_path: Path, caplog: Any
) -> None:
    """run_warm_start should override strategy runtime config by default."""
    checkpoint = tmp_path / "snapshot_runtime_override_true.pkl"
    phase1 = _make_bars("2023-01-01", 2)
    phase2 = _make_bars("2023-01-03", 2, start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=RuntimeConfigWarmConflictStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING, logger="akquant"):
        result2 = run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            strategy_runtime_config={"error_mode": "continue"},
        )

    strategy = result2.strategy
    assert strategy is not None
    assert strategy.errors == ["on_bar", "on_bar"]
    assert "overrides strategy runtime_config" in caplog.text
    assert "error_mode: raise -> continue" in caplog.text


def test_run_warm_start_runtime_config_override_false_keeps_strategy_config(
    tmp_path: Path, caplog: Any
) -> None:
    """runtime_config_override=False should keep restored strategy config."""
    checkpoint = tmp_path / "snapshot_runtime_override_false.pkl"
    phase1 = _make_bars("2023-01-01", 2)
    phase2 = _make_bars("2023-01-03", 1, start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=RuntimeConfigWarmConflictStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING, logger="akquant"):
        with pytest.raises(ValueError, match="warm_conflict_boom"):
            run_from_checkpoint(
                checkpoint_path=str(checkpoint),
                data=phase2,
                symbols="TEST",
                show_progress=False,
                strategy_runtime_config={"error_mode": "continue"},
                runtime_config_override=False,
            )

    assert "runtime_config_override=False" in caplog.text
    assert "error_mode: raise -> continue" in caplog.text


def test_run_warm_start_rejects_invalid_strategy_runtime_config_type(
    tmp_path: Path,
) -> None:
    """Invalid warm-start runtime config type should fail fast."""
    checkpoint = tmp_path / "snapshot_runtime_invalid_type.pkl"
    phase1 = _make_bars("2023-01-01", 2)
    phase2 = _make_bars("2023-01-03", 1, start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=RuntimeConfigWarmStartStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="strategy_runtime_config"):
        run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            strategy_runtime_config=cast(Any, "invalid"),
        )


def test_run_warm_start_rejects_invalid_runtime_config_from_forwarded_kwargs(
    tmp_path: Path,
) -> None:
    """Forwarded keyword map should keep strict runtime config validation."""
    checkpoint = tmp_path / "snapshot_runtime_forwarded_kwargs.pkl"
    phase1 = _make_bars("2023-01-01", 2)
    phase2 = _make_bars("2023-01-03", 1, start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=RuntimeConfigWarmStartStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    forwarded_kwargs = cast(Any, {"strategy_runtime_config": "invalid"})
    with pytest.raises(TypeError, match="strategy_runtime_config"):
        run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            **forwarded_kwargs,
        )


def test_run_warm_start_accepts_runtime_config_from_forwarded_kwargs(
    tmp_path: Path,
) -> None:
    """Forwarded keyword map should support valid runtime config injection."""
    checkpoint = tmp_path / "snapshot_runtime_forwarded_valid.pkl"
    phase1 = _make_bars("2023-01-01", 2)
    phase2 = _make_bars("2023-01-03", 2, start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=RuntimeConfigWarmStartStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    forwarded_kwargs = cast(
        Any, {"strategy_runtime_config": {"error_mode": "continue"}}
    )
    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="TEST",
        show_progress=False,
        **forwarded_kwargs,
    )

    strategy = result2.strategy
    assert strategy is not None
    assert strategy.errors == ["on_bar", "on_bar"]


def test_run_warm_start_rejects_unknown_strategy_runtime_config_fields(
    tmp_path: Path,
) -> None:
    """Unknown warm-start runtime config fields should fail with field-level error."""
    checkpoint = tmp_path / "snapshot_runtime_unknown_field.pkl"
    phase1 = _make_bars("2023-01-01", 2)
    phase2 = _make_bars("2023-01-03", 1, start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=RuntimeConfigWarmStartStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unknown fields: unknown_flag"):
        run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            strategy_runtime_config=cast(Any, {"unknown_flag": True}),
        )


def test_run_warm_start_rejects_invalid_strategy_runtime_config_values(
    tmp_path: Path,
) -> None:
    """Invalid warm-start runtime config values should fail with wrapped error."""
    checkpoint = tmp_path / "snapshot_runtime_invalid_value.pkl"
    phase1 = _make_bars("2023-01-01", 2)
    phase2 = _make_bars("2023-01-03", 1, start_price=102.0)

    result1 = run_backtest(
        data=phase1,
        strategy=RuntimeConfigWarmStartStrategy,
        symbols="TEST",
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="invalid strategy_runtime_config"):
        run_from_checkpoint(
            checkpoint_path=str(checkpoint),
            data=phase2,
            symbols="TEST",
            show_progress=False,
            strategy_runtime_config={"portfolio_update_eps": -1},
        )


class WarmStartMultiSymbolStrategy(Strategy):
    """Strategy for multi-symbol warm start continuity test."""

    def __init__(self) -> None:
        """Initialize per-symbol counters."""
        self.total_bars = 0
        self.by_symbol: dict[str, int] = {}
        self.events: list[str] = []

    def on_resume(self) -> None:
        """Record resume callback."""
        self.events.append("on_resume")

    def on_start(self) -> None:
        """Record start callback."""
        self.events.append("on_start")

    def on_bar(self, bar: Bar) -> None:
        """Track processed bars by symbol."""
        self.total_bars += 1
        self.by_symbol[bar.symbol] = self.by_symbol.get(bar.symbol, 0) + 1


class WarmStartMultiSymbolDataFrameStrategy(Strategy):
    """Strategy for multi-symbol dataframe warm-start regression tests."""

    def __init__(self) -> None:
        """Initialize deterministic event capture."""
        self.set_history_depth(2)
        self.records: list[tuple[int, str, str, float | None, float | None]] = []

    def on_bar(self, bar: Bar) -> None:
        """Capture symbol resolution and per-symbol close history."""
        aaa_last = float(self.get_history(1, symbol="AAA", field="close")[-1])
        bbb_last = float(self.get_history(1, symbol="BBB", field="close")[-1])
        self.records.append(
            (
                int(bar.timestamp),
                str(bar.symbol),
                str(self.symbol),
                None if np.isnan(aaa_last) else aaa_last,
                None if np.isnan(bbb_last) else bbb_last,
            )
        )


def _make_symbol_df(
    symbol: str,
    start: str,
    periods: int,
    start_price: float,
) -> pd.DataFrame:
    """Create deterministic OHLCV dataframe for a symbol."""
    ts = pd.date_range(start=start, periods=periods, freq="D", tz="UTC")
    prices = [start_price + float(i) for i in range(periods)]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": prices,
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": [p + 0.5 for p in prices],
            "volume": [1000.0 + float(i) for i in range(periods)],
            "symbol": [symbol] * periods,
        }
    )


def _combine_symbol_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine symbol frames into one timestamp-sorted long dataframe."""
    return cast(
        pd.DataFrame,
        pd.concat(list(frames), ignore_index=True)
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True),
    )


def test_run_warm_start_multi_symbol_continuity(tmp_path: Path) -> None:
    """Warm start should preserve multi-symbol state continuity."""
    checkpoint = tmp_path / "snapshot_multi.pkl"
    phase1 = {
        "AAA": _make_symbol_df("AAA", "2023-01-01", 3, 100.0),
        "BBB": _make_symbol_df("BBB", "2023-01-01", 3, 200.0),
    }
    phase2 = {
        "AAA": _make_symbol_df("AAA", "2023-01-04", 2, 103.0),
        "BBB": _make_symbol_df("BBB", "2023-01-04", 2, 203.0),
    }

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartMultiSymbolStrategy,
        initial_cash=100000.0,
        show_progress=False,
    )

    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        show_progress=False,
    )

    strategy = result2.strategy
    assert strategy is not None
    assert strategy.total_bars == 10
    assert strategy.by_symbol == {"AAA": 5, "BBB": 5}
    assert "on_resume" in strategy.events
    resume_idx = strategy.events.index("on_resume")
    assert strategy.events[resume_idx + 1] == "on_start"
    assert result2.metrics.initial_market_value == result1.metrics.end_market_value


def test_run_warm_start_multi_symbol_dataframe_continuity(tmp_path: Path) -> None:
    """Warm start should preserve symbol routing for multi-symbol dataframe input."""
    checkpoint = tmp_path / "snapshot_multi_dataframe.pkl"
    phase1_aaa = _make_symbol_df("AAA", "2023-01-01", 3, 100.0)
    phase1_bbb = _make_symbol_df("BBB", "2023-01-01", 3, 200.0)
    phase2_aaa = _make_symbol_df("AAA", "2023-01-04", 2, 103.0)
    phase2_bbb = _make_symbol_df("BBB", "2023-01-04", 2, 203.0)
    phase1 = _combine_symbol_frames(phase1_aaa, phase1_bbb)
    phase2 = _combine_symbol_frames(phase2_aaa, phase2_bbb)
    full = _combine_symbol_frames(phase1, phase2)

    full_result = run_backtest(
        data=full,
        strategy=WarmStartMultiSymbolDataFrameStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        show_progress=False,
    )
    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartMultiSymbolDataFrameStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    warm_result = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols=["AAA", "BBB"],
        show_progress=False,
    )

    full_strategy = full_result.strategy
    warm_strategy = warm_result.strategy
    assert full_strategy is not None
    assert warm_strategy is not None
    assert warm_strategy.records == full_strategy.records
    assert all(
        bar_symbol == self_symbol
        for _, bar_symbol, self_symbol, _, _ in warm_strategy.records
    )


class WarmStartOpenPositionMetricsStrategy(Strategy):
    """Strategy for open-position warm-start metric continuity tests."""

    def __init__(self) -> None:
        """Initialize deterministic trading state."""
        self.submitted = False

    def on_bar(self, bar: Bar) -> None:
        """Open one futures position and keep it across warm start."""
        if not self.submitted:
            self.buy(symbol="FUT", quantity=1.0)
            self.submitted = True


def test_run_warm_start_open_position_preserves_initial_market_value(
    tmp_path: Path,
) -> None:
    """Warm start should use prior end equity as the new initial market value."""
    checkpoint = tmp_path / "snapshot_open_position.pkl"
    phase1 = _make_bars("2023-01-01", 3, symbol="FUT", start_price=100.0)
    phase2 = _make_bars("2023-01-04", 2, symbol="FUT", start_price=103.0)
    config = BacktestConfig(
        strategy_config=StrategyConfig(exit_on_last_bar=False),
        instruments_config=[
            InstrumentConfig(
                symbol="FUT",
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
                tick_size=0.2,
            )
        ],
    )

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartOpenPositionMetricsStrategy,
        symbols="FUT",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
        config=config,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="FUT",
        show_progress=False,
        fill_policy=CurrentClose(),
        config=config,
    )

    assert result1.metrics.end_market_value > result1.strategy.get_account()["cash"]  # type: ignore[union-attr]
    assert result2.metrics.initial_market_value == pytest.approx(
        result1.metrics.end_market_value
    )
    assert result2.metrics.total_return == pytest.approx(
        (result2.metrics.end_market_value - result2.metrics.initial_market_value)
        / result2.metrics.initial_market_value
    )


class WarmStartEventIdempotencyStrategy(Strategy):
    """Strategy for warm-start event idempotency checks."""

    def __init__(self) -> None:
        """Initialize state and duplicate trackers."""
        self.bar_seen_by_symbol: dict[str, int] = {}
        self.trade_callback_total = 0
        self.trade_duplicate_count = 0
        self.order_callback_total = 0
        self.order_duplicate_count = 0
        self.trade_keys: set[tuple[str, int, str, str, float, float]] = set()
        self.order_keys: set[tuple[str, str, float]] = set()
        self.events: list[str] = []

    def on_resume(self) -> None:
        """Record resume callback."""
        self.events.append("on_resume")

    def on_start(self) -> None:
        """Record start callback."""
        self.events.append("on_start")

    def on_bar(self, bar: Bar) -> None:
        """Submit deterministic round-trip orders across two phases."""
        count = self.bar_seen_by_symbol.get(bar.symbol, 0) + 1
        self.bar_seen_by_symbol[bar.symbol] = count
        if count in (1, 3):
            self.buy(bar.symbol, 1)
        elif count in (2, 4):
            self.sell(bar.symbol, 1)

    def on_order(self, order: Any) -> None:
        """Track duplicate order callbacks for identical state transitions."""
        self.order_callback_total += 1
        key = (order.id, str(order.status), float(order.filled_quantity))
        if key in self.order_keys:
            self.order_duplicate_count += 1
        else:
            self.order_keys.add(key)

    def on_trade(self, trade: Any) -> None:
        """Track duplicate trade callbacks."""
        self.trade_callback_total += 1
        key = (
            trade.order_id,
            int(trade.timestamp),
            trade.symbol,
            str(trade.side),
            float(trade.quantity),
            float(trade.price),
        )
        if key in self.trade_keys:
            self.trade_duplicate_count += 1
        else:
            self.trade_keys.add(key)


def test_run_warm_start_multi_symbol_event_idempotency(tmp_path: Path) -> None:
    """Warm start should not duplicate order/trade callbacks."""
    checkpoint = tmp_path / "snapshot_event_idempotency.pkl"
    phase1 = {
        "AAA": _make_symbol_df("AAA", "2023-01-01", 2, 100.0),
        "BBB": _make_symbol_df("BBB", "2023-01-01", 2, 200.0),
    }
    phase2 = {
        "AAA": _make_symbol_df("AAA", "2023-01-03", 2, 102.0),
        "BBB": _make_symbol_df("BBB", "2023-01-03", 2, 202.0),
    }

    result1 = run_backtest(
        data=phase1,
        strategy=WarmStartEventIdempotencyStrategy,
        initial_cash=100000.0,
        fill_policy=CurrentClose(),
        show_progress=False,
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        show_progress=False,
    )

    strategy = result2.strategy
    assert strategy is not None
    assert strategy.bar_seen_by_symbol == {"AAA": 4, "BBB": 4}
    assert strategy.trade_callback_total > 0
    assert strategy.order_callback_total > 0
    assert strategy.trade_duplicate_count == 0
    assert strategy.order_duplicate_count == 0
    assert "on_resume" in strategy.events


class TimerIdempotencyStrategy(Strategy):
    """Strategy for timer registration idempotency tests."""

    def __init__(self) -> None:
        """Initialize counters."""
        self.timer_count = 0
        self.events: list[str] = []

    def on_resume(self) -> None:
        """Record resume callback."""
        self.events.append("on_resume")

    def on_start(self) -> None:
        """Register timers once."""
        self.events.append("on_start")
        self.schedule("2023-01-01 10:00:00", "manual_timer")
        self.schedule_daily("14:55:00", "daily_timer")

    def on_timer(self, payload: str) -> None:
        """Count timer callbacks."""
        self.timer_count += 1
        self.events.append(f"on_timer:{payload}")


def test_timer_registration_not_duplicated_in_warm_start() -> None:
    """Internal start should not double-register timers when called repeatedly."""
    strategy = TimerIdempotencyStrategy()
    strategy._is_restored = True
    ctx = MagicMock(spec=StrategyContext)
    strategy.ctx = ctx
    strategy._trading_days = [
        pd.Timestamp("2023-01-01").tz_localize("Asia/Shanghai"),
        pd.Timestamp("2023-01-02").tz_localize("Asia/Shanghai"),
    ]

    strategy._on_start_internal()
    strategy._on_start_internal()

    calls = strategy.ctx.schedule.call_args_list
    assert len(calls) == 3
    payloads = [c.args[1] for c in calls]
    assert payloads.count("manual_timer") == 1
    assert payloads.count("__daily__|14:55:00|daily_timer") == 2


def test_daily_timer_payload_processed_once_per_event() -> None:
    """Daily wrapped payload should trigger one timer callback per event."""
    strategy = TimerIdempotencyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    strategy.ctx = ctx
    strategy._trading_days = [
        pd.Timestamp("2023-01-01").tz_localize("Asia/Shanghai"),
    ]

    strategy._on_timer_event("__daily__|14:55:00|daily_timer", ctx)
    assert strategy.timer_count == 1
    assert strategy.events[-1] == "on_timer:daily_timer"


def test_live_mode_timer_registration_not_duplicated() -> None:
    """Live mode timer registration should run once in internal start."""
    strategy = TimerIdempotencyStrategy()
    strategy._is_restored = True
    ctx = MagicMock(spec=StrategyContext)
    strategy.ctx = ctx
    strategy._trading_days = []

    strategy._on_start_internal()
    strategy._on_start_internal()

    calls = strategy.ctx.schedule.call_args_list
    assert len(calls) == 2
    payloads = [c.args[1] for c in calls]
    assert payloads.count("manual_timer") == 1
    assert payloads.count("__daily__|14:55:00|daily_timer") == 1


def test_on_start_timer_registration_is_deferred_until_context_ready() -> None:
    """Allow on_start schedule calls before context is injected."""
    strategy = TimerIdempotencyStrategy()

    strategy._on_start_internal()

    assert len(strategy._pending_schedules) == 1
    assert len(strategy._pending_daily_timers) == 1

    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []

    ts = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    bar = Bar(
        timestamp=ts,
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        volume=1000.0,
        symbol="AAPL",
    )
    strategy._on_bar_event(bar, ctx)

    assert len(strategy._pending_schedules) == 0
    assert len(strategy._pending_daily_timers) == 0
    payloads = [call.args[1] for call in ctx.schedule.call_args_list]
    assert "manual_timer" in payloads
    assert "__daily__|14:55:00|daily_timer" in payloads


def test_live_mode_daily_timer_reschedules_once_per_trigger() -> None:
    """Live mode daily timer should reschedule once per single trigger."""
    strategy = TimerIdempotencyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    strategy.ctx = ctx
    strategy._trading_days = []

    strategy._on_timer_event("__daily__|14:55:00|daily_timer", ctx)

    assert strategy.timer_count == 1
    assert strategy.events[-1] == "on_timer:daily_timer"
    assert strategy.ctx.schedule.call_count == 1
    schedule_call = strategy.ctx.schedule.call_args
    assert schedule_call is not None
    assert isinstance(schedule_call.args[0], int)
    assert schedule_call.args[1] == "__daily__|14:55:00|daily_timer"


def test_live_mode_daily_timer_logs_reschedule_failures(caplog: Any) -> None:
    """Live daily timer reschedule failures should be logged with strategy context."""
    strategy = TimerIdempotencyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    strategy.ctx = ctx
    strategy._trading_days = []
    strategy._owner_strategy_id = "alpha"
    cast(Any, strategy).schedule = MagicMock(
        side_effect=RuntimeError("schedule failed")
    )

    with caplog.at_level(logging.WARNING, logger="akquant.strategy.events"):
        strategy._on_timer_event("__daily__|14:55:00|daily_timer", ctx)

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to reschedule live daily timer"
    )
    assert record.phase == "strategy"
    assert record.strategy_id == "alpha"
    assert record.slot == "alpha"


class TimerOrderTradeMixedStrategy(Strategy):
    """Strategy for mixed timer/order/trade event ordering tests."""

    def __init__(self) -> None:
        """Initialize counters."""
        self.order_count = 0
        self.trade_count = 0
        self.timer_count = 0
        self.events: list[str] = []

    def on_order(self, order: Any) -> None:
        """Record order callback."""
        self.order_count += 1
        self.events.append(f"on_order:{order.id}")

    def on_trade(self, trade: Any) -> None:
        """Record trade callback."""
        self.trade_count += 1
        self.events.append(f"on_trade:{trade.order_id}")

    def on_timer(self, payload: str) -> None:
        """Record timer callback."""
        self.timer_count += 1
        self.events.append(f"on_timer:{payload}")


def test_mixed_events_order_on_daily_timer_in_live_mode() -> None:
    """Daily timer should process order/trade first, then timer, and reschedule once."""
    strategy = TimerOrderTradeMixedStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = [
        SimpleNamespace(id="o1", status="Submitted", filled_quantity=0.0)
    ]
    ctx.recent_trades = [SimpleNamespace(order_id="o1")]
    strategy.ctx = ctx
    strategy._trading_days = []

    strategy._on_timer_event("__daily__|14:55:00|mixed_timer", ctx)

    assert strategy.order_count == 1
    assert strategy.trade_count == 1
    assert strategy.timer_count == 1
    assert strategy.events == ["on_order:o1", "on_trade:o1", "on_timer:mixed_timer"]
    assert strategy.ctx.schedule.call_count == 1
    schedule_call = strategy.ctx.schedule.call_args
    assert schedule_call is not None
    assert schedule_call.args[1] == "__daily__|14:55:00|mixed_timer"


def test_mixed_events_partial_fill_then_cancel_sequence() -> None:
    """Partial fill and cancel should emit deterministic callbacks."""
    strategy = TimerOrderTradeMixedStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = [
        SimpleNamespace(id="o2", status="Submitted", filled_quantity=0.0)
    ]
    ctx.recent_trades = []
    strategy.ctx = ctx
    strategy._trading_days = []

    strategy._on_timer_event("__daily__|14:55:00|mixed_timer2", ctx)

    ctx.canceled_order_ids = []
    ctx.active_orders = [
        SimpleNamespace(id="o2", status="Submitted", filled_quantity=1.0)
    ]
    ctx.recent_trades = [SimpleNamespace(order_id="o2")]
    strategy._on_timer_event("__daily__|14:55:00|mixed_timer2", ctx)

    ctx.canceled_order_ids = ["o2"]
    ctx.active_orders = []
    ctx.recent_trades = []
    strategy._on_timer_event("__daily__|14:55:00|mixed_timer2", ctx)

    assert strategy.order_count == 3
    assert strategy.trade_count == 1
    assert strategy.timer_count == 3
    assert strategy.events == [
        "on_order:o2",
        "on_timer:mixed_timer2",
        "on_order:o2",
        "on_trade:o2",
        "on_timer:mixed_timer2",
        "on_order:o2",
        "on_timer:mixed_timer2",
    ]
    assert strategy.ctx.schedule.call_count == 3


def test_daily_timer_malformed_payload_falls_back_to_raw_timer() -> None:
    """Malformed daily payload should fallback to raw timer callback."""
    strategy = TimerOrderTradeMixedStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    strategy.ctx = ctx
    strategy._trading_days = []

    strategy._on_timer_event("__daily__|bad_payload", ctx)

    assert strategy.order_count == 0
    assert strategy.trade_count == 0
    assert strategy.timer_count == 1
    assert strategy.events == ["on_timer:__daily__|bad_payload"]
    assert strategy.ctx.schedule.call_count == 0


def test_daily_timer_invalid_time_fires_once_without_raw_fallback() -> None:
    """Invalid daily time should not duplicate timer callback."""
    strategy = TimerOrderTradeMixedStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    strategy.ctx = ctx
    strategy._trading_days = []

    strategy._on_timer_event("__daily__|not_a_time|mixed_timer3", ctx)

    assert strategy.order_count == 0
    assert strategy.trade_count == 0
    assert strategy.timer_count == 1
    assert strategy.events == ["on_timer:mixed_timer3"]
    assert strategy.ctx.schedule.call_count == 0


def test_unknown_canceled_order_does_not_emit_order_callback() -> None:
    """Unknown canceled order id should not trigger on_order callback."""
    strategy = TimerOrderTradeMixedStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = ["ghost_order"]
    ctx.active_orders = []
    ctx.recent_trades = []
    strategy.ctx = ctx
    strategy._trading_days = []

    strategy._on_timer_event("plain_timer", ctx)

    assert strategy.order_count == 0
    assert strategy.trade_count == 0
    assert strategy.timer_count == 1
    assert strategy.events == ["on_timer:plain_timer"]


class FrameworkHooksStrategy(Strategy):
    """Strategy for framework-level hooks tests."""

    def __init__(self) -> None:
        """Initialize records."""
        self.events: list[str] = []
        self.errors: list[tuple[str, str]] = []
        self.portfolio_updates = 0

    def on_before_trading(self, trading_date: Any, timestamp: int) -> None:
        """Record before trading hook."""
        self.events.append(f"before:{trading_date}:{timestamp}")

    def on_after_trading(self, trading_date: Any, timestamp: int) -> None:
        """Record after trading hook."""
        self.events.append(f"after:{trading_date}:{timestamp}")

    def on_portfolio_update(self, snapshot: dict[str, Any]) -> None:
        """Record portfolio update."""
        self.portfolio_updates += 1
        self.events.append(f"portfolio:{snapshot['cash']}:{snapshot['equity']}")

    def on_reject(self, order: Any) -> None:
        """Record reject callback."""
        self.events.append(f"reject:{order.id}")

    def on_order(self, order: Any) -> None:
        """Record order callback."""
        self.events.append(f"order:{order.id}")

    def on_tick(self, tick: Tick) -> None:
        """Record tick callback."""
        self.events.append("tick")

    def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
        """Record error callback."""
        self.errors.append((source, type(error).__name__))

    def on_stop(self) -> None:
        """Record stop callback."""
        self.events.append("stop")


class FrameworkPreOpenStrategy(Strategy):
    """Strategy for pre-open framework hook tests."""

    def __init__(self) -> None:
        """Initialize captured pre-open events."""
        self.pre_open_events: list[dict[str, Any]] = []

    def on_pre_open(self, event: dict[str, Any]) -> None:
        """Record pre-open event and place a default market order."""
        self.pre_open_events.append(dict(event))
        self.buy(symbol="AAPL", quantity=1.0)


def test_framework_hooks_session_day_reject_and_portfolio() -> None:
    """Framework hooks should fire with expected transitions."""
    strategy = FrameworkHooksStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.recent_trades = []
    ctx.positions = {"AAPL": 1.0}
    ctx.available_positions = {"AAPL": 1.0}
    rejected = SimpleNamespace(
        id="rej1",
        status=OrderStatus.Rejected,
        filled_quantity=0.0,
        average_filled_price=None,
    )
    ctx.active_orders = [rejected]
    ctx.cash = 1000.0
    ctx.session = "normal"
    ctx.current_time = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value

    tick1 = Tick(timestamp=ctx.current_time, price=100.0, volume=1.0, symbol="AAPL")
    strategy._on_tick_event(tick1, ctx)

    assert any(e.startswith("before:") for e in strategy.events)
    assert "order:rej1" in strategy.events
    assert "reject:rej1" in strategy.events
    assert strategy.events.index("order:rej1") < strategy.events.index("reject:rej1")
    assert strategy.events.index("reject:rej1") < strategy.events.index("tick")
    assert strategy.portfolio_updates == 1

    ctx.current_time = pd.Timestamp("2023-01-01 15:10:00", tz="Asia/Shanghai").value
    ctx.session = "postmarket"
    tick2 = Tick(timestamp=ctx.current_time, price=101.0, volume=1.0, symbol="AAPL")
    strategy._on_tick_event(tick2, ctx)
    assert any(e.startswith("after:") for e in strategy.events)
    same_day_before_count = len(
        [e for e in strategy.events if e.startswith("before:2023-01-01")]
    )
    assert same_day_before_count == 1

    before_count = len([e for e in strategy.events if e.startswith("before:")])
    ctx.current_time = pd.Timestamp("2023-01-02 09:31:00", tz="Asia/Shanghai").value
    ctx.session = "normal"
    tick3 = Tick(timestamp=ctx.current_time, price=102.0, volume=1.0, symbol="AAPL")
    strategy._on_tick_event(tick3, ctx)
    after_count = len([e for e in strategy.events if e.startswith("before:")])
    assert after_count == before_count + 1


def test_framework_hooks_emits_reject_from_recent_rejected_orders() -> None:
    """Reject callback should be emitted from recent_rejected_orders snapshots."""
    strategy = FrameworkHooksStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.cash = 1000.0
    ctx.session = "normal"
    ctx.current_time = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    ctx.recent_rejected_orders = [
        SimpleNamespace(
            id="rej_from_snapshot",
            status=OrderStatus.Rejected,
            filled_quantity=0.0,
            average_filled_price=None,
        )
    ]

    tick = Tick(timestamp=ctx.current_time, price=100.0, volume=1.0, symbol="AAPL")
    strategy._on_tick_event(tick, ctx)

    assert "order:rej_from_snapshot" in strategy.events
    assert "reject:rej_from_snapshot" in strategy.events
    assert strategy.events.index("order:rej_from_snapshot") < strategy.events.index(
        "reject:rej_from_snapshot"
    )
    assert strategy.events.index("reject:rej_from_snapshot") < strategy.events.index(
        "tick"
    )


def test_before_trading_fallback_when_session_missing() -> None:
    """When session is missing, on_before_trading still runs once per day."""
    strategy = FrameworkHooksStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.cash = 1000.0
    ctx.session = None

    ts1 = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    ts2 = pd.Timestamp("2023-01-01 10:30:00", tz="Asia/Shanghai").value
    ts3 = pd.Timestamp("2023-01-02 09:30:00", tz="Asia/Shanghai").value

    tick1 = Tick(timestamp=ts1, price=100.0, volume=1.0, symbol="AAPL")
    tick2 = Tick(timestamp=ts2, price=100.0, volume=1.0, symbol="AAPL")
    tick3 = Tick(timestamp=ts3, price=100.0, volume=1.0, symbol="AAPL")

    ctx.current_time = ts1
    strategy._on_tick_event(tick1, ctx)
    ctx.current_time = ts2
    strategy._on_tick_event(tick2, ctx)
    ctx.current_time = ts3
    strategy._on_tick_event(tick3, ctx)

    day1 = len([e for e in strategy.events if e.startswith("before:2023-01-01")])
    day2 = len([e for e in strategy.events if e.startswith("before:2023-01-02")])
    assert day1 == 1
    assert day2 == 1


def test_portfolio_update_skips_clean_tick_without_changes() -> None:
    """Clean tick without price/position/order changes should not emit update."""
    strategy = FrameworkHooksStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.cash = 1000.0
    ctx.session = "normal"
    ts1 = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    ts2 = pd.Timestamp("2023-01-01 09:30:01", tz="Asia/Shanghai").value
    tick1 = Tick(timestamp=ts1, price=100.0, volume=1.0, symbol="AAPL")
    tick2 = Tick(timestamp=ts2, price=100.0, volume=1.0, symbol="AAPL")

    ctx.current_time = ts1
    strategy._on_tick_event(tick1, ctx)
    assert strategy.portfolio_updates == 1

    ctx.current_time = ts2
    strategy._on_tick_event(tick2, ctx)
    assert strategy.portfolio_updates == 1


def test_portfolio_update_emits_on_price_change_with_position() -> None:
    """Price move with open position should emit portfolio update."""
    strategy = FrameworkHooksStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 1.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.positions = {"AAPL": 1.0}
    ctx.available_positions = {"AAPL": 1.0}
    ctx.cash = 1000.0
    ctx.session = "normal"
    ts1 = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    ts2 = pd.Timestamp("2023-01-01 09:30:01", tz="Asia/Shanghai").value
    tick1 = Tick(timestamp=ts1, price=100.0, volume=1.0, symbol="AAPL")
    tick2 = Tick(timestamp=ts2, price=101.0, volume=1.0, symbol="AAPL")

    ctx.current_time = ts1
    strategy._on_tick_event(tick1, ctx)
    assert strategy.portfolio_updates == 1

    ctx.current_time = ts2
    strategy._on_tick_event(tick2, ctx)
    assert strategy.portfolio_updates == 2


def test_portfolio_update_respects_eps_threshold() -> None:
    """Small equity changes under eps should not emit updates."""
    strategy = FrameworkHooksStrategy()
    strategy.portfolio_update_eps = 5.0
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 1.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.positions = {"AAPL": 1.0}
    ctx.available_positions = {"AAPL": 1.0}
    ctx.cash = 1000.0
    ctx.session = "normal"
    ts1 = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value
    ts2 = pd.Timestamp("2023-01-01 09:30:01", tz="Asia/Shanghai").value
    tick1 = Tick(timestamp=ts1, price=100.0, volume=1.0, symbol="AAPL")
    tick2 = Tick(timestamp=ts2, price=101.0, volume=1.0, symbol="AAPL")

    ctx.current_time = ts1
    strategy._on_tick_event(tick1, ctx)
    assert strategy.portfolio_updates == 1

    ctx.current_time = ts2
    strategy._on_tick_event(tick2, ctx)
    assert strategy.portfolio_updates == 1


class ErrorHookStrategy(Strategy):
    """Strategy for on_error hook tests."""

    def __init__(self) -> None:
        """Initialize captured errors."""
        self.captured: list[tuple[str, str]] = []

    def on_tick(self, tick: Tick) -> None:
        """Raise an exception for testing."""
        raise ValueError("boom")

    def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
        """Capture on_error callback."""
        self.captured.append((source, str(error)))


class RuntimeConfigBarErrorStrategy(Strategy):
    """Strategy for runtime config injection test via run_backtest."""

    def __init__(self) -> None:
        """Initialize error counter."""
        self.errors: list[str] = []

    def on_bar(self, bar: Bar) -> None:
        """Raise error on each bar."""
        raise ValueError("bar_boom")

    def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
        """Record callback source."""
        self.errors.append(source)


class RuntimeConfigConflictStrategy(Strategy):
    """Strategy for runtime config conflict behavior tests."""

    def __init__(self) -> None:
        """Initialize with strict default config."""
        self.errors: list[str] = []
        self.runtime_config = StrategyRuntimeConfig(error_mode="raise")

    def on_bar(self, bar: Bar) -> None:
        """Raise on every bar."""
        raise ValueError("conflict_boom")

    def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
        """Record callback source."""
        self.errors.append(source)


def test_runtime_config_validation_rejects_invalid_values() -> None:
    """Runtime config should validate mode and eps."""
    with pytest.raises(ValueError, match="portfolio_update_eps"):
        StrategyRuntimeConfig(portfolio_update_eps=-1)
    with pytest.raises(ValueError, match="error_mode"):
        StrategyRuntimeConfig(error_mode=cast(Any, "bad"))


def test_runtime_alias_properties_sync_to_runtime_config() -> None:
    """Legacy alias fields should update runtime_config."""
    strategy = ErrorHookStrategy()
    strategy.enable_precise_day_boundary_hooks = True
    strategy.portfolio_update_eps = 1.5
    strategy.error_mode = "continue"
    strategy.re_raise_on_error = False

    cfg = strategy.runtime_config
    assert cfg.enable_precise_day_boundary_hooks is True
    assert cfg.portfolio_update_eps == 1.5
    assert cfg.error_mode == "continue"
    assert cfg.re_raise_on_error is False


def test_runtime_config_assignment_syncs_alias_properties() -> None:
    """runtime_config assignment should reflect on alias property getters."""
    strategy = ErrorHookStrategy()
    strategy.runtime_config = StrategyRuntimeConfig(
        enable_precise_day_boundary_hooks=True,
        portfolio_update_eps=2.0,
        error_mode="legacy",
        re_raise_on_error=False,
    )

    assert strategy.enable_precise_day_boundary_hooks is True
    assert strategy.portfolio_update_eps == 2.0
    assert strategy.error_mode == "legacy"
    assert strategy.re_raise_on_error is False


def test_on_error_hook_called_and_exception_re_raised() -> None:
    """Errors in user callback should call on_error then re-raise."""
    strategy = ErrorHookStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.cash = 1000.0
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.session = "normal"
    ctx.current_time = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value

    tick = Tick(timestamp=ctx.current_time, price=100.0, volume=1.0, symbol="AAPL")
    with pytest.raises(ValueError, match="boom"):
        strategy._on_tick_event(tick, ctx)

    assert strategy.captured == [("on_tick", "boom")]


def test_on_error_hook_can_swallow_user_callback_exception() -> None:
    """Errors can be swallowed when re_raise_on_error is disabled."""
    strategy = ErrorHookStrategy()
    strategy.error_mode = "legacy"
    strategy.re_raise_on_error = False
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.cash = 1000.0
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.session = "normal"
    ctx.current_time = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value

    tick = Tick(timestamp=ctx.current_time, price=100.0, volume=1.0, symbol="AAPL")
    strategy._on_tick_event(tick, ctx)

    assert strategy.captured == [("on_tick", "boom")]


def test_error_mode_continue_overrides_re_raise_flag() -> None:
    """error_mode=continue should swallow even when re_raise_on_error=True."""
    strategy = ErrorHookStrategy()
    strategy.error_mode = "continue"
    strategy.re_raise_on_error = True
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.cash = 1000.0
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.session = "normal"
    ctx.current_time = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value

    tick = Tick(timestamp=ctx.current_time, price=100.0, volume=1.0, symbol="AAPL")
    strategy._on_tick_event(tick, ctx)

    assert strategy.captured == [("on_tick", "boom")]


def test_error_mode_raise_overrides_re_raise_flag() -> None:
    """error_mode=raise should re-raise even when re_raise_on_error=False."""
    strategy = ErrorHookStrategy()
    strategy.error_mode = "raise"
    strategy.re_raise_on_error = False
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.cash = 1000.0
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.session = "normal"
    ctx.current_time = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value

    tick = Tick(timestamp=ctx.current_time, price=100.0, volume=1.0, symbol="AAPL")
    with pytest.raises(ValueError, match="boom"):
        strategy._on_tick_event(tick, ctx)

    assert strategy.captured == [("on_tick", "boom")]


def test_runtime_config_continue_mode_swallow_exception() -> None:
    """runtime_config should drive error handling when legacy fields stay default."""
    strategy = ErrorHookStrategy()
    strategy.runtime_config = StrategyRuntimeConfig(error_mode="continue")
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.cash = 1000.0
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.session = "normal"
    ctx.current_time = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value

    tick = Tick(timestamp=ctx.current_time, price=100.0, volume=1.0, symbol="AAPL")
    strategy._on_tick_event(tick, ctx)

    assert strategy.captured == [("on_tick", "boom")]


def test_legacy_error_mode_overrides_runtime_config() -> None:
    """Legacy fields should override runtime_config for compatibility."""
    strategy = ErrorHookStrategy()
    strategy.runtime_config = StrategyRuntimeConfig(error_mode="raise")
    strategy.error_mode = "continue"
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.cash = 1000.0
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.session = "normal"
    ctx.current_time = pd.Timestamp("2023-01-01 09:30:00", tz="Asia/Shanghai").value

    tick = Tick(timestamp=ctx.current_time, price=100.0, volume=1.0, symbol="AAPL")
    strategy._on_tick_event(tick, ctx)

    assert strategy.captured == [("on_tick", "boom")]


def test_run_backtest_accepts_strategy_runtime_config() -> None:
    """run_backtest should inject runtime config into strategy instance."""
    bars = _make_bars("2023-01-01", 3, symbol="TEST")
    result = run_backtest(
        data=bars,
        strategy=RuntimeConfigBarErrorStrategy,
        symbols="TEST",
        show_progress=False,
        strategy_runtime_config={"error_mode": "continue"},
    )

    strategy = result.strategy
    assert strategy is not None
    assert strategy.errors == ["on_bar", "on_bar", "on_bar"]


def test_run_backtest_rejects_invalid_strategy_runtime_config_type() -> None:
    """Invalid runtime config type should fail fast."""
    bars = _make_bars("2023-01-01", 1, symbol="TEST")
    with pytest.raises(TypeError, match="strategy_runtime_config"):
        run_backtest(
            data=bars,
            strategy=RuntimeConfigBarErrorStrategy,
            symbols="TEST",
            show_progress=False,
            strategy_runtime_config=cast(Any, "invalid"),
        )


def test_run_backtest_rejects_invalid_runtime_config_from_strategy_params() -> None:
    """strategy_params path should validate runtime config strictly."""
    bars = _make_bars("2023-01-01", 1, symbol="TEST")
    with pytest.raises(TypeError, match="strategy_runtime_config"):
        run_backtest(
            data=bars,
            strategy=RuntimeConfigBarErrorStrategy,
            symbols="TEST",
            show_progress=False,
            strategy_params={"strategy_runtime_config": "invalid"},
        )


def test_run_backtest_explicit_runtime_config_has_higher_priority_than_kwargs() -> None:
    """Explicit backtest runtime config should override forwarded values."""
    bars = _make_bars("2023-01-01", 2, symbol="TEST")
    result = run_backtest(
        data=bars,
        strategy=RuntimeConfigBarErrorStrategy,
        symbols="TEST",
        show_progress=False,
        strategy_runtime_config={"error_mode": "continue"},
        strategy_params={"strategy_runtime_config": {"error_mode": "raise"}},
    )

    strategy = result.strategy
    assert strategy is not None
    assert strategy.errors == ["on_bar", "on_bar"]


def test_run_backtest_rejects_unknown_strategy_runtime_config_fields() -> None:
    """Unknown runtime config fields should produce field-level error."""
    bars = _make_bars("2023-01-01", 1, symbol="TEST")
    with pytest.raises(ValueError, match="unknown fields: unknown_flag"):
        run_backtest(
            data=bars,
            strategy=RuntimeConfigBarErrorStrategy,
            symbols="TEST",
            show_progress=False,
            strategy_runtime_config=cast(Any, {"unknown_flag": True}),
        )


def test_run_backtest_rejects_invalid_strategy_runtime_config_values() -> None:
    """Invalid runtime config values should produce wrapped validation error."""
    bars = _make_bars("2023-01-01", 1, symbol="TEST")
    with pytest.raises(ValueError, match="invalid strategy_runtime_config"):
        run_backtest(
            data=bars,
            strategy=RuntimeConfigBarErrorStrategy,
            symbols="TEST",
            show_progress=False,
            strategy_runtime_config={"portfolio_update_eps": -1},
        )


def test_run_backtest_runtime_config_override_true_by_default(
    caplog: Any,
) -> None:
    """run_backtest should override strategy runtime config by default."""
    bars = _make_bars("2023-01-01", 2, symbol="TEST")
    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = run_backtest(
            data=bars,
            strategy=RuntimeConfigConflictStrategy,
            symbols="TEST",
            show_progress=False,
            strategy_runtime_config={"error_mode": "continue"},
        )

    strategy = result.strategy
    assert strategy is not None
    assert strategy.errors == ["on_bar", "on_bar"]
    assert "overrides strategy runtime_config" in caplog.text


def test_runtime_config_conflict_warning_deduplicated_per_strategy_instance(
    caplog: Any,
) -> None:
    """Conflict warning should be emitted once for same strategy instance."""
    strategy = RuntimeConfigConflictStrategy()
    bars = _make_bars("2023-01-01", 1, symbol="TEST")
    with caplog.at_level(logging.WARNING, logger="akquant"):
        run_backtest(
            data=bars,
            strategy=strategy,
            symbols="TEST",
            show_progress=False,
            strategy_runtime_config={"error_mode": "continue"},
        )
        run_backtest(
            data=bars,
            strategy=strategy,
            symbols="TEST",
            show_progress=False,
            strategy_runtime_config={"error_mode": "continue"},
        )

    logs = caplog.text
    assert logs.count("overrides strategy runtime_config") == 1
    assert strategy.errors == ["on_bar", "on_bar"]


def test_run_backtest_runtime_config_override_false_keeps_strategy_config(
    caplog: Any,
) -> None:
    """runtime_config_override=False should keep strategy-side config."""
    bars = _make_bars("2023-01-01", 1, symbol="TEST")
    with caplog.at_level(logging.WARNING, logger="akquant"):
        with pytest.raises(ValueError, match="conflict_boom"):
            run_backtest(
                data=bars,
                strategy=RuntimeConfigConflictStrategy,
                symbols="TEST",
                show_progress=False,
                strategy_runtime_config={"error_mode": "continue"},
                runtime_config_override=False,
            )

    assert "runtime_config_override=False" in caplog.text


def test_stop_internal_flushes_on_after_trading_hook() -> None:
    """Stop phase should flush a pending on_after_trading."""
    strategy = FrameworkHooksStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.cash = 1000.0
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.session = "normal"
    ctx.current_time = pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Shanghai").value

    tick = Tick(timestamp=ctx.current_time, price=100.0, volume=1.0, symbol="AAPL")
    strategy._on_tick_event(tick, ctx)
    assert any(e.startswith("before:") for e in strategy.events)
    assert not any(e.startswith("after:") for e in strategy.events)

    strategy._on_stop_internal()

    assert any(e.startswith("after:") for e in strategy.events)
    assert strategy.events[-1] == "stop"


def test_boundary_timers_register_and_drive_day_hooks() -> None:
    """Boundary timers should register once and trigger day hooks precisely."""
    strategy = FrameworkHooksStrategy()
    strategy.enable_precise_day_boundary_hooks = True
    start_ts = pd.Timestamp("2023-01-03 09:30:00", tz="Asia/Shanghai").value
    end_ts = pd.Timestamp("2023-01-03 15:00:00", tz="Asia/Shanghai").value
    strategy._trading_day_bounds = {"2023-01-03": (start_ts, end_ts)}

    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = 0.0
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.cash = 1000.0
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.session = "closed"
    ctx.current_time = start_ts

    tick = Tick(timestamp=start_ts, price=100.0, volume=1.0, symbol="AAPL")
    strategy._on_tick_event(tick, ctx)

    scheduled = [call.args for call in ctx.schedule.call_args_list]
    assert (start_ts, "__framework_boundary__|before|2023-01-03") in scheduled
    assert (end_ts + 1, "__framework_boundary__|after|2023-01-03") in scheduled

    strategy._on_timer_event("__framework_boundary__|before|2023-01-03", ctx)
    assert any(e.startswith("before:2023-01-03") for e in strategy.events)

    ctx.current_time = end_ts + 1
    strategy._on_timer_event("__framework_boundary__|after|2023-01-03", ctx)
    assert any(e.startswith("after:2023-01-03") for e in strategy.events)


def test_collect_boundary_timers_and_prime_globally_once() -> None:
    """Boundary timers should be deduplicated before the event loop starts."""
    start_ts = pd.Timestamp("2023-01-03 09:30:00", tz="Asia/Shanghai").value
    end_ts = pd.Timestamp("2023-01-03 15:00:00", tz="Asia/Shanghai").value
    strategy_a = FrameworkHooksStrategy()
    strategy_b = FrameworkHooksStrategy()
    strategy_a.enable_precise_day_boundary_hooks = True
    strategy_b.enable_precise_day_boundary_hooks = True
    strategy_a._trading_day_bounds = {"2023-01-03": (start_ts, end_ts)}
    strategy_b._trading_day_bounds = {"2023-01-03": (start_ts, end_ts)}

    entries = collect_boundary_timer_entries(strategy_a)
    assert entries == [
        (start_ts, "__framework_boundary__|before|2023-01-03"),
        (end_ts + 1, "__framework_boundary__|after|2023-01-03"),
    ]

    fake_engine = SimpleNamespace(add_timer=MagicMock())
    _prime_framework_boundary_timers([strategy_a, strategy_b], fake_engine)

    scheduled = [call.args for call in fake_engine.add_timer.call_args_list]
    assert scheduled == [
        (start_ts, "__framework_boundary__|before|2023-01-03"),
        (end_ts + 1, "__framework_boundary__|after|2023-01-03"),
    ]


def test_build_trading_day_metadata_merges_multi_symbol_day_bounds() -> None:
    """Trading day metadata should merge per-day bounds across symbols."""
    data_map = {
        "AAA": pd.DataFrame(
            {"close": [10.0, 10.5]},
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2023-01-03 01:30:00", tz="UTC"),
                    pd.Timestamp("2023-01-03 07:00:00", tz="UTC"),
                ]
            ),
        ),
        "BBB": pd.DataFrame(
            {"close": [20.0, 21.0]},
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2023-01-03 02:00:00", tz="UTC"),
                    pd.Timestamp("2023-01-03 06:00:00", tz="UTC"),
                ]
            ),
        ),
    }

    trading_days, day_bounds, day_rebalance_timestamps = _build_trading_day_metadata(
        data_map, "Asia/Shanghai"
    )

    assert trading_days == [pd.Timestamp("2023-01-03", tz="Asia/Shanghai")]
    assert day_bounds == {
        "2023-01-03": (
            pd.Timestamp("2023-01-03 01:30:00", tz="UTC").value,
            pd.Timestamp("2023-01-03 07:00:00", tz="UTC").value,
        )
    }
    assert day_rebalance_timestamps == {
        "2023-01-03": pd.Timestamp("2023-01-03 02:00:00", tz="UTC").value
    }


def _reference_build_trading_day_metadata(
    data_map: dict[str, pd.DataFrame], timezone: str
) -> tuple[list, dict, dict]:
    """原逐组迭代实现（对照用，与向量化替换前的生产实现一致）."""
    all_dates: set = set()
    day_bounds: dict = {}
    day_first_symbol_timestamps: dict = {}

    for symbol, df in data_map.items():
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            continue
        local_index = df.index
        if local_index.tz is None:
            local_index = local_index.tz_localize("UTC")
        local_index = local_index.tz_convert(timezone)
        normalized_index = local_index.normalize()
        all_dates.update(normalized_index.unique())
        grouped = df.groupby(normalized_index, sort=False)
        for day_ts, day_df in grouped:
            day_key = pd.Timestamp(day_ts).date().isoformat()
            start_ns = int(day_df.index.min().value)
            end_ns = int(day_df.index.max().value)
            day_first_symbol_timestamps.setdefault(day_key, {})[str(symbol)] = start_ns
            if day_key in day_bounds:
                prev_start, prev_end = day_bounds[day_key]
                day_bounds[day_key] = (min(prev_start, start_ns), max(prev_end, end_ns))
            else:
                day_bounds[day_key] = (start_ns, end_ns)

    day_rebalance_timestamps: dict = {}
    for day_key, symbol_timestamps in day_first_symbol_timestamps.items():
        if symbol_timestamps:
            day_rebalance_timestamps[day_key] = max(symbol_timestamps.values())
    return sorted(all_dates), day_bounds, day_rebalance_timestamps


def _metadata_fixture() -> dict[str, pd.DataFrame]:
    days = pd.date_range("2023-01-03", periods=5, freq="B", tz="UTC")
    data_map = {}
    for i, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        idx = days.append(days[:2] + pd.Timedelta(hours=1))
        data_map[symbol] = pd.DataFrame(
            {"close": [10.0 + i + j * 0.01 for j in range(len(idx))]}, index=idx
        )
    return data_map


def test_build_trading_day_metadata_matches_reference_groupby() -> None:
    """向量化实现与原逐组迭代实现在多标的多日数据上输出一致."""
    data_map = _metadata_fixture()
    expected = _reference_build_trading_day_metadata(data_map, "Asia/Shanghai")
    actual = _build_trading_day_metadata(data_map, "Asia/Shanghai")
    assert actual == expected


def test_build_trading_day_metadata_us_unit_and_naive() -> None:
    """微秒(us)单位 datetime64 与 tz-naive 索引口径与原实现一致."""
    data_map = _metadata_fixture()
    for df in data_map.values():
        index = cast(pd.DatetimeIndex, df.index)
        df.index = index.as_unit("us").tz_localize(None)
    expected = _reference_build_trading_day_metadata(data_map, "Asia/Shanghai")
    actual = _build_trading_day_metadata(data_map, "Asia/Shanghai")
    assert actual == expected


def test_build_trading_day_metadata_skips_empty_and_non_datetime() -> None:
    """空 frame 与非 DatetimeIndex 被跳过；全空时返回空三元组."""
    fixture = _metadata_fixture()
    data_map = {
        "EMPTY": pd.DataFrame({"close": []}),
        "RANGER": pd.DataFrame({"close": [1.0, 2.0]}),
        "AAA": fixture["AAA"],
    }
    expected = _reference_build_trading_day_metadata(data_map, "Asia/Shanghai")
    actual = _build_trading_day_metadata(data_map, "Asia/Shanghai")
    assert actual == expected

    assert _build_trading_day_metadata(
        {"X": pd.DataFrame({"close": []})}, "Asia/Shanghai"
    ) == ([], {}, {})


def test_build_trading_day_metadata_rebalance_takes_max_of_symbol_first_bars() -> None:
    """调仓时间戳应取「各标的当日首笔的最大值」，而非全局首笔.

    锁定 day_rebalance_timestamps 的双层 groupby 语义：先按 (day, sym) 取各
    标的当日首 bar，再按 day 取这些首 bar 的 max。三标的首笔错峰，正确结果应是
    最晚开始那只标的的首 bar，而非最早的全局首 bar。
    """
    day_bounds_start = pd.Timestamp("2023-03-01 01:30:00", tz="UTC")
    data_map = {
        "S1": pd.DataFrame(
            {"close": [1.0, 1.1]},
            index=pd.DatetimeIndex(
                [day_bounds_start, pd.Timestamp("2023-03-01 07:00:00", tz="UTC")]
            ),
        ),
        "S2": pd.DataFrame(
            {"close": [2.0, 2.1]},
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2023-03-01 02:15:00", tz="UTC"),
                    pd.Timestamp("2023-03-01 06:30:00", tz="UTC"),
                ]
            ),
        ),
        "S3": pd.DataFrame(
            {"close": [3.0, 3.1]},
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2023-03-01 03:45:00", tz="UTC"),
                    pd.Timestamp("2023-03-01 05:00:00", tz="UTC"),
                ]
            ),
        ),
    }

    _, _, day_rebalance_timestamps = _build_trading_day_metadata(
        data_map, "Asia/Shanghai"
    )

    assert day_rebalance_timestamps == {
        "2023-03-01": pd.Timestamp("2023-03-01 03:45:00", tz="UTC").value
    }


def test_build_trading_day_metadata_drops_nat_rows() -> None:
    """含 NaT 的 bar 应被丢弃，不污染同一交易日其它标的的 bounds.

    NaT 经 asi8 会映射为 int64 哨兵最小值；向量化路径若不剔除，会把它折进当日
    的全标的 min，横向污染 bounds。剔除后结果应与不含该 NaT 行时逐字段一致。
    """
    clean = {
        "AAA": pd.DataFrame(
            {"close": [10.0, 10.5]},
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2023-01-03 01:30:00", tz="UTC"),
                    pd.Timestamp("2023-01-03 07:00:00", tz="UTC"),
                ]
            ),
        ),
    }
    dirty = {
        "AAA": pd.DataFrame(
            {"close": [10.0, 10.5, 99.0]},
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2023-01-03 01:30:00", tz="UTC"),
                    pd.Timestamp("2023-01-03 07:00:00", tz="UTC"),
                    pd.NaT,
                ]
            ),
        ),
    }

    assert _build_trading_day_metadata(
        dirty, "Asia/Shanghai"
    ) == _build_trading_day_metadata(clean, "Asia/Shanghai")

    # 全为 NaT 时退化为空三元组，与全空输入一致。
    all_nat = {
        "AAA": pd.DataFrame(
            {"close": [1.0]}, index=pd.DatetimeIndex([pd.NaT], dtype="datetime64[ns]")
        ),
    }
    assert _build_trading_day_metadata(all_nat, "Asia/Shanghai") == ([], {}, {})


def test_precise_boundary_hooks_delay_after_trading_until_day_end() -> None:
    """Precise boundary hooks should not emit after_trading during same-day timers."""

    class PreciseBoundaryRegressionStrategy(Strategy):
        def __init__(self) -> None:
            self.enable_precise_day_boundary_hooks = True
            self.events: list[tuple[str, object, int]] = []

        def on_start(self) -> None:
            self.subscribe("HOOKS_DEMO")

        def on_before_trading(self, trading_date: object, timestamp: int) -> None:
            self.events.append(("before", trading_date, timestamp))

        def on_after_trading(self, trading_date: object, timestamp: int) -> None:
            self.events.append(("after", trading_date, timestamp))

        def on_bar(self, bar: Bar) -> None:
            self.events.append(("bar", bar.symbol, int(bar.timestamp)))

    day1_open = pd.Timestamp("2023-01-03 09:30:00", tz="Asia/Shanghai").value
    day1_mid = pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai").value
    day1_close = pd.Timestamp("2023-01-03 15:00:00", tz="Asia/Shanghai").value
    day2_open = pd.Timestamp("2023-01-04 09:30:00", tz="Asia/Shanghai").value
    day2_mid = pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai").value
    day2_close = pd.Timestamp("2023-01-04 15:00:00", tz="Asia/Shanghai").value

    rows = [
        (day1_open, 10.0),
        (day1_mid, 10.2),
        (day1_close, 10.5),
        (day2_open, 10.8),
        (day2_mid, 10.0),
        (day2_close, 10.6),
    ]
    bars = [
        Bar(
            timestamp=timestamp,
            open=close,
            high=close + 0.2,
            low=close - 0.2,
            close=close,
            volume=1000.0,
            symbol="HOOKS_DEMO",
        )
        for timestamp, close in rows
    ]

    strategy = PreciseBoundaryRegressionStrategy()
    run_backtest(
        data=bars,
        strategy=strategy,
        symbols=["HOOKS_DEMO"],
        initial_cash=1000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    day1_before = ("before", pd.Timestamp("2023-01-03").date(), day1_open)
    day1_last_bar = ("bar", "HOOKS_DEMO", day1_close)
    day1_after = ("after", pd.Timestamp("2023-01-03").date(), day1_close)

    assert day1_before in strategy.events
    assert day1_last_bar in strategy.events
    assert day1_after in strategy.events
    assert strategy.events.index(day1_before) < strategy.events.index(day1_last_bar)
    assert strategy.events.index(day1_before) < strategy.events.index(day1_after)


def test_non_precise_boundary_hooks_fire_with_continuous_session_backtest() -> None:
    """Default backtest path should emit day hooks even when session is Continuous."""

    class NonPreciseBoundaryRegressionStrategy(Strategy):
        def __init__(self) -> None:
            self.enable_precise_day_boundary_hooks = False
            self.events: list[tuple[str, object, int]] = []

        def on_start(self) -> None:
            self.subscribe("HOOKS_DEMO")

        def on_before_trading(self, trading_date: object, timestamp: int) -> None:
            self.events.append(("before", trading_date, timestamp))

        def on_after_trading(self, trading_date: object, timestamp: int) -> None:
            self.events.append(("after", trading_date, timestamp))

        def on_bar(self, bar: Bar) -> None:
            self.events.append(("bar", bar.symbol, int(bar.timestamp)))

    day1_open = pd.Timestamp("2023-01-03 09:30:00", tz="Asia/Shanghai").value
    day1_close = pd.Timestamp("2023-01-03 15:00:00", tz="Asia/Shanghai").value
    day2_open = pd.Timestamp("2023-01-04 09:30:00", tz="Asia/Shanghai").value
    bars = [
        Bar(
            timestamp=timestamp,
            open=close,
            high=close + 0.2,
            low=close - 0.2,
            close=close,
            volume=1000.0,
            symbol="HOOKS_DEMO",
        )
        for timestamp, close in [
            (day1_open, 10.0),
            (day1_close, 10.5),
            (day2_open, 10.8),
        ]
    ]

    strategy = NonPreciseBoundaryRegressionStrategy()
    run_backtest(
        data=bars,
        strategy=strategy,
        symbols=["HOOKS_DEMO"],
        initial_cash=1000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    day1 = pd.Timestamp("2023-01-03").date()
    day2 = pd.Timestamp("2023-01-04").date()
    day1_before = ("before", day1, day1_open)
    day2_before = ("before", day2, day2_open)

    assert day1_before in strategy.events
    assert day2_before in strategy.events
    assert strategy.events.index(day1_before) < strategy.events.index(
        ("bar", "HOOKS_DEMO", day1_open)
    )

    # #324: in the default (non-precise) mode on_after_trading now fires at the
    # day's session close (within day1, at/after its last bar), not lazily on the
    # next day's bar. Firing at day1's close keeps a next-open order's created_at
    # inside day1 so it fills on day2 (T+1) rather than T+2.
    after_day1 = [e for e in strategy.events if e[0] == "after" and e[1] == day1]
    assert len(after_day1) == 1, after_day1
    after_day1_ts = after_day1[0][2]
    assert day1_close <= after_day1_ts < day2_open, after_day1_ts
    assert strategy.events.index(after_day1[0]) < strategy.events.index(day2_before)


@pytest.mark.parametrize("precise_boundaries", [False, True])
def test_day_boundary_hooks_hide_current_bar_from_history_and_current_bar(
    precise_boundaries: bool,
) -> None:
    """Day-boundary hooks should only observe prior-day history in both modes."""

    class BoundaryHistoryVisibilityStrategy(Strategy):
        def __init__(self) -> None:
            self.enable_precise_day_boundary_hooks = precise_boundaries
            self.before_histories: list[np.ndarray] = []
            self.rebalance_histories: list[np.ndarray] = []
            self.before_current_bar_states: list[bool] = []
            self.rebalance_current_bar_states: list[bool] = []
            self.events: list[tuple[str, int]] = []
            self.set_history_depth(1)

        def on_start(self) -> None:
            self.subscribe("HOOKS_DEMO")

        def on_before_trading(self, trading_date: object, timestamp: int) -> None:
            self.before_histories.append(self.get_history(1, "HOOKS_DEMO"))
            self.before_current_bar_states.append(self.current_bar is None)
            self.events.append(("before", int(timestamp)))

        def on_bar(self, bar: Bar) -> None:
            self.events.append(("bar", int(bar.timestamp)))

    bars = [
        Bar(
            timestamp=pd.Timestamp(day_text, tz="Asia/Shanghai").value,
            open=close,
            high=close + 0.2,
            low=close - 0.2,
            close=close,
            volume=1000.0,
            symbol="HOOKS_DEMO",
        )
        for day_text, close in [
            ("2023-01-03", 10.0),
            ("2023-01-04", 10.5),
            ("2023-01-05", 10.8),
        ]
    ]

    strategy = BoundaryHistoryVisibilityStrategy()
    run_backtest(
        data=bars,
        strategy=strategy,
        symbols=["HOOKS_DEMO"],
        initial_cash=1000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    before_values = [float(history[0]) for history in strategy.before_histories[1:]]
    assert len(strategy.before_histories) == 3
    assert np.isnan(strategy.before_histories[0][0])
    assert before_values == [10.0, 10.5]
    assert strategy.before_current_bar_states == [True, True, True]
    assert strategy.events[:2] == [
        ("before", bars[0].timestamp),
        ("bar", bars[0].timestamp),
    ]


@pytest.mark.parametrize("precise_boundaries", [False, True])
def test_day_boundary_hooks_use_previous_account_snapshot(
    precise_boundaries: bool,
) -> None:
    """Day-boundary hooks should not see current-day account marking in either mode."""

    class BoundaryAccountVisibilityStrategy(Strategy):
        def __init__(self) -> None:
            self.enable_precise_day_boundary_hooks = precise_boundaries
            self.before_equities: list[float] = []
            self.before_portfolio_values: list[float] = []
            self.has_bought = False

        def on_start(self) -> None:
            self.subscribe("HOOKS_DEMO")

        def on_before_trading(self, trading_date: object, timestamp: int) -> None:
            account = self.get_account()
            self.before_equities.append(float(account["equity"]))
            self.before_portfolio_values.append(float(self.equity))

        def on_bar(self, bar: Bar) -> None:
            if not self.has_bought:
                self.buy(bar.symbol, 1.0)
                self.has_bought = True

    bars = [
        Bar(
            timestamp=pd.Timestamp(day_text, tz="Asia/Shanghai").value,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1000.0,
            symbol="HOOKS_DEMO",
        )
        for day_text, close in [
            ("2023-01-03", 10.0),
            ("2023-01-04", 11.0),
            ("2023-01-05", 12.0),
        ]
    ]

    strategy = BoundaryAccountVisibilityStrategy()
    run_backtest(
        data=bars,
        strategy=strategy,
        symbols=["HOOKS_DEMO"],
        initial_cash=1000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    assert strategy.before_equities == [1000.0, 1000.0, 1001.0]
    assert strategy.before_portfolio_values == [1000.0, 1000.0, 1001.0]


@pytest.mark.parametrize("precise_boundaries", [False, True])
def test_cross_section_sees_current_day_history_and_runs_after_bar(
    precise_boundaries: bool,
) -> None:
    """After-bar rebalance should observe the current completed slice."""

    class AfterBarHistoryVisibilityStrategy(Strategy):
        def __init__(self) -> None:
            self.enable_precise_day_boundary_hooks = precise_boundaries
            self.after_bar_histories: list[np.ndarray] = []
            self.after_bar_current_bar_states: list[bool] = []
            self.events: list[tuple[str, int]] = []
            self.set_history_depth(1)

        def on_start(self) -> None:
            self.subscribe("HOOKS_DEMO")

        def on_cross_section(self, trading_date: object, timestamp: int) -> None:
            self.after_bar_histories.append(self.get_history(1, "HOOKS_DEMO"))
            self.after_bar_current_bar_states.append(self.current_bar is None)
            self.events.append(("after_bar", int(timestamp)))

        def on_bar(self, bar: Bar) -> None:
            self.events.append(("bar", int(bar.timestamp)))

    bars = [
        Bar(
            timestamp=pd.Timestamp(day_text, tz="Asia/Shanghai").value,
            open=close,
            high=close + 0.2,
            low=close - 0.2,
            close=close,
            volume=1000.0,
            symbol="HOOKS_DEMO",
        )
        for day_text, close in [
            ("2023-01-03", 10.0),
            ("2023-01-04", 10.5),
            ("2023-01-05", 10.8),
        ]
    ]

    strategy = AfterBarHistoryVisibilityStrategy()
    run_backtest(
        data=bars,
        strategy=strategy,
        symbols=["HOOKS_DEMO"],
        initial_cash=1000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    assert [float(history[0]) for history in strategy.after_bar_histories] == [
        10.0,
        10.5,
        10.8,
    ]
    assert strategy.after_bar_current_bar_states == [True, True, True]
    assert strategy.events[:2] == [
        ("bar", bars[0].timestamp),
        ("after_bar", bars[0].timestamp),
    ]


@pytest.mark.parametrize("precise_boundaries", [False, True])
def test_cross_section_uses_current_account_snapshot(
    precise_boundaries: bool,
) -> None:
    """After-bar rebalance should see the current-day marked account view."""

    class AfterBarAccountVisibilityStrategy(Strategy):
        def __init__(self) -> None:
            self.enable_precise_day_boundary_hooks = precise_boundaries
            self.after_bar_equities: list[float] = []
            self.after_bar_portfolio_values: list[float] = []
            self.has_bought = False

        def on_start(self) -> None:
            self.subscribe("HOOKS_DEMO")

        def on_cross_section(self, trading_date: object, timestamp: int) -> None:
            account = self.get_account()
            self.after_bar_equities.append(float(account["equity"]))
            self.after_bar_portfolio_values.append(float(self.equity))

        def on_bar(self, bar: Bar) -> None:
            if not self.has_bought:
                self.buy(bar.symbol, 1.0)
                self.has_bought = True

    bars = [
        Bar(
            timestamp=pd.Timestamp(day_text, tz="Asia/Shanghai").value,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1000.0,
            symbol="HOOKS_DEMO",
        )
        for day_text, close in [
            ("2023-01-03", 10.0),
            ("2023-01-04", 11.0),
            ("2023-01-05", 12.0),
        ]
    ]

    strategy = AfterBarAccountVisibilityStrategy()
    run_backtest(
        data=bars,
        strategy=strategy,
        symbols=["HOOKS_DEMO"],
        initial_cash=1000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    assert strategy.after_bar_equities == [1000.0, 1001.0, 1002.0]
    assert strategy.after_bar_portfolio_values == [1000.0, 1001.0, 1002.0]


def test_get_account_uses_previous_cash_when_framework_snapshot_enabled() -> None:
    """Framework snapshot mode should return the previous cash value consistently."""

    class SnapshotCashStrategy(Strategy):
        pass

    strategy = SnapshotCashStrategy()
    strategy.ctx = cast(
        Any,
        SimpleNamespace(
            cash=950.0,
            previous_cash=900.0,
            positions={},
            active_orders=[],
            canceled_order_ids=[],
            account_equity=1050.0,
            account_market_value=100.0,
            account_notional_value=100.0,
            account_used_margin=0.0,
            account_unrealized_pnl=0.0,
            account_maintenance_ratio=0.0,
            previous_account_equity=1000.0,
            previous_account_market_value=100.0,
            previous_account_notional_value=100.0,
            previous_account_used_margin=0.0,
            previous_account_unrealized_pnl=0.0,
            previous_account_maintenance_ratio=0.0,
        ),
    )
    strategy._framework_use_previous_account_snapshot = True

    account = strategy.get_account()

    assert float(account["cash"]) == 900.0
    assert float(account["equity"]) == 1000.0


def test_get_account_uses_cached_previous_account_details_when_enabled() -> None:
    """Framework snapshot mode should reuse cached derived account details."""

    class SnapshotDerivedStrategy(Strategy):
        pass

    strategy = SnapshotDerivedStrategy()
    strategy.ctx = cast(
        Any,
        SimpleNamespace(
            cash=950.0,
            previous_cash=900.0,
            positions={"SHORT": -2.0},
            active_orders=[],
            canceled_order_ids=[],
            margin_accrued_interest=9.0,
            margin_daily_interest=4.0,
            account_equity=1050.0,
            account_market_value=100.0,
            account_notional_value=100.0,
            account_used_margin=12.0,
            account_unrealized_pnl=0.0,
            account_maintenance_ratio=0.0,
            previous_account_equity=1000.0,
            previous_account_market_value=80.0,
            previous_account_notional_value=80.0,
            previous_account_used_margin=10.0,
            previous_account_unrealized_pnl=0.0,
            previous_account_maintenance_ratio=0.0,
        ),
    )
    strategy._framework_use_previous_account_snapshot = True
    strategy._framework_previous_account_details = {
        "frozen_cash": 33.0,
        "short_market_value": 44.0,
        "margin_accrued_interest": 1.5,
        "margin_daily_interest": 0.5,
    }

    account = strategy.get_account()

    assert float(account["frozen_cash"]) == 33.0
    assert float(account["short_market_value"]) == 44.0
    assert float(account["accrued_interest"]) == 1.5
    assert float(account["daily_interest"]) == 0.5


def test_collect_pre_open_timers_and_prime_globally_once() -> None:
    """Pre-open timers should be deduplicated before the event loop starts."""
    start_ts = pd.Timestamp("2023-01-03 09:30:00", tz="Asia/Shanghai").value
    strategy_a = FrameworkPreOpenStrategy()
    strategy_b = FrameworkPreOpenStrategy()
    strategy_a._trading_day_bounds = {"2023-01-03": (start_ts, start_ts + 1)}
    strategy_b._trading_day_bounds = {"2023-01-03": (start_ts, start_ts + 1)}

    # #324: the pre-open timer fires 1ns before the day's first bar so an order
    # placed in on_pre_open fills on this day's open; the payload keeps start_ts
    # as the expected_open_at / history-cutoff source timestamp.
    entries = collect_pre_open_timer_entries(strategy_a)
    assert entries == [(start_ts - 1, f"__framework_pre_open__|2023-01-03|{start_ts}")]

    fake_engine = SimpleNamespace(add_timer=MagicMock())
    _prime_framework_pre_open_timers([strategy_a, strategy_b], fake_engine)

    scheduled = [call.args for call in fake_engine.add_timer.call_args_list]
    assert scheduled == [
        (start_ts - 1, f"__framework_pre_open__|2023-01-03|{start_ts}")
    ]
    assert strategy_a._framework_pre_open_timers_registered is True
    assert strategy_b._framework_pre_open_timers_registered is True


def test_pre_open_timer_dispatch_uses_next_open_fill_policy_by_default() -> None:
    """Pre-open callback should default market orders to next-open semantics."""
    strategy = FrameworkPreOpenStrategy()
    start_ts = pd.Timestamp("2023-01-03 09:30:00", tz="Asia/Shanghai").value

    ctx = MagicMock(spec=StrategyContext)
    ctx.buy.return_value = "pre-open-order"
    ctx.cash = 1000.0
    ctx.positions = {}
    ctx.available_positions = {}
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.canceled_order_ids = []
    ctx.session = "preopen"
    ctx.current_time = start_ts
    strategy.ctx = ctx

    consumed = dispatch_pre_open_timer(
        strategy,
        f"__framework_pre_open__|2023-01-03|{start_ts}",
    )

    assert consumed is True
    assert len(strategy.pre_open_events) == 1
    assert strategy.pre_open_events[0]["trading_date"] == dt.date(2023, 1, 3)
    assert strategy.pre_open_events[0]["timestamp"] == start_ts
    assert strategy.pre_open_events[0]["expected_open_at"] == start_ts
    assert strategy._framework_in_pre_open_phase is False
    args = ctx.buy.call_args.args
    assert args[0:2] == ("AAPL", 1.0)
    # Task 4: ctx.buy 的 3 个 fill 参数 (price_basis, bar_offset, temporal)
    # 收敛为 2 个 (fill_mode: ExecutionMode, fill_timer_timing: str)。
    # 默认 pre-open 注入 {open,1,same_cycle} → NextOpen + same_cycle。
    assert args[9] == ExecutionMode.NextOpen
    assert args[10] == "same_cycle"

    dispatch_pre_open_timer(strategy, f"__framework_pre_open__|2023-01-03|{start_ts}")
    assert ctx.buy.call_count == 1


@pytest.mark.parametrize(
    ("payload", "expected_event", "expected_schedule_calls"),
    [
        ("__daily__|14:55:00|daily_ok", "on_timer:daily_ok", 1),
        ("__daily__|bad_payload", "on_timer:__daily__|bad_payload", 0),
        ("__daily__|not_a_time|daily_bad_time", "on_timer:daily_bad_time", 0),
    ],
)
def test_daily_timer_paths_parameterized(
    payload: str, expected_event: str, expected_schedule_calls: int
) -> None:
    """Daily timer paths should produce stable callback and reschedule behavior."""
    strategy = TimerOrderTradeMixedStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    strategy.ctx = ctx
    strategy._trading_days = []

    strategy._on_timer_event(payload, ctx)

    assert strategy.order_count == 0
    assert strategy.trade_count == 0
    assert strategy.timer_count == 1
    assert strategy.events == [expected_event]
    assert strategy.ctx.schedule.call_count == expected_schedule_calls


def test_strategy_submit_order_default_behavior() -> None:
    """Unified submit_order should expose stable default behavior."""
    strategy = MyStrategy()
    capabilities = strategy.get_execution_capabilities()

    assert strategy.can_submit_client_order("coid-default")
    assert capabilities["broker_live"] is False
    assert capabilities["client_order_id"] is False

    with pytest.raises(
        RuntimeError, match="client_order_id is not supported in current execution mode"
    ):
        strategy.submit_order(
            symbol="000001.SZ",
            side="Buy",
            quantity=10.0,
            client_order_id="coid-unified",
        )


def test_strategy_buy_sell_delegate_to_submit_order() -> None:
    """buy/sell should route through unified submit_order method."""

    class _SubmitSpyStrategy(Strategy):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, float, str, bool]] = []

        def submit_order(
            self,
            symbol: str | None = None,
            side: str = "Buy",
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: TimeInForce | str | None = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            client_order_id: str | None = None,
            order_type: str | None = None,
            extra: dict[str, Any] | None = None,
            broker_options: dict[str, Any] | None = None,
            trail_offset: float | None = None,
            trail_reference_price: float | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
            position_effect: str | None = None,
            reduce_only: bool = False,
            asset_type: str = "stock",
        ) -> Any:
            _ = price
            _ = time_in_force
            _ = trigger_price
            _ = tag
            _ = client_order_id
            _ = order_type
            _ = extra
            _ = broker_options
            _ = trail_offset
            _ = trail_reference_price
            _ = fill_mode
            _ = slippage
            _ = commission
            assert symbol is not None
            assert quantity is not None
            self.calls.append(
                (side, symbol, quantity, position_effect or "auto", reduce_only)
            )
            return f"oid-{side}-{symbol}"

    strategy = _SubmitSpyStrategy()
    buy_order_id = strategy.buy(symbol="AAPL", quantity=2.0)
    sell_order_id = strategy.sell(symbol="AAPL", quantity=1.0)

    assert buy_order_id == "oid-Buy-AAPL"
    assert sell_order_id == "oid-Sell-AAPL"
    assert strategy.calls == [
        ("Buy", "AAPL", 2.0, "auto", False),
        ("Sell", "AAPL", 1.0, "auto", False),
    ]


def test_strategy_short_cover_delegate_position_effect() -> None:
    """short/cover should forward explicit position_effect semantics."""

    class _SubmitSpyStrategy(Strategy):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, float, str, bool]] = []

        def submit_order(
            self,
            symbol: str | None = None,
            side: str = "Buy",
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: TimeInForce | str | None = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            client_order_id: str | None = None,
            order_type: str | None = None,
            extra: dict[str, Any] | None = None,
            broker_options: dict[str, Any] | None = None,
            trail_offset: float | None = None,
            trail_reference_price: float | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
            position_effect: str | None = None,
            reduce_only: bool = False,
            asset_type: str = "stock",
        ) -> Any:
            _ = (
                price,
                time_in_force,
                trigger_price,
                tag,
                client_order_id,
                order_type,
                extra,
                broker_options,
                trail_offset,
                trail_reference_price,
                fill_mode,
                slippage,
                commission,
            )
            assert symbol is not None
            assert quantity is not None
            self.calls.append(
                (side, symbol, quantity, position_effect or "auto", reduce_only)
            )
            return "ok"

    strategy = _SubmitSpyStrategy()
    strategy.short(symbol="AAPL", quantity=2.0)
    strategy.cover(symbol="AAPL", quantity=1.0, reduce_only=True)

    assert strategy.calls == [
        ("Sell", "AAPL", 2.0, "open", False),
        ("Buy", "AAPL", 1.0, "close", True),
    ]


def test_strategy_buy_auto_splits_cover_then_open() -> None:
    """buy(auto) should split short-cover and long-open legs using current position."""
    strategy = MyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.get_position.return_value = -2.0
    # auto 拆腿改读可平持仓（已扣除同 bar 在途平仓单，见 #361）；此处无在途单，
    # 可平量与结算仓一致。
    ctx.get_closable_position.return_value = -2.0
    ctx.cash = 100000.0
    ctx.buy.side_effect = ["oid-close", "oid-open"]
    strategy.ctx = ctx
    strategy._last_prices["AAPL"] = 100.0

    order_id = strategy.buy(symbol="AAPL", quantity=5.0)

    # 回测出口现返回 OrderReceipt：反手拆腿产生的 close/open 两条 id 均需保留。
    assert isinstance(order_id, OrderReceipt)
    assert order_id.order_ids == ("oid-close", "oid-open")
    assert order_id.group_id == "oid-close"
    assert ctx.buy.call_count == 2
    first_call = ctx.buy.call_args_list[0]
    second_call = ctx.buy.call_args_list[1]
    assert first_call.args[:2] == ("AAPL", 2.0)
    assert first_call.kwargs["position_effect"] == PositionEffect.Close
    assert second_call.args[:2] == ("AAPL", 3.0)
    assert second_call.kwargs["position_effect"] == PositionEffect.Open


def test_strategy_submit_order_accepts_close_today_position_effect() -> None:
    """submit_order should map close_today to the extended PositionEffect enum."""
    strategy = MyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.buy.return_value = "oid-close-today"
    strategy.ctx = ctx

    order_id = strategy.submit_order(
        symbol="IF2406",
        side="Buy",
        quantity=1.0,
        position_effect="close_today",
    )

    # 回测出口现返回 OrderReceipt（封装全部拆腿 id），非纯字符串。
    assert isinstance(order_id, OrderReceipt)
    assert order_id.primary == "oid-close-today"
    first_call = ctx.buy.call_args_list[0]
    assert first_call.kwargs["position_effect"] == PositionEffect.CloseToday


def test_strategy_rebalance_positions_supports_signed_targets() -> None:
    """rebalance_positions should route long/short target deltas via order_target."""

    class _TargetSpyStrategy(MyStrategy):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, str, float, str, bool]] = []

        def submit_order(
            self,
            symbol: str | None = None,
            side: str = "Buy",
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: str | TimeInForce | None = "GTC",
            trigger_price: float | None = None,
            tag: str | None = None,
            client_order_id: str | None = None,
            order_type: str | None = None,
            extra: dict[str, Any] | None = None,
            broker_options: dict[str, Any] | None = None,
            trail_offset: float | None = None,
            trail_reference_price: float | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
            position_effect: str | None = None,
            reduce_only: bool = False,
            asset_type: str = "stock",
        ) -> Any:
            _ = (
                price,
                time_in_force,
                trigger_price,
                tag,
                client_order_id,
                order_type,
                extra,
                broker_options,
                trail_offset,
                trail_reference_price,
                fill_mode,
                slippage,
                commission,
            )
            assert symbol is not None
            assert quantity is not None
            self.calls.append(
                (side, symbol, quantity, position_effect or "auto", reduce_only)
            )
            return f"oid-{symbol}"

    strategy = _TargetSpyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.positions = {"AAA": 100.0, "BBB": 0.0}
    ctx.get_position.side_effect = lambda symbol: {"AAA": 100.0, "BBB": 0.0}.get(
        symbol, 0.0
    )
    # 目标仓位改读投影持仓（含在途单，见 #361）；此处无在途单，与结算仓一致。
    ctx.get_projected_position.side_effect = lambda symbol: {
        "AAA": 100.0,
        "BBB": 0.0,
    }.get(symbol, 0.0)
    ctx.risk_config = SimpleNamespace(account_mode="margin", enable_short_sell=True)
    strategy.ctx = ctx

    strategy.rebalance_positions({"AAA": -50.0, "BBB": 25.0})

    assert strategy.calls == [
        ("Sell", "AAA", 150.0, "auto", False),
        ("Buy", "BBB", 25.0, "auto", False),
    ]


def test_strategy_rebalance_positions_rejects_negative_targets_in_cash_mode() -> None:
    """Negative target positions should fail fast when short selling is unavailable."""
    strategy = MyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.positions = {"AAA": 0.0}
    ctx.get_position.return_value = 0.0
    ctx.risk_config = SimpleNamespace(account_mode="cash", enable_short_sell=False)
    strategy.ctx = ctx

    with pytest.raises(
        ValueError, match="negative target positions require allow_short=True"
    ):
        strategy.rebalance_positions({"AAA": -10.0})


def test_strategy_rebalance_positions_rejects_when_short_is_disallowed() -> None:
    """Negative targets should respect strict broker short-sell capability checks."""
    strategy = MyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.positions = {"AAA": 0.0}
    ctx.get_position.return_value = 0.0
    ctx.risk_config = SimpleNamespace(account_mode="margin", enable_short_sell=False)
    strategy.ctx = ctx

    class _CapExecution:
        def capabilities(self) -> dict:
            return {
                "broker_live": True,
                "broker_name": "miniqmt",
                "account_mode": "margin",
                "supports_short_sell": False,
            }

    strategy.execution = _CapExecution()

    with pytest.raises(RuntimeError, match="does not advertise short-sell support"):
        strategy.rebalance_positions({"AAA": -10.0}, allow_short=True)


def test_strategy_rebalance_positions_can_bypass_strict_short_capability() -> None:
    """Opting out should allow negative targets with unknown broker capability."""

    class _TargetSpyStrategy(MyStrategy):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, str, float]] = []

        def submit_order(
            self,
            symbol: str | None = None,
            side: str = "Buy",
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: TimeInForce | str | None = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            client_order_id: str | None = None,
            order_type: str | None = None,
            extra: dict[str, Any] | None = None,
            broker_options: dict[str, Any] | None = None,
            trail_offset: float | None = None,
            trail_reference_price: float | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
            position_effect: str | None = None,
            reduce_only: bool = False,
            asset_type: str = "stock",
        ) -> Any:
            _ = (
                price,
                time_in_force,
                trigger_price,
                tag,
                client_order_id,
                order_type,
                extra,
                broker_options,
                trail_offset,
                trail_reference_price,
                fill_mode,
                slippage,
                commission,
                position_effect,
                reduce_only,
            )
            assert symbol is not None
            assert quantity is not None
            self.calls.append((side, symbol, quantity))
            return "oid"

    strategy = _TargetSpyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.positions = {"AAA": 0.0}
    ctx.get_position.return_value = 0.0
    ctx.risk_config = SimpleNamespace(account_mode="margin", enable_short_sell=False)
    strategy.ctx = ctx

    class _CapExecution:
        def capabilities(self) -> dict:
            return {
                "broker_live": True,
                "broker_name": "unknown",
                "account_mode": "margin",
                "supports_short_sell": False,
            }

        def get_position(self, symbol: str | None = None) -> float:
            return 0.0

    strategy.execution = _CapExecution()

    strategy.rebalance_positions(
        {"AAA": -10.0},
        allow_short=True,
        strict_short_capability=False,
    )

    assert strategy.calls == [("Sell", "AAA", 10.0)]


def test_strategy_rebalance_positions_missing_price_mode_fail() -> None:
    """missing_price_mode=fail should reject symbols absent from price_map."""
    strategy = MyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.positions = {"AAA": 10.0, "BBB": 0.0}
    ctx.get_position.side_effect = lambda symbol: {"AAA": 10.0, "BBB": 0.0}.get(
        symbol, 0.0
    )
    # 目标仓位改读投影持仓（含在途单，见 #361）；此处无在途单，与结算仓一致。
    ctx.get_projected_position.side_effect = lambda symbol: {
        "AAA": 10.0,
        "BBB": 0.0,
    }.get(symbol, 0.0)
    ctx.risk_config = SimpleNamespace(account_mode="cash", enable_short_sell=False)
    strategy.ctx = ctx

    with pytest.raises(RuntimeError, match="missing price_map entry for symbol 'BBB'"):
        strategy.rebalance_positions(
            {"AAA": 0.0, "BBB": 5.0},
            price_map={"AAA": 10.0},
            missing_price_mode="fail",
        )


def test_strategy_rebalance_positions_missing_price_mode_skip() -> None:
    """missing_price_mode=skip should drop legs whose price_map entry is missing."""

    class _TargetSpyStrategy(MyStrategy):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, str, float, float | None]] = []

        def submit_order(
            self,
            symbol: str | None = None,
            side: str = "Buy",
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: TimeInForce | str | None = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            client_order_id: str | None = None,
            order_type: str | None = None,
            extra: dict[str, Any] | None = None,
            broker_options: dict[str, Any] | None = None,
            trail_offset: float | None = None,
            trail_reference_price: float | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
            position_effect: str | None = None,
            reduce_only: bool = False,
            asset_type: str = "stock",
        ) -> Any:
            _ = (
                time_in_force,
                trigger_price,
                tag,
                client_order_id,
                order_type,
                extra,
                broker_options,
                trail_offset,
                trail_reference_price,
                fill_mode,
                slippage,
                commission,
                position_effect,
                reduce_only,
            )
            assert symbol is not None
            assert quantity is not None
            self.calls.append((side, symbol, quantity, price))
            return "oid"

    strategy = _TargetSpyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.positions = {"AAA": 10.0, "BBB": 0.0}
    ctx.get_position.side_effect = lambda symbol: {"AAA": 10.0, "BBB": 0.0}.get(
        symbol, 0.0
    )
    # 目标仓位改读投影持仓（含在途单，见 #361）；此处无在途单，与结算仓一致。
    ctx.get_projected_position.side_effect = lambda symbol: {
        "AAA": 10.0,
        "BBB": 0.0,
    }.get(symbol, 0.0)
    ctx.risk_config = SimpleNamespace(account_mode="cash", enable_short_sell=False)
    strategy.ctx = ctx

    strategy.rebalance_positions(
        {"AAA": 0.0, "BBB": 5.0},
        price_map={"AAA": 10.0},
        missing_price_mode="skip",
    )

    assert strategy.calls == [("Sell", "AAA", 10.0, 10.0)]


def test_strategy_rebalance_positions_missing_price_mode_ignore() -> None:
    """Default ignore mode should still submit legs without explicit prices."""

    class _TargetSpyStrategy(MyStrategy):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, str, float, float | None]] = []

        def submit_order(
            self,
            symbol: str | None = None,
            side: str = "Buy",
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: TimeInForce | str | None = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            client_order_id: str | None = None,
            order_type: str | None = None,
            extra: dict[str, Any] | None = None,
            broker_options: dict[str, Any] | None = None,
            trail_offset: float | None = None,
            trail_reference_price: float | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
            position_effect: str | None = None,
            reduce_only: bool = False,
            asset_type: str = "stock",
        ) -> Any:
            _ = (
                time_in_force,
                trigger_price,
                tag,
                client_order_id,
                order_type,
                extra,
                broker_options,
                trail_offset,
                trail_reference_price,
                fill_mode,
                slippage,
                commission,
                position_effect,
                reduce_only,
            )
            assert symbol is not None
            assert quantity is not None
            self.calls.append((side, symbol, quantity, price))
            return "oid"

    strategy = _TargetSpyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.positions = {"AAA": 10.0, "BBB": 0.0}
    ctx.get_position.side_effect = lambda symbol: {"AAA": 10.0, "BBB": 0.0}.get(
        symbol, 0.0
    )
    # 目标仓位改读投影持仓（含在途单，见 #361）；此处无在途单，与结算仓一致。
    ctx.get_projected_position.side_effect = lambda symbol: {
        "AAA": 10.0,
        "BBB": 0.0,
    }.get(symbol, 0.0)
    ctx.risk_config = SimpleNamespace(account_mode="cash", enable_short_sell=False)
    strategy.ctx = ctx

    strategy.rebalance_positions(
        {"AAA": 0.0, "BBB": 5.0},
        price_map={"AAA": 10.0},
    )

    assert strategy.calls == [
        ("Sell", "AAA", 10.0, 10.0),
        ("Buy", "BBB", 5.0, None),
    ]


def test_strategy_rebalance_positions_records_explainable_plan() -> None:
    """rebalance_positions should expose the last generated rebalance plan."""

    class _TargetSpyStrategy(MyStrategy):
        def __init__(self) -> None:
            super().__init__()

        def submit_order(
            self,
            symbol: str | None = None,
            side: str = "Buy",
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: TimeInForce | str | None = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            client_order_id: str | None = None,
            order_type: str | None = None,
            extra: dict[str, Any] | None = None,
            broker_options: dict[str, Any] | None = None,
            trail_offset: float | None = None,
            trail_reference_price: float | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
            position_effect: str | None = None,
            reduce_only: bool = False,
            asset_type: str = "stock",
        ) -> Any:
            _ = (
                symbol,
                side,
                quantity,
                price,
                time_in_force,
                trigger_price,
                tag,
                client_order_id,
                order_type,
                extra,
                broker_options,
                trail_offset,
                trail_reference_price,
                fill_mode,
                slippage,
                commission,
                position_effect,
                reduce_only,
            )
            return "oid"

    strategy = _TargetSpyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.positions = {"AAA": 10.0, "BBB": 0.0}
    ctx.get_position.side_effect = lambda symbol: {"AAA": 10.0, "BBB": 0.0}.get(
        symbol, 0.0
    )
    # 目标仓位改读投影持仓（含在途单，见 #361）；此处无在途单，与结算仓一致。
    ctx.get_projected_position.side_effect = lambda symbol: {
        "AAA": 10.0,
        "BBB": 0.0,
    }.get(symbol, 0.0)
    ctx.risk_config = SimpleNamespace(account_mode="cash", enable_short_sell=False)
    strategy.ctx = ctx

    strategy.rebalance_positions(
        {"AAA": 0.0, "BBB": 5.0},
        price_map={"AAA": 10.0, "BBB": 20.0},
    )

    plan = strategy.get_last_target_positions_plan()
    assert plan["status"] == "submitted"
    assert plan["missing_price_mode"] == "ignore"
    assert [leg["symbol"] for leg in plan["reduce_legs"]] == ["AAA"]
    assert [leg["symbol"] for leg in plan["increase_legs"]] == ["BBB"]
    assert [leg["phase"] for leg in plan["submitted_legs"]] == ["reduce", "increase"]


def test_strategy_rebalance_positions_plan_tracks_skipped_legs() -> None:
    """Plan should record skipped legs when missing_price_mode=skip."""

    class _TargetSpyStrategy(MyStrategy):
        def __init__(self) -> None:
            super().__init__()

        def submit_order(
            self,
            symbol: str | None = None,
            side: str = "Buy",
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: TimeInForce | str | None = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            client_order_id: str | None = None,
            order_type: str | None = None,
            extra: dict[str, Any] | None = None,
            broker_options: dict[str, Any] | None = None,
            trail_offset: float | None = None,
            trail_reference_price: float | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
            position_effect: str | None = None,
            reduce_only: bool = False,
            asset_type: str = "stock",
        ) -> Any:
            _ = (
                symbol,
                side,
                quantity,
                price,
                time_in_force,
                trigger_price,
                tag,
                client_order_id,
                order_type,
                extra,
                broker_options,
                trail_offset,
                trail_reference_price,
                fill_mode,
                slippage,
                commission,
                position_effect,
                reduce_only,
            )
            return "oid"

    strategy = _TargetSpyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.positions = {"AAA": 10.0, "BBB": 0.0}
    ctx.get_position.side_effect = lambda symbol: {"AAA": 10.0, "BBB": 0.0}.get(
        symbol, 0.0
    )
    # 目标仓位改读投影持仓（含在途单，见 #361）；此处无在途单，与结算仓一致。
    ctx.get_projected_position.side_effect = lambda symbol: {
        "AAA": 10.0,
        "BBB": 0.0,
    }.get(symbol, 0.0)
    ctx.risk_config = SimpleNamespace(account_mode="cash", enable_short_sell=False)
    strategy.ctx = ctx

    strategy.rebalance_positions(
        {"AAA": 0.0, "BBB": 5.0},
        price_map={"AAA": 10.0},
        missing_price_mode="skip",
    )

    plan = strategy.get_last_target_positions_plan()
    assert plan["status"] == "submitted"
    assert plan["submitted_legs"] == [
        {
            "symbol": "AAA",
            "target_quantity": 0.0,
            "price": 10.0,
            "phase": "reduce",
            "order_id": "oid",
        }
    ]
    assert plan["skipped_legs"] == [
        {
            "symbol": "BBB",
            "target_quantity": 5.0,
            "reason": "missing_price_map",
            "phase": "increase",
        }
    ]


def test_strategy_rebalance_positions_plan_tracks_reject_reason() -> None:
    """Plan should retain reject_reason when validation fails."""
    strategy = MyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    ctx.positions = {"AAA": 0.0}
    ctx.get_position.return_value = 0.0
    ctx.risk_config = SimpleNamespace(account_mode="cash", enable_short_sell=False)
    strategy.ctx = ctx

    with pytest.raises(
        ValueError, match="negative target positions require allow_short=True"
    ):
        strategy.rebalance_positions({"AAA": -1.0})

    plan = strategy.get_last_target_positions_plan()
    assert plan["status"] == "rejected"
    assert "negative target positions require allow_short=True" in plan["reject_reason"]


def test_strategy_submit_order_trailing_validation() -> None:
    """submit_order should validate trailing order required fields."""
    strategy = MyStrategy()

    with pytest.raises(RuntimeError, match="trail_offset must be > 0"):
        strategy.submit_order(
            symbol="AAPL",
            side="Sell",
            quantity=1.0,
            order_type="StopTrail",
        )

    with pytest.raises(RuntimeError, match="price must be provided"):
        strategy.submit_order(
            symbol="AAPL",
            side="Sell",
            quantity=1.0,
            order_type="StopTrailLimit",
            trail_offset=1.0,
        )


def test_strategy_submit_order_records_broker_options_in_backtest_mode() -> None:
    """submit_order should accept broker_options and attach them to known orders."""
    strategy = MyStrategy()
    ctx = MagicMock(spec=StrategyContext)
    order = SimpleNamespace(id="oid-broker-options")
    ctx.buy.return_value = order.id
    ctx.active_orders = [order]
    strategy.ctx = ctx

    order_id = strategy.submit_order(
        symbol="AAPL",
        side="Buy",
        quantity=1.0,
        broker_options={"xt_price_type": "LATEST_PRICE", "order_remark": "demo"},
    )

    # 回测出口现返回 OrderReceipt（封装全部拆腿 id），非纯字符串。
    assert isinstance(order_id, OrderReceipt)
    assert order_id.primary == "oid-broker-options"
    queried = strategy.get_order(order_id.primary)
    assert queried is order
    assert getattr(queried, "broker_options") == {
        "xt_price_type": "LATEST_PRICE",
        "order_remark": "demo",
    }


class _OrderLevelFillPolicyStrategy(Strategy):
    """Submit two market orders with different order-level fill_policy."""

    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar: Bar) -> None:
        if self.sent:
            return
        self.sent = True
        self.submit_order(
            symbol=bar.symbol,
            side="Buy",
            quantity=1.0,
            tag="order-open",
            fill_mode=NextOpen(),
        )
        self.submit_order(
            symbol=bar.symbol,
            side="Buy",
            quantity=1.0,
            tag="order-close",
            fill_mode=NextClose(),
        )


def test_order_level_fill_policy_overrides_engine_fill_policy() -> None:
    """Order-level fill_policy should override engine-level fill_policy for fills."""
    register_logger(console=False, level="INFO")
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC"),
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [10000.0, 10000.0, 10000.0],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        }
    )
    result = run_backtest(
        data=data,
        strategy=_OrderLevelFillPolicyStrategy,
        symbols="AAPL",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=NextOpen(),
    )
    filled_orders = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    open_row = filled_orders[filled_orders["tag"] == "order-open"].iloc[0]
    close_row = filled_orders[filled_orders["tag"] == "order-close"].iloc[0]
    assert float(open_row["avg_price"]) == pytest.approx(20.0)
    assert float(close_row["avg_price"]) == pytest.approx(200.0)
    register_logger(console=False, level="INFO")


class _StrategyLevelFillPolicyStrategy(Strategy):
    """Submit one market order without order-level fill_policy override."""

    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar: Bar) -> None:
        if self.sent:
            return
        self.sent = True
        self.buy(symbol=bar.symbol, quantity=1.0, tag="strategy-level-policy")


def test_strategy_level_fill_policy_applies_when_order_policy_missing() -> None:
    """Strategy-level fill policy map should apply when order-level policy is absent."""
    register_logger(console=False, level="INFO")
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC"),
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [10000.0, 10000.0, 10000.0],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        }
    )
    result = run_backtest(
        data=data,
        strategy=_StrategyLevelFillPolicyStrategy,
        symbols="AAPL",
        initial_cash=100000.0,
        show_progress=False,
        fill_policy=NextOpen(),
        strategy_fill_policy={"_default": NextClose()},
    )
    filled_orders = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    row = filled_orders[filled_orders["tag"] == "strategy-level-policy"].iloc[0]
    assert float(row["avg_price"]) == pytest.approx(200.0)
    register_logger(console=False, level="INFO")


class _OrderLevelSlippageStrategy(Strategy):
    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar: Bar) -> None:
        if self.sent:
            return
        self.sent = True
        self.submit_order(
            symbol=bar.symbol,
            side="Buy",
            quantity=1.0,
            tag="slippage-fixed",
            fill_mode=NextOpen(),
            slippage={"type": "fixed", "value": 0.5},
        )
        self.submit_order(
            symbol=bar.symbol,
            side="Buy",
            quantity=1.0,
            tag="slippage-percent",
            fill_mode=NextOpen(),
            slippage={"type": "percent", "value": 0.1},
        )
        self.submit_order(
            symbol=bar.symbol,
            side="Buy",
            quantity=1.0,
            tag="slippage-ticks",
            fill_mode=NextOpen(),
            slippage={"type": "ticks", "value": 2},
        )


def test_order_level_slippage_overrides_engine_slippage() -> None:
    """Order-level slippage should override engine-level slippage model."""
    register_logger(console=False, level="INFO")
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC"),
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [10000.0, 10000.0, 10000.0],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        }
    )
    result = run_backtest(
        data=data,
        strategy=_OrderLevelSlippageStrategy,
        symbols="AAPL",
        initial_cash=100000.0,
        show_progress=False,
        config=BacktestConfig(
            strategy_config=StrategyConfig(),
            instruments_config=[InstrumentConfig(symbol="AAPL", tick_size=0.25)],
        ),
    )
    filled_orders = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    fixed_row = filled_orders[filled_orders["tag"] == "slippage-fixed"].iloc[0]
    percent_row = filled_orders[filled_orders["tag"] == "slippage-percent"].iloc[0]
    ticks_row = filled_orders[filled_orders["tag"] == "slippage-ticks"].iloc[0]
    assert float(fixed_row["avg_price"]) == pytest.approx(20.5)
    assert float(percent_row["avg_price"]) == pytest.approx(22.0)
    assert float(ticks_row["avg_price"]) == pytest.approx(20.5)


class _StrategyLevelSlippageStrategy(Strategy):
    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar: Bar) -> None:
        if self.sent:
            return
        self.sent = True
        self.buy(symbol=bar.symbol, quantity=1.0, tag="strategy-slippage")


def test_strategy_level_slippage_applies_when_order_slippage_missing() -> None:
    """Strategy-level slippage overrides engine-level when order slippage is missing."""
    register_logger(console=False, level="INFO")
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC"),
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [10000.0, 10000.0, 10000.0],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        }
    )
    result = run_backtest(
        data=data,
        strategy=_StrategyLevelSlippageStrategy,
        symbols="AAPL",
        initial_cash=100000.0,
        show_progress=False,
        slippage={"type": "percent", "value": 0.2},
        strategy_slippage={"_default": {"type": "fixed", "value": 0.5}},
    )
    filled_orders = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    row = filled_orders[filled_orders["tag"] == "strategy-slippage"].iloc[0]
    assert float(row["avg_price"]) == pytest.approx(20.5)


def test_strategy_level_ticks_slippage_uses_symbol_tick_size() -> None:
    """Strategy-level tick slippage should resolve via instrument tick_size."""
    register_logger(console=False, level="INFO")
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC"),
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [10000.0, 10000.0, 10000.0],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        }
    )
    result = run_backtest(
        data=data,
        strategy=_StrategyLevelSlippageStrategy,
        symbols="AAPL",
        initial_cash=100000.0,
        show_progress=False,
        config=BacktestConfig(
            strategy_config=StrategyConfig(),
            instruments_config=[InstrumentConfig(symbol="AAPL", tick_size=0.25)],
        ),
        strategy_slippage={"_default": {"type": "ticks", "value": 2}},
    )
    filled_orders = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    row = filled_orders[filled_orders["tag"] == "strategy-slippage"].iloc[0]
    assert float(row["avg_price"]) == pytest.approx(20.5)


def test_global_ticks_slippage_resolves_to_fixed_from_shared_tick_size() -> None:
    """Global tick slippage should resolve from a shared instrument tick_size."""
    register_logger(console=False, level="INFO")
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC"),
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [10000.0, 10000.0, 10000.0],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        }
    )
    result = run_backtest(
        data=data,
        strategy=_StrategyLevelSlippageStrategy,
        symbols="AAPL",
        initial_cash=100000.0,
        show_progress=False,
        config=BacktestConfig(
            strategy_config=StrategyConfig(),
            instruments_config=[InstrumentConfig(symbol="AAPL", tick_size=0.25)],
        ),
        slippage={"type": "ticks", "value": 2},
    )
    filled_orders = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    row = filled_orders[filled_orders["tag"] == "strategy-slippage"].iloc[0]
    assert float(row["avg_price"]) == pytest.approx(20.5)


class _OrderLevelCommissionStrategy(Strategy):
    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar: Bar) -> None:
        if self.sent:
            return
        self.sent = True
        self.submit_order(
            symbol=bar.symbol,
            side="Buy",
            quantity=1.0,
            tag="commission-fixed",
            fill_mode=NextOpen(),
            commission={"type": "fixed", "value": 3.0},
        )
        self.submit_order(
            symbol=bar.symbol,
            side="Buy",
            quantity=1.0,
            tag="commission-percent",
            fill_mode=NextOpen(),
            commission={"type": "percent", "value": 0.01},
        )
        self.submit_order(
            symbol=bar.symbol,
            side="Buy",
            quantity=3.0,
            tag="commission-per-unit",
            fill_mode=NextOpen(),
            commission={"type": "per_unit", "value": 0.5},
        )


def test_order_level_commission_overrides_market_model() -> None:
    """Order-level commission should override market-model commission calculation."""
    register_logger(console=False, level="INFO")
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC"),
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [10000.0, 10000.0, 10000.0],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        }
    )
    result = run_backtest(
        data=data,
        strategy=_OrderLevelCommissionStrategy,
        symbols="AAPL",
        initial_cash=100000.0,
        show_progress=False,
    )
    filled_orders = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    fixed_row = filled_orders[filled_orders["tag"] == "commission-fixed"].iloc[0]
    percent_row = filled_orders[filled_orders["tag"] == "commission-percent"].iloc[0]
    per_unit_row = filled_orders[filled_orders["tag"] == "commission-per-unit"].iloc[0]
    assert float(fixed_row["commission"]) == pytest.approx(3.0)
    assert float(percent_row["commission"]) == pytest.approx(0.2)
    assert float(per_unit_row["commission"]) == pytest.approx(1.5)


class _StrategyLevelCommissionStrategy(Strategy):
    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar: Bar) -> None:
        if self.sent:
            return
        self.sent = True
        self.buy(symbol=bar.symbol, quantity=1.0, tag="strategy-commission")


def test_strategy_level_commission_applies_when_order_commission_missing() -> None:
    """Strategy-level commission applies when order-level commission is omitted."""
    register_logger(console=False, level="INFO")
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC"),
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [10000.0, 10000.0, 10000.0],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        }
    )
    result = run_backtest(
        data=data,
        strategy=_StrategyLevelCommissionStrategy,
        symbols="AAPL",
        initial_cash=100000.0,
        show_progress=False,
        strategy_commission={"_default": {"type": "fixed", "value": 0.5}},
    )
    filled_orders = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    row = filled_orders[filled_orders["tag"] == "strategy-commission"].iloc[0]
    assert float(row["commission"]) == pytest.approx(0.5)


def test_strategy_level_per_unit_commission_applies_when_order_commission_missing() -> (
    None
):
    """Strategy-level per-unit commission should apply.

    Existing fixed semantics remain unchanged.
    """
    register_logger(console=False, level="INFO")
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC"),
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [10000.0, 10000.0, 10000.0],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        }
    )
    result = run_backtest(
        data=data,
        strategy=_StrategyLevelCommissionStrategy,
        symbols="AAPL",
        initial_cash=100000.0,
        show_progress=False,
        strategy_commission={"_default": {"type": "per_unit", "value": 0.5}},
    )
    filled_orders = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    row = filled_orders[filled_orders["tag"] == "strategy-commission"].iloc[0]
    assert float(row["commission"]) == pytest.approx(0.5)


class _CheckpointResiduePickleStrategy(Strategy):
    """Buy exactly one lot on the first bar of each phase.

    Phase 1's order must fully fill before ``save_checkpoint`` (no pending
    order pickled). Phase 2's order (tag="phase2") is the one under test:
    it exercises whichever fill_policy/slippage/commission resolution is in
    effect *this* call, with no per-order override of its own.
    """

    def __init__(self) -> None:
        self.phase1_sent = False
        self.phase2_sent = False

    def on_bar(self, bar: Bar) -> None:
        if not self.is_restored:
            if not self.phase1_sent:
                self.phase1_sent = True
                self.buy(symbol=bar.symbol, quantity=1.0, tag="phase1")
        elif not self.phase2_sent:
            self.phase2_sent = True
            self.buy(symbol=bar.symbol, quantity=1.0, tag="phase2")


def _checkpoint_residue_phase_data(symbol: str, phase: int) -> pd.DataFrame:
    """Two very-distinguishable phases: open/close values never collide."""
    if phase == 1:
        start = "2023-01-01"
        opens = [10.0, 20.0, 30.0]
        closes = [100.0, 200.0, 300.0]
    else:
        start = "2023-02-01"
        opens = [50.0, 60.0, 70.0]
        closes = [500.0, 600.0, 700.0]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=3, freq="D", tz="UTC"),
            "open": opens,
            "high": [o + 1.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": closes,
            "volume": [10000.0, 10000.0, 10000.0],
            "symbol": [symbol, symbol, symbol],
        }
    )


def _checkpoint_residue_phase2_row(result: Any) -> pd.Series:
    filled = result.orders_df[
        result.orders_df["status"].astype(str).str.lower() == "filled"
    ]
    return cast(pd.Series, filled[filled["tag"] == "phase2"].iloc[0])


def test_checkpoint_resume_explicit_fill_policy_overrides_stale_strategy_map(
    tmp_path: Path,
) -> None:
    """本次显式传的运行级 fill_policy 不能被残留的策略级 map 静默压过.

    根因: save_checkpoint 把整个策略对象 pickle 落盘, `_strategy_fill_policy_map`
    随对象整体恢复; 若 run_from_checkpoint 只在本次显式传了
    strategy_fill_policy 时才 setattr, 本次不传就不会清空旧 map —— 而
    strategy_trading_api.py 的 `_resolve_effective_order_fill_policy` 优先命中
    策略级 map, 命中后直接返回, 根本不会走到本次显式传的运行级 fill_policy。
    """
    register_logger(console=False, level="INFO")
    checkpoint = tmp_path / "fill_policy_residue.pkl"
    phase1 = _checkpoint_residue_phase_data("FPRES", 1)
    phase2 = _checkpoint_residue_phase_data("FPRES", 2)

    result1 = run_backtest(
        data=phase1,
        strategy=_CheckpointResiduePickleStrategy,
        symbols="FPRES",
        initial_cash=1_000_000.0,
        show_progress=False,
        strategy_fill_policy={"_default": NextClose()},
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="FPRES",
        show_progress=False,
        fill_policy=NextOpen(),
    )
    row = _checkpoint_residue_phase2_row(result2)
    # 本次显式传的 NextOpen() 必须生效: 成交价取 phase2 bar1 的 open=60.0。
    # 修复前会被残留的 "_default": NextClose() 压过, 成交价变成 close=600.0。
    assert float(row["avg_price"]) == pytest.approx(60.0)


def test_checkpoint_resume_without_strategy_slippage_clears_stale_map(
    tmp_path: Path,
) -> None:
    """本次不传 strategy_slippage 时必须清空残留的策略级滑点 map.

    根因与上一条 fill_policy 用例同源: `_strategy_slippage_map` 随 checkpoint
    整体 pickle 恢复; 若不清空, `_resolve_effective_order_slippage` 命中旧 map
    后直接返回旧滑点, 本次"不加滑点"的意图被静默覆盖。
    """
    register_logger(console=False, level="INFO")
    checkpoint = tmp_path / "slippage_residue.pkl"
    phase1 = _checkpoint_residue_phase_data("SLRES", 1)
    phase2 = _checkpoint_residue_phase_data("SLRES", 2)

    result1 = run_backtest(
        data=phase1,
        strategy=_CheckpointResiduePickleStrategy,
        symbols="SLRES",
        initial_cash=1_000_000.0,
        show_progress=False,
        strategy_slippage={"_default": {"type": "fixed", "value": 5.0}},
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="SLRES",
        show_progress=False,
    )
    row = _checkpoint_residue_phase2_row(result2)
    # 本次未传任何滑点: 成交价必须是干净的 phase2 bar1 open=60.0, 不带旧的
    # fixed=5.0 滑点。修复前残留 map 仍生效, 成交价变成 65.0。
    assert float(row["avg_price"]) == pytest.approx(60.0)


def test_checkpoint_resume_explicit_zero_commission_overrides_stale_strategy_map(
    tmp_path: Path,
) -> None:
    """本次显式传 commission_rate=0.0 时, 残留的策略级佣金 map 不能静默压过它.

    这是资金层面的静默错误: 用户本次明确表示"不收佣金", 但若残留的
    `_strategy_commission_map` 未被清空, `_resolve_effective_order_commission`
    命中旧 map 后直接返回旧佣金规则, 实收佣金仍按旧费率收取。
    """
    register_logger(console=False, level="INFO")
    checkpoint = tmp_path / "commission_residue.pkl"
    phase1 = _checkpoint_residue_phase_data("CORES", 1)
    phase2 = _checkpoint_residue_phase_data("CORES", 2)

    result1 = run_backtest(
        data=phase1,
        strategy=_CheckpointResiduePickleStrategy,
        symbols="CORES",
        initial_cash=1_000_000.0,
        show_progress=False,
        strategy_commission={"_default": {"type": "percent", "value": 0.02}},
    )
    save_checkpoint(result1.engine, result1.strategy, str(checkpoint))  # type: ignore[arg-type]

    result2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=phase2,
        symbols="CORES",
        show_progress=False,
        commission_rate=0.0,
    )
    row = _checkpoint_residue_phase2_row(result2)
    # 本次显式传 commission_rate=0.0: 实收佣金必须为 0。
    # 修复前残留的 "_default": percent 0.02 仍生效, 实收佣金变成 60*0.02=1.2。
    assert float(row["commission"]) == pytest.approx(0.0)


def test_strategy_trailing_helpers_delegate_to_submit_order() -> None:
    """Trailing helper APIs should call unified submit_order with trailing args."""

    class _TrailingSpyStrategy(Strategy):
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def submit_order(
            self,
            symbol: str | None = None,
            side: str = "Buy",
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: Any = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            client_order_id: str | None = None,
            order_type: str | None = None,
            extra: dict[str, Any] | None = None,
            broker_options: dict[str, Any] | None = None,
            trail_offset: float | None = None,
            trail_reference_price: float | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
            position_effect: str | None = None,
            reduce_only: bool = False,
            asset_type: str = "stock",
        ) -> Any:
            _ = time_in_force
            _ = trigger_price
            _ = client_order_id
            _ = extra
            _ = broker_options
            _ = fill_mode
            _ = slippage
            _ = commission
            _ = position_effect
            _ = reduce_only
            self.calls.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "tag": tag,
                    "order_type": order_type,
                    "trail_offset": trail_offset,
                    "trail_reference_price": trail_reference_price,
                }
            )
            return f"oid-{len(self.calls)}"

    strategy = _TrailingSpyStrategy()
    order_id_1 = strategy.place_trailing_stop(
        symbol="AAPL",
        quantity=2.0,
        trail_offset=1.5,
        trail_reference_price=101.0,
        tag="trail-stop",
    )
    order_id_2 = strategy.place_trailing_stop_limit(
        symbol="AAPL",
        quantity=3.0,
        price=99.5,
        trail_offset=2.0,
        trail_reference_price=103.0,
        tag="trail-limit",
    )

    assert order_id_1 == "oid-1"
    assert order_id_2 == "oid-2"
    assert strategy.calls == [
        {
            "symbol": "AAPL",
            "side": "Sell",
            "quantity": 2.0,
            "price": None,
            "tag": "trail-stop",
            "order_type": "StopTrail",
            "trail_offset": 1.5,
            "trail_reference_price": 101.0,
        },
        {
            "symbol": "AAPL",
            "side": "Sell",
            "quantity": 3.0,
            "price": 99.5,
            "tag": "trail-limit",
            "order_type": "StopTrailLimit",
            "trail_offset": 2.0,
            "trail_reference_price": 103.0,
        },
    ]


def test_strategy_no_legacy_broker_aliases() -> None:
    """Unified API should not expose removed broker alias methods."""
    strategy = MyStrategy()

    assert not hasattr(strategy, "submit_broker_order")
    assert not hasattr(strategy, "broker_buy")
    assert not hasattr(strategy, "broker_sell")


def test_oco_group_cancels_peer_on_trade() -> None:
    """Filled order in OCO group should cancel the peer order."""

    class _OcoSpyStrategy(Strategy):
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        def cancel_order(self, order_id: str) -> None:
            self.cancelled.append(order_id)

    strategy = _OcoSpyStrategy()
    group_id = strategy.place_oco("order-a", "order-b")
    strategy._process_order_groups(SimpleNamespace(order_id="order-a"))

    assert group_id == "oco-1"
    assert strategy.cancelled == ["order-b"]
    assert strategy._oco_groups == {}
    assert strategy._oco_order_to_group == {}


def test_oco_group_rebind_detaches_old_group() -> None:
    """Rebinding an order into new OCO group should detach old mapping."""
    strategy = MyStrategy()

    first_group = strategy.place_oco("order-a", "order-b")
    second_group = strategy.place_oco("order-b", "order-c")

    assert first_group == "oco-1"
    assert second_group == "oco-2"
    assert strategy._oco_groups == {"oco-2": {"order-b", "order-c"}}
    assert strategy._oco_order_to_group == {"order-b": "oco-2", "order-c": "oco-2"}


def test_oco_group_prefers_engine_registration_when_available() -> None:
    """Engine OCO registration should be preferred when capability exists."""

    class _FakeEngine:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        def register_oco_group(
            self, group_id: str, first_order_id: str, second_order_id: str
        ) -> None:
            self.calls.append((group_id, first_order_id, second_order_id))

    strategy = MyStrategy()
    fake_engine = _FakeEngine()
    setattr(strategy, "_engine", fake_engine)

    group_id = strategy.place_oco("order-a", "order-b")

    assert group_id == "oco-1"
    assert strategy._use_engine_oco is True
    assert fake_engine.calls == [("oco-1", "order-a", "order-b")]
    assert strategy._oco_groups == {}
    assert strategy._oco_order_to_group == {}


def test_oco_trade_processing_skips_python_fallback_when_engine_enabled() -> None:
    """Python OCO fallback should not run after engine OCO is enabled."""

    class _OcoSpyStrategy(Strategy):
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        def cancel_order(self, order_id: str) -> None:
            self.cancelled.append(order_id)

    strategy = _OcoSpyStrategy()
    strategy._use_engine_oco = True
    strategy._oco_groups = {"oco-1": {"order-a", "order-b"}}
    strategy._oco_order_to_group = {"order-a": "oco-1", "order-b": "oco-1"}

    strategy._process_oco_trade(SimpleNamespace(order_id="order-a"))

    assert strategy.cancelled == []
    assert strategy._oco_groups == {"oco-1": {"order-a", "order-b"}}
    assert strategy._oco_order_to_group == {"order-a": "oco-1", "order-b": "oco-1"}


def test_oco_group_falls_back_to_deferred_engine_queue_on_runtime_error() -> None:
    """OCO should queue engine registration when immediate call raises."""

    class _FailingEngine:
        def register_oco_group(
            self, group_id: str, first_order_id: str, second_order_id: str
        ) -> None:
            raise RuntimeError("engine borrow conflict")

    strategy = MyStrategy()
    setattr(strategy, "_engine", _FailingEngine())

    group_id = strategy.place_oco("order-a", "order-b")

    assert group_id == "oco-1"
    assert strategy._use_engine_oco is True
    assert strategy._pending_engine_oco_groups == [("oco-1", "order-a", "order-b")]
    assert strategy._oco_groups == {}
    assert strategy._oco_order_to_group == {}


def test_bracket_prefers_engine_registration_when_available() -> None:
    """Bracket plan should be registered to engine when capability exists."""

    class _FakeEngine:
        def __init__(self) -> None:
            self.calls: list[
                tuple[
                    str,
                    float | None,
                    float | None,
                    Any,
                    str | None,
                    str | None,
                ]
            ] = []

        def register_bracket_plan(
            self,
            entry_order_id: str,
            stop_trigger_price: float | None,
            take_profit_price: float | None,
            time_in_force: Any,
            stop_tag: str | None,
            take_profit_tag: str | None,
        ) -> None:
            self.calls.append(
                (
                    entry_order_id,
                    stop_trigger_price,
                    take_profit_price,
                    time_in_force,
                    stop_tag,
                    take_profit_tag,
                )
            )

    class _BracketEngineStrategy(Strategy):
        def __init__(self) -> None:
            self.buy_calls: list[dict[str, Any]] = []

        def buy(
            self,
            symbol: str | None = None,
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: Any = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
        ) -> Any:
            self.buy_calls.append(
                {
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "time_in_force": time_in_force,
                    "trigger_price": trigger_price,
                    "tag": tag,
                    "fill_mode": fill_mode,
                    "slippage": slippage,
                    "commission": commission,
                }
            )
            return "entry-1"

    strategy = _BracketEngineStrategy()
    fake_engine = _FakeEngine()
    setattr(strategy, "_engine", fake_engine)

    entry_id = strategy.place_bracket(
        symbol="AAPL",
        quantity=2.0,
        entry_price=100.0,
        stop_trigger_price=95.0,
        take_profit_price=110.0,
        entry_tag="entry",
        stop_tag="stop",
        take_profit_tag="take",
    )

    assert entry_id == "entry-1"
    assert strategy._use_engine_bracket is True
    assert strategy._pending_brackets == {}
    assert fake_engine.calls == [("entry-1", 95.0, 110.0, None, "stop", "take")]


def test_bracket_falls_back_to_deferred_engine_queue_on_runtime_error() -> None:
    """Bracket plan should queue deferred registration when engine call raises."""

    class _FailingEngine:
        def register_bracket_plan(
            self,
            entry_order_id: str,
            stop_trigger_price: float | None,
            take_profit_price: float | None,
            time_in_force: Any,
            stop_tag: str | None,
            take_profit_tag: str | None,
        ) -> None:
            _ = (
                entry_order_id,
                stop_trigger_price,
                take_profit_price,
                time_in_force,
                stop_tag,
                take_profit_tag,
            )
            raise RuntimeError("engine borrow conflict")

    class _BracketEngineStrategy(Strategy):
        def buy(
            self,
            symbol: str | None = None,
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: Any = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
        ) -> Any:
            _ = (
                symbol,
                quantity,
                price,
                time_in_force,
                trigger_price,
                tag,
                fill_mode,
                slippage,
                commission,
            )
            return "entry-1"

    strategy = _BracketEngineStrategy()
    setattr(strategy, "_engine", _FailingEngine())

    entry_id = strategy.place_bracket(
        symbol="AAPL",
        quantity=2.0,
        entry_price=100.0,
        stop_trigger_price=95.0,
        take_profit_price=110.0,
        entry_tag="entry",
        stop_tag="stop",
        take_profit_tag="take",
    )

    assert entry_id == "entry-1"
    assert strategy._use_engine_bracket is True
    assert strategy._pending_brackets == {}
    assert strategy._pending_engine_bracket_plans == [
        ("entry-1", 95.0, 110.0, None, "stop", "take")
    ]


def test_bracket_places_exit_orders_and_builds_oco() -> None:
    """Bracket entry fill should create stop/take exits and bind OCO."""

    class _BracketSpyStrategy(Strategy):
        def __init__(self) -> None:
            self.buy_calls: list[dict[str, Any]] = []
            self.sell_calls: list[dict[str, Any]] = []
            self._sell_counter = 0

        def buy(
            self,
            symbol: str | None = None,
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: Any = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
        ) -> Any:
            self.buy_calls.append(
                {
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "time_in_force": time_in_force,
                    "trigger_price": trigger_price,
                    "tag": tag,
                    "fill_mode": fill_mode,
                    "slippage": slippage,
                    "commission": commission,
                }
            )
            return "entry-1"

        def sell(
            self,
            symbol: str | None = None,
            quantity: float | None = None,
            price: float | None = None,
            time_in_force: Any = None,
            trigger_price: float | None = None,
            tag: str | None = None,
            fill_mode: FillMode | None = None,
            slippage: float | dict[str, Any] | None = None,
            commission: dict[str, Any] | None = None,
        ) -> Any:
            self._sell_counter += 1
            order_id = f"exit-{self._sell_counter}"
            self.sell_calls.append(
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "time_in_force": time_in_force,
                    "trigger_price": trigger_price,
                    "tag": tag,
                    "fill_mode": fill_mode,
                    "slippage": slippage,
                    "commission": commission,
                }
            )
            return order_id

    strategy = _BracketSpyStrategy()
    entry_id = strategy.place_bracket(
        symbol="AAPL",
        quantity=2.0,
        entry_price=100.0,
        stop_trigger_price=95.0,
        take_profit_price=110.0,
        entry_tag="entry",
        stop_tag="stop",
        take_profit_tag="take",
    )
    strategy._process_order_groups(
        SimpleNamespace(order_id=entry_id, symbol="AAPL", quantity=2.0)
    )

    assert strategy.buy_calls == [
        {
            "symbol": "AAPL",
            "quantity": 2.0,
            "price": 100.0,
            "time_in_force": None,
            "trigger_price": None,
            "tag": "entry",
            "fill_mode": None,
            "slippage": None,
            "commission": None,
        }
    ]
    assert strategy.sell_calls == [
        {
            "order_id": "exit-1",
            "symbol": "AAPL",
            "quantity": 2.0,
            "price": None,
            "time_in_force": None,
            "trigger_price": 95.0,
            "tag": "stop",
            "fill_mode": None,
            "slippage": None,
            "commission": None,
        },
        {
            "order_id": "exit-2",
            "symbol": "AAPL",
            "quantity": 2.0,
            "price": 110.0,
            "time_in_force": None,
            "trigger_price": None,
            "tag": "take",
            "fill_mode": None,
            "slippage": None,
            "commission": None,
        },
    ]
    assert strategy._pending_brackets == {}
    assert strategy._oco_groups == {"oco-1": {"exit-1", "exit-2"}}
    assert strategy._oco_order_to_group == {"exit-1": "oco-1", "exit-2": "oco-1"}
