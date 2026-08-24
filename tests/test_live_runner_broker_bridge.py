import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import akquant.live._runner as live_module
import pytest
from akquant.akquant import OrderSide, OrderStatus
from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.models import BrokerCapability, UnifiedPosition, UnifiedTrade
from akquant.live._runner import LiveRunner
from akquant.strategy import Strategy


def test_live_runner_broker_bridge_dispatches_events() -> None:
    """Dispatch broker events to strategy callbacks."""

    class _DummyTraderGateway:
        def __init__(self) -> None:
            self._on_order: Callable[[Any], None] | None = None
            self._on_trade: Callable[[Any], None] | None = None
            self._on_execution_report: Callable[[Any], None] | None = None

        def on_order(self, callback: Callable[[Any], None]) -> None:
            self._on_order = callback

        def on_trade(self, callback: Callable[[Any], None]) -> None:
            self._on_trade = callback

        def on_execution_report(self, callback: Callable[[Any], None]) -> None:
            self._on_execution_report = callback

        def emit_order(self, payload: Any) -> None:
            if self._on_order is not None:
                self._on_order(payload)

        def emit_trade(self, payload: Any) -> None:
            if self._on_trade is not None:
                self._on_trade(payload)

        def emit_execution_report(self, payload: Any) -> None:
            if self._on_execution_report is not None:
                self._on_execution_report(payload)

    class _DummyStrategy:
        def __init__(self) -> None:
            self.orders: list[Any] = []
            self.trades: list[Any] = []
            self.reports: list[Any] = []
            self.errors: list[tuple[str, Any]] = []

        def on_order(self, order: Any) -> None:
            self.orders.append(order)

        def on_trade(self, trade: Any) -> None:
            self.trades.append(trade)

        def on_execution_report(self, report: Any) -> None:
            self.reports.append(report)

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._bind_broker_callbacks(gateway, cast(Any, strategy))

    gateway.emit_order(
        {
            "broker_order_id": "o1",
            "client_order_id": "c-o1",
            "symbol": "IF2406",
            "status": "Filled",
            "filled_quantity": 3.0,
            "avg_fill_price": 10.0,
        }
    )
    gateway.emit_trade(
        {
            "trade_id": "t1",
            "broker_order_id": "o1",
            "client_order_id": "c-o1",
            "symbol": "IF2406",
            "side": "Buy",
            "quantity": 3.0,
            "price": 10.0,
        }
    )
    gateway.emit_execution_report({"id": "r1"})
    time.sleep(0.2)
    runner._stop_broker_dispatcher()

    # broker_live dispatch adapts order/trade payloads to the same shape as
    # backtest Order/Trade objects before calling on_order/on_trade.
    assert len(strategy.orders) == 1
    order = strategy.orders[0]
    assert order.symbol == "IF2406"
    assert order.status is OrderStatus.Filled
    assert order.filled_quantity == 3.0

    assert len(strategy.trades) == 1
    trade = strategy.trades[0]
    assert trade.symbol == "IF2406"
    assert trade.side is OrderSide.Buy
    assert trade.price == 10.0

    # execution_report is dispatched unchanged (raw payload).
    assert strategy.reports == [{"id": "r1"}]


def test_live_runner_broker_bridge_forwards_errors(caplog: Any) -> None:
    """Forward callback exceptions to strategy on_error."""

    class _DummyTraderGateway:
        def __init__(self) -> None:
            self._on_trade: Callable[[Any], None] | None = None

        def on_order(self, callback: Callable[[Any], None]) -> None:
            return None

        def on_trade(self, callback: Callable[[Any], None]) -> None:
            self._on_trade = callback

        def on_execution_report(self, callback: Callable[[Any], None]) -> None:
            return None

        def emit_trade(self, payload: Any) -> None:
            if self._on_trade is not None:
                self._on_trade(payload)

    class _DummyErrorStrategy:
        def __init__(self) -> None:
            self.errors: list[tuple[str, Any]] = []

        def on_trade(self, trade: Any) -> None:
            raise RuntimeError("trade callback failed")

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ptrade"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyErrorStrategy()
    runner._bind_broker_callbacks(gateway, cast(Any, strategy))

    with caplog.at_level("WARNING", logger="akquant.gateway.live"):
        gateway.emit_trade(
            {"id": "t2", "symbol": "IF2406", "client_order_id": "coid-t2"}
        )
        time.sleep(0.2)
        runner._stop_broker_dispatcher()

    assert strategy.errors
    assert strategy.errors[0][0] == "on_trade"
    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Strategy broker callback failed"
    )
    assert record.phase == "gateway"
    assert record.symbol == "IF2406"
    assert record.client_order_id == "coid-t2"


def test_live_runner_broker_bridge_deduplicates_events() -> None:
    """Deduplicate repeated broker events by semantic keys."""

    class _DummyStrategy:
        def __init__(self) -> None:
            self.orders: list[Any] = []

        def on_order(self, order: Any) -> None:
            self.orders.append(order)

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    strategy = _DummyStrategy()
    payload = SimpleNamespace(
        broker_order_id="b1",
        status="Submitted",
        filled_quantity=0.0,
        timestamp_ns=100,
    )

    runner._queue_broker_event("order", payload)
    runner._queue_broker_event("order", payload)
    runner._drain_broker_events(cast(Any, strategy))

    assert len(strategy.orders) == 1


def test_live_runner_broker_bridge_recovers_from_sync() -> None:
    """Recover order and trade snapshots from trader gateway sync methods."""

    class _DummyTraderGateway:
        def __init__(self) -> None:
            self._on_order: Callable[[Any], None] | None = None
            self._on_trade: Callable[[Any], None] | None = None
            self._on_execution_report: Callable[[Any], None] | None = None
            self.connected = False

        def on_order(self, callback: Callable[[Any], None]) -> None:
            self._on_order = callback

        def on_trade(self, callback: Callable[[Any], None]) -> None:
            self._on_trade = callback

        def on_execution_report(self, callback: Callable[[Any], None]) -> None:
            self._on_execution_report = callback

        def heartbeat(self) -> bool:
            return self.connected

        def connect(self) -> None:
            self.connected = True

        def sync_open_orders(self) -> list[Any]:
            return [
                SimpleNamespace(
                    broker_order_id="b-sync-1",
                    status="Submitted",
                    filled_quantity=0.0,
                    timestamp_ns=101,
                )
            ]

        def sync_today_trades(self) -> list[Any]:
            return [
                SimpleNamespace(
                    trade_id="t-sync-1",
                    broker_order_id="b-sync-1",
                    timestamp_ns=102,
                )
            ]

    class _DummyStrategy:
        def __init__(self) -> None:
            self.orders: list[Any] = []
            self.trades: list[Any] = []

        def on_order(self, order: Any) -> None:
            self.orders.append(order)

        def on_trade(self, trade: Any) -> None:
            self.trades.append(trade)

        def on_execution_report(self, report: Any) -> None:
            return None

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            return None

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ptrade"
    runner._init_broker_bridge_state()
    runner._broker_recovery_interval_sec = 0.05
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._bind_broker_callbacks(gateway, cast(Any, strategy))
    runner._broker_baseline_done = True
    runner._run_broker_recovery_cycle()
    runner._drain_broker_events(cast(Any, strategy))
    runner._stop_broker_dispatcher()

    assert strategy.orders
    assert strategy.trades
    assert "b-sync-1" in runner._broker_order_states
    assert any(getattr(t, "id", None) == "t-sync-1" for t in strategy.trades)


def test_live_runner_recovery_syncs_account_snapshot() -> None:
    """Recovery should cache the latest broker account snapshot."""

    class _DummyTraderGateway:
        def heartbeat(self) -> bool:
            return True

        def sync_open_orders(self) -> list[Any]:
            return []

        def sync_today_trades(self) -> list[Any]:
            return []

        def query_account(self) -> Any:
            return SimpleNamespace(
                account_id="acct-live-1",
                equity=100000.0,
                cash=100000.0,
                available_cash=80000.0,
                timestamp_ns=200,
            )

    class _DummyStrategy:
        def __init__(self) -> None:
            self.portfolio_updates: list[Any] = []

        def on_portfolio_update(self, payload: Any) -> None:
            self.portfolio_updates.append(payload)

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            return None

    observed: list[dict[str, Any]] = []
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.gateway_options = {"recovery_mode": "compatible"}
    runner.on_broker_event = observed.append
    runner._init_broker_bridge_state()
    runner._broker_trader_gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()

    runner._run_broker_recovery_cycle(cast(Any, strategy))
    runner._drain_broker_events(cast(Any, strategy))

    assert runner._broker_account_state is not None
    assert (
        runner._payload_field(runner._broker_account_state, "account_id")
        == "acct-live-1"
    )
    assert strategy.portfolio_updates
    account_events = [event for event in observed if event["event_type"] == "account"]
    assert account_events
    assert account_events[0]["payload"]["account_id"] == "acct-live-1"


def test_live_runner_strict_recovery_reports_sync_failure() -> None:
    """Strict recovery mode should surface sync failures to strategy and observer."""

    class _DummyTraderGateway:
        def heartbeat(self) -> bool:
            return True

        def sync_open_orders(self) -> list[Any]:
            return []

        def sync_today_trades(self) -> list[Any]:
            raise RuntimeError("sync trades failed")

    class _DummyStrategy:
        def __init__(self) -> None:
            self.errors: list[tuple[str, Any]] = []
            self.execution: Any | None = None

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    broker_events: list[dict[str, Any]] = []
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.gateway_options = {"recovery_mode": "strict"}
    runner.on_broker_event = broker_events.append
    runner._init_broker_bridge_state()
    runner._broker_trader_gateway = _DummyTraderGateway()
    runner._broker_baseline_done = True
    strategy = _DummyStrategy()

    runner._run_broker_recovery_cycle(cast(Any, strategy))

    assert strategy.errors
    assert strategy.errors[0][0] == "broker_recovery.sync_today_trades"
    assert broker_events
    assert broker_events[0]["event_type"] == "recovery_error"
    assert broker_events[0]["payload"]["source"] == "broker_recovery.sync_today_trades"


def test_live_runner_strict_recovery_reports_account_query_failure() -> None:
    """Strict recovery mode should surface account query failures."""

    class _DummyTraderGateway:
        def heartbeat(self) -> bool:
            return True

        def sync_open_orders(self) -> list[Any]:
            return []

        def sync_today_trades(self) -> list[Any]:
            return []

        def query_account(self) -> Any:
            raise RuntimeError("account query failed")

    class _DummyStrategy:
        def __init__(self) -> None:
            self.errors: list[tuple[str, Any]] = []
            self.execution: Any | None = None

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    observed: list[dict[str, Any]] = []
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.gateway_options = {"recovery_mode": "strict"}
    runner.on_broker_event = observed.append
    runner._init_broker_bridge_state()
    runner._broker_trader_gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()

    runner._run_broker_recovery_cycle(cast(Any, strategy))

    assert strategy.errors
    assert strategy.errors[0][0] == "broker_recovery.query_account"
    assert observed
    assert observed[0]["event_type"] == "recovery_error"
    assert observed[0]["payload"]["source"] == "broker_recovery.query_account"


def test_live_runner_compatible_recovery_keeps_sync_failure_silent() -> None:
    """Compatible recovery mode should keep sync failures non-fatal and silent."""

    class _DummyTraderGateway:
        def heartbeat(self) -> bool:
            return True

        def sync_open_orders(self) -> list[Any]:
            return []

        def sync_today_trades(self) -> list[Any]:
            raise RuntimeError("sync trades failed")

    class _DummyStrategy:
        def __init__(self) -> None:
            self.errors: list[tuple[str, Any]] = []
            self.execution: Any | None = None

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    broker_events: list[dict[str, Any]] = []
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.gateway_options = {"recovery_mode": "compatible"}
    runner.on_broker_event = broker_events.append
    runner._init_broker_bridge_state()
    runner._broker_trader_gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()

    runner._run_broker_recovery_cycle(cast(Any, strategy))

    assert strategy.errors == []
    assert broker_events == []


def test_live_runner_init_normalizes_legacy_gateway_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LiveRunner should fold legacy broker args into gateway_options at init."""

    class _DummyDataFeed:
        @staticmethod
        def create_live() -> object:
            return object()

    class _DummyEngine:
        pass

    monkeypatch.setattr(live_module, "DataFeed", _DummyDataFeed)
    monkeypatch.setattr(live_module, "Engine", _DummyEngine)

    runner = LiveRunner(
        strategy_cls=None,
        instruments=[],
        md_front="tcp://md-front",
        broker_id="9999",
        user_id="trader-a",
        password="secret",
        gateway_options={"recovery_mode": "strict", "custom": "value"},
    )

    assert runner.gateway_options == {
        "recovery_mode": "strict",
        "custom": "value",
        "md_front": "tcp://md-front",
        "broker_id": "9999",
        "user_id": "trader-a",
        "password": "secret",
    }
    assert runner.md_front == "tcp://md-front"
    assert runner.broker_id == "9999"
    assert runner.user_id == "trader-a"
    assert runner.password == "secret"
    assert runner._build_gateway_kwargs() == runner.gateway_options


def test_live_runner_init_prefers_gateway_options_over_legacy_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit gateway_options values should win over legacy init args."""

    class _DummyDataFeed:
        @staticmethod
        def create_live() -> object:
            return object()

    class _DummyEngine:
        pass

    monkeypatch.setattr(live_module, "DataFeed", _DummyDataFeed)
    monkeypatch.setattr(live_module, "Engine", _DummyEngine)

    runner = LiveRunner(
        strategy_cls=None,
        instruments=[],
        md_front="tcp://legacy-md-front",
        broker_id="legacy-broker",
        user_id="legacy-user",
        gateway_options={
            "md_front": "tcp://explicit-md-front",
            "broker_id": "explicit-broker",
            "user_id": "explicit-user",
        },
    )

    assert runner.gateway_options["md_front"] == "tcp://explicit-md-front"
    assert runner.gateway_options["broker_id"] == "explicit-broker"
    assert runner.gateway_options["user_id"] == "explicit-user"
    assert runner.md_front == "tcp://explicit-md-front"
    assert runner.broker_id == "explicit-broker"
    assert runner.user_id == "explicit-user"


def test_live_runner_syncs_client_broker_order_id_mapping() -> None:
    """Sync id mapping from order, report and trade events."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()

    runner._update_broker_state(
        "order",
        {
            "client_order_id": "c-map-1",
            "broker_order_id": "b-map-1",
            "status": "Submitted",
        },
    )
    runner._update_broker_state(
        "execution_report",
        {
            "client_order_id": "c-map-1",
            "broker_order_id": "b-map-1",
            "status": "Submitted",
            "timestamp_ns": 1,
        },
    )
    runner._update_broker_state(
        "trade",
        {
            "trade_id": "t-map-1",
            "broker_order_id": "b-map-1",
        },
    )

    assert runner._resolve_broker_order_id("c-map-1") == "b-map-1"
    assert runner._resolve_client_order_id("b-map-1") == "c-map-1"


def test_live_runner_cleans_mapping_on_terminal_status() -> None:
    """Cleanup active mapping when order enters terminal status."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()

    runner._update_broker_state(
        "order",
        {
            "client_order_id": "c-term-1",
            "broker_order_id": "b-term-1",
            "status": "Submitted",
        },
    )
    assert not runner.can_submit_client_order("c-term-1")

    runner._update_broker_state(
        "execution_report",
        {
            "client_order_id": "c-term-1",
            "broker_order_id": "b-term-1",
            "status": "Cancelled",
            "timestamp_ns": 2,
        },
    )

    assert runner.can_submit_client_order("c-term-1")
    assert "b-term-1" in runner._closed_broker_order_ids
    assert runner._resolve_broker_order_id("c-term-1") == ""


def test_live_runner_submitter_checks_idempotency_and_maps() -> None:
    """Install submitter and map ids after broker placement."""

    class _DummyTraderGateway:
        def __init__(self) -> None:
            self.last_client_order_id = ""

        def place_order(self, req: Any) -> str:
            self.last_client_order_id = req.client_order_id
            return f"b-{req.client_order_id}"

    class _DummyStrategy:
        def __init__(self) -> None:
            self.errors: list[tuple[str, Any]] = []
            self.execution: Any | None = None

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._install_broker_order_submitter(cast(Any, gateway), cast(Any, strategy))
    assert isinstance(strategy.execution, BrokerExecution)
    strategy_any = cast(Any, strategy)

    broker_order_id = strategy_any.execution.submit_order(
        symbol="000001.SZ",
        side="Buy",
        quantity=10.0,
        client_order_id="coid-1",
    )

    assert broker_order_id.primary == "b-coid-1"
    assert runner._resolve_broker_order_id("coid-1") == "b-coid-1"
    assert runner._resolve_client_order_id("b-coid-1") == "coid-1"


def test_live_runner_submitter_forwards_duplicate_error(caplog: Any) -> None:
    """Raise and forward error when submitting duplicate active client order id."""

    class _DummyTraderGateway:
        def place_order(self, req: Any) -> str:
            return f"b-{req.client_order_id}"

    class _DummyStrategy:
        def __init__(self) -> None:
            self.errors: list[tuple[str, Any]] = []

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ptrade"
    runner._init_broker_bridge_state()
    runner._sync_order_id_mapping("coid-dup", "b-coid-dup")
    runner._broker_order_states["b-coid-dup"] = {"status": "Submitted"}
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._install_broker_order_submitter(cast(Any, gateway), cast(Any, strategy))
    strategy_any = cast(Any, strategy)

    with caplog.at_level("WARNING", logger="akquant.gateway.live"):
        try:
            strategy_any.execution.submit_order(
                symbol="000002.SZ",
                side="Sell",
                quantity=5.0,
                client_order_id="coid-dup",
            )
        except RuntimeError as exc:
            assert "duplicate active client_order_id" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for duplicate client_order_id")

    assert strategy.errors
    assert strategy.errors[0][0] == "submit_order"
    record = next(
        record
        for record in caplog.records
        if record.getMessage()
        == "Rejected live submit_order because client_order_id is already active"
    )
    assert record.phase == "gateway"
    assert record.symbol == "000002.SZ"
    assert record.client_order_id == "coid-dup"


def test_live_runner_submit_order_supports_buy_and_sell_side() -> None:
    """Unified submit_order should support both buy and sell side."""

    class _DummyTraderGateway:
        def __init__(self) -> None:
            self.last_side = ""
            self.last_client_order_id = ""

        def place_order(self, req: Any) -> str:
            self.last_side = req.side
            self.last_client_order_id = req.client_order_id
            return f"b-{req.side}-{req.client_order_id}"

    class _DummyStrategy:
        def __init__(self) -> None:
            self.errors: list[tuple[str, Any]] = []

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._install_broker_order_submitter(cast(Any, gateway), cast(Any, strategy))
    strategy_any = cast(Any, strategy)

    buy_broker_order_id = strategy_any.execution.submit_order(
        symbol="000001.SZ",
        side="Buy",
        quantity=10.0,
        client_order_id="coid-buy-1",
    )
    sell_broker_order_id = strategy_any.execution.submit_order(
        symbol="000001.SZ",
        side="Sell",
        quantity=5.0,
        client_order_id="coid-sell-1",
    )

    assert buy_broker_order_id.primary == "b-Buy-coid-buy-1"
    assert sell_broker_order_id.primary == "b-Sell-coid-sell-1"
    assert runner._resolve_broker_order_id("coid-buy-1") == "b-Buy-coid-buy-1"
    assert runner._resolve_broker_order_id("coid-sell-1") == "b-Sell-coid-sell-1"


def test_live_runner_submit_order_forwards_position_effect() -> None:
    """Unified submit_order should forward position_effect to gateway request."""

    class _DummyTraderGateway:
        def __init__(self) -> None:
            self.last_position_effect = ""
            self.last_reduce_only = False

        def place_order(self, req: Any) -> str:
            self.last_position_effect = req.position_effect
            self.last_reduce_only = req.reduce_only
            return f"b-{req.client_order_id}"

    class _DummyStrategy:
        def __init__(self) -> None:
            self.errors: list[tuple[str, Any]] = []

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._install_broker_order_submitter(cast(Any, gateway), cast(Any, strategy))
    strategy_any = cast(Any, strategy)

    broker_order_id = strategy_any.execution.submit_order(
        symbol="000001.SZ",
        side="Buy",
        quantity=10.0,
        client_order_id="coid-effect-1",
        position_effect="close",
        reduce_only=True,
    )

    assert broker_order_id.primary == "b-coid-effect-1"
    assert gateway.last_position_effect == "close"
    assert gateway.last_reduce_only is True


def test_live_runner_submit_order_auto_splits_close_today_and_yesterday() -> None:
    """Live submitter should split close into close_today and close_yesterday."""

    class _DummyTraderGateway:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        def place_order(self, req: Any) -> str:
            self.requests.append(req)
            return f"b-{req.client_order_id}"

        def query_positions(self) -> list[UnifiedPosition]:
            return [
                UnifiedPosition(
                    symbol="au2606",
                    quantity=5.0,
                    available_quantity=5.0,
                    direction="Buy",
                    today_quantity=2.0,
                    yesterday_quantity=3.0,
                    available_today_quantity=2.0,
                    available_yesterday_quantity=3.0,
                )
            ]

        def get_capabilities(self) -> BrokerCapability:
            return BrokerCapability(
                broker_name="ctp",
                broker_live=True,
                client_order_id=True,
                order_type=True,
                time_in_force_str=True,
                position_effect=True,
                reduce_only=False,
                position_details=True,
                supports_short_sell=True,
                supported_position_effects=(
                    "auto",
                    "open",
                    "close",
                    "close_today",
                    "close_yesterday",
                ),
            )

    class _DummyStrategy:
        def __init__(self) -> None:
            self.errors: list[tuple[str, Any]] = []

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._install_broker_order_submitter(cast(Any, gateway), cast(Any, strategy))
    strategy_any = cast(Any, strategy)

    broker_order_id = strategy_any.execution.submit_order(
        symbol="au2606",
        side="Sell",
        quantity=4.0,
        client_order_id="coid-close-split",
        position_effect="close",
    )

    assert broker_order_id.primary == "b-coid-close-split"
    assert len(gateway.requests) == 2
    assert gateway.requests[0].client_order_id == "coid-close-split"
    assert gateway.requests[0].position_effect == "close_today"
    assert gateway.requests[0].quantity == 2.0
    assert gateway.requests[1].client_order_id == "coid-close-split-close-yesterday-2"
    assert gateway.requests[1].position_effect == "close_yesterday"
    assert gateway.requests[1].quantity == 2.0
    assert runner._resolve_broker_order_id("coid-close-split") == "b-coid-close-split"
    assert (
        runner._resolve_broker_order_id("coid-close-split-close-yesterday-2")
        == "b-coid-close-split-close-yesterday-2"
    )


def test_live_runner_close_position_prefers_direction_match() -> None:
    """Close leg resolution should prefer the matching position direction."""
    runner = LiveRunner.__new__(LiveRunner)
    positions = [
        UnifiedPosition(
            symbol="IF2406",
            quantity=-1.0,
            available_quantity=1.0,
            direction="Sell",
        ),
        UnifiedPosition(
            symbol="IF2406",
            quantity=2.0,
            available_quantity=2.0,
            direction="Buy",
        ),
    ]

    sell_side_match = runner._find_live_close_position(positions, "IF2406", "Sell")
    buy_side_match = runner._find_live_close_position(positions, "IF2406", "Buy")

    assert sell_side_match is not None
    assert sell_side_match.direction == "Buy"
    assert buy_side_match is not None
    assert buy_side_match.direction == "Sell"


def test_live_runner_submit_order_falls_back_to_close_when_position_query_fails() -> (
    None
):
    """Live submitter should keep a plain close without position details."""

    class _DummyTraderGateway:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        def place_order(self, req: Any) -> str:
            self.requests.append(req)
            return f"b-{req.client_order_id}"

        def query_positions(self) -> list[UnifiedPosition]:
            raise RuntimeError("query failed")

        def get_capabilities(self) -> BrokerCapability:
            return BrokerCapability(
                broker_name="ctp",
                broker_live=True,
                client_order_id=True,
                order_type=True,
                time_in_force_str=True,
                position_effect=True,
                reduce_only=False,
                position_details=True,
                supports_short_sell=True,
                supported_position_effects=(
                    "auto",
                    "open",
                    "close",
                    "close_today",
                    "close_yesterday",
                ),
            )

    class _DummyStrategy:
        def __init__(self) -> None:
            self.errors: list[tuple[str, Any]] = []

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._install_broker_order_submitter(cast(Any, gateway), cast(Any, strategy))
    strategy_any = cast(Any, strategy)

    broker_order_id = strategy_any.execution.submit_order(
        symbol="au2606",
        side="Sell",
        quantity=4.0,
        client_order_id="coid-close-fallback",
        position_effect="close",
    )

    assert broker_order_id.primary == "b-coid-close-fallback"
    assert len(gateway.requests) == 1
    assert gateway.requests[0].client_order_id == "coid-close-fallback"
    assert gateway.requests[0].position_effect == "close"
    assert gateway.requests[0].quantity == 4.0


def test_live_runner_submitter_respects_gateway_capabilities() -> None:
    """Injected submit_order should reject semantics not supported by broker."""

    class _DummyTraderGateway:
        def place_order(self, req: Any) -> str:
            return f"b-{req.client_order_id}"

        def get_capabilities(self) -> BrokerCapability:
            return BrokerCapability(
                broker_name="miniqmt",
                broker_live=True,
                client_order_id=True,
                order_type=True,
                time_in_force_str=True,
                position_effect=False,
                reduce_only=False,
                supported_position_effects=("auto",),
            )

    class _DummyStrategy:
        def __init__(self) -> None:
            self.errors: list[tuple[str, Any]] = []

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._install_broker_order_submitter(cast(Any, gateway), cast(Any, strategy))
    strategy_any = cast(Any, strategy)

    with pytest.raises(RuntimeError, match="does not support explicit position_effect"):
        strategy_any.execution.submit_order(
            symbol="000001.SZ",
            side="Buy",
            quantity=10.0,
            client_order_id="coid-effect-unsupported",
            position_effect="close",
        )

    with pytest.raises(RuntimeError, match="does not support reduce_only"):
        strategy_any.execution.submit_order(
            symbol="000001.SZ",
            side="Sell",
            quantity=10.0,
            client_order_id="coid-reduce-only-unsupported",
            reduce_only=True,
        )


def test_live_runner_injects_execution_capabilities() -> None:
    """Expose broker-live capabilities after submitter injection."""

    class _DummyTraderGateway:
        def place_order(self, req: Any) -> str:
            return f"b-{req.client_order_id}"

        def get_capabilities(self) -> BrokerCapability:
            return BrokerCapability(
                broker_name="ctp",
                broker_live=True,
                client_order_id=True,
                order_type=True,
                time_in_force_str=True,
                position_effect=True,
                reduce_only=False,
                position_details=True,
                supports_short_sell=True,
                supported_position_effects=(
                    "auto",
                    "open",
                    "close",
                    "close_today",
                    "close_yesterday",
                ),
            )

    class _DummyStrategy:
        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            return None

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ptrade"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._install_broker_order_submitter(cast(Any, gateway), cast(Any, strategy))
    strategy_any = cast(Any, strategy)
    capabilities = strategy_any.execution.capabilities()

    assert capabilities["broker_live"] is True
    assert capabilities["client_order_id"] is True
    assert capabilities["position_effect"] is True
    assert capabilities["reduce_only"] is False
    assert capabilities["position_details"] is True
    assert capabilities["supports_short_sell"] is True
    assert capabilities["supported_position_effects"] == [
        "auto",
        "open",
        "close",
        "close_today",
        "close_yesterday",
    ]


def test_live_runner_does_not_inject_removed_broker_aliases() -> None:
    """Keep BrokerExecution as the sole install target (no direct strategy setattrs)."""

    class _DummyTraderGateway:
        def place_order(self, req: Any) -> str:
            return f"b-{req.client_order_id}"

    class _DummyStrategy:
        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            return None

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ptrade"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._install_broker_order_submitter(cast(Any, gateway), cast(Any, strategy))
    strategy_any = cast(Any, strategy)

    assert isinstance(strategy_any.execution, BrokerExecution)
    assert hasattr(strategy_any.execution, "submit_order")
    assert "submit_order" not in strategy_any.__dict__
    assert not hasattr(strategy_any, "submit_broker_order")
    assert not hasattr(strategy_any, "broker_buy")
    assert not hasattr(strategy_any, "broker_sell")


def test_live_runner_builds_strategy_instance_from_class() -> None:
    """Build strategy instance from class input."""

    class _DummyStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = _DummyStrategy
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    strategy = runner._build_strategy_instance(runner.strategy_cls)
    assert isinstance(strategy, _DummyStrategy)


def test_live_runner_builds_strategy_instance_from_existing_instance() -> None:
    """Reuse provided strategy instance input."""

    class _DummyStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    instance = _DummyStrategy()
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = instance
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    strategy = runner._build_strategy_instance(runner.strategy_cls)
    assert strategy is instance


def test_live_runner_builds_functional_strategy_instance() -> None:
    """Build functional strategy wrapper from callable input."""
    events: list[str] = []

    def initialize(ctx: Any) -> None:
        events.append("initialize")
        ctx.seed = 7

    def on_start(ctx: Any) -> None:
        _ = ctx
        events.append("on_start")

    def on_stop(ctx: Any) -> None:
        _ = ctx
        events.append("on_stop")

    def on_bar(ctx: Any, bar: Any) -> None:
        _ = bar
        events.append(f"bar:{getattr(ctx, 'seed', 0)}")

    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = on_bar
    runner.initialize = initialize
    runner.on_start = on_start
    runner.on_stop = on_stop
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {"flag": "ok"}
    strategy = runner._build_strategy_instance(runner.strategy_cls)

    assert isinstance(strategy, Strategy)
    assert getattr(strategy, "flag") == "ok"
    assert events == ["initialize"]
    strategy.on_start()
    strategy.on_bar(cast(Any, SimpleNamespace(symbol="TEST")))
    strategy.on_stop()
    assert events == ["initialize", "on_start", "bar:7", "on_stop"]


def test_live_runner_builds_strategy_instance_from_strategy_source(
    tmp_path: Path,
) -> None:
    """Build strategy instance from configured strategy_source."""
    strategy_file = tmp_path / "live_source_strategy.py"
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

    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = None
    runner.strategy_source = str(strategy_file)
    runner.strategy_loader = "python_plain"
    runner.strategy_loader_options = None
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    strategy = runner._build_strategy_instance(runner.strategy_cls)

    assert isinstance(strategy, Strategy)
    assert type(strategy).__name__ == "Strategy"


def test_live_runner_builds_strategy_from_encrypted_external_loader() -> None:
    """Build strategy instance using encrypted_external loader callback."""

    class _LoadedStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    def _decrypt_loader(source: Any, options: dict[str, Any]) -> type[Strategy]:
        _ = source
        _ = options
        return _LoadedStrategy

    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = None
    runner.strategy_source = b"cipher"
    runner.strategy_loader = "encrypted_external"
    runner.strategy_loader_options = {"decrypt_and_load": _decrypt_loader}
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    strategy = runner._build_strategy_instance(runner.strategy_cls)

    assert isinstance(strategy, _LoadedStrategy)


def test_live_runner_rejects_missing_strategy_and_source() -> None:
    """Live runner should fail when both strategy and source are missing."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = None
    runner.strategy_source = None
    runner.strategy_loader = None
    runner.strategy_loader_options = None
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    with pytest.raises(ValueError, match="Strategy must be provided"):
        runner._build_strategy_instance(runner.strategy_cls)


def test_live_runner_builds_strategy_topology_with_slots() -> None:
    """Build primary and slot strategies with explicit strategy ids."""

    def on_bar(ctx: Any, bar: Any) -> None:
        _ = ctx
        _ = bar

    def slot_on_bar(ctx: Any, bar: Any) -> None:
        _ = ctx
        _ = bar

    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_cls = on_bar
    runner.strategy_id = "alpha"
    runner.strategies_by_slot = {"beta": slot_on_bar}
    runner.initialize = None
    runner.on_start = None
    runner.on_stop = None
    runner.on_tick = None
    runner.on_order = None
    runner.on_trade = None
    runner.on_timer = None
    runner.context = {}
    strategy, slots, strategy_id = runner._build_strategy_topology()

    assert isinstance(strategy, Strategy)
    assert strategy_id == "alpha"
    assert set(slots.keys()) == {"beta"}
    assert isinstance(slots["beta"], Strategy)


def test_live_runner_configures_engine_slots_for_primary_and_secondary() -> None:
    """Configure slot metadata and strategy binding on engine."""

    class _DummyEngine:
        def __init__(self) -> None:
            self.slot_ids: list[str] = []
            self.default_strategy_id = ""
            self.slot_strategies: dict[int, Any] = {}

        def set_strategy_slots(self, slot_ids: list[str]) -> None:
            self.slot_ids = slot_ids

        def set_default_strategy_id(self, strategy_id: str) -> None:
            self.default_strategy_id = strategy_id

        def set_strategy_for_slot(self, slot_index: int, strategy: Any) -> None:
            self.slot_strategies[slot_index] = strategy

    class _DummyStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    runner = LiveRunner.__new__(LiveRunner)
    runner.engine = cast(Any, _DummyEngine())
    runner.context = {"shared_flag": "ok"}
    primary = _DummyStrategy()
    secondary = _DummyStrategy()
    runner._configure_strategy_slots(primary, {"beta": secondary}, "alpha")
    engine = cast(_DummyEngine, runner.engine)

    assert engine.slot_ids == ["alpha", "beta"]
    assert engine.default_strategy_id == "alpha"
    assert engine.slot_strategies[0] is primary
    assert engine.slot_strategies[1] is secondary
    assert getattr(primary, "_owner_strategy_id") == "alpha"
    assert getattr(secondary, "_owner_strategy_id") == "beta"
    assert getattr(primary, "shared_flag") == "ok"
    assert getattr(secondary, "shared_flag") == "ok"


def test_live_runner_applies_strategy_risk_controls_for_slots() -> None:
    """Apply strategy-level risk controls using configured slot ids."""

    class _DummyEngine:
        def __init__(self) -> None:
            self.slot_ids: list[str] = []
            self.default_strategy_id = ""
            self.slot_strategies: dict[int, Any] = {}
            self.max_order_value_limits: dict[str, float] = {}
            self.max_order_size_limits: dict[str, float] = {}
            self.max_position_size_limits: dict[str, float] = {}
            self.max_daily_loss_limits: dict[str, float] = {}
            self.max_drawdown_limits: dict[str, float] = {}
            self.reduce_only_flags: dict[str, bool] = {}
            self.cooldown_bars: dict[str, int] = {}
            self.strategy_priorities: dict[str, int] = {}
            self.strategy_risk_budget_limits: dict[str, float] = {}
            self.portfolio_risk_budget_limit: float | None = None
            self.risk_budget_mode = ""
            self.risk_budget_reset_daily = False

        def set_strategy_slots(self, slot_ids: list[str]) -> None:
            self.slot_ids = slot_ids

        def set_default_strategy_id(self, strategy_id: str) -> None:
            self.default_strategy_id = strategy_id

        def set_strategy_for_slot(self, slot_index: int, strategy: Any) -> None:
            self.slot_strategies[slot_index] = strategy

        def set_strategy_max_order_value_limits(self, limits: dict[str, float]) -> None:
            self.max_order_value_limits = limits

        def set_strategy_max_order_size_limits(self, limits: dict[str, float]) -> None:
            self.max_order_size_limits = limits

        def set_strategy_max_position_size_limits(
            self, limits: dict[str, float]
        ) -> None:
            self.max_position_size_limits = limits

        def set_strategy_max_daily_loss_limits(self, limits: dict[str, float]) -> None:
            self.max_daily_loss_limits = limits

        def set_strategy_max_drawdown_limits(self, limits: dict[str, float]) -> None:
            self.max_drawdown_limits = limits

        def set_strategy_reduce_only_after_risk(self, flags: dict[str, bool]) -> None:
            self.reduce_only_flags = flags

        def set_strategy_risk_cooldown_bars(self, bars: dict[str, int]) -> None:
            self.cooldown_bars = bars

        def set_strategy_priorities(self, priorities: dict[str, int]) -> None:
            self.strategy_priorities = priorities

        def set_strategy_risk_budget_limits(self, limits: dict[str, float]) -> None:
            self.strategy_risk_budget_limits = limits

        def set_portfolio_risk_budget_limit(self, limit: float | None) -> None:
            self.portfolio_risk_budget_limit = limit

        def set_risk_budget_mode(self, mode: str) -> None:
            self.risk_budget_mode = mode

        def set_risk_budget_reset_daily(self, enabled: bool) -> None:
            self.risk_budget_reset_daily = enabled

    class _DummyStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    runner = LiveRunner.__new__(LiveRunner)
    runner.engine = cast(Any, _DummyEngine())
    runner.context = {}
    runner.strategy_max_order_value = {"alpha": 1000.0, "beta": 2000.0}
    runner.strategy_max_order_size = {"alpha": 10.0, "beta": 20.0}
    runner.strategy_max_position_size = {"alpha": 100.0, "beta": 200.0}
    runner.strategy_max_daily_loss = {"alpha": 0.02, "beta": 0.03}
    runner.strategy_max_drawdown = {"alpha": 0.1, "beta": 0.15}
    runner.strategy_reduce_only_after_risk = {"alpha": True, "beta": False}
    runner.strategy_risk_cooldown_bars = {"alpha": 3, "beta": 5}
    runner.strategy_priority = {"alpha": 1, "beta": 2}
    runner.strategy_risk_budget = {"alpha": 50000.0, "beta": 60000.0}
    runner.portfolio_risk_budget = 120000.0
    runner.risk_budget_mode = "order_notional"
    runner.risk_budget_reset_daily = True
    primary = _DummyStrategy()
    secondary = _DummyStrategy()
    runner._configure_strategy_slots(primary, {"beta": secondary}, "alpha")
    engine = cast(_DummyEngine, runner.engine)

    assert engine.max_order_value_limits == {"alpha": 1000.0, "beta": 2000.0}
    assert engine.max_order_size_limits == {"alpha": 10.0, "beta": 20.0}
    assert engine.max_position_size_limits == {"alpha": 100.0, "beta": 200.0}
    assert engine.max_daily_loss_limits == {"alpha": 0.02, "beta": 0.03}
    assert engine.max_drawdown_limits == {"alpha": 0.1, "beta": 0.15}
    assert engine.reduce_only_flags == {"alpha": True, "beta": False}
    assert engine.cooldown_bars == {"alpha": 3, "beta": 5}
    assert engine.strategy_priorities == {"alpha": 1, "beta": 2}
    assert engine.strategy_risk_budget_limits == {"alpha": 50000.0, "beta": 60000.0}
    assert engine.portfolio_risk_budget_limit == 120000.0
    assert engine.risk_budget_mode == "order_notional"
    assert engine.risk_budget_reset_daily is True


def test_live_runner_rejects_unknown_strategy_ids_in_risk_controls() -> None:
    """Reject strategy-level maps containing ids outside configured slots."""

    class _DummyEngine:
        def set_strategy_slots(self, slot_ids: list[str]) -> None:
            _ = slot_ids

        def set_default_strategy_id(self, strategy_id: str) -> None:
            _ = strategy_id

        def set_strategy_for_slot(self, slot_index: int, strategy: Any) -> None:
            _ = slot_index
            _ = strategy

    class _DummyStrategy(Strategy):
        def on_bar(self, bar: Any) -> None:
            _ = bar

    runner = LiveRunner.__new__(LiveRunner)
    runner.engine = cast(Any, _DummyEngine())
    runner.context = {}
    runner.strategy_max_order_value = {"ghost": 123.0}
    runner.strategy_max_order_size = {}
    runner.strategy_max_position_size = {}
    runner.strategy_max_daily_loss = {}
    runner.strategy_max_drawdown = {}
    runner.strategy_reduce_only_after_risk = {}
    runner.strategy_risk_cooldown_bars = {}
    runner.strategy_priority = {}
    runner.strategy_risk_budget = {}
    runner.portfolio_risk_budget = None
    runner.risk_budget_mode = "order_notional"
    runner.risk_budget_reset_daily = False
    primary = _DummyStrategy()
    secondary = _DummyStrategy()

    try:
        runner._configure_strategy_slots(primary, {"beta": secondary}, "alpha")
        assert False, "expected ValueError for unknown strategy id"
    except ValueError as exc:
        assert "unknown strategy ids: ghost" in str(exc)


def test_live_runner_submitter_binds_owner_strategy_id_mapping() -> None:
    """Bind strategy owner mapping when submit_order is called."""

    class _DummyTraderGateway:
        def place_order(self, req: Any) -> str:
            return f"b-{req.client_order_id}"

    class _DummyStrategy:
        def __init__(self) -> None:
            self._owner_strategy_id = "alpha"
            self.errors: list[tuple[str, Any]] = []

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._install_broker_order_submitter(cast(Any, gateway), cast(Any, strategy))
    strategy_any = cast(Any, strategy)
    broker_order_id = strategy_any.execution.submit_order(
        symbol="000001.SZ",
        side="Buy",
        quantity=10.0,
        client_order_id="coid-owner-1",
    )

    assert broker_order_id.primary == "b-coid-owner-1"
    assert runner._client_to_strategy_ids["coid-owner-1"] == "alpha"
    assert runner._broker_to_strategy_ids["b-coid-owner-1"] == "alpha"


def test_live_runner_submitter_syncs_group_mapping() -> None:
    """submit_order 应把每腿 client_order_id -> 根 client_order_id 映射同步进 runner."""

    class _DummyTraderGateway:
        def place_order(self, req: Any) -> str:
            return f"b-{req.client_order_id}"

    class _DummyStrategy:
        def __init__(self) -> None:
            self._owner_strategy_id = "alpha"
            self.errors: list[tuple[str, Any]] = []

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            self.errors.append((source, payload))

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    gateway = _DummyTraderGateway()
    strategy = _DummyStrategy()
    runner._install_broker_order_submitter(cast(Any, gateway), cast(Any, strategy))
    strategy_any = cast(Any, strategy)
    strategy_any.execution.submit_order(
        symbol="000001.SZ",
        side="Buy",
        quantity=10.0,
        client_order_id="coid-group-1",
    )

    assert runner._client_to_group_ids["coid-group-1"] == "coid-group-1"


def test_live_runner_lookup_group_id_falls_back_to_broker_order_id() -> None:
    """Payload 无 client_order_id 时, 经 broker_order_id 反查再取 group_id."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    runner._sync_group_mapping("c1-open-2", "c1")
    runner._broker_to_client_order_ids["b2"] = "c1-open-2"

    assert runner._lookup_group_id({"broker_order_id": "b2"}) == "c1"
    assert runner._lookup_group_id({"client_order_id": "c1-open-2"}) == "c1"
    # 未登记映射时退化为 client_order_id 本身（单腿场景 group_id==root cid）。
    assert runner._lookup_group_id({"client_order_id": "unmapped"}) == "unmapped"


def test_live_runner_emits_order_and_trade_with_group_id() -> None:
    """order/trade 广播回填 group_id, 供策略按逻辑单据聚合分腿成交."""

    class _DummyStrategy:
        def __init__(self) -> None:
            self.orders: list[Any] = []
            self.trades: list[Any] = []

        def on_order(self, order: Any) -> None:
            self.orders.append(order)

        def on_trade(self, trade: Any) -> None:
            self.trades.append(trade)

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    runner._sync_group_mapping("coid-leg-2", "coid-root-1")
    strategy = _DummyStrategy()

    order_payload = {
        "client_order_id": "coid-leg-2",
        "broker_order_id": "b-leg-2",
        "symbol": "000001.SZ",
        "status": "Submitted",
    }
    runner._queue_broker_event("order", order_payload)
    runner._drain_broker_events(cast(Any, strategy))

    trade_payload = {
        "trade_id": "t1",
        "client_order_id": "coid-leg-2",
        "broker_order_id": "b-leg-2",
        "symbol": "000001.SZ",
        "side": "Buy",
        "quantity": 1.0,
        "price": 10.0,
        "timestamp_ns": 1,
    }
    runner._queue_broker_event("trade", trade_payload)
    runner._drain_broker_events(cast(Any, strategy))

    assert strategy.orders
    assert strategy.orders[0].group_id == "coid-root-1"
    assert strategy.trades
    assert strategy.trades[0].group_id == "coid-root-1"


def test_group_id_survives_terminal_cleanup_for_open_leg() -> None:
    """乱序到达场景 (N1 回归): 终态 ORDER 先清理映射, 末笔 TRADE 仍需正确关联 group_id.

    反手 open 腿的终态 ORDER/execution-report 事件先到达并触发
    _close_order_mapping 清理, 随后到达的 TRADE 事件仍应能通过
    _client_to_group_ids 解析出根 group_id, 而不是退化为该腿自己的
    client_order_id (_lookup_group_id 的 get(cid, cid) 兜底)。
    """
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()

    # 2 腿组: 根腿 "root" 与反手 open 腿 "root-open-2", 均归组到 "root"。
    runner._sync_group_mapping("root", "root")
    runner._sync_group_mapping("root-open-2", "root")
    runner._sync_order_id_mapping("root-open-2", "B-open")

    # 模拟该 open 腿的终态 ORDER/execution-report 事件先被清理掉映射。
    runner._close_order_mapping("root-open-2", "B-open")

    # 随后到达同一条腿的末笔成交事件。
    trade = UnifiedTrade(
        trade_id="t-open-2",
        broker_order_id="B-open",
        client_order_id="root-open-2",
        symbol="000001.SZ",
        side="Buy",
        quantity=1.0,
        price=10.0,
        timestamp_ns=1,
    )
    adapted = runner._adapt_strategy_payload("trade", trade)

    assert adapted.group_id == "root"


def test_live_runner_emits_observable_broker_events_with_owner_strategy_id() -> None:
    """Emit broker event snapshots with resolved owner strategy id."""
    observed: list[dict[str, Any]] = []

    class _DummyStrategy:
        def __init__(self) -> None:
            self.orders: list[Any] = []

        def on_order(self, order: Any) -> None:
            self.orders.append(order)

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    runner.on_broker_event = observed.append
    runner._client_to_strategy_ids["coid-obs-1"] = "beta"
    strategy = _DummyStrategy()
    payload = {
        "client_order_id": "coid-obs-1",
        "broker_order_id": "b-obs-1",
        "status": "Submitted",
    }
    runner._queue_broker_event("order", payload)
    runner._drain_broker_events(cast(Any, strategy))

    assert observed
    event = observed[0]
    assert event["event_type"] == "order"
    assert event["owner_strategy_id"] == "beta"
    assert event["payload"]["client_order_id"] == "coid-obs-1"


def test_live_runner_logs_broker_event_observer_failures(caplog: Any) -> None:
    """Observer callback failures should be logged with broker context."""

    def _observer(_: dict[str, Any]) -> None:
        raise RuntimeError("observer failed")

    class _DummyStrategy:
        def on_order(self, order: Any) -> None:
            return None

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "miniqmt"
    runner._init_broker_bridge_state()
    runner.on_broker_event = _observer
    runner._client_to_strategy_ids["coid-obs-2"] = "gamma"
    strategy = _DummyStrategy()
    payload = {
        "client_order_id": "coid-obs-2",
        "broker_order_id": "b-obs-2",
        "symbol": "000001.SZ",
        "status": "Submitted",
    }

    with caplog.at_level("WARNING", logger="akquant.gateway.live"):
        runner._queue_broker_event("order", payload)
        runner._drain_broker_events(cast(Any, strategy))

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Broker event observer failed"
    )
    assert record.phase == "gateway"
    assert record.strategy_id == "gamma"
    assert record.slot == "gamma"
    assert record.symbol == "000001.SZ"
    assert record.order_id == "b-obs-2"
    assert record.client_order_id == "coid-obs-2"


def test_live_runner_logs_invalid_duration_format(caplog: Any) -> None:
    """Invalid live duration should be logged instead of printed."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_id = "alpha"

    with caplog.at_level("WARNING", logger="akquant.gateway.live"):
        runner._apply_time_limit(cast(Any, SimpleNamespace()), "not-a-duration")

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Ignored invalid live duration format: not-a-duration"
    )
    assert record.phase == "live"
    assert record.strategy_id == "alpha"
    assert record.slot == "alpha"


def test_live_runner_logs_summary_with_structured_context(caplog: Any) -> None:
    """Live summary should be emitted through the gateway logger."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_id = "beta"
    runner.engine = cast(
        Any,
        SimpleNamespace(
            get_results=lambda: SimpleNamespace(
                metrics=SimpleNamespace(
                    total_return_pct=0.12,
                    annualized_return=0.08,
                    max_drawdown_pct=-0.04,
                    sharpe_ratio=1.23,
                    win_rate=0.67,
                ),
                trades=[object(), object()],
                snapshots=[
                    (
                        0,
                        [
                            SimpleNamespace(symbol="IF2406", quantity=2.0),
                            SimpleNamespace(symbol="rb2406", quantity=0.0),
                        ],
                    )
                ],
            )
        ),
    )

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._print_summary()

    record = next(
        record for record in caplog.records if "TRADING SUMMARY" in record.getMessage()
    )
    assert record.phase == "live"
    assert record.strategy_id == "beta"
    assert record.slot == "beta"
    assert "Current Positions:" in record.getMessage()
    assert "IF2406: 2.0" in record.getMessage()


def _summary_runner_with_bridge(dropped_event_counts: Callable[[], Any]) -> LiveRunner:
    """Build a ``_print_summary``-ready runner stub with a fake broker event bridge."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_id = "beta"
    runner._broker_event_bridge = SimpleNamespace(
        dropped_event_counts=dropped_event_counts
    )
    runner.engine = cast(
        Any,
        SimpleNamespace(
            get_results=lambda: SimpleNamespace(
                metrics=SimpleNamespace(
                    total_return_pct=0.12,
                    annualized_return=0.08,
                    max_drawdown_pct=-0.04,
                    sharpe_ratio=1.23,
                    win_rate=0.67,
                ),
                trades=[object(), object()],
                snapshots=[],
            )
        ),
    )
    return runner


def test_live_runner_summary_reports_dropped_event_counts(caplog: Any) -> None:
    """摘要在 Total Trades 之后、结尾分隔线之前插入两行丢弃计数."""
    runner = _summary_runner_with_bridge(
        lambda: {"foreign_symbol": 3, "duplicate_order": 5}
    )

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._print_summary()

    record = next(
        record for record in caplog.records if "TRADING SUMMARY" in record.getMessage()
    )
    message = record.getMessage()
    total_trades_pos = message.index("Total Trades")
    foreign_pos = message.index("Dropped (foreign symbol): 3")
    duplicate_pos = message.index("Dropped (duplicate order): 5")
    trailing_rule_pos = message.rindex("=" * 50)

    assert total_trades_pos < foreign_pos < duplicate_pos < trailing_rule_pos


def test_live_runner_summary_survives_dropped_event_counts_exception(
    caplog: Any,
) -> None:
    """``dropped_event_counts()`` 抛异常时摘要主体仍完整输出, 只丢计数两行."""

    def boom() -> dict[str, int]:
        raise RuntimeError("dropped_event_counts exploded")

    runner = _summary_runner_with_bridge(boom)

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._print_summary()

    record = next(
        record for record in caplog.records if "TRADING SUMMARY" in record.getMessage()
    )
    message = record.getMessage()
    assert "Total Trades: 2" in message
    assert "Sharpe Ratio: 1.2300" in message
    assert "Dropped (foreign symbol)" not in message
    assert "Dropped (duplicate order)" not in message


def test_live_runner_summary_survives_dropped_event_counts_bad_shape(
    caplog: Any,
) -> None:
    """``dropped_event_counts()`` 不抛异常但返回非 dict-like 时摘要主体仍完整.

    格式化 ``counts.get(...)`` 必须与「读」共用同一个 try: 否则返回值格式不对
    (如 list)会在 ``else`` 分支里抛 ``AttributeError``, 冒泡到最外层大 try,
    造成与「读抛异常」完全相同的后果——整段 TRADING SUMMARY 一起丢失。
    """
    runner = _summary_runner_with_bridge(lambda: ["not", "a", "dict"])

    with caplog.at_level("INFO", logger="akquant.gateway.live"):
        runner._print_summary()

    record = next(
        record for record in caplog.records if "TRADING SUMMARY" in record.getMessage()
    )
    message = record.getMessage()
    assert "Total Trades: 2" in message
    assert "Sharpe Ratio: 1.2300" in message
    assert "Dropped (foreign symbol)" not in message
    assert "Dropped (duplicate order)" not in message
