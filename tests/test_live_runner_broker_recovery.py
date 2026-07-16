from types import SimpleNamespace
from typing import Any, Callable, cast

from akquant.live import LiveRunner


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
