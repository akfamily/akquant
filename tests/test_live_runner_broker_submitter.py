from typing import Any, cast

import pytest
from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.models import BrokerCapability, UnifiedPosition
from akquant.live._runner import LiveRunner


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
