from typing import Any, Callable

import pytest
from akquant.gateway.broker_models import BrokerCapability, UnifiedPosition
from akquant.gateway.order_submitter import BrokerOrderSubmitter


class _DummyStrategy:
    def __init__(self) -> None:
        self._owner_strategy_id = "alpha"
        self.errors: list[tuple[str, Any]] = []

    def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
        self.errors.append((source, payload))


def _build_submitter(
    trader_gateway: Any,
    strategy: _DummyStrategy | None = None,
    *,
    can_submit_client_order: Callable[[str], bool] | None = None,
) -> tuple[BrokerOrderSubmitter, _DummyStrategy, dict[str, str], dict[str, str]]:
    strategy = strategy or _DummyStrategy()
    client_to_broker: dict[str, str] = {}
    broker_to_owner: dict[str, str] = {}
    capability = BrokerCapability(
        broker_name="ctp",
        broker_live=True,
        client_order_id=True,
        order_type=True,
        time_in_force_str=True,
        position_effect=True,
        reduce_only=True,
        position_details=True,
        supported_position_effects=(
            "auto",
            "open",
            "close",
            "close_today",
            "close_yesterday",
        ),
    )
    submitter = BrokerOrderSubmitter(
        trader_gateway=trader_gateway,
        strategy=strategy,
        resolve_trader_capabilities=lambda _: capability,
        next_client_order_id=lambda: "auto-coid-1",
        can_submit_client_order=can_submit_client_order or (lambda _: True),
        sync_order_id_mapping=lambda client_order_id, broker_order_id: (
            client_to_broker.__setitem__(client_order_id, broker_order_id)
        ),
        bind_order_owner=lambda client_order_id, broker_order_id, owner_strategy_id: (
            broker_to_owner.__setitem__(broker_order_id, owner_strategy_id)
        ),
        notify_strategy_error=lambda target, error, source, payload: target.on_error(
            error,
            source,
            payload,
        ),
        payload_field=lambda payload, field: (
            payload.get(field, "")
            if isinstance(payload, dict)
            else getattr(payload, field, "")
        ),
        get_execution_capabilities=lambda: capability.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
    )
    submitter.install()
    return submitter, strategy, client_to_broker, broker_to_owner


def test_submitter_contract_injects_strategy_api_and_maps_order_ids() -> None:
    """Submitter should inject strategy API and keep order id mapping consistent."""

    class _DummyTraderGateway:
        def place_order(self, req: Any) -> str:
            return f"b-{req.client_order_id}"

    submitter, _strategy, client_to_broker, broker_to_owner = _build_submitter(
        _DummyTraderGateway()
    )

    broker_order_id = submitter.submit_order(
        symbol="IF2406",
        side="Buy",
        quantity=1.0,
        client_order_id="coid-1",
    )

    assert broker_order_id == "b-coid-1"
    assert submitter._can_submit_client_order("free-coid") is True
    assert submitter._get_execution_capabilities()["broker_live"] is True
    assert client_to_broker == {"coid-1": "b-coid-1"}
    assert broker_to_owner == {"b-coid-1": "alpha"}


def test_submitter_contract_rejects_duplicate_active_client_order_id() -> None:
    """Submitter should reject duplicate active client_order_id before placement."""

    class _DummyTraderGateway:
        def place_order(self, req: Any) -> str:
            raise AssertionError(
                f"place_order should not be called: {req.client_order_id}"
            )

    submitter, strategy, _, _ = _build_submitter(
        _DummyTraderGateway(),
        can_submit_client_order=lambda client_order_id: client_order_id != "coid-dup",
    )

    with pytest.raises(
        RuntimeError, match="duplicate active client_order_id: coid-dup"
    ):
        submitter.submit_order(
            symbol="IF2406",
            side="Buy",
            quantity=1.0,
            client_order_id="coid-dup",
        )

    assert strategy.errors
    assert strategy.errors[0][0] == "submit_order"
    assert strategy.errors[0][1]["client_order_id"] == "coid-dup"


def test_submitter_contract_splits_close_legs_when_position_details_available() -> None:
    """Submitter should split close legs into today/yesterday."""

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

    gateway = _DummyTraderGateway()
    submitter, _strategy, client_to_broker, _ = _build_submitter(gateway)

    broker_order_id = submitter.submit_order(
        symbol="au2606",
        side="Sell",
        quantity=4.0,
        client_order_id="coid-close",
        position_effect="close",
    )

    assert broker_order_id == "b-coid-close"
    assert len(gateway.requests) == 2
    assert gateway.requests[0].position_effect == "close_today"
    assert gateway.requests[0].quantity == 2.0
    assert gateway.requests[1].client_order_id == "coid-close-close-yesterday-2"
    assert gateway.requests[1].position_effect == "close_yesterday"
    assert gateway.requests[1].quantity == 2.0
    assert client_to_broker == {
        "coid-close": "b-coid-close",
        "coid-close-close-yesterday-2": "b-coid-close-close-yesterday-2",
    }
