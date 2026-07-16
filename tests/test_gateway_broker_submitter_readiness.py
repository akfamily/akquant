import pytest
from akquant.gateway.broker_models import BrokerCapability, UnifiedOrderRequest
from akquant.gateway.order_submitter import BrokerOrderSubmitter


class _Strategy:
    """Strategy stub carrying broker_ready."""

    _owner_strategy_id = "_default"

    def __init__(self, ready: bool) -> None:
        """Set readiness flag."""
        self.broker_ready = ready


def _submitter(strategy: _Strategy) -> BrokerOrderSubmitter:
    """Build a submitter over a no-op gateway."""

    class _Gw:
        def place_order(self, req: UnifiedOrderRequest) -> str:
            return "b1"

    cap = BrokerCapability(broker_name="qmf")
    return BrokerOrderSubmitter(
        trader_gateway=_Gw(),
        strategy=strategy,
        resolve_trader_capabilities=lambda _gw: cap,
        next_client_order_id=lambda: "c1",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda *_a, **_k: None,
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: cap.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
    )


def test_submit_before_ready_raises_clear_error() -> None:
    """Submitting before broker_ready raises a clear RuntimeError."""
    sub = _submitter(_Strategy(ready=False))
    with pytest.raises(RuntimeError, match="就绪|ready"):
        sub.submit_order(
            symbol="600000.SH",
            side="Buy",
            quantity=100,
            price=10.5,
            order_type="Limit",
        )


def test_submit_when_ready_ok() -> None:
    """Submitting when ready returns the broker order id."""
    sub = _submitter(_Strategy(ready=True))
    bid = sub.submit_order(
        symbol="600000.SH",
        side="Buy",
        quantity=100,
        price=10.5,
        order_type="Limit",
    )
    assert bid.primary == "b1"
