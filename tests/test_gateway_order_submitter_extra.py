import pytest
from akquant.gateway.broker_models import BrokerCapability, UnifiedOrderRequest
from akquant.gateway.order_submitter import BrokerOrderSubmitter


class _Strategy:
    """Bare strategy object for id/owner attributes."""

    _owner_strategy_id = "_default"


def _make_submitter(
    captured: list, capability: BrokerCapability
) -> BrokerOrderSubmitter:
    """Build a submitter whose gateway records the placed request."""

    class _Gw:
        def place_order(self, req: UnifiedOrderRequest) -> str:
            captured.append(req)
            return "b1"

    return BrokerOrderSubmitter(
        trader_gateway=_Gw(),
        strategy=_Strategy(),
        resolve_trader_capabilities=lambda _gw: capability,
        next_client_order_id=lambda: "c1",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda *_a, **_k: None,
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: capability.as_execution_capabilities(),
    )


def test_submit_with_declared_extra_reaches_place_order() -> None:
    """Declared extra + asset_type are normalized onto the request."""
    captured: list = []
    cap = BrokerCapability(broker_name="qmf", broker_extra_fields=("entrust_oc",))
    sub = _make_submitter(captured, cap)
    sub.submit_order(
        symbol="10004321.SH",
        side="Buy",
        quantity=1,
        price=0.05,
        order_type="Limit",
        asset_type="opt",
        extra={"entrust_oc": "O"},
    )
    assert len(captured) == 1
    assert captured[0].asset_type == "option"  # 归一
    assert captured[0].extra == {"entrust_oc": "O"}


def test_submit_with_undeclared_extra_raises() -> None:
    """Undeclared extra keys are rejected."""
    captured: list = []
    cap = BrokerCapability(broker_name="qmf", broker_extra_fields=())
    sub = _make_submitter(captured, cap)
    with pytest.raises(RuntimeError):
        sub.submit_order(
            symbol="600000.SH",
            side="Buy",
            quantity=100,
            price=10.5,
            order_type="Limit",
            extra={"nope": "1"},
        )
