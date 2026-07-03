import pytest
from akquant.gateway.broker_models import BrokerCapability, UnifiedOrderRequest
from akquant.gateway.order_submitter import BrokerOrderSubmitter


class _Strategy:
    """Ready strategy stub."""

    _owner_strategy_id = "_default"
    broker_ready = True


def _submitter(captured: list) -> BrokerOrderSubmitter:
    """Build a submitter whose gateway records placed requests."""

    class _Gw:
        def place_order(self, req: UnifiedOrderRequest) -> str:
            captured.append(req)
            return "b1"

    cap = BrokerCapability(broker_name="qmf")
    return BrokerOrderSubmitter(
        trader_gateway=_Gw(),
        strategy=_Strategy(),
        resolve_trader_capabilities=lambda _gw: cap,
        next_client_order_id=lambda: "c1",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda *_a, **_k: None,
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: cap.as_execution_capabilities(),
    )


def test_accepts_full_signature_with_sim_knobs_none() -> None:
    """The full backtest signature (sim knobs None) reaches place_order."""
    captured: list = []
    sub = _submitter(captured)
    bid = sub.submit_order(
        symbol="600000.SH",
        side="Buy",
        quantity=100,
        price=10.5,
        order_type="Limit",
        fill_policy=None,
        slippage=None,
        commission=None,
        trail_offset=None,
        trail_reference_price=None,
        broker_options=None,
    )
    assert bid == "b1"
    assert len(captured) == 1


def test_sim_knobs_ignored_with_warning(caplog) -> None:
    """Non-None sim knobs are ignored (order still placed) and warned."""
    import logging

    captured: list = []
    sub = _submitter(captured)
    with caplog.at_level(logging.WARNING):
        sub.submit_order(
            symbol="600000.SH",
            side="Buy",
            quantity=100,
            price=10.5,
            order_type="Limit",
            slippage=0.01,
            commission={"rate": 0.001},
        )
    assert len(captured) == 1
    assert any(
        "slippage" in r.message or "commission" in r.message for r in caplog.records
    )


def test_trigger_price_rejected() -> None:
    """A stop/conditional order (trigger_price) is clearly rejected."""
    sub = _submitter([])
    with pytest.raises(RuntimeError, match="条件|触发|trigger"):
        sub.submit_order(
            symbol="600000.SH",
            side="Sell",
            quantity=100,
            price=10.0,
            order_type="Limit",
            trigger_price=9.5,
        )


def test_trailing_rejected() -> None:
    """A trailing-stop order is clearly rejected."""
    sub = _submitter([])
    with pytest.raises(RuntimeError, match="追踪|trail"):
        sub.submit_order(
            symbol="600000.SH",
            side="Sell",
            quantity=100,
            order_type="StopTrail",
            trail_offset=0.2,
        )


def test_buy_convenience_path_no_typeerror() -> None:
    """buy() forwarding sim-knob kwargs no longer TypeErrors in broker_live."""
    from akquant import strategy_trading_api as api

    captured: list = []
    strategy = _Strategy()
    strategy.submit_order = _submitter(captured).submit_order
    result = api.buy(strategy, symbol="600000.SH", quantity=100, price=10.5)
    assert result == "b1"
    assert len(captured) == 1


def test_order_type_none_defaults_to_market() -> None:
    """order_type=None (as buy()/sell() forward) becomes the documented default."""
    captured: list = []
    sub = _submitter(captured)
    sub.submit_order(
        symbol="600000.SH", side="Buy", quantity=100, price=10.5, order_type=None
    )
    assert captured[0].order_type == "Market"
