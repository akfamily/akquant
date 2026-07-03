import pytest
from akquant.gateway.broker_models import UnifiedOrderRequest, UnifiedOrderStatus
from akquant.gateway.brokers.qmf import mapper


def test_build_option_order_payload() -> None:
    """An option buy-open maps to the OptOrderRequest fields from extra."""
    req = UnifiedOrderRequest(
        client_order_id="c1",
        symbol="10003456.SH",
        side="Buy",
        quantity=1,
        price=0.05,
        order_type="Limit",
        asset_type="option",
        extra={"entrust_oc": "O", "covered_flag": "0", "entrust_prop": "F0"},
    )
    payload = mapper.build_option_order_payload(req)
    assert payload == {
        "exchange_type": "1",
        "option_code": "10003456",
        "entrust_bs": "1",
        "entrust_oc": "O",
        "covered_flag": "0",
        "entrust_prop": "F0",
        "opt_entrust_price": "0.05",
        "entrust_amount": "1",
    }


def test_build_option_order_requires_entrust_oc_and_prop() -> None:
    """Missing entrust_oc or entrust_prop raises a clear error."""
    base = dict(
        client_order_id="c1",
        symbol="10003456.SH",
        side="Buy",
        quantity=1,
        price=0.05,
        order_type="Limit",
        asset_type="option",
    )
    with pytest.raises(ValueError):
        mapper.build_option_order_payload(
            UnifiedOrderRequest(**base, extra={"entrust_prop": "F0"})
        )
    with pytest.raises(ValueError):
        mapper.build_option_order_payload(
            UnifiedOrderRequest(**base, extra={"entrust_oc": "O"})
        )


def test_parse_option_order() -> None:
    """An option order row maps to a snapshot using opt_business_price."""
    snap = mapper.parse_option_order(
        {
            "entrust_no": "9000000001",
            "exchange_type": "1",
            "option_code": "10003456",
            "entrust_status": "2",
            "business_amount": "0",
            "opt_business_price": "0.0000",
            "error_no": "0",
        },
        client_order_id="c1",
    )
    assert snap.broker_order_id == "9000000001"
    assert snap.client_order_id == "c1"
    assert snap.symbol == "10003456.SH"
    assert snap.status == UnifiedOrderStatus.PARTIALLY_FILLED


def test_parse_option_trade() -> None:
    """An option trade row maps using serial_no + opt_business_price."""
    trade = mapper.parse_option_trade(
        {
            "serial_no": "T0000001",
            "entrust_no": "9000000001",
            "exchange_type": "1",
            "option_code": "10003456",
            "entrust_bs": "1",
            "business_amount": "1",
            "opt_business_price": "0.0500",
        },
        client_order_id="c1",
    )
    assert trade.trade_id == "T0000001"
    assert trade.symbol == "10003456.SH"
    assert trade.side == "Buy"
    assert trade.quantity == 1.0
    assert trade.price == 0.05


def test_parse_option_position() -> None:
    """An option position row maps using enable_amount + opt_cost_price."""
    pos = mapper.parse_option_position(
        {
            "exchange_type": "1",
            "option_code": "10003456",
            "current_amount": "1",
            "enable_amount": "1",
            "opt_cost_price": "0.0500",
        }
    )
    assert pos.symbol == "10003456.SH"
    assert pos.quantity == 1.0
    assert pos.available_quantity == 1.0
    assert pos.avg_price == 0.05
