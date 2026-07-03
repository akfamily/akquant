from akquant.gateway.broker_models import UnifiedOrderStatus
from akquant.gateway.brokers.qmf import mapper


def test_parse_account() -> None:
    """Funds query maps into UnifiedAccount cash/equity fields."""
    acct = mapper.parse_account(
        {
            "fund_account": "8888000001",
            "asset_balance": "1500000.00",
            "current_balance": "1000000.00",
            "enable_balance": "850000.00",
        }
    )
    assert acct.account_id == "8888000001"
    assert acct.equity == 1500000.0
    assert acct.cash == 1000000.0
    assert acct.available_cash == 850000.0


def test_parse_position() -> None:
    """A position row maps into UnifiedPosition with reconstructed symbol."""
    pos = mapper.parse_position(
        {
            "exchange_type": "1",
            "stock_code": "600000",
            "current_amount": "1000",
            "enable_amount": "1000",
            "cost_price": "10.20",
        }
    )
    assert pos.symbol == "600000.SH"
    assert pos.quantity == 1000.0
    assert pos.available_quantity == 1000.0
    assert pos.avg_price == 10.20
    assert pos.direction == "long"


def test_parse_order() -> None:
    """An order row maps into UnifiedOrderSnapshot with mapped status."""
    snap = mapper.parse_order(
        {
            "entrust_no": "100000001",
            "exchange_type": "1",
            "stock_code": "600000",
            "entrust_status": "8",
            "business_amount": "100",
            "business_price": "10.50",
            "error_no": "0",
        },
        client_order_id="c1",
    )
    assert snap.broker_order_id == "100000001"
    assert snap.client_order_id == "c1"
    assert snap.symbol == "600000.SH"
    assert snap.status == UnifiedOrderStatus.FILLED
    assert snap.filled_quantity == 100.0
    assert snap.avg_fill_price == 10.50


def test_parse_trade() -> None:
    """A trade row maps into UnifiedTrade with side/price/quantity."""
    trade = mapper.parse_trade(
        {
            "serial_no": "T1",
            "entrust_no": "100000001",
            "exchange_type": "1",
            "stock_code": "600000",
            "entrust_bs": "1",
            "business_amount": "100",
            "business_price": "10.50",
        },
        client_order_id="c1",
    )
    assert trade.trade_id == "T1"
    assert trade.broker_order_id == "100000001"
    assert trade.client_order_id == "c1"
    assert trade.symbol == "600000.SH"
    assert trade.side == "Buy"
    assert trade.quantity == 100.0
    assert trade.price == 10.50
