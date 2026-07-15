from akquant.gateway.broker_models import (
    UnifiedOrderRequest,
    UnifiedOrderStatus,
)
from akquant.gateway.brokers.middleware import mapper


def test_symbol_to_instrument_stock_and_option() -> None:
    """Map akquant symbol to middleware instrument_id by market + asset type."""
    assert mapper.symbol_to_instrument("600000.SH", "stock") == "SSE:600000"
    assert mapper.symbol_to_instrument("000001.SZ", "stock") == "SZSE:000001"
    assert mapper.symbol_to_instrument("10003456.SH", "option") == "SSE_OPT:10003456"


def test_instrument_to_symbol_roundtrip() -> None:
    """Map middleware instrument_id back to akquant symbol (stock + option)."""
    assert mapper.instrument_to_symbol("SSE:600000") == "600000.SH"
    assert mapper.instrument_to_symbol("SZSE:000001") == "000001.SZ"
    assert mapper.instrument_to_symbol("SSE_OPT:10003456") == "10003456.SH"


def test_map_status_covers_middleware_values() -> None:
    """Map middleware order statuses to UnifiedOrderStatus with safe default."""
    assert mapper.map_status("submitted") == UnifiedOrderStatus.SUBMITTED
    assert mapper.map_status("partially_filled") == UnifiedOrderStatus.PARTIALLY_FILLED
    assert mapper.map_status("filled") == UnifiedOrderStatus.FILLED
    assert mapper.map_status("partially_cancelled") == UnifiedOrderStatus.CANCELLED
    assert mapper.map_status("cancelled") == UnifiedOrderStatus.CANCELLED
    assert mapper.map_status("rejected") == UnifiedOrderStatus.REJECTED
    assert mapper.map_status("pending") == UnifiedOrderStatus.NEW
    assert mapper.map_status("weird") == UnifiedOrderStatus.SUBMITTED


def test_build_order_body_stock_limit() -> None:
    """Build the /orders body from a stock limit UnifiedOrderRequest."""
    req = UnifiedOrderRequest(
        client_order_id="cli-1",
        symbol="600000.SH",
        side="Buy",
        quantity=100,
        price=11.2,
        order_type="Limit",
        position_effect="open",
        asset_type="stock",
    )
    body = mapper.build_order_body(req)
    assert body == {
        "client_order_id": "cli-1",
        "instrument_id": "SSE:600000",
        "side": "buy",
        "offset": "open",
        "order_type": "limit",
        "quantity": 100,
        "price": 11.2,
        "time_in_force": "GTC",
        "legs": [],
    }


def test_build_order_body_market_omits_price_and_forwards_extra() -> None:
    """Market order omits price; extra is forwarded opaquely."""
    req = UnifiedOrderRequest(
        client_order_id="cli-2",
        symbol="000001.SZ",
        side="sell",
        quantity=200,
        price=None,
        order_type="Market",
        position_effect="close",
        asset_type="stock",
        extra={"business_id": "stock.sell"},
    )
    body = mapper.build_order_body(req)
    assert "price" not in body
    assert body["side"] == "sell"
    assert body["offset"] == "close"
    assert body["order_type"] == "market"
    assert body["extra"] == {"business_id": "stock.sell"}


def test_parse_order_snapshot() -> None:
    """Parse a middleware Order into UnifiedOrderSnapshot."""
    snap = mapper.parse_order(
        {
            "order_id": "hengsheng:1:security:123456",
            "broker_order_id": "123456",
            "client_order_id": "cli-1",
            "instrument_id": "SSE:600000",
            "status": "partially_filled",
            "filled_quantity": 30,
            "status_msg": "",
        }
    )
    assert snap.broker_order_id == "123456"
    assert snap.client_order_id == "cli-1"
    assert snap.symbol == "600000.SH"
    assert snap.status == UnifiedOrderStatus.PARTIALLY_FILLED
    assert snap.filled_quantity == 30.0


def test_parse_trade() -> None:
    """Parse a middleware trade into UnifiedTrade with capitalized side."""
    trade = mapper.parse_trade(
        {
            "trade_id": "t-1",
            "broker_order_id": "123456",
            "client_order_id": "cli-1",
            "instrument_id": "SSE:600000",
            "side": "buy",
            "quantity": 100,
            "price": 11.2,
        }
    )
    assert trade.trade_id == "t-1"
    assert trade.symbol == "600000.SH"
    assert trade.side == "Buy"
    assert trade.quantity == 100.0
    assert trade.price == 11.2


def test_parse_position_and_account() -> None:
    """Parse position row and summary into Unified models."""
    pos = mapper.parse_position(
        {"instrument_id": "SSE:600000", "quantity": 1000, "available_quantity": 800}
    )
    assert pos.symbol == "600000.SH"
    assert pos.quantity == 1000.0
    assert pos.available_quantity == 800.0

    acct = mapper.parse_account(
        {
            "account_id": "hengsheng:1:security",
            "net_asset": 701000.0,
            "available": 420000.0,
            "cash_balance": 450500.0,
        }
    )
    assert acct.account_id == "hengsheng:1:security"
    assert acct.equity == 701000.0
    assert acct.available_cash == 420000.0
    assert acct.cash == 450500.0
