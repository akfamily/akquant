import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

from akquant.gateway.broker_models import (
    UnifiedOrderRequest,
    UnifiedOrderStatus,
)
from akquant.gateway.brokers.qmf.adapter import QMFTraderGateway


class _FakeClient:
    def __init__(self):
        self.fund_account = "8888000001"
        self.placed = []
        self.cancelled = []

    def place_order(self, fields):
        self.placed.append(fields)
        return {"entrust_no": "100000001", "error_no": "0"}

    def cancel_order(self, entrust_no, exchange_type=None):
        self.cancelled.append(entrust_no)
        return {"error_no": "0"}

    def query_funds(self):
        return {
            "fund_account": "8888000001",
            "asset_balance": "1500000.00",
            "current_balance": "1000000.00",
            "enable_balance": "850000.00",
        }

    def query_positions(self):
        return [
            {
                "exchange_type": "1",
                "stock_code": "600000",
                "current_amount": "1000",
                "enable_amount": "1000",
                "cost_price": "10.2",
            }
        ]

    def query_orders(self):
        return [
            {
                "entrust_no": "100000001",
                "exchange_type": "1",
                "stock_code": "600000",
                "entrust_status": "1",
                "business_amount": "0",
                "business_price": "0",
                "error_no": "0",
            }
        ]

    def query_trades(self):
        return [
            {
                "serial_no": "T1",
                "entrust_no": "100000001",
                "exchange_type": "1",
                "stock_code": "600000",
                "entrust_bs": "1",
                "business_amount": "100",
                "business_price": "10.5",
            }
        ]

    def auth_status(self, keepalive=True):
        return True


def _gateway() -> QMFTraderGateway:
    """Build a trader gateway backed by the fake client."""
    return QMFTraderGateway(client=_FakeClient(), ws_url="ws://gw.test/api/v1/stream")


def _limit_buy() -> UnifiedOrderRequest:
    """Build a canonical limit buy order request."""
    return UnifiedOrderRequest(
        client_order_id="c1",
        symbol="600000.SH",
        side="Buy",
        quantity=100,
        price=10.5,
        order_type="Limit",
    )


def test_place_order_returns_entrust_no_and_maps_id() -> None:
    """place_order returns entrust_no and records the id reverse-map."""
    gw = _gateway()
    broker_id = gw.place_order(_limit_buy())
    assert broker_id == "100000001"
    assert gw._client_id_by_broker["100000001"] == "c1"


def test_push_trade_dispatches_on_trade() -> None:
    """A trade push resolves the client_order_id and fires on_trade."""
    gw = _gateway()
    gw.place_order(_limit_buy())
    trades = []
    gw.on_trade(trades.append)
    gw._dispatch_push(
        "trade_update",
        {
            "serial_no": "T1",
            "entrust_no": "100000001",
            "exchange_type": "1",
            "stock_code": "600000",
            "entrust_bs": "1",
            "business_amount": "100",
            "business_price": "10.5",
        },
    )
    assert len(trades) == 1
    assert trades[0].client_order_id == "c1"
    assert trades[0].symbol == "600000.SH"


def test_query_account_and_positions() -> None:
    """query_account/query_positions map counter data to Unified models."""
    gw = _gateway()
    acct = gw.query_account()
    assert acct.available_cash == 850000.0
    positions = gw.query_positions()
    assert positions[0].symbol == "600000.SH"


def test_sync_open_orders() -> None:
    """sync_open_orders returns snapshots from the orders query."""
    gw = _gateway()
    snaps = gw.sync_open_orders()
    assert snaps[0].broker_order_id == "100000001"
    assert snaps[0].status == UnifiedOrderStatus.SUBMITTED


def test_capabilities() -> None:
    """Capabilities advertise the Phase 1 securities matrix."""
    gw = _gateway()
    cap = gw.get_capabilities()
    assert cap.broker_name == "qmf"
    assert cap.position_effect is False
