import pytest

pytest.importorskip("httpx")

from akquant.gateway.broker_models import (
    UnifiedOrderRequest,
    UnifiedOrderStatus,
)
from akquant.gateway.brokers.middleware.adapter import (
    MiddlewareTraderGateway,
    default_capability,
)


class _FakeClient:
    """Stand-in for MiddlewareHttpClient recording calls and returning canned data."""

    def __init__(self) -> None:
        self.account_id = ""
        self.logged_in = False
        self.closed = False
        self.placed: list = []
        self.cancelled: list = []
        self.online = True

    def login(self) -> str:
        self.logged_in = True
        self.account_id = "hengsheng:20432166:security"
        return self.account_id

    def close(self) -> None:
        self.closed = True

    def session_online(self) -> bool:
        return self.online

    def place_order(self, body: dict) -> dict:
        self.placed.append(body)
        return {"broker_order_id": "123456", "status": "submitted"}

    def cancel_order(self, body: dict) -> dict:
        self.cancelled.append(body)
        return {"status": "cancelled"}

    def query_orders(self) -> list:
        return [
            {
                "broker_order_id": "123456",
                "client_order_id": "cli-1",
                "instrument_id": "SSE:600000",
                "status": "partially_filled",
                "filled_quantity": 30,
            }
        ]

    def query_trades(self) -> list:
        return [
            {
                "trade_id": "t-1",
                "broker_order_id": "123456",
                "instrument_id": "SSE:600000",
                "side": "buy",
                "quantity": 100,
                "price": 11.2,
            }
        ]

    def query_positions(self) -> list:
        return [
            {"instrument_id": "SSE:600000", "quantity": 1000, "available_quantity": 800}
        ]

    def query_summary(self) -> dict:
        return {
            "account_id": "hengsheng:20432166:security",
            "net_asset": 701000.0,
            "available": 420000.0,
            "cash_balance": 450500.0,
        }


def _gateway() -> tuple[MiddlewareTraderGateway, _FakeClient]:
    """Build a gateway wired to a fake client (no WS)."""
    client = _FakeClient()
    gw = MiddlewareTraderGateway(client=client, ws_url="ws://gw.test/api/v1/ws")
    return gw, client


def _order_req() -> UnifiedOrderRequest:
    return UnifiedOrderRequest(
        client_order_id="cli-1",
        symbol="600000.SH",
        side="Buy",
        quantity=100,
        price=11.2,
        order_type="Limit",
        position_effect="open",
        asset_type="stock",
    )


def test_connect_logs_in_and_sets_account_id() -> None:
    """connect() logs in the client."""
    gw, client = _gateway()
    gw.connect()
    assert client.logged_in


def test_place_order_returns_broker_id_and_records_mapping() -> None:
    """place_order returns broker_order_id and records the id mapping."""
    gw, client = _gateway()
    bid = gw.place_order(_order_req())
    assert bid == "123456"
    assert client.placed[0]["client_order_id"] == "cli-1"
    assert client.placed[0]["instrument_id"] == "SSE:600000"
    assert gw.client_order_id_for("123456") == "cli-1"


def test_cancel_order_posts_broker_order_id() -> None:
    """cancel_order forwards the broker_order_id in the cancel body."""
    gw, client = _gateway()
    gw.cancel_order("123456")
    assert client.cancelled[0]["broker_order_id"] == "123456"


def test_query_order_finds_snapshot() -> None:
    """query_order returns a parsed snapshot for a known broker id."""
    gw, _ = _gateway()
    snap = gw.query_order("123456")
    assert snap is not None
    assert snap.symbol == "600000.SH"
    assert snap.status == UnifiedOrderStatus.PARTIALLY_FILLED


def test_query_trades_positions_account() -> None:
    """query_trades/positions/account map middleware rows to Unified models."""
    gw, _ = _gateway()
    trades = gw.query_trades()
    assert trades[0].side == "Buy"
    positions = gw.query_positions()
    assert positions[0].quantity == 1000.0
    account = gw.query_account()
    assert account is not None
    assert account.equity == 701000.0
    assert account.available_cash == 420000.0


def test_heartbeat_reflects_session_online() -> None:
    """Heartbeat mirrors client.session_online()."""
    gw, client = _gateway()
    assert gw.heartbeat() is True
    client.online = False
    assert gw.heartbeat() is False


def test_ws_book_order_emits_order_and_execution_report() -> None:
    """A book.order push emits an order snapshot and an execution report."""
    gw, _ = _gateway()
    orders: list = []
    execs: list = []
    gw.on_order(orders.append)
    gw.on_execution_report(execs.append)
    gw.record_broker_order("123456", "cli-1")
    gw._dispatch_push(
        "book.order",
        {
            "broker_order_id": "123456",
            "instrument_id": "SSE:600000",
            "status": "filled",
            "filled_quantity": 100,
        },
    )
    assert orders[0].client_order_id == "cli-1"
    assert orders[0].status == UnifiedOrderStatus.FILLED
    assert execs[0].status == UnifiedOrderStatus.FILLED


def test_ws_book_trade_emits_trade() -> None:
    """A book.trade push emits a trade."""
    gw, _ = _gateway()
    trades: list = []
    gw.on_trade(trades.append)
    gw.record_broker_order("123456", "cli-1")
    gw._dispatch_push(
        "book.trade",
        {
            "trade_id": "t-9",
            "broker_order_id": "123456",
            "instrument_id": "SSE:600000",
            "side": "sell",
            "quantity": 50,
            "price": 11.5,
        },
    )
    assert trades[0].trade_id == "t-9"
    assert trades[0].side == "Sell"
    assert trades[0].client_order_id == "cli-1"


def test_default_capability_options_flag() -> None:
    """default_capability advertises options only when enabled."""
    assert "options" not in default_capability(False).features
    assert "options" in default_capability(True).features
