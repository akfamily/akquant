import base64
import json

import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

import httpx
from akquant.gateway.brokers.qmf.client import QMFClientConfig, QMFHttpClient

_KEY = base64.b64encode(b"0" * 32).decode("ascii")


def _capture(data: object) -> tuple[QMFHttpClient, dict]:
    """Build a logged-in securities client recording path/body, returning data."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/login":
            return httpx.Response(
                200,
                json={
                    "result": "0",
                    "msg": "success",
                    "data": {"user_token": "gw-a", "fund_account": "8888000001"},
                },
            )
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"result": "0", "msg": "success", "data": data})

    cfg = QMFClientConfig(
        base_url="http://gw.test",
        qmf_user_id="u",
        account_content="8888000001",
        password="pw",
        input_content="1",
        content_type="1",
        password_key=_KEY,
    )
    client = QMFHttpClient(cfg, transport=httpx.MockTransport(handler))
    client.login()
    return client, seen


def test_place_composite_order_required_fields() -> None:
    """Composite order posts the 7 required fields plus fund_account; returns dict."""
    client, seen = _capture({"entrust_no": "C1"})
    result = client.place_composite_order(
        exchange_type="1",
        stock_account="A1",
        stock_code="600000",
        entrust_price="10.0",
        entrust_amount="100",
        entrust_prop="PFP",
        entrust_bs="1",
    )
    assert seen["path"] == "/api/v1/trading/composite-order"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "exchange_type": "1",
        "stock_account": "A1",
        "stock_code": "600000",
        "entrust_price": "10.0",
        "entrust_amount": "100",
        "entrust_prop": "PFP",
        "entrust_bs": "1",
    }
    assert result == {"entrust_no": "C1"}


def test_place_composite_order_extra_merged() -> None:
    """Optional fields provided via extra are merged into the payload."""
    client, seen = _capture({})
    client.place_composite_order(
        exchange_type="1",
        stock_account="A1",
        stock_code="600000",
        entrust_price="10.0",
        entrust_amount="100",
        entrust_prop="PFP",
        entrust_bs="1",
        extra={"cbpconfer_id": "AG9", "agreement_id": "P1"},
    )
    assert seen["body"]["cbpconfer_id"] == "AG9"
    assert seen["body"]["agreement_id"] == "P1"


def test_cancel_composite_order() -> None:
    """Composite cancel posts entrust_no; entrust_reference only when given."""
    client, seen = _capture({"error_no": "0"})
    result = client.cancel_composite_order("C1")
    assert seen["path"] == "/api/v1/trading/composite-cancel"
    assert seen["body"] == {"fund_account": "8888000001", "entrust_no": "C1"}
    assert result == {"error_no": "0"}

    client2, seen2 = _capture({})
    client2.cancel_composite_order("C1", entrust_reference="R9")
    assert seen2["body"]["entrust_reference"] == "R9"


def test_query_composite_orders_filters() -> None:
    """Composite orders query sends fund_account and forwards filters when given."""
    client, seen = _capture([{"entrust_no": "C1"}])
    result = client.query_composite_orders()
    assert seen["path"] == "/api/v1/account/composite-orders"
    assert seen["body"] == {"fund_account": "8888000001"}
    assert result == [{"entrust_no": "C1"}]

    client2, seen2 = _capture([])
    client2.query_composite_orders(stock_code="600000", query_kind="1")
    assert seen2["body"]["stock_code"] == "600000"
    assert seen2["body"]["query_kind"] == "1"


def test_query_composite_trades_filters() -> None:
    """Composite trades query sends fund_account and forwards filters when given."""
    client, seen = _capture([{"serial_no": "S1"}])
    result = client.query_composite_trades()
    assert seen["path"] == "/api/v1/account/composite-trades"
    assert seen["body"] == {"fund_account": "8888000001"}
    assert result == [{"serial_no": "S1"}]

    client2, seen2 = _capture([])
    client2.query_composite_trades(stock_code="600000")
    assert seen2["body"]["stock_code"] == "600000"


class _FakeSecClient:
    """Securities client stub recording composite calls."""

    token = "gw-sec"
    fund_account = "8888000001"

    def __init__(self) -> None:
        """Init the call log."""
        self.calls: list = []

    def place_composite_order(self, *args, **kwargs):
        """Stub composite order."""
        self.calls.append(("place", args, kwargs))
        return {"entrust_no": "C1"}

    def cancel_composite_order(self, entrust_no, entrust_reference=None):
        """Stub composite cancel."""
        self.calls.append(("cancel", entrust_no, entrust_reference))
        return {"error_no": "0"}

    def query_composite_orders(self, **filters):
        """Stub composite orders query."""
        self.calls.append(("orders", filters))
        return [{"m": "orders"}]

    def query_composite_trades(self, **filters):
        """Stub composite trades query."""
        self.calls.append(("trades", filters))
        return [{"m": "trades"}]

    def close(self):
        """Ignore close."""


def test_adapter_delegates_composite_without_option_session() -> None:
    """Composite convenience methods delegate to securities client, no option needed."""
    from akquant.gateway.brokers.qmf.adapter import QMFTraderGateway

    sec = _FakeSecClient()
    gw = QMFTraderGateway(client=sec, ws_url="ws://gw.test/api/v1/stream")

    assert gw.place_composite_order("1", "A1", "600000", "10.0", "100", "PFP", "1") == {
        "entrust_no": "C1"
    }
    assert gw.cancel_composite_order("C1") == {"error_no": "0"}
    assert gw.query_composite_orders(stock_code="600000") == [{"m": "orders"}]
    assert gw.query_composite_trades() == [{"m": "trades"}]

    assert sec.calls[0][0] == "place"
    assert ("cancel", "C1", None) in sec.calls
    assert ("orders", {"stock_code": "600000"}) in sec.calls
