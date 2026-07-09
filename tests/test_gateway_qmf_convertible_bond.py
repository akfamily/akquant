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


def test_place_convertible_bond_order_write() -> None:
    """Place a convertible-bond order with required fields; return the data dict."""
    client, seen = _capture({"entrust_no": "100000009"})
    result = client.place_convertible_bond_order(
        stock_code="113050",
        exchange_type="1",
        entrust_prop="0",
        entrust_amount="10",
    )
    assert seen["path"] == "/api/v1/trading/convertible-bond-order"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "stock_code": "113050",
        "exchange_type": "1",
        "entrust_prop": "0",
        "entrust_amount": "10",
    }
    assert result == {"entrust_no": "100000009"}


def test_place_convertible_bond_order_optional_fields() -> None:
    """Optional stock_account/stb_stock_property are forwarded only when provided."""
    client, seen = _capture({})
    client.place_convertible_bond_order(
        stock_code="113050",
        exchange_type="1",
        entrust_prop="0",
        entrust_amount="10",
        stock_account="A1",
        stb_stock_property="0",
    )
    assert seen["body"]["stock_account"] == "A1"
    assert seen["body"]["stb_stock_property"] == "0"


def test_cancel_convertible_bond_order_write() -> None:
    """Cancel a convertible-bond order by entrust_no; return the data dict."""
    client, seen = _capture({"error_no": "0"})
    result = client.cancel_convertible_bond_order("100000009")
    assert seen["path"] == "/api/v1/trading/convertible-bond-cancel"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "entrust_no": "100000009",
    }
    assert result == {"error_no": "0"}


def test_query_convertible_bond_orders_read() -> None:
    """Default sends only fund_account; optional filters forwarded when given."""
    client, seen = _capture([{"entrust_no": "100000009"}])
    result = client.query_convertible_bond_orders()
    assert seen["path"] == "/api/v1/account/convertible-bond-orders"
    assert seen["body"] == {"fund_account": "8888000001"}
    assert result == [{"entrust_no": "100000009"}]

    client2, seen2 = _capture([])
    client2.query_convertible_bond_orders(stock_code="113050", entrust_no="1")
    assert seen2["body"]["stock_code"] == "113050"
    assert seen2["body"]["entrust_no"] == "1"


def test_query_bond_putback_info_read() -> None:
    """Query bond putback info; optional stock_code forwarded only when provided."""
    client, seen = _capture([{"stock_code": "113050"}])
    result = client.query_bond_putback_info()
    assert seen["path"] == "/api/v1/account/bond-putback-info"
    assert seen["body"] == {"fund_account": "8888000001"}
    assert result == [{"stock_code": "113050"}]

    client2, seen2 = _capture([])
    client2.query_bond_putback_info(stock_code="113050")
    assert seen2["body"]["stock_code"] == "113050"
