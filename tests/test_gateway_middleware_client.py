import json

import pytest

pytest.importorskip("httpx")

import httpx
from akquant.gateway.brokers.middleware.client import (
    MiddlewareApiError,
    MiddlewareClientConfig,
    MiddlewareHttpClient,
)


def _envelope(data: object, success: bool = True, code: str = "0", msg: str = "ok"):
    """Wrap data in the middleware unified envelope."""
    return httpx.Response(
        200, json={"success": success, "code": code, "msg": msg, "data": data}
    )


def _client(handler, token: str = "") -> MiddlewareHttpClient:
    """Build a logged-in middleware client bound to a mock transport."""
    cfg = MiddlewareClientConfig(
        base_url="http://gw.test/api/v1",
        broker_id="hengsheng",
        fund_account="20432166",
        password="pw",
        account_type="security",
        qmf_user_id="u1",
        token=token,
    )
    client = MiddlewareHttpClient(cfg, transport=httpx.MockTransport(handler))
    return client


def test_login_posts_sessions_and_stores_account_id() -> None:
    """login() posts /sessions with the standard body and stores account_id."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return _envelope({"account": {"account_id": "hengsheng:20432166:security"}})

    client = _client(handler)
    account_id = client.login()
    assert seen["path"] == "/api/v1/sessions"
    assert seen["body"] == {
        "broker_id": "hengsheng",
        "fund_account": "20432166",
        "password": "pw",
        "account_type": "security",
        "qmf_user_id": "u1",
        "extra": {},
    }
    assert account_id == "hengsheng:20432166:security"
    assert client.account_id == "hengsheng:20432166:security"


def test_business_failure_raises() -> None:
    """success=false envelope raises MiddlewareApiError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _envelope({}, success=False, code="E1", msg="拒单")

    client = _client(handler)
    with pytest.raises(MiddlewareApiError) as exc:
        client.login()
    assert exc.value.code == "E1"


def test_bearer_token_header_when_configured() -> None:
    """A configured token is sent as Authorization: Bearer."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return _envelope({"account": {"account_id": "hengsheng:1:security"}})

    client = _client(handler, token="jwt-123")
    client.login()
    assert seen["auth"] == "Bearer jwt-123"


def _logged_in(handler) -> MiddlewareHttpClient:
    """Return a client that has account_id set (skip real login round-trip)."""
    client = _client(handler)
    client.account_id = "hengsheng:20432166:security"
    return client


def test_place_order_posts_to_account_orders() -> None:
    """place_order posts body to /accounts/{id}/orders and returns data."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return _envelope({"broker_order_id": "123456", "status": "submitted"})

    client = _logged_in(handler)
    data = client.place_order(
        {"client_order_id": "cli-1", "instrument_id": "SSE:600000"}
    )
    assert seen["path"].endswith("/orders")
    assert "20432166" in seen["path"]
    assert seen["body"]["client_order_id"] == "cli-1"
    assert data["broker_order_id"] == "123456"


def test_cancel_order_posts_to_account_cancel() -> None:
    """cancel_order posts to /accounts/{id}/cancel."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return _envelope({"status": "cancelled"})

    client = _logged_in(handler)
    client.cancel_order({"broker_order_id": "123456"})
    assert seen["path"].endswith("/cancel")
    assert seen["body"]["broker_order_id"] == "123456"


@pytest.mark.parametrize(
    "method,tail,key",
    [
        ("query_positions", "/positions", "positions"),
        ("query_trades", "/trades", "trades"),
        ("query_orders", "/orders", "orders"),
    ],
)
def test_list_queries_unwrap_data_key(method: str, tail: str, key: str) -> None:
    """List queries GET /accounts/{id}/<x> and unwrap data[<key>]."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return _envelope({key: [{"row": "1"}]})

    client = _logged_in(handler)
    result = getattr(client, method)()
    assert seen["path"].endswith(tail)
    assert result == [{"row": "1"}]


def test_query_summary_returns_data_object() -> None:
    """query_summary GET /accounts/{id}/summary returns the data object."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/summary")
        return _envelope({"net_asset": 701000.0})

    client = _logged_in(handler)
    assert client.query_summary() == {"net_asset": 701000.0}
