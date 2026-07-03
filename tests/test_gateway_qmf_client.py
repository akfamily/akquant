import base64
import json

import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

import httpx
from akquant.gateway.brokers.qmf.client import (
    QMFApiError,
    QMFClientConfig,
    QMFHttpClient,
)

_KEY = base64.b64encode(b"0" * 32).decode("ascii")


def _config() -> QMFClientConfig:
    """Build a minimal client config for tests."""
    return QMFClientConfig(
        base_url="http://gw.test",
        qmf_user_id="u",
        account_content="8888000001",
        password="pw",
        input_content="1",
        content_type="1",
        password_key=_KEY,
    )


def _envelope(data: object) -> httpx.Response:
    """Wrap data in the chibi_quant success envelope."""
    return httpx.Response(200, json={"result": "0", "msg": "success", "data": data})


def test_login_stores_token_and_account() -> None:
    """Login encrypts the password and stores token + fund_account."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/auth/login"
        body = json.loads(request.content)
        assert body["password"] != "pw"  # 已加密
        return _envelope({"user_token": "gw-abc", "fund_account": "8888000001"})

    client = QMFHttpClient(_config(), transport=httpx.MockTransport(handler))
    data = client.login()
    assert data["user_token"] == "gw-abc"
    assert client.token == "gw-abc"
    assert client.fund_account == "8888000001"


def test_place_order_injects_bearer_and_account() -> None:
    """place_order sends the Bearer token and injects fund_account."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/login":
            return _envelope({"user_token": "gw-abc", "fund_account": "8888000001"})
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _envelope({"entrust_no": "100000001", "error_no": "0"})

    client = QMFHttpClient(_config(), transport=httpx.MockTransport(handler))
    client.login()
    data = client.place_order(
        {
            "exchange_type": "1",
            "stock_code": "600000",
            "entrust_bs": "1",
            "entrust_prop": "0",
            "entrust_price": "10.5",
            "entrust_amount": "100",
        }
    )
    assert data["entrust_no"] == "100000001"
    assert seen["auth"] == "Bearer gw-abc"
    assert seen["body"]["fund_account"] == "8888000001"


def test_query_positions_returns_list() -> None:
    """query_positions returns the data array as a list."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/login":
            return _envelope({"user_token": "gw-abc", "fund_account": "8888000001"})
        return _envelope([{"stock_code": "600000"}, {"stock_code": "000001"}])

    client = QMFHttpClient(_config(), transport=httpx.MockTransport(handler))
    client.login()
    rows = client.query_positions()
    assert len(rows) == 2


def test_error_envelope_raises() -> None:
    """A non-zero result envelope raises QMFApiError carrying the code."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"result": "331100", "msg": "登录失败", "data": {}}
        )

    client = QMFHttpClient(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(QMFApiError) as exc:
        client.login()
    assert exc.value.result == "331100"
