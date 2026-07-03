import base64
import json

import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

import httpx
from akquant.gateway.brokers.qmf.client import QMFClientConfig, QMFHttpClient

_KEY = base64.b64encode(b"0" * 32).decode("ascii")


def _client(handler) -> QMFHttpClient:
    """Build a logged-in option client bound to a mock transport."""
    cfg = QMFClientConfig(
        base_url="http://gw.test",
        qmf_user_id="u",
        account_content="8888000001",
        password="pw",
        input_content="1",
        content_type="1",
        password_key=_KEY,
        asset_prop="B",
    )
    client = QMFHttpClient(cfg, transport=httpx.MockTransport(handler))
    client.login()
    return client


def _envelope(data: object) -> httpx.Response:
    """Wrap data in the chibi_quant success envelope."""
    return httpx.Response(200, json={"result": "0", "msg": "success", "data": data})


def test_place_option_order() -> None:
    """place_option_order posts to /option/order with fund_account injected."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/login":
            return _envelope({"user_token": "gw-b", "fund_account": "8888000001"})
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return _envelope({"entrust_no": "9000000001"})

    client = _client(handler)
    data = client.place_option_order(
        {
            "exchange_type": "1",
            "option_code": "10003456",
            "entrust_bs": "1",
            "entrust_oc": "O",
            "covered_flag": "0",
            "entrust_prop": "F0",
            "opt_entrust_price": "0.05",
            "entrust_amount": "1",
        }
    )
    assert data["entrust_no"] == "9000000001"
    assert seen["path"] == "/api/v1/option/order"
    assert seen["body"]["fund_account"] == "8888000001"


def test_query_option_positions_and_assets() -> None:
    """Option position query returns a list; assets query passes money_type."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/login":
            return _envelope({"user_token": "gw-b", "fund_account": "8888000001"})
        if request.url.path == "/api/v1/option/positions":
            return _envelope([{"option_code": "10003456"}])
        seen["assets_body"] = json.loads(request.content)
        return _envelope({"enable_balance": "750000.00"})

    client = _client(handler)
    assert len(client.query_option_positions()) == 1
    client.query_option_assets()
    assert seen["assets_body"]["money_type"] == "0"
