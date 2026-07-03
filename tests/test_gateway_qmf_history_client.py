import base64
import json

import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

import httpx
from akquant.gateway.brokers.qmf.client import QMFClientConfig, QMFHttpClient

_KEY = base64.b64encode(b"0" * 32).decode("ascii")


def _client(handler) -> QMFHttpClient:
    """Build a logged-in client bound to a mock transport."""
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
    return client


def _envelope(data: object) -> httpx.Response:
    """Wrap data in the chibi_quant success envelope."""
    return httpx.Response(200, json={"result": "0", "msg": "success", "data": data})


def test_query_settlements() -> None:
    """Settlements query posts start/end dates and fund_account."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/login":
            return _envelope({"user_token": "gw", "fund_account": "8888000001"})
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return _envelope([{"stock_code": "600000"}])

    client = _client(handler)
    rows = client.query_settlements("20260401", "20260419")
    assert len(rows) == 1
    assert seen["path"] == "/api/v1/account/settlements"
    assert seen["body"]["start_date"] == "20260401"
    assert seen["body"]["end_date"] == "20260419"
    assert seen["body"]["fund_account"] == "8888000001"


def test_query_fund_flow_dates_optional() -> None:
    """Fund-flow omits date keys when not provided."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/login":
            return _envelope({"user_token": "gw", "fund_account": "8888000001"})
        seen["body"] = json.loads(request.content)
        return _envelope([{"serial_no": "FF-0001"}])

    client = _client(handler)
    rows = client.query_fund_flow()
    assert len(rows) == 1
    assert "start_date" not in seen["body"]
    assert "end_date" not in seen["body"]


def test_query_option_history_orders() -> None:
    """Option history-orders posts to the option history endpoint."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/login":
            return _envelope({"user_token": "gw", "fund_account": "8888000001"})
        seen["path"] = request.url.path
        return _envelope([{"entrust_no": "9000000900"}])

    client = _client(handler)
    rows = client.query_option_history_orders("20260401", "20260419")
    assert len(rows) == 1
    assert seen["path"] == "/api/v1/option/history-orders"


def test_query_option_history_trades_and_settlements() -> None:
    """History trades/settlements post to their endpoints with fund_account."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/login":
            return _envelope({"user_token": "gw", "fund_account": "8888000001"})
        seen[request.url.path] = json.loads(request.content)
        return _envelope([{"row": "1"}])

    client = _client(handler)
    assert len(client.query_option_history_trades("20260401", "20260419")) == 1
    assert len(client.query_option_history_settlements("20260401", "20260419")) == 1
    assert seen["/api/v1/option/history-trades"]["fund_account"] == "8888000001"
    assert seen["/api/v1/option/history-settlements"]["fund_account"] == "8888000001"
    assert seen["/api/v1/option/history-trades"]["start_date"] == "20260401"
