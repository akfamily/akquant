import base64
import json

import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

import httpx
from akquant.gateway.brokers.qmf.client import QMFClientConfig, QMFHttpClient

_KEY = base64.b64encode(b"0" * 32).decode("ascii")


def _capture(data: object) -> tuple[QMFHttpClient, dict]:
    """Build a logged-in option client recording path/body, returning data."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/login":
            return httpx.Response(
                200,
                json={
                    "result": "0",
                    "msg": "success",
                    "data": {"user_token": "gw-b", "fund_account": "8888000001"},
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
        asset_prop="B",
    )
    client = QMFHttpClient(cfg, transport=httpx.MockTransport(handler))
    client.login()
    return client, seen


def test_option_contract_confirm() -> None:
    """Contract confirm posts exchange_type/option_code; returns the list."""
    client, seen = _capture([{"option_code": "10003456"}])
    result = client.option_contract_confirm("1", "10003456")
    assert seen["path"] == "/api/v1/option/contract-confirm"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "exchange_type": "1",
        "option_code": "10003456",
    }
    assert result == [{"option_code": "10003456"}]


def test_query_option_history_bill() -> None:
    """History bill posts begin_date/end_date/money_type; returns the list."""
    client, seen = _capture([{"bill": "1"}])
    result = client.query_option_history_bill("20260101", "20260131")
    assert seen["path"] == "/api/v1/option/history-bill"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "begin_date": "20260101",
        "end_date": "20260131",
        "money_type": "0",
    }
    assert result == [{"bill": "1"}]


def test_query_option_history_statements() -> None:
    """History statements posts begin_date/end_date/query_mode; returns the list."""
    client, seen = _capture([])
    result = client.query_option_history_statements("20260101", "20260131", "1")
    assert seen["path"] == "/api/v1/option/history-statements"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "begin_date": "20260101",
        "end_date": "20260131",
        "query_mode": "1",
    }
    assert result == []


class _FakeOptionClient:
    """Option client stub recording straggler calls."""

    token = "gw-opt"
    fund_account = "8888000001"

    def __init__(self) -> None:
        """Init the call log."""
        self.calls: list = []

    def option_contract_confirm(self, exchange_type, option_code):
        """Stub contract confirm."""
        self.calls.append(("confirm", exchange_type, option_code))
        return [{"m": "confirm"}]

    def query_option_history_bill(self, begin_date, end_date, money_type="0"):
        """Stub history bill."""
        self.calls.append(("bill", begin_date, end_date, money_type))
        return [{"m": "bill"}]

    def query_option_history_statements(self, begin_date, end_date, query_mode):
        """Stub history statements."""
        self.calls.append(("stmts", begin_date, end_date, query_mode))
        return [{"m": "stmts"}]

    def close(self):
        """Ignore close."""


def _gateway(option_client):
    """Build a gateway with a securities stub and given option client."""
    from akquant.gateway.brokers.qmf.adapter import QMFTraderGateway

    class _Sec:
        token = "gw-sec"
        fund_account = "8888000001"

        def close(self):
            """Ignore close."""

    return QMFTraderGateway(
        client=_Sec(), ws_url="ws://gw.test/api/v1/stream", option_client=option_client
    )


def test_adapter_delegates_stragglers() -> None:
    """Adapter straggler methods delegate to the option client."""
    opt = _FakeOptionClient()
    gw = _gateway(opt)

    assert gw.option_contract_confirm("1", "10003456") == [{"m": "confirm"}]
    assert gw.query_option_history_bill("20260101", "20260131") == [{"m": "bill"}]
    assert gw.query_option_history_statements("20260101", "20260131", "1") == [
        {"m": "stmts"}
    ]

    assert ("confirm", "1", "10003456") in opt.calls
    assert ("bill", "20260101", "20260131", "0") in opt.calls
    assert ("stmts", "20260101", "20260131", "1") in opt.calls


def test_adapter_stragglers_require_option_session() -> None:
    """Without an option session, straggler methods raise RuntimeError."""
    gw = _gateway(None)
    with pytest.raises(RuntimeError):
        gw.option_contract_confirm("1", "10003456")
    with pytest.raises(RuntimeError):
        gw.query_option_history_bill("20260101", "20260131")
