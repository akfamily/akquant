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


def _capture(data: object) -> tuple[QMFHttpClient, dict]:
    """Build a client whose handler records path/body and returns data."""
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

    return _client(handler), seen


@pytest.mark.parametrize(
    "method,path",
    [
        ("query_option_exercise_assignments", "/api/v1/option/exercise-assignments"),
        ("query_option_exercise_settlements", "/api/v1/option/exercise-settlements"),
        ("query_option_exercise_debts", "/api/v1/option/exercise-debts"),
        ("query_option_covered_shortages", "/api/v1/option/covered-shortages"),
    ],
)
def test_no_arg_option_reads(method: str, path: str) -> None:
    """No-arg option reads hit the right path, inject fund_account, return the list."""
    client, seen = _capture([{"row": "1"}])
    result = getattr(client, method)()
    assert seen["path"] == path
    assert seen["body"] == {"fund_account": "8888000001"}
    assert result == [{"row": "1"}]


@pytest.mark.parametrize(
    "method,path",
    [
        (
            "query_option_history_exercise_assignments",
            "/api/v1/option/history-exercise-assignments",
        ),
        (
            "query_option_history_exercise_settlements",
            "/api/v1/option/history-exercise-settlements",
        ),
    ],
)
def test_history_exercise_reads(method: str, path: str) -> None:
    """History exercise reads pass start_date/end_date plus fund_account."""
    client, seen = _capture([])
    result = getattr(client, method)("20260101", "20260131")
    assert seen["path"] == path
    assert seen["body"] == {
        "fund_account": "8888000001",
        "start_date": "20260101",
        "end_date": "20260131",
    }
    assert result == []


def test_covered_transferable_passes_required_and_optional() -> None:
    """Pass exchange_type/lock_direction; omit stock_code when None."""
    client, seen = _capture([{"t": "1"}])
    result = client.query_option_covered_transferable("1", "1")
    assert seen["path"] == "/api/v1/option/covered-transferable"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "exchange_type": "1",
        "lock_direction": "1",
    }
    assert result == [{"t": "1"}]

    client2, seen2 = _capture([])
    client2.query_option_covered_transferable("1", "2", stock_code="600000")
    assert seen2["body"]["stock_code"] == "600000"


def test_covered_transfer_write() -> None:
    """Post all four fields plus fund_account and return the data dict."""
    client, seen = _capture({"entrust_no": "9000000009"})
    result = client.covered_transfer("1", "600000", "1000", "1")
    assert seen["path"] == "/api/v1/option/covered-transfer"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "exchange_type": "1",
        "stock_code": "600000",
        "entrust_amount": "1000",
        "lock_direction": "1",
    }
    assert result == {"entrust_no": "9000000009"}
