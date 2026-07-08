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


class _FakeOptionClient:
    """Option client stub recording calls for adapter delegation tests."""

    def __init__(self) -> None:
        """Init token, fund account and call log."""
        self.token = "gw-opt"
        self.fund_account = "8888000001"
        self.calls: list = []

    def _record(self, name, *args):
        """Record a call and return a name-tagged row."""
        self.calls.append((name, args))
        return [{"m": name}]

    def query_option_exercise_assignments(self):
        """Stub exercise assignments."""
        return self._record("assignments")

    def query_option_exercise_settlements(self):
        """Stub exercise settlements."""
        return self._record("settlements")

    def query_option_exercise_debts(self):
        """Stub exercise debts."""
        return self._record("debts")

    def query_option_history_exercise_assignments(self, start_date, end_date):
        """Stub history exercise assignments."""
        return self._record("hist_assignments", start_date, end_date)

    def query_option_history_exercise_settlements(self, start_date, end_date):
        """Stub history exercise settlements."""
        return self._record("hist_settlements", start_date, end_date)

    def query_option_covered_shortages(self):
        """Stub covered shortages."""
        return self._record("shortages")

    def query_option_covered_transferable(
        self, exchange_type, lock_direction, stock_code=None
    ):
        """Stub covered transferable."""
        return self._record("transferable", exchange_type, lock_direction, stock_code)

    def covered_transfer(
        self, exchange_type, stock_code, entrust_amount, lock_direction
    ):
        """Stub covered transfer write."""
        self.calls.append(
            ("transfer", (exchange_type, stock_code, entrust_amount, lock_direction))
        )
        return {"entrust_no": "9000000009"}

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


def test_adapter_delegates_exercise_and_covered() -> None:
    """Adapter convenience methods delegate to the option client with args intact."""
    opt = _FakeOptionClient()
    gw = _gateway(opt)

    assert gw.query_option_exercise_assignments() == [{"m": "assignments"}]
    assert gw.query_option_exercise_settlements() == [{"m": "settlements"}]
    assert gw.query_option_exercise_debts() == [{"m": "debts"}]
    assert gw.query_option_history_exercise_assignments("20260101", "20260131") == [
        {"m": "hist_assignments"}
    ]
    assert gw.query_option_history_exercise_settlements("20260101", "20260131") == [
        {"m": "hist_settlements"}
    ]
    assert gw.query_option_covered_shortages() == [{"m": "shortages"}]
    assert gw.query_option_covered_transferable("1", "1", stock_code="600000") == [
        {"m": "transferable"}
    ]
    assert gw.covered_transfer("1", "600000", "1000", "1") == {
        "entrust_no": "9000000009"
    }

    assert ("hist_assignments", ("20260101", "20260131")) in opt.calls
    assert ("transferable", ("1", "1", "600000")) in opt.calls
    assert ("transfer", ("1", "600000", "1000", "1")) in opt.calls


def test_adapter_exercise_covered_require_option_session() -> None:
    """Without an option session, 3c convenience methods raise RuntimeError."""
    gw = _gateway(None)
    for call in (
        lambda: gw.query_option_exercise_assignments(),
        lambda: gw.query_option_covered_shortages(),
        lambda: gw.covered_transfer("1", "600000", "1000", "1"),
    ):
        with pytest.raises(RuntimeError):
            call()
