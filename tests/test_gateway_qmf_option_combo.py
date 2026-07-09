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


def test_place_option_combo_order_two_legs() -> None:
    """Place a two-leg combo order with all required legs; return the data dict."""
    client, seen = _capture({"report_no": "R1"})
    result = client.place_option_combo_order(
        exchange_type="1",
        optcomb_code="CNSJC",
        first_option_code="10003456",
        first_opthold_type="1",
        second_option_code="10003457",
        second_opthold_type="2",
        entrust_amount="1",
        comb_bs="1",
    )
    assert seen["path"] == "/api/v1/option/combo-order"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "exchange_type": "1",
        "optcomb_code": "CNSJC",
        "first_option_code": "10003456",
        "first_opthold_type": "1",
        "second_option_code": "10003457",
        "second_opthold_type": "2",
        "entrust_amount": "1",
        "comb_bs": "1",
    }
    assert result == {"report_no": "R1"}


def test_place_option_combo_order_optcomb_id_when_given() -> None:
    """optcomb_id (required on split) is forwarded only when provided."""
    client, seen = _capture({})
    client.place_option_combo_order(
        exchange_type="1",
        optcomb_code="CNSJC",
        first_option_code="10003456",
        first_opthold_type="1",
        second_option_code="10003457",
        second_opthold_type="2",
        entrust_amount="1",
        comb_bs="2",
        optcomb_id="CB1",
    )
    assert seen["body"]["optcomb_id"] == "CB1"


def test_confirm_option_combo_write() -> None:
    """Confirm a combo with required fields; optional legs omitted when None."""
    client, seen = _capture({"ok": "1"})
    result = client.confirm_option_combo(
        exchange_type="1", optcomb_code="CNSJC", comb_bs="1"
    )
    assert seen["path"] == "/api/v1/option/combo-confirm"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "exchange_type": "1",
        "optcomb_code": "CNSJC",
        "comb_bs": "1",
    }
    assert result == {"ok": "1"}


def test_query_option_combo_orders_read() -> None:
    """Combo orders query: default only fund_account; filters forwarded when given."""
    client, seen = _capture([{"report_no": "R1"}])
    result = client.query_option_combo_orders()
    assert seen["path"] == "/api/v1/option/combo-orders"
    assert seen["body"] == {"fund_account": "8888000001"}
    assert result == [{"report_no": "R1"}]

    client2, seen2 = _capture([])
    client2.query_option_combo_orders(optcomb_code="CNSJC", optcomb_id="CB1")
    assert seen2["body"]["optcomb_code"] == "CNSJC"
    assert seen2["body"]["optcomb_id"] == "CB1"


def test_query_option_combo_positions_read() -> None:
    """Combo positions query forwards optcomb_code/query_mode when provided."""
    client, seen = _capture([{"pos": "1"}])
    result = client.query_option_combo_positions(optcomb_code="CNSJC", query_mode="1")
    assert seen["path"] == "/api/v1/option/combo-positions"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "optcomb_code": "CNSJC",
        "query_mode": "1",
    }
    assert result == [{"pos": "1"}]


def test_query_option_history_combo_orders_read() -> None:
    """History combo orders query passes start/end dates plus fund_account."""
    client, seen = _capture([])
    result = client.query_option_history_combo_orders("20260101", "20260131")
    assert seen["path"] == "/api/v1/option/history-combo-orders"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "start_date": "20260101",
        "end_date": "20260131",
    }
    assert result == []
