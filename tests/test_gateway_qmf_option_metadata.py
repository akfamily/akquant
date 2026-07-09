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


@pytest.mark.parametrize(
    "method,path",
    [
        ("query_option_contracts", "/api/v1/option/contracts"),
        ("query_option_underlyings", "/api/v1/option/underlyings"),
        ("query_option_strategies", "/api/v1/option/strategies"),
        ("query_option_position_limits", "/api/v1/option/position-limits"),
    ],
)
def test_all_optional_reads_default_to_fund_account_only(
    method: str, path: str
) -> None:
    """Send only fund_account by default and return the list."""
    client, seen = _capture([{"row": "1"}])
    result = getattr(client, method)()
    assert seen["path"] == path
    assert seen["body"] == {"fund_account": "8888000001"}
    assert result == [{"row": "1"}]


def test_contracts_optional_filters_passed_when_present() -> None:
    """query_option_contracts forwards stock_code/option_code only when provided."""
    client, seen = _capture([])
    client.query_option_contracts(stock_code="510300", option_code="10003456")
    assert seen["path"] == "/api/v1/option/contracts"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "stock_code": "510300",
        "option_code": "10003456",
    }


def test_contract_tips_sends_money_type_default() -> None:
    """Send money_type (default '0') and return the list."""
    client, seen = _capture([{"tip": "x"}])
    result = client.query_option_contract_tips()
    assert seen["path"] == "/api/v1/option/contract-tips"
    assert seen["body"] == {"fund_account": "8888000001", "money_type": "0"}
    assert result == [{"tip": "x"}]


def test_enable_amount_returns_dict_with_required_fields() -> None:
    """query_option_enable_amount posts required order fields and returns the dict."""
    client, seen = _capture({"enable_amount": "10"})
    result = client.query_option_enable_amount(
        exchange_type="1",
        option_code="10003456",
        opt_entrust_price="0.05",
        entrust_prop="F0",
        entrust_bs="1",
        entrust_oc="O",
    )
    assert seen["path"] == "/api/v1/option/enable-amount"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "exchange_type": "1",
        "option_code": "10003456",
        "opt_entrust_price": "0.05",
        "entrust_prop": "F0",
        "entrust_bs": "1",
        "entrust_oc": "O",
    }
    assert result == {"enable_amount": "10"}


def test_enable_amount_includes_covered_flag_when_given() -> None:
    """query_option_enable_amount forwards covered_flag only when provided."""
    client, seen = _capture({})
    client.query_option_enable_amount(
        exchange_type="1",
        option_code="10003456",
        opt_entrust_price="0.05",
        entrust_prop="F0",
        entrust_bs="2",
        entrust_oc="O",
        covered_flag="1",
    )
    assert seen["body"]["covered_flag"] == "1"


def test_underlying_amount_tip_returns_dict() -> None:
    """query_option_underlying_amount_tip posts required fields and returns the dict."""
    client, seen = _capture({"tip_amount": "100"})
    result = client.query_option_underlying_amount_tip(
        exchange_type="1",
        option_code="10003456",
        entrust_amount="1",
        entrust_bs="1",
        entrust_oc="O",
    )
    assert seen["path"] == "/api/v1/option/underlying-amount-tip"
    assert seen["body"] == {
        "fund_account": "8888000001",
        "exchange_type": "1",
        "option_code": "10003456",
        "entrust_amount": "1",
        "entrust_bs": "1",
        "entrust_oc": "O",
    }
    assert result == {"tip_amount": "100"}
