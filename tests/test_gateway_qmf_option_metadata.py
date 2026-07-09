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


class _FakeOptionClient:
    """Option client stub recording metadata calls for adapter delegation tests."""

    def __init__(self) -> None:
        """Init token/fund account and call log."""
        self.token = "gw-opt"
        self.fund_account = "8888000001"
        self.calls: list = []

    def query_option_contracts(self, stock_code=None, option_code=None):
        """Stub contracts."""
        self.calls.append(("contracts", stock_code, option_code))
        return [{"m": "contracts"}]

    def query_option_underlyings(self, stock_code=None):
        """Stub underlyings."""
        self.calls.append(("underlyings", stock_code))
        return [{"m": "underlyings"}]

    def query_option_strategies(self, optcomb_code=None):
        """Stub strategies."""
        self.calls.append(("strategies", optcomb_code))
        return [{"m": "strategies"}]

    def query_option_position_limits(self, stock_code=None):
        """Stub position limits."""
        self.calls.append(("position_limits", stock_code))
        return [{"m": "position_limits"}]

    def query_option_contract_tips(self, money_type="0"):
        """Stub contract tips."""
        self.calls.append(("contract_tips", money_type))
        return [{"m": "contract_tips"}]

    def query_option_enable_amount(
        self,
        exchange_type,
        option_code,
        opt_entrust_price,
        entrust_prop,
        entrust_bs,
        entrust_oc,
        covered_flag=None,
    ):
        """Stub enable amount."""
        self.calls.append(("enable_amount", exchange_type, option_code, covered_flag))
        return {"enable_amount": "10"}

    def query_option_underlying_amount_tip(
        self, exchange_type, option_code, entrust_amount, entrust_bs, entrust_oc
    ):
        """Stub underlying amount tip."""
        self.calls.append(("amount_tip", exchange_type, option_code, entrust_amount))
        return {"tip_amount": "100"}

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


def test_adapter_delegates_metadata_reads() -> None:
    """Adapter metadata convenience methods delegate to the option client."""
    opt = _FakeOptionClient()
    gw = _gateway(opt)

    assert gw.query_option_contracts(stock_code="510300") == [{"m": "contracts"}]
    assert gw.query_option_underlyings() == [{"m": "underlyings"}]
    assert gw.query_option_strategies() == [{"m": "strategies"}]
    assert gw.query_option_position_limits() == [{"m": "position_limits"}]
    assert gw.query_option_contract_tips() == [{"m": "contract_tips"}]
    assert gw.query_option_enable_amount("1", "10003456", "0.05", "F0", "1", "O") == {
        "enable_amount": "10"
    }
    assert gw.query_option_underlying_amount_tip("1", "10003456", "1", "1", "O") == {
        "tip_amount": "100"
    }

    assert ("contracts", "510300", None) in opt.calls
    assert ("amount_tip", "1", "10003456", "1") in opt.calls


def test_adapter_metadata_requires_option_session() -> None:
    """Without an option session, metadata convenience methods raise RuntimeError."""
    gw = _gateway(None)
    for call in (
        lambda: gw.query_option_contracts(),
        lambda: gw.query_option_enable_amount("1", "c", "0.05", "F0", "1", "O"),
    ):
        with pytest.raises(RuntimeError):
            call()
