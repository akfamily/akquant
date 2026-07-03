import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

from akquant.gateway.broker_models import UnifiedAccount
from akquant.gateway.brokers.qmf import mapper
from akquant.gateway.brokers.qmf.adapter import QMFTraderGateway


def test_merge_option_assets() -> None:
    """Option assets add onto the securities account totals."""
    sec = UnifiedAccount(
        account_id="8888000001",
        equity=1500000.0,
        cash=1000000.0,
        available_cash=850000.0,
    )
    merged = mapper.merge_option_assets(
        sec,
        {
            "total_asset": "1000000.00",
            "current_balance": "850000.00",
            "enable_balance": "750000.00",
        },
    )
    assert merged.account_id == "8888000001"
    assert merged.equity == 2500000.0
    assert merged.cash == 1850000.0
    assert merged.available_cash == 1600000.0


class _SecClient:
    """Securities client stub for query_account."""

    def query_funds(self) -> dict:
        """Return securities funds."""
        return {
            "fund_account": "8888000001",
            "asset_balance": "1500000.00",
            "current_balance": "1000000.00",
            "enable_balance": "850000.00",
        }


class _OptClient(_SecClient):
    """Option client stub adding option assets."""

    def query_option_assets(self, money_type: str = "0") -> dict:
        """Return option assets."""
        return {
            "total_asset": "1000000.00",
            "current_balance": "850000.00",
            "enable_balance": "750000.00",
        }


def test_query_account_merges_when_options_enabled() -> None:
    """query_account merges option assets when an option session exists."""
    gw = QMFTraderGateway(
        client=_SecClient(), ws_url="ws://x", option_client=_OptClient()
    )
    acct = gw.query_account()
    assert acct.available_cash == 1600000.0
    assert acct.equity == 2500000.0


def test_query_account_securities_only_without_options() -> None:
    """Without an option session query_account returns securities only."""
    gw = QMFTraderGateway(client=_SecClient(), ws_url="ws://x")
    acct = gw.query_account()
    assert acct.available_cash == 850000.0
    assert acct.equity == 1500000.0


class _HistSecClient(_SecClient):
    """Securities client stub serving settlements/fund-flow."""

    def query_settlements(self, start_date, end_date, stock_type=None):
        """Return one settlement row."""
        return [{"stock_code": "600000", "start": start_date, "end": end_date}]

    def query_fund_flow(self, start_date=None, end_date=None):
        """Return one fund-flow row."""
        return [{"serial_no": "FF-0001"}]


class _HistOptClient(_HistSecClient):
    """Option client stub serving option history-orders."""

    def query_option_assets(self, money_type: str = "0"):
        """Return option assets."""
        return {"total_asset": "0", "current_balance": "0", "enable_balance": "0"}

    def query_option_history_orders(self, start_date, end_date):
        """Return one option history order row."""
        return [{"entrust_no": "9000000900"}]

    def query_option_history_trades(self, start_date, end_date):
        """Return no option history trades."""
        return []

    def query_option_history_settlements(self, start_date, end_date):
        """Return no option history settlements."""
        return []


def test_adapter_settlements_and_fund_flow_delegate_securities() -> None:
    """Settlements/fund-flow delegate to the securities client."""
    gw = QMFTraderGateway(client=_HistSecClient(), ws_url="ws://x")
    assert gw.query_settlements("20260401", "20260419")[0]["stock_code"] == "600000"
    assert gw.query_fund_flow()[0]["serial_no"] == "FF-0001"


def test_adapter_option_history_requires_option_session() -> None:
    """Option history raises clearly without an option session."""
    gw = QMFTraderGateway(client=_HistSecClient(), ws_url="ws://x")
    with pytest.raises(RuntimeError):
        gw.query_option_history_orders("20260401", "20260419")


def test_adapter_option_history_delegates_option_client() -> None:
    """Option history delegates to the option client when enabled."""
    gw = QMFTraderGateway(
        client=_HistSecClient(), ws_url="ws://x", option_client=_HistOptClient()
    )
    rows = gw.query_option_history_orders("20260401", "20260419")
    assert rows[0]["entrust_no"] == "9000000900"
