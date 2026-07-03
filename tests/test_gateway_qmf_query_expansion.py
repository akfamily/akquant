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
