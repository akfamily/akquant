"""BrokerExecution：读走 cache、submit 走 submitter、cancel 走 gateway."""

from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_models import UnifiedAccount, UnifiedPosition
from akquant.gateway.broker_state_cache import BrokerStateCache


class _Gw:
    def __init__(self):
        self.canceled = []

    def query_positions(self):
        return [
            UnifiedPosition(symbol="600000.SH", quantity=1000, available_quantity=800)
        ]

    def query_account(self):
        return UnifiedAccount(
            account_id="a", equity=1500.0, cash=1000.0, available_cash=850.0
        )

    def sync_open_orders(self):
        return []

    def cancel_order(self, bid):
        self.canceled.append(bid)


class _Submitter:
    def __init__(self):
        self.submitted = None

    def submit_order(self, **kwargs):
        self.submitted = kwargs
        return "BID-1"

    def _get_execution_capabilities(self):
        return {"broker_live": True, "client_order_id": True}


class _S:
    current_bar = None
    current_tick = None


def test_broker_execution_reads_and_writes() -> None:
    """读走 cache、submit 走 submitter、cancel 走 gateway."""
    gw = _Gw()
    ex = BrokerExecution(_S(), gw, BrokerStateCache(gw), _Submitter())
    assert ex.get_position("600000.SH") == 1000.0
    assert ex.get_available_position("600000.SH") == 800.0
    assert ex.get_account()["cash"] == 1000.0
    assert ex.get_cash() == 1000.0
    assert ex.get_portfolio_value() == 1500.0
    assert ex.capabilities()["broker_live"] is True
    assert ex.submit_order(symbol="600000.SH", side="Buy", quantity=100) == "BID-1"
    ex.cancel_order("BID-1")
    assert gw.canceled == ["BID-1"]
