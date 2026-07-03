from akquant.gateway.broker_models import UnifiedAccount, UnifiedPosition
from akquant.gateway.broker_state_cache import BrokerStateCache
from akquant.gateway.broker_strategy_api import install_broker_state_reads


class _Gw:
    """Fake trader gateway serving positions/account/orders."""

    def query_positions(self):
        """Return one position."""
        return [
            UnifiedPosition(symbol="600000.SH", quantity=1000, available_quantity=800)
        ]

    def query_account(self):
        """Return an account."""
        return UnifiedAccount(
            account_id="a", equity=1500.0, cash=1000.0, available_cash=850.0
        )

    def sync_open_orders(self):
        """Return no open orders."""
        return []


class _Strategy:
    """Bare strategy target."""


def test_broker_live_reads_route_to_gateway() -> None:
    """After install, get_position/get_account read the broker snapshot."""
    strategy = _Strategy()
    install_broker_state_reads(strategy, BrokerStateCache(_Gw()))
    assert strategy.get_position("600000.SH") == 1000.0
    assert strategy.get_available_position("600000.SH") == 800.0
    assert strategy.get_account()["cash"] == 1000.0
    assert strategy.get_account()["available_cash"] == 850.0
    assert strategy.get_portfolio_value() == 1500.0
    assert strategy.get_position("000001.SZ") == 0.0
    assert strategy.get_open_orders() == []
