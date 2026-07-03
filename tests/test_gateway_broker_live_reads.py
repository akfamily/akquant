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


def test_get_account_has_backtest_key_parity() -> None:
    """broker_live get_account exposes all backtest keys (0-default) for parity."""
    strategy = _Strategy()
    install_broker_state_reads(strategy, BrokerStateCache(_Gw()))
    acct = strategy.get_account()
    for key in (
        "cash",
        "equity",
        "market_value",
        "notional_value",
        "frozen_cash",
        "margin",
        "used_margin",
        "free_margin",
        "unrealized_pnl",
        "borrowed_cash",
        "short_market_value",
        "maintenance_ratio",
        "account_mode",
        "accrued_interest",
        "daily_interest",
    ):
        assert key in acct


def test_get_position_defaults_to_current_bar_symbol() -> None:
    """get_position() with no symbol resolves to the current bar (backtest parity)."""

    class _Bar:
        symbol = "600000.SH"

    strategy = _Strategy()
    strategy.current_bar = _Bar()
    install_broker_state_reads(strategy, BrokerStateCache(_Gw()))
    assert strategy.get_position() == 1000.0
    assert strategy.get_available_position() == 800.0
