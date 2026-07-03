from akquant.gateway.broker_models import UnifiedAccount, UnifiedPosition
from akquant.gateway.broker_state_cache import BrokerStateCache


class _Gw:
    """Fake gateway counting query calls."""

    def __init__(self) -> None:
        """Init counters."""
        self.pos_calls = 0
        self.acct_calls = 0

    def query_positions(self):
        """Return one position, counting calls."""
        self.pos_calls += 1
        return [
            UnifiedPosition(symbol="600000.SH", quantity=1000, available_quantity=800)
        ]

    def query_account(self):
        """Return an account, counting calls."""
        self.acct_calls += 1
        return UnifiedAccount(account_id="a", equity=2.0, cash=1.0, available_cash=0.5)

    def sync_open_orders(self):
        """Return no open orders."""
        return []


def test_positions_cached_until_invalidate() -> None:
    """positions() caches the gateway query until invalidate()."""
    gw = _Gw()
    cache = BrokerStateCache(gw)
    assert cache.positions()["600000.SH"] == 1000.0
    assert cache.available_positions()["600000.SH"] == 800.0
    cache.positions()
    assert gw.pos_calls == 1  # cached
    cache.invalidate()
    cache.positions()
    assert gw.pos_calls == 2  # refreshed


def test_account_and_error_fallback() -> None:
    """account() caches; a raising query returns the last cache (or None)."""
    gw = _Gw()
    cache = BrokerStateCache(gw)
    assert cache.account().equity == 2.0
    assert gw.acct_calls == 1

    def _boom():
        raise RuntimeError("down")

    gw.query_account = _boom
    cache.invalidate()
    # 异常 → 返回上次缓存(不抛)
    assert cache.account().equity == 2.0
