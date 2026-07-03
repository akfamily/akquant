"""wrap_state_invalidation：order/trade 事件后失效状态缓存（独立可测）."""

from akquant.gateway.broker_strategy_api import wrap_state_invalidation


class _Cache:
    """Fake cache tracking invalidate() calls."""

    def __init__(self) -> None:
        """Start with zero invalidate calls."""
        self.invalidate_calls = 0

    def invalidate(self) -> None:
        """Record an invalidate call."""
        self.invalidate_calls += 1


def test_trade_event_invalidates_cache_and_calls_original() -> None:
    """A 'trade' event calls the original callback and invalidates the cache."""
    calls = []
    cache = _Cache()
    wrapped = wrap_state_invalidation(
        lambda name, payload: calls.append((name, payload)), lambda: cache
    )

    wrapped("trade", {"symbol": "600000.SH"})

    assert calls == [("trade", {"symbol": "600000.SH"})]
    assert cache.invalidate_calls == 1


def test_order_event_invalidates_cache() -> None:
    """An 'order' event also invalidates the cache."""
    cache = _Cache()
    wrapped = wrap_state_invalidation(lambda name, payload: None, lambda: cache)

    wrapped("order", {})

    assert cache.invalidate_calls == 1


def test_bar_event_does_not_invalidate_cache() -> None:
    """Non order/trade events (e.g. 'bar') do not touch the cache."""
    cache = _Cache()
    wrapped = wrap_state_invalidation(lambda name, payload: None, lambda: cache)

    wrapped("bar", {})

    assert cache.invalidate_calls == 0


def test_none_cache_does_not_crash_on_trade_event() -> None:
    """Before install_submitter runs, get_cache() returns None — no crash."""
    wrapped = wrap_state_invalidation(lambda name, payload: None, lambda: None)

    wrapped("trade", {})  # should not raise
