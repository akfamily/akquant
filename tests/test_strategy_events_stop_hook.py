"""strategy_events._drive_local_stops 调 execution.check_stop_triggers(仅 broker_live)."""  # noqa: E501

from types import SimpleNamespace

from akquant import strategy_events


class _Exec:
    """Fake broker_live execution exposing check_stop_triggers."""

    def __init__(self):
        """Track calls to check_stop_triggers."""
        self.calls = []

    def check_stop_triggers(self, symbol, last, high=None, low=None):
        """Record the arguments passed by the hook."""
        self.calls.append((symbol, last, high, low))


def test_drive_local_stops_calls_when_present() -> None:
    """When execution exposes check_stop_triggers, the hook calls it."""
    ex = _Exec()
    strategy_events._drive_local_stops(
        SimpleNamespace(execution=ex), "X", 10.5, high=11.0, low=9.0
    )
    assert ex.calls == [("X", 10.5, 11.0, 9.0)]


def test_drive_local_stops_noop_when_absent() -> None:
    """SimExecution 无 check_stop_triggers → 不调用, 不抛；execution=None 也安全."""
    strategy_events._drive_local_stops(SimpleNamespace(execution=object()), "X", 10.5)
    strategy_events._drive_local_stops(SimpleNamespace(execution=None), "X", 10.5)
