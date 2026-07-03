import time
from typing import Any, cast

from akquant.live import LiveRunner


class _FakeTrader:
    """Fake trader gateway recording connect/start order and heartbeat."""

    def __init__(self, hb: bool = True) -> None:
        """Init call log and heartbeat toggle."""
        self.calls: list = []
        self._hb = hb

    def connect(self) -> None:
        """Record connect (login)."""
        self.calls.append("connect")

    def start(self) -> None:
        """Record start (stream)."""
        self.calls.append("start")

    def heartbeat(self) -> bool:
        """Report readiness."""
        return self._hb


def _runner() -> LiveRunner:
    """Construct a bare LiveRunner, mirroring test_live_runner_broker_bridge.py."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "qmf"
    return runner


def test_connect_called_before_start() -> None:
    """_connect_and_start_trader calls connect() before start()."""
    fake = _FakeTrader()
    _runner()._connect_and_start_trader(cast(Any, fake))
    for _ in range(50):  # start() runs on a daemon thread; wait briefly
        if "start" in fake.calls:
            break
        time.sleep(0.02)
    assert fake.calls[0] == "connect"
    assert "start" in fake.calls
