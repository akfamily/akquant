import time
from typing import Any, cast

from akquant.live._runner import LiveRunner


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


def _runner(**over: Any) -> LiveRunner:
    """Construct a bare LiveRunner, mirroring test_live_runner_broker_bridge.py."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "qmf"
    runner.on_broker_connected = None
    runner.broker_ready_timeout = 10.0
    runner.broker_ready_required = False
    # _await_broker_ready 就绪激活需要 _broker_runtime
    runner._init_broker_bridge_state()
    for key, value in over.items():
        setattr(runner, key, value)
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


class _Target:
    """Minimal strategy target holding broker_ready."""

    def __init__(self) -> None:
        self.broker_ready = False


def test_broker_ready_set_and_callback_fired() -> None:
    """Heartbeat True → targets.broker_ready=True and on_broker_connected fires."""
    fired: list[bool] = []
    runner = _runner(on_broker_connected=lambda ctx: fired.append(ctx.broker_ready))
    target = _Target()
    runner._await_broker_ready(_FakeTrader(hb=True), [target])
    assert target.broker_ready is True
    assert fired == [True]


def test_broker_not_ready_on_heartbeat_timeout() -> None:
    """Heartbeat always False → broker_ready False, callback not fired."""
    fired: list[int] = []
    runner = _runner(
        on_broker_connected=lambda ctx: fired.append(1),
        broker_ready_timeout=0.3,
    )
    target = _Target()
    runner._await_broker_ready(_FakeTrader(hb=False), [target])
    assert target.broker_ready is False
    assert fired == []


def test_blocking_connect_does_not_block_main_thread() -> None:
    """A blocking connect() (CTP-shaped) runs on the thread, not the caller."""
    import threading

    started = threading.Event()

    class _BlockingTrader:
        """Trader whose connect() blocks like CTP's Join()."""

        def connect(self) -> None:
            """Block until released (simulates CTP Join)."""
            started.set()
            time.sleep(5.0)

        def start(self) -> None:
            """Unreached while connect() blocks (CTP connect==start)."""

        def heartbeat(self) -> bool:
            """Report ready once connect() has begun."""
            return started.is_set()

    t0 = time.monotonic()
    _runner()._connect_and_start_trader(cast(Any, _BlockingTrader()))
    # Caller must return immediately, not wait on the blocking connect().
    assert time.monotonic() - t0 < 1.0
    assert started.wait(2.0)
