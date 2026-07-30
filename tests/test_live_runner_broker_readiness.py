import inspect
import logging
import threading
import time
from typing import Any, cast

import pytest
from akquant.gateway.trader_base import TraderGatewayBase
from akquant.live import run_live
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


def test_broker_ready_defaults_are_fail_fast() -> None:
    """默认必须 fail-fast: broker 未就绪即中止启动, 且超时对真实柜台足够宽.

    未就绪时 order_submitter 会对每次下单抛 RuntimeError, 继续运行没有任何
    可用性——只会把一条易被淹没的 warning 换成一堆难归因的异常。
    """
    for func in (LiveRunner.__init__, run_live):
        params = inspect.signature(func).parameters
        assert params["broker_ready_required"].default is True, (
            f"{func.__qualname__}: broker_ready_required 默认应为 True"
        )
        assert params["broker_ready_timeout"].default >= 30.0, (
            f"{func.__qualname__}: 超时应足够真实柜台登录(QMF 双会话 + WS 建连)"
        )


def test_not_ready_raises_when_required() -> None:
    """required=True 且心跳始终为 False → 抛 RuntimeError 中止启动."""
    runner = _runner(broker_ready_timeout=0.3, broker_ready_required=True)
    target = _Target()

    with pytest.raises(RuntimeError, match="broker not ready"):
        runner._await_broker_ready(_FakeTrader(hb=False), [target])

    assert target.broker_ready is False


def test_not_ready_stops_broker_threads_before_raising() -> None:
    """中止启动前必须停掉 dispatcher/recovery 线程.

    这两个线程在 _bind_broker_callbacks 里就已启动, 而 raise 发生在 run() 的
    try 之外, finally 的清理不会执行。若不在此处收尾, 一次失败的启动会留下一个
    每秒轮询柜台的 recovery 线程。
    """
    runner = _runner(broker_ready_timeout=0.3, broker_ready_required=True)
    runner._start_broker_dispatcher(cast(Any, _Target()))
    assert runner._broker_dispatch_thread is not None, "线程未启动, 测试前提不成立"

    with pytest.raises(RuntimeError):
        runner._await_broker_ready(_FakeTrader(hb=False), [_Target()])

    assert runner._broker_dispatch_thread is None, "dispatcher 线程未被清理"
    assert runner._broker_recovery_thread is None, "recovery 线程未被清理"


def test_gateway_not_overriding_heartbeat_is_warned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """未覆写 heartbeat() 的 gateway 会永远判定就绪, 必须告警.

    TraderGatewayBase.heartbeat 默认 return True, 自定义 broker 若忘了覆写,
    broker_ready_required 对其完全失效——登录失败照样放行到下单。
    """

    class _NoHeartbeatGateway(TraderGatewayBase):
        """自定义 broker 的典型疏漏: 没覆写 heartbeat."""

    runner = _runner(broker_ready_required=True)
    target = _Target()

    with caplog.at_level(logging.WARNING):
        runner._await_broker_ready(cast(Any, _NoHeartbeatGateway()), [target])

    assert target.broker_ready is True, "基类心跳恒真, 应判定就绪"
    assert any("heartbeat" in record.getMessage() for record in caplog.records), (
        "未覆写 heartbeat 的 gateway 应产生告警"
    )


def test_blocking_connect_does_not_block_main_thread() -> None:
    """A blocking connect() (CTP-shaped) runs on the thread, not the caller."""
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
