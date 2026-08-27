"""平台用 SIGTERM 停任务时的收尾: 装 handler -> KeyboardInterrupt -> on_stop.

反馈「任务手动停止, 没有触发 on_stop 回调」的最后一段: v0.3.47 已覆盖正常结束、
``KeyboardInterrupt``(Ctrl+C / duration 到点)、异常中止三条路径, 但全仓零个
``signal.signal`` / ``atexit`` ⇒ 平台若用 ``terminate()`` / ``kill`` 停任务,
``finally`` 根本不执行, ``on_stop`` 里的撤单/平仓全部落空。

**为什么不再等平台澄清停止方式**: 注册 SIGTERM handler 是通用做法, 对
SIGINT/SIGTERM 都有效, 只有 ``kill -9``(SIGKILL) / Windows ``TerminateProcess``
任何框架都无解 —— 先把能覆盖的覆盖掉, 不必阻塞在这个问题上。

handler 只做一件事: 抛 ``KeyboardInterrupt``, 复用 ``run()`` 里已验证的收尾
路径(``duration`` 到点也是这么做的, 见 ``_apply_time_limit``), 不在信号处理器
里直接跑用户的 ``on_stop``。
"""

import signal
import threading
from typing import Any

import pytest
from akquant.live._runner import LiveRunner


def _bare_runner() -> LiveRunner:
    """绕开 ``__init__`` 的最小 runner(只需要日志 extra 用得到的两个属性)."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ctp"
    runner.strategy_id = "_default"
    return runner


def test_guard_installs_sigterm_handler_and_restores_it() -> None:
    """进入时装上自己的 handler, 退出时**原样恢复**.

    恢复是硬要求: ``run_live`` 可能被嵌进宿主进程(平台的任务进程里不止我们
    一个组件), 留着不还就是污染全局状态。
    """
    runner = _bare_runner()
    original = signal.getsignal(signal.SIGTERM)

    with runner._termination_signal_guard():
        installed = signal.getsignal(signal.SIGTERM)
        assert callable(installed)
        assert installed is not original

    assert signal.getsignal(signal.SIGTERM) is original


def test_installed_handler_raises_keyboard_interrupt() -> None:
    """Handler 抛 ``KeyboardInterrupt``, 而不是在信号处理器里跑收尾逻辑.

    收尾要撤单/平仓、要打摘要, 都得在正常栈上跑; 而 ``run()`` 已经有一条被
    ``duration`` 与 Ctrl+C 验证过的 ``KeyboardInterrupt`` 收尾路径, 复用它
    比新开一条停止机制可靠。
    """
    runner = _bare_runner()

    with runner._termination_signal_guard():
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        with pytest.raises(KeyboardInterrupt):
            handler(int(signal.SIGTERM), None)


def test_handler_restored_even_if_the_body_raises() -> None:
    """Body 抛异常(正是 SIGTERM 那条路径)时 handler 依然被恢复."""
    runner = _bare_runner()
    original = signal.getsignal(signal.SIGTERM)

    with pytest.raises(KeyboardInterrupt):
        with runner._termination_signal_guard():
            raise KeyboardInterrupt("terminated")

    assert signal.getsignal(signal.SIGTERM) is original


def test_guard_is_a_noop_off_the_main_thread(caplog: Any) -> None:
    """非主线程装不了 handler(``signal.signal`` 抛 ValueError) -> 告警但放行.

    平台完全可能把 ``run_live`` 跑在工作线程里(线程池承载多个任务)。那种场景
    下装不上 handler 是既定事实, 但**绝不能因此让会话起不来** —— 收不到
    SIGTERM 收尾远好过任务根本跑不起来。
    """
    runner = _bare_runner()
    original = signal.getsignal(signal.SIGTERM)
    outcome: list[str] = []

    def body() -> None:
        try:
            with runner._termination_signal_guard():
                outcome.append("entered")
        except BaseException as exc:  # noqa: BLE001 - 测试要捕获一切
            outcome.append(f"raised:{type(exc).__name__}")

    with caplog.at_level("WARNING", logger="akquant.gateway.live"):
        worker = threading.Thread(target=body)
        worker.start()
        worker.join()

    assert outcome == ["entered"]
    assert signal.getsignal(signal.SIGTERM) is original
    assert [r for r in caplog.records if "SIGTERM" in r.getMessage()] != []
