"""信号接入的协议定义(形状对齐 gateway 的 MarketGateway/TraderGateway)."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from .models import Signal, SignalResult


@runtime_checkable
class OrderSink(Protocol):
    """下单出口: 屏蔽 paper(引擎注入)与 broker_live(柜台报单)的差异.

    调度器只认这个协议, 因此两种模式共用同一套幂等/审计/回执逻辑。
    """

    def submit(self, signal: Signal) -> str:
        """提交一笔委托, 返回订单 id.

        抛异常表示提交失败; 返回空字符串表示被同步拒绝(如风控拦下)。
        """
        ...

    @property
    def mode(self) -> str:
        """出口模式标识(``paper`` / ``broker_live``), 仅用于日志与回执."""
        ...


@runtime_checkable
class SignalSource(Protocol):
    """信号来源: 从外部拿指令并交给 ``dispatch``.

    生命周期由 ``run_live`` 托管:``bind`` → ``start``(独立线程)→ ``stop``。

    ``start`` **必须在自己的线程里跑**, 且要先确认线程已就绪才返回 —— 见
    :func:`akquant.signal.sources.queue.QueueSignalSource.start` 的实现与
    ``run_live(signal_port_ready=...)`` 的文档: 主线程一旦进入引擎循环就会长期
    持有 GIL, 未及时就绪的线程可能整场会话拿不到执行机会。
    """

    def bind(self, dispatch: Callable[[Signal], SignalResult]) -> None:
        """接收调度入口。在 ``start`` 之前调用一次."""
        ...

    def start(self) -> None:
        """开始接收信号(非阻塞: 自己起线程, 就绪后返回)."""
        ...

    def stop(self) -> None:
        """停止接收并回收资源(须幂等, 可能被调多次)."""
        ...

    def on_result(self, result: SignalResult) -> None:
        """接收处理结果回执(含异步拒单)。默认实现可为 no-op."""
        ...


class SignalSourceBase:
    """``SignalSource`` 的便利基类: 实现 bind/on_result 的常规部分."""

    def __init__(self) -> None:
        """初始化未绑定状态."""
        self._dispatch: Callable[[Signal], SignalResult] | None = None

    def bind(self, dispatch: Callable[[Signal], SignalResult]) -> None:
        """记住调度入口."""
        self._dispatch = dispatch

    def dispatch(self, signal: Signal) -> SignalResult:
        """把信号交给调度器; 未 bind 即调用是编程错误."""
        if self._dispatch is None:
            raise RuntimeError("SignalSource 未 bind, 不能 dispatch")
        return self._dispatch(signal)

    def start(self) -> None:
        """默认无需启动."""

    def stop(self) -> None:
        """默认无需清理."""

    def on_result(self, result: SignalResult) -> None:
        """默认忽略回执."""

    @property
    def bound(self) -> bool:
        """是否已绑定调度入口."""
        return self._dispatch is not None


SignalSourceFactory = Callable[..., Any]
