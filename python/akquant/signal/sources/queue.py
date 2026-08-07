"""进程内队列信号源: 参考实现 + 测试基座.

也是最简单的生产可用形态: 外部进程收 HTTP/WS, 把规范化后的 ``Signal`` 投进这个
队列(或经 Redis 转发, 见 signal-ingestion-rfc.md 4.5 的部署建议)。
"""

from __future__ import annotations

import queue
import threading
from typing import Any, List, Optional

from ...log import build_log_extra, get_logger
from ..models import Signal, SignalResult
from ..protocols import SignalSourceBase

logger = get_logger("signal.source.queue")

_SENTINEL = object()


class QueueSignalSource(SignalSourceBase):
    """从 ``queue.Queue`` 取信号并派发的信号源.

    调用 :meth:`put` 投信号(可从任意线程), 消费在自己的 daemon 线程里进行。
    """

    def __init__(self, maxsize: int = 0) -> None:
        """建队列与线程占位."""
        super().__init__()
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._stopped = threading.Event()
        self.results: List[SignalResult] = []

    def put(self, signal: Signal, timeout: Optional[float] = None) -> None:
        """投一条信号(线程安全)."""
        self._queue.put(signal, timeout=timeout)

    def start(self) -> None:
        """起消费线程, **确认其已就绪后**才返回.

        必须等就绪: ``run_live`` 在引擎循环启动前同步调用本方法, 一返回主线程
        就进 Rust 主循环并长期持有 GIL; 若消费线程此刻尚未被调度, 它可能整场会话
        拿不到执行机会(实测结论, 见 signal-ingestion-rfc.md 4.1.1)。
        """
        if self._thread is not None:
            return
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._consume, name="akquant-signal-queue", daemon=True
        )
        self._thread.start()
        if not self._running.wait(timeout=5.0):
            logger.warning(
                "信号消费线程 5 秒内未就绪, 信号可能不会被处理",
                extra=build_log_extra(phase="signal"),
            )

    def stop(self) -> None:
        """停止消费(幂等)."""
        if self._thread is None:
            return
        self._stopped.set()
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=2.0)
        self._thread = None
        self._running.clear()

    def on_result(self, result: SignalResult) -> None:
        """收集回执(便于调用方/测试断言)."""
        self.results.append(result)

    def _consume(self) -> None:
        """消费循环: 取信号 → dispatch, 异常隔离到单条."""
        self._running.set()
        while not self._stopped.is_set():
            item = self._queue.get()
            if item is _SENTINEL:
                break
            try:
                self.dispatch(item)
            except Exception:  # noqa: BLE001 — 单条失败不终止消费线程
                logger.error(
                    "信号处理抛出未捕获异常, 已跳过该条",
                    exc_info=True,
                    extra=build_log_extra(phase="signal"),
                )
