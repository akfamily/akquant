"""Redis Stream 信号源(生产推荐形态).

对标 vn.py 的 WebTrader 取舍: 把 Web 服务放在**独立进程**, 与交易进程之间用消息
通道解耦 —— HTTP 服务的故障与负载都被进程边界隔开, 不会波及交易主循环
(见 signal-ingestion-rfc.md 4.5)。

用 Stream 而非 List: ``XREADGROUP`` 提供消费组与显式 ack, 崩溃重启后未 ack 的消息
仍在 pending 中可被重投; ``BLPOP`` 取走即丢, 崩在处理中途就丢单了。

``redis`` 是可选依赖。构造时可注入任意实现了 ``xreadgroup`` / ``xack`` /
``xgroup_create`` 的客户端(测试即如此), 不传才按 ``url`` 建真实连接。
"""

from __future__ import annotations

import json
import threading
from typing import Any, List, Optional

from ...log import build_log_extra, get_logger
from ..models import Signal, SignalResult
from ..protocols import SignalSourceBase

logger = get_logger("signal.source.redis")

DEFAULT_STREAM = "akquant:signals"
DEFAULT_GROUP = "akquant"


class RedisSignalSource(SignalSourceBase):
    """从 Redis Stream 消费信号."""

    def __init__(
        self,
        *,
        url: str = "redis://127.0.0.1:6379/0",
        stream: str = DEFAULT_STREAM,
        group: str = DEFAULT_GROUP,
        consumer: str = "akquant-1",
        client: Optional[Any] = None,
        block_ms: int = 1000,
        field: str = "signal",
    ) -> None:
        """准备连接参数; ``client`` 非空则直接用它(便于测试与自定义连接池)."""
        super().__init__()
        self._url = url
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._client = client
        self._block_ms = block_ms
        self._field = field
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._stopped = threading.Event()
        self.results: List[SignalResult] = []

    def _ensure_client(self) -> Any:
        """惰性建客户端: 让 ``import akquant.signal`` 不依赖 redis."""
        if self._client is not None:
            return self._client
        try:
            import redis  # noqa: PLC0415 — 可选依赖必须惰性导入
        except ImportError as exc:  # pragma: no cover — 取决于环境
            raise RuntimeError(
                "RedisSignalSource 需要 redis 包: pip install 'akquant[signal-redis]'"
                "(或自行注入 client=...)"
            ) from exc
        self._client = redis.Redis.from_url(self._url, decode_responses=True)
        return self._client

    def start(self) -> None:
        """建消费组并起消费线程, 确认就绪后返回."""
        if self._thread is not None:
            return
        client = self._ensure_client()
        try:
            client.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except Exception as exc:  # noqa: BLE001 — 组已存在是正常路径
            if "BUSYGROUP" not in str(exc):
                raise
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._consume, name="akquant-signal-redis", daemon=True
        )
        self._thread.start()
        if not self._running.wait(timeout=5.0):
            logger.warning(
                "Redis 信号消费线程 5 秒内未就绪",
                extra=build_log_extra(phase="signal"),
            )

    def stop(self) -> None:
        """停止消费(幂等)."""
        if self._thread is None:
            return
        self._stopped.set()
        self._thread.join(timeout=3.0)
        self._thread = None
        self._running.clear()

    def on_result(self, result: SignalResult) -> None:
        """收集回执."""
        self.results.append(result)

    def _parse(self, fields: dict[str, Any]) -> Optional[Signal]:
        """把 stream entry 的 fields 解析成 Signal; 解析失败返回 None."""
        raw = fields.get(self._field)
        try:
            payload = json.loads(raw) if isinstance(raw, str) else dict(fields)
            return Signal(**payload)
        except Exception:  # noqa: BLE001 — 坏消息不能卡住整条流
            logger.error(
                "Redis 信号解析失败, 已 ack 丢弃: %r",
                fields,
                exc_info=True,
                extra=build_log_extra(phase="signal"),
            )
            return None

    def _consume(self) -> None:
        """消费循环: XREADGROUP → dispatch → XACK."""
        self._running.set()
        client = self._ensure_client()
        while not self._stopped.is_set():
            try:
                batches = client.xreadgroup(
                    self._group,
                    self._consumer,
                    {self._stream: ">"},
                    count=16,
                    block=self._block_ms,
                )
            except Exception:  # noqa: BLE001 — 断连不应终止线程, 下轮重试
                logger.error(
                    "Redis 读取失败, 稍后重试",
                    exc_info=True,
                    extra=build_log_extra(phase="signal"),
                )
                if self._stopped.wait(timeout=1.0):
                    break
                continue
            for _stream, entries in batches or ():
                for entry_id, fields in entries:
                    signal = self._parse(dict(fields))
                    if signal is not None:
                        try:
                            self.dispatch(signal)
                        except Exception:  # noqa: BLE001 — 单条失败不停流
                            logger.error(
                                "信号处理抛出未捕获异常, 已跳过该条",
                                exc_info=True,
                                extra=build_log_extra(phase="signal"),
                            )
                    # 无论成功失败都 ack: dispatch 内部已按 signal_id 幂等,
                    # 不 ack 会让坏消息永远堵在 pending 里反复重投。
                    try:
                        client.xack(self._stream, self._group, entry_id)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "Redis ack 失败: %s",
                            entry_id,
                            exc_info=True,
                            extra=build_log_extra(phase="signal"),
                        )
