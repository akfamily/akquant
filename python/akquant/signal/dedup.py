"""按 signal_id 幂等去重.

思路与 ``gateway/broker_strategy_api.py`` 的 ``applied_fill_ids`` 同源: 信号平台
重连/重推是常态, 加性操作(下单)必须靠会话级已见集合兜住, 否则一次重推就是一笔
重复委托。
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from ..log import build_log_extra, get_logger

logger = get_logger("signal.dedup")

DEFAULT_CAPACITY = 100_000


class SignalDedup:
    """线程安全的 signal_id 已见集合(有界, LRU 淘汰).

    **为何有界**: 长跑会话(数周)下无界集合会持续吃内存。容量按"一天最多几千条
    信号"估, 默认 10 万条足够覆盖很长的窗口。

    **淘汰的风险是真实的**: 一旦开始淘汰, 极老的 signal_id 重推会被当成新信号
    再下一次单。因此首次淘汰时打 WARNING —— 静默淘汰等于静默重复下单。
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        """按容量初始化; capacity <= 0 视为不限(仅测试用)."""
        self._capacity = capacity
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
        self._warned_evict = False
        self._evicted = 0

    def check_and_mark(self, signal_id: str) -> bool:
        """首次见到返回 True(可放行); 已见过返回 False(应丢弃).

        原子操作: 检查与标记在同一把锁内, 故并发投递同一 id 只有一个能放行。
        """
        with self._lock:
            if signal_id in self._seen:
                self._seen.move_to_end(signal_id)
                return False
            self._seen[signal_id] = None
            if self._capacity > 0 and len(self._seen) > self._capacity:
                self._seen.popitem(last=False)
                self._evicted += 1
                if not self._warned_evict:
                    self._warned_evict = True
                    logger.warning(
                        "信号去重集合已达容量 %s 并开始淘汰: 极老的 signal_id "
                        "若被重推将被当作新信号再次下单。如需更长窗口请调大 "
                        "capacity",
                        self._capacity,
                        extra=build_log_extra(phase="signal"),
                    )
            return True

    def forget(self, signal_id: str) -> None:
        """移除一个已见 id(用于投递失败后允许平台重推)."""
        with self._lock:
            self._seen.pop(signal_id, None)

    def __len__(self) -> int:
        """当前已见 id 数量."""
        with self._lock:
            return len(self._seen)

    @property
    def evicted(self) -> int:
        """已淘汰条数(>0 意味着存在重复下单的理论可能)."""
        with self._lock:
            return self._evicted
