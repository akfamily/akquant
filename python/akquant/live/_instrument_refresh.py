"""实盘运行中按时刻刷新合约表: 新挂合约登记进引擎、策略快照与行情订阅."""

from __future__ import annotations

import datetime as dt
import threading
from typing import Any, Callable, Dict, List, Sequence

from ..log import get_logger

logger = get_logger("live.instrument_refresh")

# 判定合约参数是否变化时逐项比较的字段(Instrument 的可读属性)
COMPARED_FIELDS = (
    "asset_type",
    "multiplier",
    "margin_ratio",
    "tick_size",
    "option_type",
    "strike_price",
    "expiry_date",
    "underlying_symbol",
    "option_margin_model",
)


def instrument_changes(old: Any, new: Any) -> List[str]:
    """逐项列出两个同名合约不同的字段, 形如 ``margin_ratio 0.09 -> 0.12``."""
    return [
        f"{name} {getattr(old, name)} -> {getattr(new, name)}"
        for name in COMPARED_FIELDS
        if getattr(old, name) != getattr(new, name)
    ]


class InstrumentRefresher:
    """按时刻刷新合约表的后台线程.

    每到 ``source.next_refresh()`` 调 ``source.refresh()``, 把新增与参数变化的合约
    交给 ``apply``(由实盘运行器登记进引擎、策略与行情订阅)。

    :param source: 提供 ``refresh() -> List[Instrument]`` 与 ``next_refresh()``
        的合约来源(如 ``CTPOptionUniverse``)
    :param current: 会话已登记的合约
    :param apply: 接收需要登记的合约列表
    """

    def __init__(
        self,
        source: Any,
        current: Sequence[Any],
        apply: Callable[[List[Any]], None],
    ) -> None:
        """记下已登记合约; 线程在 ``start()`` 时启动."""
        self.source = source
        self.known: Dict[str, Any] = {str(inst.symbol): inst for inst in current}
        self.apply = apply
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="akquant-instrument-refresh", daemon=True
        )

    def start(self) -> None:
        """启动刷新线程."""
        self._thread.start()

    def stop(self) -> None:
        """通知刷新线程退出(正在进行的查询结束后退出)."""
        self._stop.set()

    def _loop(self) -> None:
        while True:
            at = self.source.next_refresh()
            wait = (at - dt.datetime.now(at.tzinfo)).total_seconds()
            logger.info("下次刷新合约表: %s", at.isoformat())
            if self._stop.wait(max(wait, 0.0)):
                return
            try:
                self.refresh_once()
            except Exception:  # noqa: BLE001 刷新失败不中断交易, 下个时刻再查
                logger.exception("刷新合约表失败, 本时段新挂合约未登记")

    def refresh_once(self) -> List[Any]:
        """查询一次, 返回已交给 ``apply`` 的合约(新增与参数变化)."""
        updates: List[Any] = []
        added: List[str] = []
        for inst in self.source.refresh():
            symbol = str(inst.symbol)
            old = self.known.get(symbol)
            if old is None:
                added.append(symbol)
            else:
                changes = instrument_changes(old, inst)
                if not changes:
                    continue
                logger.warning("合约 %s 参数变化: %s", symbol, "; ".join(changes))
            updates.append(inst)
        if updates:
            self.apply(updates)
            for inst in updates:
                self.known[str(inst.symbol)] = inst
        logger.info(
            "合约表刷新完成: 新增 %d 个%s, 参数变化 %d 个",
            len(added),
            f"({', '.join(added)})" if added else "",
            len(updates) - len(added),
        )
        return updates
