"""信号调度器: 幂等 → 下单 → 审计 → 回执.

这是 ``akquant.signal`` 的唯一决策点。两种出口(paper / broker_live)、任意传输层
(队列 / HTTP / Redis)都汇到这里, 因此幂等与审计只有一份实现。
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

from ..gateway.order_audit import record_reject
from ..log import build_log_extra, get_logger
from .dedup import SignalDedup
from .models import Signal, SignalResult, SignalStatus

logger = get_logger("signal.dispatcher")


class SignalDispatcher:
    """把外部信号翻译成委托, 并把结果回吐给来源."""

    def __init__(
        self,
        sink: Any,
        *,
        dedup: Optional[SignalDedup] = None,
        on_result: Optional[Callable[[SignalResult], None]] = None,
    ) -> None:
        """绑定下单出口、去重器与回执回调."""
        self._sink = sink
        self._dedup = dedup if dedup is not None else SignalDedup()
        self._on_result = on_result
        # signal_id ↔ order_id 双向映射: 供异步拒单反查原始信号。
        self._by_order: Dict[str, str] = {}
        self._lock = threading.Lock()

    def dispatch(self, signal: Signal) -> SignalResult:
        """处理一条信号(线程安全, 可从任意来源线程调用)."""
        log_extra = build_log_extra(phase="signal")

        if not self._dedup.check_and_mark(signal.signal_id):
            logger.info("信号 %s 重复, 已幂等丢弃", signal.signal_id, extra=log_extra)
            return self._finish(
                SignalResult(signal_id=signal.signal_id, status=SignalStatus.DUPLICATE)
            )

        try:
            order_id = self._sink.submit(signal)
        except Exception as exc:  # noqa: BLE001 — 单条信号失败不应拖垮来源线程
            # 放开去重标记: 投递本身没成功, 平台重推应当被受理。
            self._dedup.forget(signal.signal_id)
            logger.error(
                "信号 %s 投递失败: %s",
                signal.signal_id,
                exc,
                exc_info=True,
                extra=log_extra,
            )
            record_reject(
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                side=signal.side,
                quantity=signal.quantity,
                client_order_id="",
                reason=f"signal dispatch error: {exc}",
            )
            return self._finish(
                SignalResult(
                    signal_id=signal.signal_id,
                    status=SignalStatus.ERROR,
                    reason=str(exc),
                )
            )

        if not order_id:
            # 出口同步返回"未下单"(broker 侧被前置风控拦下)。不放开去重标记:
            # 指令已被判定为不可执行, 重推同一 id 只会再被拒一次。
            return self._finish(
                SignalResult(
                    signal_id=signal.signal_id,
                    status=SignalStatus.REJECTED,
                    reason="rejected before reaching the venue (risk limits)",
                )
            )

        with self._lock:
            self._by_order[order_id] = signal.signal_id
        logger.info(
            "信号 %s 已投递 → 订单 %s (%s)",
            signal.signal_id,
            order_id,
            self._sink.mode,
            extra=log_extra,
        )
        return self._finish(
            SignalResult(
                signal_id=signal.signal_id,
                status=SignalStatus.ACCEPTED,
                order_id=order_id,
            )
        )

    def handle_reject(self, order: Any) -> None:
        """把引擎/柜台的异步拒单回吐给信号来源.

        挂到策略的 ``on_reject`` 上即可(``run_live`` 已自动接线)。靠 ``tag``
        或 order_id 反查 signal_id —— 找不到说明该单不是信号产生的, 忽略。
        """
        order_id = str(getattr(order, "id", "") or "")
        tag = str(getattr(order, "tag", "") or "")
        with self._lock:
            signal_id = self._by_order.get(order_id, "") or tag
        if not signal_id:
            return
        reason = str(getattr(order, "reject_reason", "") or "rejected")
        logger.warning(
            "信号 %s 对应的订单被拒: %s",
            signal_id,
            reason,
            extra=build_log_extra(phase="signal"),
        )
        self._finish(
            SignalResult(
                signal_id=signal_id, status=SignalStatus.REJECTED, reason=reason
            )
        )

    def _finish(self, result: SignalResult) -> SignalResult:
        """派发回执(回调异常不影响调度结果)."""
        if self._on_result is not None:
            try:
                self._on_result(result)
            except Exception:  # noqa: BLE001 — 回执失败不改变已发生的事实
                logger.warning(
                    "信号 %s 回执派发失败",
                    result.signal_id,
                    exc_info=True,
                    extra=build_log_extra(phase="signal"),
                )
        return result

    @property
    def dedup(self) -> SignalDedup:
        """暴露去重器(便于监控已见/已淘汰数量)."""
        return self._dedup
