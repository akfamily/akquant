"""broker_live 客户端本地止损簿：盯价触发,触发后提交底层市价/限价单.

broker_live 下订单经 BrokerExecution→柜台、不进 Rust 引擎, 故 Rust 原生止损
用不上;本簿在客户端持有条件单并按 bar/tick 价盯触发, 语义对齐 Rust.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

_STOP_ORDER_TYPES = {
    "stop",
    "stopmarket",
    "stop_limit",
    "stoplimit",
    "stoptrail",
    "stoptraillimit",
}
_TRAIL_TYPES = {"stoptrail", "stoptraillimit"}
_LIMIT_ON_TRIGGER = {"stop_limit", "stoplimit", "stoptraillimit"}


@dataclass
class LocalStopOrder:
    """客户端挂着的条件/止损单."""

    local_id: str
    symbol: str
    side: str  # "Buy"/"Sell"
    quantity: float
    order_type: str  # 归一小写, 如 "stopmarket"
    trigger_price: Optional[float] = None
    price: Optional[float] = None
    trail_offset: Optional[float] = None
    trail_reference_price: Optional[float] = None
    time_in_force: Any = None
    status: str = "Submitted"
    extra: dict = field(default_factory=dict)  # 透传其余下单参数
    submit_attempts: int = 0


def is_stop_order_type(order_type: Any) -> bool:
    """判断 order_type 是否为条件/止损类型."""
    return str(order_type or "").strip().lower() in _STOP_ORDER_TYPES


def underlying_order_type(stop_order_type: str) -> str:
    """触发后底层单类型: 限价类 → Limit, 其余 → Market."""
    return (
        "Limit"
        if str(stop_order_type or "").strip().lower() in _LIMIT_ON_TRIGGER
        else "Market"
    )


class LocalStopBook:
    """客户端止损簿: 注册/撤销/列单/盯价触发."""

    def __init__(self) -> None:
        """初始化空簿."""
        self._orders: dict[str, LocalStopOrder] = {}
        self._lock = threading.Lock()

    def register(self, order: LocalStopOrder) -> None:
        """登记一个本地止损单."""
        with self._lock:
            self._orders[order.local_id] = order

    def cancel(self, local_id: str) -> bool:
        """按本地 id 撤销; 存在返回 True."""
        with self._lock:
            return self._orders.pop(str(local_id), None) is not None

    def open_orders(self, symbol: Optional[str] = None) -> list[LocalStopOrder]:
        """列出挂着的本地止损单(可按 symbol 过滤)."""
        with self._lock:
            vals = list(self._orders.values())
        if symbol is not None:
            return [o for o in vals if o.symbol == symbol]
        return vals

    def check(
        self,
        symbol: str,
        last: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
    ) -> list[LocalStopOrder]:
        """按最新价盯触发该 symbol 挂单; 返回并移除已触发单."""
        with self._lock:
            triggered: list[LocalStopOrder] = []
            for order in list(self._orders.values()):
                if order.symbol != symbol:
                    continue
                # 对齐 Rust common.rs: 同一 bar 内先用本 bar high/low 棘轮更新
                # trailing trigger, 再用刚更新出的 trigger 判触发(同 bar 语义,
                # 允许本 bar 触发). 非 trailing 单 _update_trailing 为空操作.
                self._update_trailing(order, last, high, low)
                if order.trigger_price is not None and self._is_triggered(
                    order.side, order.trigger_price, last, high, low
                ):
                    self._orders.pop(order.local_id, None)
                    order.status = "Triggered"
                    triggered.append(order)
            return triggered

    @staticmethod
    def _is_triggered(
        side: str,
        trigger: float,
        last: float,
        high: Optional[float],
        low: Optional[float],
    ) -> bool:
        if str(side).strip().lower() == "buy":
            ref = high if high is not None else last
            return ref >= trigger
        ref = low if low is not None else last
        return ref <= trigger

    def _update_trailing(
        self,
        order: LocalStopOrder,
        last: float,
        high: Optional[float],
        low: Optional[float],
    ) -> None:
        if (
            order.order_type not in _TRAIL_TYPES
            or not order.trail_offset
            or order.trail_offset <= 0
        ):
            return
        if str(order.side).strip().lower() == "sell":
            observed = high if high is not None else last
            ref = (
                observed
                if order.trail_reference_price is None
                else max(order.trail_reference_price, observed)
            )
            order.trail_reference_price = ref
            order.trigger_price = ref - order.trail_offset
        else:  # buy
            observed = low if low is not None else last
            ref = (
                observed
                if order.trail_reference_price is None
                else min(order.trail_reference_price, observed)
            )
            order.trail_reference_price = ref
            order.trigger_price = ref + order.trail_offset
