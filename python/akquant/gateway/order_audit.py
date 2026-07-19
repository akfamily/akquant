"""实盘订单生命周期审计日志 (RFC G1).

在 submit / update / fill / cancel / reject 五个跃迁点产出结构化 INFO 审计,
统一走 ``akquant.audit.order`` 命名空间, 可经 ``LogConfig.order_audit_file``
落到独立 JSON 审计文件, 用于事后对账、复盘与追责。

审计是**旁路**关切: 任何埋点异常都不得中断下单/回报主流程, 因此所有对外
函数都经 ``_safe_emit`` 吞掉自身异常(仅退化为一条 debug), 绝不上抛。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..log import ORDER_AUDIT_LOGGER_NAME, build_order_audit_extra, get_logger

audit_logger = get_logger(ORDER_AUDIT_LOGGER_NAME)


def _safe_emit(level: int, message: str, extra: dict[str, Any]) -> None:
    """Emit one audit record, never propagating an audit-side failure."""
    try:
        audit_logger.log(level, message, extra=extra)
    except Exception:  # noqa: BLE001 审计旁路: 失败绝不影响交易主流程
        try:
            audit_logger.debug("order audit emit failed", exc_info=True)
        except Exception:  # noqa: BLE001
            pass


def _pick(payload: Any, name: str) -> Any:
    """Read a field from a dataclass-like or dict payload (getattr then dict)."""
    value = getattr(payload, name, None)
    if value is None and isinstance(payload, dict):
        value = payload.get(name)
    return value


def _status_text(value: Any) -> Optional[str]:
    """Render an order-status enum/value as a stable string."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    return str(value)


def record_submit(
    *,
    strategy_id: Optional[str],
    symbol: str,
    side: str,
    quantity: float,
    price: float | None,
    client_order_id: str,
    broker_order_id: str,
    order_type: str,
    trace_id: Optional[str] = None,
) -> None:
    """Audit a successful order leg submission to the broker."""
    _safe_emit(
        logging.INFO,
        "order submitted",
        build_order_audit_extra(
            event="order_submit",
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            client_order_id=client_order_id,
            order_id=broker_order_id,
            order_type=order_type,
            trace_id=trace_id,
        ),
    )


def record_reject(
    *,
    strategy_id: Optional[str],
    symbol: str,
    client_order_id: str,
    reason: str,
    side: Optional[str] = None,
    quantity: float | None = None,
    trace_id: Optional[str] = None,
) -> None:
    """Audit a locally-rejected order (never reached the broker)."""
    _safe_emit(
        logging.WARNING,
        "order rejected",
        build_order_audit_extra(
            event="order_reject",
            strategy_id=strategy_id,
            symbol=symbol,
            client_order_id=client_order_id,
            reason=reason,
            side=side,
            quantity=quantity,
            trace_id=trace_id,
        ),
    )


def record_cancel(
    *,
    broker_order_id: str,
    symbol: Optional[str] = None,
    strategy_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> None:
    """Audit a cancel request dispatched to the broker."""
    _safe_emit(
        logging.INFO,
        "order cancel requested",
        build_order_audit_extra(
            event="order_cancel",
            strategy_id=strategy_id,
            symbol=symbol,
            order_id=broker_order_id,
            trace_id=trace_id,
        ),
    )


def record_broker_event(
    event_name: str,
    payload: Any,
    *,
    owner_strategy_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> None:
    """Audit an inbound broker push (order update / trade fill / exec report)."""
    if event_name == "order":
        _safe_emit(
            logging.INFO,
            "order update",
            build_order_audit_extra(
                event="order_update",
                strategy_id=owner_strategy_id,
                symbol=_pick(payload, "symbol"),
                order_id=_pick(payload, "broker_order_id"),
                client_order_id=_pick(payload, "client_order_id"),
                trace_id=trace_id,
                order_status=_status_text(_pick(payload, "status")),
                quantity=_pick(payload, "filled_quantity"),
                price=_pick(payload, "avg_fill_price"),
                reason=_pick(payload, "reject_reason") or None,
            ),
        )
    elif event_name == "trade":
        _safe_emit(
            logging.INFO,
            "order fill",
            build_order_audit_extra(
                event="order_fill",
                strategy_id=owner_strategy_id,
                symbol=_pick(payload, "symbol"),
                order_id=_pick(payload, "broker_order_id"),
                client_order_id=_pick(payload, "client_order_id"),
                trace_id=trace_id,
                side=_pick(payload, "side"),
                quantity=_pick(payload, "quantity"),
                price=_pick(payload, "price"),
                trade_id=_pick(payload, "trade_id"),
            ),
        )
    elif event_name == "execution_report":
        _safe_emit(
            logging.INFO,
            "order execution report",
            build_order_audit_extra(
                event="order_update",
                strategy_id=owner_strategy_id,
                symbol=_pick(payload, "symbol"),
                order_id=_pick(payload, "broker_order_id"),
                client_order_id=_pick(payload, "client_order_id"),
                trace_id=trace_id,
                order_status=_status_text(_pick(payload, "status")),
                quantity=_pick(payload, "filled_quantity"),
                price=_pick(payload, "avg_fill_price"),
                reason=_pick(payload, "reject_reason") or None,
            ),
        )
