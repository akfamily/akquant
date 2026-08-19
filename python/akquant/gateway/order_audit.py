"""实盘订单生命周期审计日志 (RFC G1 / G6).

在 submit / update / fill / cancel / reject 五个跃迁点产出结构化 INFO 审计,
统一走 ``akquant.audit.order`` 命名空间, 可经 ``LogConfig.order_audit_file``
落到独立 JSON 审计文件, 用于事后对账、复盘与追责。

**语言中立**(G6): 审计记录只带英文 ``event`` 码 + 结构化字段; 人类可读的
message 由 ``log.render_audit_message`` 渲染, 英文为 canonical(文件/JSON/grep),
控制台可选按 ``LogConfig.language`` 渲染其他语言——业务代码不写任何语言散文。

审计是**旁路**关切: 任何埋点异常都不得中断下单/回报主流程, 因此所有对外
函数都经 ``_safe_emit`` 吞掉自身异常(仅退化为一条 debug), 绝不上抛。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..log import (
    ORDER_AUDIT_LOGGER_NAME,
    build_order_audit_extra,
    get_logger,
    render_audit_message,
)

audit_logger = get_logger(ORDER_AUDIT_LOGGER_NAME)


def _safe_emit(level: int, extra: dict[str, Any]) -> None:
    """Emit one audit record with an english canonical message, never raising."""
    try:
        message = render_audit_message(extra.get("event"), extra)
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
    origin: str = "local",
) -> None:
    """Audit a rejected order.

    ``origin="local"``: 本地校验/风控拒的, 单从未到达柜台。
    ``origin="broker"``: 报文已发出、被柜台明确回绝。
    """
    _safe_emit(
        logging.WARNING,
        build_order_audit_extra(
            event="order_reject",
            strategy_id=strategy_id,
            symbol=symbol,
            client_order_id=client_order_id,
            reason=reason,
            side=side,
            quantity=quantity,
            trace_id=trace_id,
            origin=origin,
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
        build_order_audit_extra(
            event="order_cancel",
            strategy_id=strategy_id,
            symbol=symbol,
            order_id=broker_order_id,
            trace_id=trace_id,
        ),
    )


def record_submit_unknown(
    *,
    strategy_id: Optional[str],
    symbol: str,
    side: str,
    quantity: float,
    price: float | None,
    client_order_id: str,
    reason: str,
    trace_id: Optional[str] = None,
) -> None:
    """Audit an order whose broker-side fate is unknown (timeout/transport).

    与 ``record_reject`` 刻意分开: 这笔单**可能已经在柜台里**, 对账时必须能和
    「确定没报出去」的拒单区分开, 否则会按拒单口径重下单。
    """
    _safe_emit(
        logging.WARNING,
        build_order_audit_extra(
            event="order_submit_unknown",
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            client_order_id=client_order_id,
            reason=reason,
            trace_id=trace_id,
        ),
    )


def record_cancel_failed(
    *,
    broker_order_id: str,
    reason: str,
    symbol: Optional[str] = None,
    strategy_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> None:
    """Audit a cancel request the broker refused or that failed in transport.

    ``record_cancel``(意图)照常在调用前发出, 本函数补的是**结果**。两条成对,
    审计流水里才能看出「撤过、但没撤成」。
    """
    _safe_emit(
        logging.WARNING,
        build_order_audit_extra(
            event="order_cancel_failed",
            strategy_id=strategy_id,
            symbol=symbol,
            order_id=broker_order_id,
            reason=reason,
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
    symbol = _pick(payload, "symbol")
    broker_order_id = _pick(payload, "broker_order_id")
    client_order_id = _pick(payload, "client_order_id")
    if event_name == "trade":
        _safe_emit(
            logging.INFO,
            build_order_audit_extra(
                event="order_fill",
                strategy_id=owner_strategy_id,
                symbol=symbol,
                order_id=broker_order_id,
                client_order_id=client_order_id,
                trace_id=trace_id,
                side=_pick(payload, "side"),
                quantity=_pick(payload, "quantity"),
                price=_pick(payload, "price"),
                trade_id=_pick(payload, "trade_id"),
            ),
        )
    elif event_name in ("order", "execution_report"):
        _safe_emit(
            logging.INFO,
            build_order_audit_extra(
                event="order_update",
                strategy_id=owner_strategy_id,
                symbol=symbol,
                order_id=broker_order_id,
                client_order_id=client_order_id,
                trace_id=trace_id,
                order_status=_status_text(_pick(payload, "status")),
                quantity=_pick(payload, "filled_quantity"),
                price=_pick(payload, "avg_fill_price"),
                reason=_pick(payload, "reject_reason") or None,
            ),
        )
