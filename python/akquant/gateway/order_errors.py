"""网关调用失败的分类: 区分「柜台明确回绝」与「订单状态不可知」.

为什么必须区分: 超时与连接断开往往发生在**报文已经发出之后**, 订单可能真的
躺在柜台里。此时若回吐一个 `Rejected` 事件, 策略会以为这单没报出去而重下,
变成重复委托——这比让异常直接抛出去更危险。

分类知识归插件: 只有适配器懂自己柜台的错误码, 而核心仓不能 import 插件。
故约定一个**可选**方法 `TraderGateway.classify_order_error(exc)`, 用 `getattr`
探测而不加进 Protocol 必需成员——既有插件(ctp/miniqmt/ptrade/qmf)不改也不会挂,
行为自动落到保守分支。

保守缺省: 未实现该方法、返回值无法识别、分类方法自身抛错, 一律归入
`RETRYABLE`(状态未知)。宁可让策略多等一轮 recovery 对账, 也不谎报
「这单没报出去」。
"""

from __future__ import annotations

from typing import Any

from ..log import build_log_extra, get_logger
from .broker_models import UnifiedErrorType

logger = get_logger("gateway.live")

#: 被视为「订单确定不存在」的分类。其余(含未知)一律当状态未知处理。
_DEFINITE_REJECT_TYPES = frozenset(
    {UnifiedErrorType.RISK_REJECTED, UnifiedErrorType.NON_RETRYABLE}
)


def classify_gateway_error(trader_gateway: Any, exc: BaseException) -> UnifiedErrorType:
    """把一个网关调用异常分类成 :class:`UnifiedErrorType`.

    :param trader_gateway: 交易网关实例, 可选实现 ``classify_order_error``
    :param exc: 网关调用抛出的异常
    :return: 分类结果; 无法判定时返回 ``RETRYABLE``(状态未知)
    """
    classify = getattr(trader_gateway, "classify_order_error", None)
    if not callable(classify):
        return UnifiedErrorType.RETRYABLE
    try:
        raw = classify(exc)
    except Exception:  # noqa: BLE001 分类器自身出错不得二次崩, 退回保守分支
        logger.warning(
            "classify_order_error 自身抛异常, 按状态未知处理",
            exc_info=True,
            extra=build_log_extra(phase="gateway"),
        )
        return UnifiedErrorType.RETRYABLE
    try:
        return UnifiedErrorType(raw)
    except (ValueError, TypeError):
        logger.warning(
            "classify_order_error 返回无法识别的值 %r, 按状态未知处理",
            raw,
            extra=build_log_extra(phase="gateway"),
        )
        return UnifiedErrorType.RETRYABLE


def is_definite_reject(error_type: UnifiedErrorType) -> bool:
    """该分类是否意味着「订单确定没有到达柜台」."""
    return error_type in _DEFINITE_REJECT_TYPES
