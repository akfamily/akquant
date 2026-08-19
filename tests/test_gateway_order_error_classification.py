"""网关异常分类: 只有插件明确说「柜台回绝了」才算明确拒单, 其余一律状态未知.

保守缺省是安全前提: 超时/连接断开往往发生在报文已发出之后, 订单可能真的躺在
柜台里。此时若判成「明确拒单」并回吐 Rejected, 策略会以为没报出去而重下单,
造成重复委托——比异常直接抛出去更危险。
"""

from typing import Any

from akquant.gateway.broker_models import UnifiedErrorType
from akquant.gateway.order_errors import classify_gateway_error, is_definite_reject


class _NoClassify:
    """不实现 classify_order_error 的网关(ctp/miniqmt/ptrade/qmf 现状)."""


class _Classifying:
    """按构造参数返回固定分类的网关."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def classify_order_error(self, exc: BaseException) -> Any:
        return self._result


class _Exploding:
    """分类方法自身抛异常的网关(插件 bug)."""

    def classify_order_error(self, exc: BaseException) -> Any:
        raise ValueError("classifier boom")


def test_missing_method_falls_back_to_unknown() -> None:
    """网关没实现分类方法 → 状态未知(保守)."""
    result = classify_gateway_error(_NoClassify(), RuntimeError("x"))
    assert result == UnifiedErrorType.RETRYABLE
    assert is_definite_reject(result) is False


def test_non_retryable_is_definite_reject() -> None:
    """柜台明确回绝 → 明确拒单."""
    gw = _Classifying(UnifiedErrorType.NON_RETRYABLE)
    assert is_definite_reject(classify_gateway_error(gw, RuntimeError("x"))) is True


def test_risk_rejected_is_definite_reject() -> None:
    """柜台风控拒单 → 明确拒单."""
    gw = _Classifying(UnifiedErrorType.RISK_REJECTED)
    assert is_definite_reject(classify_gateway_error(gw, RuntimeError("x"))) is True


def test_retryable_is_not_definite_reject() -> None:
    """可重试(超时/网络) → 状态未知, 不得当成拒单."""
    gw = _Classifying(UnifiedErrorType.RETRYABLE)
    assert is_definite_reject(classify_gateway_error(gw, RuntimeError("x"))) is False


def test_raw_string_result_is_accepted() -> None:
    """插件返回裸字符串也认(UnifiedErrorType 是 str 枚举)."""
    gw = _Classifying("non_retryable")
    assert classify_gateway_error(gw, RuntimeError("x")) == (
        UnifiedErrorType.NON_RETRYABLE
    )


def test_garbage_result_falls_back_to_unknown() -> None:
    """插件返回无法识别的值 → 状态未知, 不猜."""
    gw = _Classifying("完全不认识的东西")
    assert classify_gateway_error(gw, RuntimeError("x")) == UnifiedErrorType.RETRYABLE


def test_classifier_exception_falls_back_to_unknown() -> None:
    """分类方法自身炸了不得二次崩, 退回状态未知."""
    result = classify_gateway_error(_Exploding(), RuntimeError("x"))
    assert result == UnifiedErrorType.RETRYABLE
