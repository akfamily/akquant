"""验证 CTP 网关的 classify_order_error 实现.

CTP 的所有 insert_order/cancel_order 异常都是本地错误(报文未出进程),
应归类为 NON_RETRYABLE(明确拒单), 而非 RETRYABLE(状态未知)。
这消除流控场景下的告警风暴: -2/-3 流控是高频场景常态, 若归入状态未知
会打 CRITICAL, 接入告警通道即造成风暴。

CTP 真正的柜台拒单走 OnRspOrderInsert 异步回报, 不经异常通道。
"""

import pytest
from akquant.gateway.broker_models import UnifiedErrorType


def test_ctp_classify_all_exceptions_as_definite_reject() -> None:
    """CTP 的所有异常都应归类为明确拒单(报文未出进程)."""
    try:
        from akquant.gateway.brokers.ctp.native import CTPTraderGateway
    except ImportError:
        pytest.skip("openctp-ctp not installed")

    # 构造一个未连接的 CTP 实例(只为调用 classify_order_error)
    # 不调用 connect(), 避免真实连接
    gateway = CTPTraderGateway.__new__(CTPTraderGateway)

    # 测试各种典型异常
    test_cases = [
        RuntimeError("CTP trader is not ready for trading"),
        RuntimeError("ReqOrderInsert failed with code=-2"),  # 流控: 未处理请求超限
        RuntimeError("ReqOrderInsert failed with code=-3"),  # 流控: 每秒请求超限
        RuntimeError("ReqOrderInsert failed with code=-1"),  # 网络连接失败
        RuntimeError("CTP trader API is not initialized"),
        ValueError("invalid broker_order_id=xxx"),
        TimeoutError("connection timeout"),  # 即使是超时, CTP 的也是本地判定
    ]

    for exc in test_cases:
        result = gateway.classify_order_error(exc)
        assert result == UnifiedErrorType.NON_RETRYABLE.value, (
            f"{type(exc).__name__}('{exc}') 应归类为 NON_RETRYABLE, 实际返回 {result}"
        )


def test_ctp_classify_returns_enum_value_not_enum() -> None:
    """classify_order_error 应返回枚举的 .value(字符串), 而非枚举实例."""
    try:
        from akquant.gateway.brokers.ctp.native import CTPTraderGateway
    except ImportError:
        pytest.skip("openctp-ctp not installed")

    gateway = CTPTraderGateway.__new__(CTPTraderGateway)
    result = gateway.classify_order_error(RuntimeError("test"))

    # 返回值应该是字符串, 而非枚举实例
    assert isinstance(result, str), f"应返回 str, 实际返回 {type(result)}"
    assert result == "non_retryable", f"应返回 'non_retryable', 实际返回 {result!r}"


def test_ctp_has_classify_order_error_method() -> None:
    """CTP 网关应实现 classify_order_error 方法(可被 getattr 探测到)."""
    try:
        from akquant.gateway.brokers.ctp.native import CTPTraderGateway
    except ImportError:
        pytest.skip("openctp-ctp not installed")

    gateway = CTPTraderGateway.__new__(CTPTraderGateway)
    # 模拟 order_errors.classify_gateway_error 的探测逻辑
    classify = getattr(gateway, "classify_order_error", None)
    assert callable(classify), "classify_order_error 应该存在且可调用"


if __name__ == "__main__":
    test_ctp_classify_all_exceptions_as_definite_reject()
    test_ctp_classify_returns_enum_value_not_enum()
    test_ctp_has_classify_order_error_method()
    print("✅ 所有测试通过")
