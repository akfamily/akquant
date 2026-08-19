"""验证本地止损重试时 client_order_id 的生成逻辑.

本测试锁住问题 2 的核心矛盾: 止损单触发后经 check_stop_triggers 调用
submit_order, 而 submit_order 内部按 `client_order_id or self._next_client_order_id()`
生成 client_order_id。重试时:
  1. 若策略显式传了 client_order_id(落进 order.extra), 它会被复用, 柜台可去重;
  2. 若策略未传, 每次触发都调用 _next_client_order_id() 生成**新** id, 柜台无从去重。

当报单状态未知(超时/断连)时, 报文可能已到达柜台, 重试即真实的重复委托。
现状: 已在 check_stop_triggers 加 `receipt.failure == "unknown"` 分支放弃重试;
但这是**缓解**不是根治——该单首次提交就状态未知时确实不重试了, 但若首次
柜台明确拒单、第二次状态未知, 仍会以两个不同的 client_order_id 报出去。

本测试证明:
  - 策略显式传 client_order_id 时, 重试复用同一 id(边界场景, 正确)
  - 策略未传时, 每次触发生成新 id(主流场景, **有风险**)
"""

from typing import Any, cast
from unittest.mock import MagicMock

from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_models import BrokerCapability, UnifiedErrorType
from akquant.gateway.broker_state_cache import BrokerStateCache
from akquant.gateway.order_submitter import BrokerOrderSubmitter


class _MinimalStrategy:
    """喂给 BrokerOrderSubmitter 的最小策略桩."""

    _owner_strategy_id = "_default"
    broker_ready = True


class _RejectOnceGateway:
    """第一次调用 place_order 抛明确拒单, 第二次成功(模拟重试)."""

    def __init__(self) -> None:
        self.call_count = 0
        self.received_client_order_ids: list[str] = []

    def place_order(self, req: Any) -> str:
        self.call_count += 1
        cid = getattr(req, "client_order_id", "?")
        self.received_client_order_ids.append(cid)
        if self.call_count == 1:
            raise RuntimeError("柜台明确拒单")
        return f"broker-order-{self.call_count}"

    def classify_order_error(self, exc: BaseException) -> UnifiedErrorType:
        return UnifiedErrorType.NON_RETRYABLE  # 明确拒单, 允许重试


class _Cache:
    def positions(self) -> dict[str, float]:
        return {}

    def available_positions(self) -> dict[str, float]:
        return {}

    def open_orders(self) -> list[object]:
        return []

    def account(self) -> None:
        return None


def _build_submitter(gateway: Any, id_generator: Any) -> BrokerOrderSubmitter:
    """构造真实 BrokerOrderSubmitter, 注入自定义 id 生成器."""
    capability = BrokerCapability(broker_name="test")
    return BrokerOrderSubmitter(
        trader_gateway=gateway,
        strategy=_MinimalStrategy(),
        resolve_trader_capabilities=lambda _: capability,
        next_client_order_id=id_generator,
        can_submit_client_order=lambda _: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda *_a, **_k: None,
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: capability.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
    )


def test_retry_without_explicit_client_order_id_generates_new_ids() -> None:
    """策略未显式传 client_order_id 时, 重试会生成新 id(问题现场)."""
    gateway = _RejectOnceGateway()
    # 模拟单调递增的 id 生成器
    counter = {"n": 0}

    def next_id() -> str:
        counter["n"] += 1
        return f"auto-{counter['n']}"

    submitter = _build_submitter(gateway, next_id)
    strategy = MagicMock()
    strategy.on_error = MagicMock()

    ex = BrokerExecution(
        strategy,
        gateway,
        cast(BrokerStateCache, _Cache()),
        submitter,
        record_stop_remap=None,
    )
    # 注册止损单, **未传** client_order_id
    ex.submit_order(
        symbol="AAPL",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=95.0,
    )
    # 第一次触发: 柜台拒单, 重入簿
    ex.check_stop_triggers("AAPL", last=94.5)
    assert gateway.call_count == 1
    assert len(ex.get_open_orders("AAPL")) == 1  # 仍在簿中

    # 第二次触发(重试): 成功
    ex.check_stop_triggers("AAPL", last=94.5)
    assert gateway.call_count == 2
    assert len(ex.get_open_orders("AAPL")) == 0  # 已移出簿

    # 关键断言: 两次提交用了**不同的** client_order_id
    assert len(gateway.received_client_order_ids) == 2
    assert gateway.received_client_order_ids[0] == "auto-1"
    assert gateway.received_client_order_ids[1] == "auto-2"
    # ☝️ 若第一次报文其实已到柜台(状态未知误判成拒单), 这就是重复委托


def test_retry_with_explicit_client_order_id_reuses_same_id() -> None:
    """策略显式传 client_order_id 时, 重试复用同一 id(边界场景, 正确)."""
    gateway = _RejectOnceGateway()

    def next_id() -> str:
        return "fallback-id"  # 不应被调用

    submitter = _build_submitter(gateway, next_id)
    strategy = MagicMock()
    strategy.on_error = MagicMock()

    ex = BrokerExecution(
        strategy,
        gateway,
        cast(BrokerStateCache, _Cache()),
        submitter,
        record_stop_remap=None,
    )
    # 注册止损单, **显式传** client_order_id
    ex.submit_order(
        symbol="AAPL",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=95.0,
        client_order_id="my-stop-1",
    )
    # 第一次触发: 柜台拒单
    ex.check_stop_triggers("AAPL", last=94.5)
    assert gateway.call_count == 1

    # 第二次触发(重试): 成功
    ex.check_stop_triggers("AAPL", last=94.5)
    assert gateway.call_count == 2

    # 关键断言: 两次提交用了**相同的** client_order_id
    assert len(gateway.received_client_order_ids) == 2
    assert gateway.received_client_order_ids[0] == "my-stop-1"
    assert gateway.received_client_order_ids[1] == "my-stop-1"
    # ☝️ 柜台可按 client_order_id 去重, 即使第一次状态未知也不会重复委托


def test_current_code_already_prevents_retry_on_first_unknown() -> None:
    """现有代码已阻止「首次状态未知」时重试(缓解, 但不彻底)."""

    class _UnknownGateway:
        def __init__(self) -> None:
            self.call_count = 0

        def place_order(self, req: Any) -> str:
            self.call_count += 1
            raise TimeoutError("状态未知")

        def classify_order_error(self, exc: BaseException) -> UnifiedErrorType:
            return UnifiedErrorType.RETRYABLE  # 状态未知

    gateway = _UnknownGateway()
    submitter = _build_submitter(gateway, lambda: "auto-id")
    strategy = MagicMock()
    strategy.on_error = MagicMock()

    ex = BrokerExecution(
        strategy,
        gateway,
        cast(BrokerStateCache, _Cache()),
        submitter,
        record_stop_remap=None,
    )
    ex.submit_order(
        symbol="AAPL",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=95.0,
    )
    # 第一次触发: 状态未知, 放弃该单
    ex.check_stop_triggers("AAPL", last=94.5)
    assert gateway.call_count == 1
    assert len(ex.get_open_orders("AAPL")) == 0  # 已放弃, 不重试 ✅

    # 第二次触发: 该单已不在簿中, 不会再调用 place_order
    ex.check_stop_triggers("AAPL", last=94.5)
    assert gateway.call_count == 1  # 仍是 1, 没有第二次调用


def test_solution_b_reject_then_unknown_stops_at_second_attempt() -> None:
    """方案 B: 首次拒单、第二次状态未知时在第二次就放弃(不会有第三次)."""

    class _RejectThenUnknownGateway:
        """第一次明确拒单, 第二次状态未知(最危险的场景)."""

        def __init__(self) -> None:
            self.call_count = 0
            self.received_client_order_ids: list[str] = []

        def place_order(self, req: Any) -> str:
            self.call_count += 1
            cid = getattr(req, "client_order_id", "?")
            self.received_client_order_ids.append(cid)
            if self.call_count == 1:
                raise RuntimeError("柜台明确拒单")
            # 第二次: 状态未知
            raise TimeoutError("超时")

        def classify_order_error(self, exc: BaseException) -> UnifiedErrorType:
            if "拒单" in str(exc):
                return UnifiedErrorType.NON_RETRYABLE  # 明确拒单
            return UnifiedErrorType.RETRYABLE  # 状态未知

    gateway = _RejectThenUnknownGateway()
    counter = {"n": 0}

    def next_id() -> str:
        counter["n"] += 1
        return f"auto-{counter['n']}"

    submitter = _build_submitter(gateway, next_id)
    strategy = MagicMock()
    strategy.on_error = MagicMock()

    ex = BrokerExecution(
        strategy,
        gateway,
        cast(BrokerStateCache, _Cache()),
        submitter,
        record_stop_remap=None,
    )
    ex.submit_order(
        symbol="AAPL",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=95.0,
    )
    # 第一次触发: 柜台明确拒单, 重入簿
    ex.check_stop_triggers("AAPL", last=94.5)
    assert gateway.call_count == 1
    assert len(ex.get_open_orders("AAPL")) == 1  # 仍在簿中, 会重试

    # 第二次触发(重试): 状态未知 → 方案 B 在此放弃
    ex.check_stop_triggers("AAPL", last=94.5)
    assert gateway.call_count == 2
    assert len(ex.get_open_orders("AAPL")) == 0  # 已放弃, 不再重试 ✅

    # 第三次触发: 该单已不在簿中, 不会再调用 place_order
    ex.check_stop_triggers("AAPL", last=94.5)
    assert gateway.call_count == 2  # 仍是 2, 没有第三次

    # 关键断言: 只用了两个 client_order_id(而非三个)
    assert len(gateway.received_client_order_ids) == 2
    assert gateway.received_client_order_ids == ["auto-1", "auto-2"]
    # ☝️ 方案 B 把风险窗口从"无限次"降到"最多两次"


if __name__ == "__main__":
    test_retry_without_explicit_client_order_id_generates_new_ids()
    test_retry_with_explicit_client_order_id_reuses_same_id()
    test_current_code_already_prevents_retry_on_first_unknown()
    print("✅ 所有测试通过")
