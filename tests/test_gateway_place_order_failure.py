"""柜台报单失败必须落成拒单事件或状态未知, 不得把异常抛穿策略回调.

**修复前的缺陷**: `order_submitter.py` 里 `self._trader_gateway.place_order(request)`
是整条下单链路上唯一没有保护的跨边界调用。中间件返回 HTTP 400 时异常直接从
`self.buy()` 炸给策略: 不进审计流水、不触发 `on_reject`、多腿单前面的腿已发出
而回执根本没构造出来。

同文件 `_validate_price_tick` 与前置风控两条路径早就做对了(record_reject +
_emit_risk_reject + 空回执), 该注释就写在 `:634`。本测试锁住网关调用点也照此办理。
"""

from typing import Any

from akquant.gateway.broker_models import (
    BrokerCapability,
    UnifiedErrorType,
    UnifiedOrderRequest,
)
from akquant.gateway.order_submitter import BrokerOrderSubmitter


class _Strategy:
    """记录 on_reject 的最小策略对象."""

    _owner_strategy_id = "_default"

    def __init__(self) -> None:
        self.rejected: list[Any] = []

    def on_reject(self, order: Any) -> None:
        self.rejected.append(order)


class _FailingGateway:
    """place_order 必失败的网关; 分类结果由构造参数决定."""

    def __init__(self, classify: Any = "__omit__") -> None:
        self.calls = 0
        self._classify = classify
        if classify != "__omit__":
            self.classify_order_error = self._classify_impl  # type: ignore[method-assign]

    def _classify_impl(self, exc: BaseException) -> Any:
        if self._classify == "__raise__":
            raise ValueError("classifier boom")
        return self._classify

    def place_order(self, req: UnifiedOrderRequest) -> str:
        self.calls += 1
        raise RuntimeError("HTTP 400 [251005] 证券可用数量不足")


def _make_submitter(
    gateway: Any, strategy: _Strategy, errors: list[tuple[Exception, str]]
) -> BrokerOrderSubmitter:
    """构造一个只接桩网关的 submitter(夹具口径同 test_gateway_order_submitter_extra)."""
    capability = BrokerCapability(broker_name="failing-fake")
    return BrokerOrderSubmitter(
        trader_gateway=gateway,
        strategy=strategy,
        resolve_trader_capabilities=lambda _gw: capability,
        next_client_order_id=lambda: "c1",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda _s, exc, source, _p: errors.append((exc, source)),
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: capability.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
    )


def _submit(submitter: BrokerOrderSubmitter) -> Any:
    """报一笔最普通的限价买单."""
    return submitter.submit_order(
        symbol="600000.SH", side="Buy", quantity=100, price=10.5, order_type="Limit"
    )


def test_definite_reject_becomes_on_reject_not_exception() -> None:
    """柜台明确回绝 → on_reject + 空回执, 不抛异常."""
    strategy = _Strategy()
    errors: list[tuple[Exception, str]] = []
    gateway = _FailingGateway(UnifiedErrorType.NON_RETRYABLE)
    receipt = _submit(_make_submitter(gateway, strategy, errors))

    assert len(receipt) == 0, "明确拒单应返回空回执"
    assert str(receipt) == "", "空回执的 group_id 应为空串(同风控拒单口径)"
    assert len(strategy.rejected) == 1, "明确拒单必须触发 on_reject"
    assert "251005" in str(strategy.rejected[0].reject_reason)
    assert strategy.rejected[0].symbol == "600000.SH"


def test_unknown_state_does_not_fake_a_reject() -> None:
    """状态未知 → 只走 on_error, 绝不回吐 Rejected(否则策略会重下单)."""
    strategy = _Strategy()
    errors: list[tuple[Exception, str]] = []
    gateway = _FailingGateway(UnifiedErrorType.RETRYABLE)
    receipt = _submit(_make_submitter(gateway, strategy, errors))

    assert len(receipt) == 0
    assert strategy.rejected == [], "状态未知不得谎报拒单"
    assert [source for _exc, source in errors] == ["order_submit"]


def test_gateway_without_classifier_is_treated_as_unknown() -> None:
    """网关没实现分类方法 → 保守走状态未知."""
    strategy = _Strategy()
    errors: list[tuple[Exception, str]] = []
    receipt = _submit(_make_submitter(_FailingGateway(), strategy, errors))

    assert len(receipt) == 0
    assert strategy.rejected == []
    assert [source for _exc, source in errors] == ["order_submit"]


def test_exploding_classifier_is_treated_as_unknown() -> None:
    """分类方法自身抛错也不得二次崩."""
    strategy = _Strategy()
    errors: list[tuple[Exception, str]] = []
    gateway = _FailingGateway("__raise__")
    receipt = _submit(_make_submitter(gateway, strategy, errors))

    assert len(receipt) == 0
    assert strategy.rejected == []
    assert [source for _exc, source in errors] == ["order_submit"]


class _Position:
    """今仓/昨仓各 5 手的多头持仓, 用于把一笔平仓拆成两腿."""

    symbol = "600000.SH"
    direction = "long"
    quantity = 10.0
    available_quantity = 10.0
    available_today_quantity = 5.0
    available_yesterday_quantity = 5.0


class _SecondLegFailsGateway:
    """第一腿正常回单号, 第二腿抛柜台业务错误."""

    def __init__(self, classify: Any) -> None:
        self.placed: list[UnifiedOrderRequest] = []
        self._classify = classify

    def classify_order_error(self, exc: BaseException) -> Any:
        return self._classify

    def query_positions(self) -> list[Any]:
        return [_Position()]

    def place_order(self, req: UnifiedOrderRequest) -> str:
        self.placed.append(req)
        if len(self.placed) == 1:
            return "broker-1"
        raise RuntimeError("HTTP 400 [251020] 昨仓不可平")


def _make_split_submitter(
    gateway: Any, strategy: _Strategy, errors: list[tuple[Exception, str]]
) -> BrokerOrderSubmitter:
    """能力声明支持今昨拆腿的 submitter."""
    capability = BrokerCapability(
        broker_name="split-fake",
        position_effect=True,
        position_details=True,
        supported_position_effects=(
            "auto",
            "open",
            "close",
            "close_today",
            "close_yesterday",
        ),
    )
    return BrokerOrderSubmitter(
        trader_gateway=gateway,
        strategy=strategy,
        resolve_trader_capabilities=lambda _gw: capability,
        next_client_order_id=lambda: "c1",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda _s, exc, source, _p: errors.append((exc, source)),
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: capability.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
    )


def _submit_close(submitter: BrokerOrderSubmitter) -> Any:
    """报一笔会被拆成 close_today + close_yesterday 两腿的平仓单."""
    return submitter.submit_order(
        symbol="600000.SH",
        side="Sell",
        quantity=10,
        price=10.5,
        order_type="Limit",
        position_effect="close",
    )


def test_multi_leg_keeps_successful_leg_on_definite_reject() -> None:
    """第二腿被柜台拒 → 回执仍带第一腿 id, 且失败腿有 Rejected 事件."""
    strategy = _Strategy()
    errors: list[tuple[Exception, str]] = []
    gateway = _SecondLegFailsGateway(UnifiedErrorType.NON_RETRYABLE)
    receipt = _submit_close(_make_split_submitter(gateway, strategy, errors))

    assert len(gateway.placed) == 2, "应确实拆成了两腿并都尝试报出"
    assert len(receipt) == 1, "已成功的第一腿必须保留在回执里, 否则策略无从撤它"
    assert receipt.primary == "broker-1"
    assert len(receipt.legs) == 1
    assert len(strategy.rejected) == 1, "失败腿应有 Rejected 事件"


def test_multi_leg_unknown_state_still_returns_partial_receipt() -> None:
    """第二腿状态未知 → 回执同样保留第一腿, 但不谎报 Rejected."""
    strategy = _Strategy()
    errors: list[tuple[Exception, str]] = []
    gateway = _SecondLegFailsGateway(UnifiedErrorType.RETRYABLE)
    receipt = _submit_close(_make_split_submitter(gateway, strategy, errors))

    assert len(receipt) == 1
    assert receipt.primary == "broker-1"
    assert strategy.rejected == [], "状态未知不得谎报拒单, 多腿也一样"
    assert [source for _exc, source in errors] == ["order_submit"]


class _Instrument:
    """带 tick_size 的股票标的桩, 用于触发本地 tick 校验拒单."""

    asset_type = "STOCK"
    tick_size = 0.01


class _TickStrategy(_Strategy):
    """能返回标的信息的策略桩(基类没有 get_instrument, tick 校验会被跳过)."""

    def get_instrument(self, symbol: str) -> _Instrument:
        return _Instrument()


class _OkGateway:
    """place_order 正常回单号的网关桩."""

    def place_order(self, req: UnifiedOrderRequest) -> str:
        return "broker-1"


def test_normal_receipt_has_no_failure() -> None:
    """正常回执 failure 为 None."""
    errors: list[tuple[Exception, str]] = []
    submitter = _make_submitter(_OkGateway(), _Strategy(), errors)
    receipt = _submit(submitter)
    assert receipt.primary == "broker-1"
    assert receipt.failure is None


def test_zero_quantity_is_not_a_failure() -> None:
    """quantity<=0 返回空回执, 但那是「无需交易」而非失败."""
    errors: list[tuple[Exception, str]] = []
    submitter = _make_submitter(_OkGateway(), _Strategy(), errors)
    receipt = submitter.submit_order(
        symbol="600000.SH", side="Buy", quantity=0, price=10.5, order_type="Limit"
    )
    assert len(receipt) == 0
    assert receipt.failure is None


def test_tick_reject_is_rejected() -> None:
    """本地 tick 校验拒单 → rejected(订单确定不存在)."""
    errors: list[tuple[Exception, str]] = []
    strategy = _TickStrategy()
    submitter = _make_submitter(_OkGateway(), strategy, errors)
    receipt = submitter.submit_order(
        symbol="600000.SH",
        side="Buy",
        quantity=100,
        price=10.503,  # 不是 0.01 的整数倍
        order_type="Limit",
    )
    assert receipt.failure == "rejected"
    assert len(strategy.rejected) == 1


def test_risk_reject_is_rejected() -> None:
    """前置风控拒单 → rejected."""
    errors: list[tuple[Exception, str]] = []
    strategy = _Strategy()
    capability = BrokerCapability(broker_name="risk-fake")
    submitter = BrokerOrderSubmitter(
        trader_gateway=_OkGateway(),
        strategy=strategy,
        resolve_trader_capabilities=lambda _gw: capability,
        next_client_order_id=lambda: "c1",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda _s, exc, source, _p: errors.append((exc, source)),
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: capability.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
        strategy_limits={"max_order_size": {"_default": 1.0}},
    )
    receipt = submitter.submit_order(
        symbol="600000.SH", side="Buy", quantity=100, price=10.5, order_type="Limit"
    )
    assert receipt.failure == "rejected"
    assert len(strategy.rejected) == 1


def test_broker_definite_reject_is_rejected() -> None:
    """柜台明确回绝 → rejected."""
    errors: list[tuple[Exception, str]] = []
    gateway = _FailingGateway(UnifiedErrorType.NON_RETRYABLE)
    receipt = _submit(_make_submitter(gateway, _Strategy(), errors))
    assert receipt.failure == "rejected"


def test_unknown_state_is_unknown() -> None:
    """状态未知 → unknown(订单可能已在柜台)."""
    errors: list[tuple[Exception, str]] = []
    gateway = _FailingGateway(UnifiedErrorType.RETRYABLE)
    receipt = _submit(_make_submitter(gateway, _Strategy(), errors))
    assert receipt.failure == "unknown"
