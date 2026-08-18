"""check_stop_triggers: 失败重试(不崩)+ on_error + 成功 remap 记录."""

from typing import Any, cast

from akquant.gateway.broker_execution import MAX_STOP_SUBMIT_ATTEMPTS, BrokerExecution
from akquant.gateway.broker_models import BrokerCapability, UnifiedErrorType
from akquant.gateway.broker_state_cache import BrokerStateCache
from akquant.gateway.order_receipt import OrderReceipt
from akquant.gateway.order_submitter import BrokerOrderSubmitter


class _Cache:
    def positions(self) -> dict[str, float]:
        return {}

    def available_positions(self) -> dict[str, float]:
        return {}

    def open_orders(self) -> list[object]:
        return []

    def account(self) -> None:
        return None


class _Gw:
    def cancel_order(self, bid: str) -> None:
        return None

    def sync_open_orders(self) -> list[object]:
        return []


class _OkSub:
    def __init__(self) -> None:
        self.n = 0

    def submit_order(self, **kw: Any) -> OrderReceipt:
        self.n += 1
        # group_id(client) 与 broker_order_id 故意不同: remaps 断言锁定
        # broker_order_id(.primary), 避免 str(receipt)==broker_order_id 掩盖回归。
        return OrderReceipt.single(group_id="CID-9", broker_order_id="BID-9")


class _FailSub:
    def __init__(self) -> None:
        self.n = 0

    def submit_order(self, **kw: Any) -> str:
        self.n += 1
        raise RuntimeError("broker not ready")


class _S:
    current_bar = None
    current_tick = None

    def __init__(self) -> None:
        self.errors: list[tuple[str, Any]] = []

    def on_error(self, exc: Exception, source: str, payload: Any = None) -> None:
        self.errors.append((source, payload))


def test_success_records_remap() -> None:
    """止损触发提交成功后应调用 record_stop_remap(local_id, broker_order_id)."""
    remaps: list[tuple[str, str]] = []
    ex = BrokerExecution(
        _S(),
        _Gw(),
        cast(BrokerStateCache, _Cache()),
        _OkSub(),
        record_stop_remap=lambda lid, bid: remaps.append((lid, bid)),
    )
    oid = ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    ex.check_stop_triggers("X", last=9.4, high=9.6, low=9.3)
    assert remaps == [(oid.primary, "BID-9")]


def test_failure_requeues_and_notifies_then_gives_up() -> None:
    """止损触发提交失败应重试(上限 MAX_STOP_SUBMIT_ATTEMPTS)+on_error, 不崩溃."""
    s = _S()
    ex = BrokerExecution(s, _Gw(), cast(BrokerStateCache, _Cache()), _FailSub())
    ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    # attempt 1: fails → requeued, on_error, still in book, no raise
    ex.check_stop_triggers("X", last=9.4, high=9.6, low=9.3)
    assert len(ex.get_open_orders("X")) == 1
    assert len(s.errors) == 1
    # attempts 2..MAX: keeps failing; after MAX total attempts, dropped
    for _ in range(MAX_STOP_SUBMIT_ATTEMPTS):
        ex.check_stop_triggers("X", last=9.4, high=9.6, low=9.3)
    assert ex.get_open_orders("X") == []  # given up
    assert len(s.errors) == MAX_STOP_SUBMIT_ATTEMPTS


class _SubmitterStrategy:
    """喂给真实 BrokerOrderSubmitter 的最小策略桩(与 BrokerExecution 的 _S 无关)."""

    _owner_strategy_id = "_default"


class _BrokerRejectingGateway:
    """place_order 必抛柜台异常, classify_order_error 归类为明确拒单."""

    def place_order(self, req: Any) -> str:
        raise RuntimeError("HTTP 400 [251005] broker rejected")

    def classify_order_error(self, exc: BaseException) -> UnifiedErrorType:
        return UnifiedErrorType.NON_RETRYABLE


def _real_submitter(gateway: Any) -> BrokerOrderSubmitter:
    """构造一个真实 BrokerOrderSubmitter(非桩), 复现 Task 3 改动后的空回执路径."""
    capability = BrokerCapability(broker_name="reject-fake")
    return BrokerOrderSubmitter(
        trader_gateway=gateway,
        strategy=_SubmitterStrategy(),
        resolve_trader_capabilities=lambda _gw: capability,
        next_client_order_id=lambda: "c1",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda *_a, **_k: None,
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: capability.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
    )


def test_broker_reject_requeues_and_notifies_via_real_submitter() -> None:
    """柜台明确拒单(空回执, 非异常穿透)仍须走止损重试+on_error(Task 3 回归).

    Task 3 把 `BrokerOrderSubmitter.submit_order` 对可分类柜台失败的处置从
    "异常穿透" 改成 "返回空 OrderReceipt"; `BrokerExecution.check_stop_triggers`
    此前只在 `except Exception` 里做重试记账+on_error, 对空回执视而不见——本
    用例用真实 submitter(而非上面 `_FailSub` 的桩异常)复现该链路, 锁住修复。
    """
    s = _S()
    submitter = _real_submitter(_BrokerRejectingGateway())
    ex = BrokerExecution(s, _Gw(), cast(BrokerStateCache, _Cache()), submitter)
    ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    ex.check_stop_triggers("X", last=9.4, high=9.6, low=9.3)
    assert len(ex.get_open_orders("X")) == 1, "空回执应等价于提交失败, 重入 stop book"
    assert len(s.errors) == 1
    assert s.errors[0][0] == "stop_trigger"
