"""撤单失败必须逐单隔离并留下结果审计, 不得中断整轮 cancel_all_orders.

**修复前的三个缺陷**(`broker_execution.py`):
1. `:275` `self._gw.cancel_order(...)` 裸调用, 撤单失败直接炸给策略;
2. `:271` `record_cancel` 发在调用之前, 撤单被拒时审计里却写着"已派发撤单",
   没有任何失败记录;
3. `:286-295` `cancel_all_orders` 循环无隔离——第一单撤单报错, 剩余单全都不撤。

平台 8/11 反馈的 `cancel_all_orders` 报 HTTP 400
`[251020][委托状态错误不能撤单]` 正是踩在第 3 条。
"""

import logging
from typing import Any

import pytest
from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_state_cache import BrokerStateCache


class _Strategy:
    """记录 on_error 的最小策略对象."""

    strategy_id = "alpha"

    def __init__(self) -> None:
        self.errors: list[tuple[Exception, str]] = []

    def on_error(self, exc: Exception, source: str, payload: Any = None) -> None:
        self.errors.append((exc, source))


class _OpenOrder:
    """sync_open_orders 返回的挂单快照."""

    def __init__(self, broker_order_id: str, symbol: str = "600000.SH") -> None:
        self.broker_order_id = broker_order_id
        self.symbol = symbol


class _Gateway:
    """指定哪些单号撤单会失败的网关桩."""

    def __init__(self, failing: set[str], open_orders: list[_OpenOrder]) -> None:
        self.failing = failing
        self._open_orders = open_orders
        self.cancelled: list[str] = []

    def sync_open_orders(self) -> list[_OpenOrder]:
        return list(self._open_orders)

    def cancel_order(self, broker_order_id: str) -> None:
        if broker_order_id in self.failing:
            raise RuntimeError("HTTP 400 [251020] 委托状态错误不能撤单")
        self.cancelled.append(broker_order_id)


def _make_execution(strategy: _Strategy, gateway: _Gateway) -> BrokerExecution:
    """构造只用到撤单路径的 BrokerExecution."""
    # BrokerStateCache 需要网关引用(查柜台失效), 撤单路径不碰它, 传同一个桩即可。
    return BrokerExecution(
        strategy=strategy,
        trader_gateway=gateway,
        state_cache=BrokerStateCache(gateway),
        submitter=None,
    )


def test_cancel_order_failure_is_not_raised() -> None:
    """单笔撤单失败 → on_error, 不抛."""
    strategy = _Strategy()
    gateway = _Gateway(failing={"B1"}, open_orders=[])
    _make_execution(strategy, gateway).cancel_order("B1")

    assert [source for _exc, source in strategy.errors] == ["order_cancel"]


def test_cancel_order_failure_records_intent_and_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """审计要成对: 调用前的意图 + 失败后的结果."""
    strategy = _Strategy()
    gateway = _Gateway(failing={"B1"}, open_orders=[])
    with caplog.at_level(logging.DEBUG, logger="akquant.audit.order"):
        _make_execution(strategy, gateway).cancel_order("B1")

    events = [getattr(rec, "event", None) for rec in caplog.records]
    assert "order_cancel" in events, "调用前的意图记录要保留"
    assert "order_cancel_failed" in events, "失败结果必须也留一条"


def test_cancel_all_continues_after_a_failure() -> None:
    """中间一单撤单失败不得中断整轮——用户调 cancel_all 就是要全撤."""
    strategy = _Strategy()
    gateway = _Gateway(
        failing={"B2"},
        open_orders=[_OpenOrder("B1"), _OpenOrder("B2"), _OpenOrder("B3")],
    )
    _make_execution(strategy, gateway).cancel_all_orders()

    assert gateway.cancelled == ["B1", "B3"], "失败单之后的单仍必须被撤"
    assert [source for _exc, source in strategy.errors] == ["order_cancel"]


def test_cancel_all_survives_sync_open_orders_failure() -> None:
    """查挂单本身失败也不得抛穿策略."""

    class _BrokenGateway(_Gateway):
        def sync_open_orders(self) -> list[_OpenOrder]:
            raise RuntimeError("HTTP 500 网关不可用")

    strategy = _Strategy()
    gateway = _BrokenGateway(failing=set(), open_orders=[])
    _make_execution(strategy, gateway).cancel_all_orders()

    assert [source for _exc, source in strategy.errors] == ["order_cancel_all"]
