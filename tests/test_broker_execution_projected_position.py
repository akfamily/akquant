"""经柜台交易时的在途量投影: 只计本策略自己报出的未完成委托.

回测里 ``get_projected_position()`` / ``get_closable_position()`` 把同一根 bar 内
已报未成的单计入, 目标仓位和 auto 拆腿靠它避免重复下单。实盘此前两者都退回结算仓。
"""

from __future__ import annotations

from typing import Any

from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_models import (
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedPosition,
)
from akquant.gateway.broker_state_cache import BrokerStateCache
from akquant.gateway.broker_strategy_api import wrap_state_invalidation
from akquant.gateway.order_receipt import OrderLeg, OrderReceipt

SYMBOL = "rb2601.SHFE"


class _Gw:
    """假柜台: 多头 10 手, 另有一笔别人报的未完成卖单."""

    def __init__(self, quantity: float = 10.0) -> None:
        self.quantity = quantity

    def query_positions(self) -> list[UnifiedPosition]:
        return [
            UnifiedPosition(
                symbol=SYMBOL,
                quantity=abs(self.quantity),
                available_quantity=abs(self.quantity),
                direction="long" if self.quantity >= 0 else "short",
            )
        ]

    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        return [
            UnifiedOrderSnapshot(
                client_order_id="other",
                broker_order_id="FOREIGN-1",
                symbol=SYMBOL,
                status=UnifiedOrderStatus.SUBMITTED,
                side="Sell",
                quantity=5.0,
            )
        ]


class _Submitter:
    """每次下单按给定的腿返回回执; 不给腿时整单一条腿."""

    def __init__(self) -> None:
        self.seq = 0
        self.next_legs: list[tuple[str, float]] | None = None
        self.fail = False

    def submit_order(self, **kwargs: Any) -> OrderReceipt:
        if self.fail:
            return OrderReceipt(group_id="", order_ids=(), legs=(), failure="rejected")
        plan = self.next_legs or [
            (str(kwargs.get("position_effect") or "auto"), float(kwargs["quantity"]))
        ]
        self.next_legs = None
        legs = []
        for effect, quantity in plan:
            self.seq += 1
            legs.append(OrderLeg(effect, quantity, f"C{self.seq}", f"B{self.seq}"))
        return OrderReceipt(
            group_id=legs[0].client_order_id,
            order_ids=tuple(leg.broker_order_id for leg in legs),
            legs=tuple(legs),
        )

    def _get_execution_capabilities(self) -> dict[str, bool]:
        return {"broker_live": True}


class _S:
    ctx = None
    current_bar = None
    current_tick = None


def _setup(quantity: float = 10.0) -> tuple[BrokerExecution, _Submitter, Any]:
    gw = _Gw(quantity)
    cache = BrokerStateCache(gw)
    submitter = _Submitter()
    execution = BrokerExecution(_S(), gw, cache, submitter)
    events = wrap_state_invalidation(lambda name, payload: None, lambda: [cache])
    return execution, submitter, events


def _views(execution: BrokerExecution) -> tuple[float, float, float]:
    return (
        execution.get_position(SYMBOL),
        execution.get_closable_position(SYMBOL),
        execution.get_projected_position(SYMBOL),
    )


def test_without_own_orders_projection_equals_position() -> None:
    """别人报的挂单(手工单、其他策略)不计入."""
    execution, _, _ = _setup()
    assert _views(execution) == (10.0, 10.0, 10.0)


def test_pending_open_order_counts_in_projected_but_not_closable() -> None:
    """在途开仓单: 计入投影持仓, 不计入可平持仓."""
    execution, _, _ = _setup()
    execution.submit_order(symbol=SYMBOL, side="Buy", quantity=3.0)
    assert _views(execution) == (10.0, 10.0, 13.0)


def test_pending_close_order_reduces_closable_and_projected() -> None:
    """在途平仓单: 可平持仓与投影持仓都扣掉."""
    execution, _, _ = _setup()
    execution.submit_order(
        symbol=SYMBOL, side="Sell", quantity=4.0, position_effect="close"
    )
    assert _views(execution) == (10.0, 6.0, 6.0)


def test_reversal_legs_project_through_zero() -> None:
    """先平后开的反手(卖 15 拆成平 10 + 开 5): 投影持仓为 -5, 可平为 0."""
    execution, submitter, _ = _setup()
    submitter.next_legs = [("close", 10.0), ("open", 5.0)]
    execution.submit_order(symbol=SYMBOL, side="Sell", quantity=15.0)
    assert _views(execution) == (10.0, 0.0, -5.0)


def test_fill_moves_quantity_from_pending_to_position() -> None:
    """成交后在途量相应减少: 投影持仓不因成交而变, 不重复计."""
    execution, _, events = _setup()
    execution.submit_order(
        symbol=SYMBOL, side="Sell", quantity=4.0, position_effect="close"
    )
    # 先读一次, 让持仓从柜台加载(之后的成交按增量叠加, 不重查假柜台)
    assert _views(execution) == (10.0, 6.0, 6.0)
    events(
        "trade",
        {
            "trade_id": "T1",
            "broker_order_id": "B1",
            "symbol": SYMBOL,
            "side": "Sell",
            "quantity": 1.0,
        },
    )
    assert _views(execution) == (9.0, 6.0, 6.0)
    # 同一笔成交重放(断线补齐)不再扣一次
    events(
        "trade",
        {
            "trade_id": "T1",
            "broker_order_id": "B1",
            "symbol": SYMBOL,
            "side": "Sell",
            "quantity": 1.0,
        },
    )
    assert _views(execution) == (9.0, 6.0, 6.0)


def test_terminal_order_event_releases_pending_quantity() -> None:
    """撤单/拒单/过期/全部成交后, 剩余在途量不再计入."""
    for status in (
        UnifiedOrderStatus.CANCELLED,
        UnifiedOrderStatus.REJECTED,
        UnifiedOrderStatus.EXPIRED,
        UnifiedOrderStatus.FILLED,
    ):
        execution, _, events = _setup()
        execution.submit_order(symbol=SYMBOL, side="Buy", quantity=3.0)
        events(
            "order",
            UnifiedOrderSnapshot("C1", "B1", SYMBOL, UnifiedOrderStatus.SUBMITTED),
        )
        assert execution.get_projected_position(SYMBOL) == 13.0
        events("order", UnifiedOrderSnapshot("C1", "B1", SYMBOL, status))
        assert execution.get_projected_position(SYMBOL) == 10.0, status


def test_failed_submit_leaves_no_pending_quantity() -> None:
    """没报出去的单(空回执)不计入."""
    execution, submitter, _ = _setup()
    submitter.fail = True
    execution.submit_order(symbol=SYMBOL, side="Buy", quantity=3.0)
    assert _views(execution) == (10.0, 10.0, 10.0)


def test_other_strategy_orders_are_not_counted() -> None:
    """同一账户的两个策略各有缓存: 一个策略的在途单不影响另一个的投影."""
    gw = _Gw()
    first = BrokerExecution(_S(), gw, BrokerStateCache(gw), _Submitter())
    second = BrokerExecution(_S(), gw, BrokerStateCache(gw), _Submitter())
    first.submit_order(symbol=SYMBOL, side="Buy", quantity=3.0)
    assert first.get_projected_position(SYMBOL) == 13.0
    assert second.get_projected_position(SYMBOL) == 10.0
