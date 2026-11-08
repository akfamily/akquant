"""StrategyContext.order_events / find_order 绑定层(经真实上下文驱动).

账本算法的逐步行为见 src/order_ledger.rs 的单元测试; 这里验证 PyO3 绑定:
合并路径与两段式、无事件返回 None、去重跨调用生效、find_order、非法 stage,
以及上下文订单列表为空(接真实柜台时上下文就是这样)不产出任何事件。
"""

from typing import Any

import pytest
from akquant import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Strategy,
    StrategyContext,
    Trade,
)

_OPTIONAL_ARGS = (
    "previous_cash position_entry_prices session current_time active_orders "
    "closed_trades recent_trades recent_expiry_events risk_config strategy_id "
    "account_equity account_market_value account_notional_value "
    "account_used_margin account_unrealized_pnl account_maintenance_ratio "
    "previous_account_equity previous_account_market_value "
    "previous_account_notional_value previous_account_used_margin "
    "previous_account_unrealized_pnl previous_account_maintenance_ratio "
    "margin_accrued_interest margin_daily_interest"
).split()

LIMITS = (10000, 50000)


def make_ctx(**kwargs: Any) -> StrategyContext:
    """构造真实上下文; 未给出的可选参数一律 None."""
    params: dict[str, Any] = dict.fromkeys(_OPTIONAL_ARGS)
    params.update(kwargs)
    return StrategyContext(
        cash=100000.0, positions={}, available_positions={}, **params
    )


def make_order(oid: str) -> Order:
    """构造一张限价买单."""
    return Order(oid, "X", OrderSide.Buy, OrderType.Limit, 100, price=1.0)


def make_trade(tid: str, oid: str) -> Trade:
    """构造一笔成交."""
    return Trade(tid, oid, "X", OrderSide.Buy, 100, 10.0, 0.0, 1, 0)


def test_empty_context_yields_no_events() -> None:
    """订单列表全空(接真实柜台时的情形): 两段都返回 None, 查不到订单."""
    ctx = make_ctx()
    assert ctx.order_events(0, *LIMITS) is None
    assert ctx.order_events(1, *LIMITS) is None
    assert ctx.find_order("A") is None


def test_merged_path_returns_orders_and_trades_once() -> None:
    """第一段无事件时同一次调用接着跑第二段(done=True); 再调一次全被去重."""
    ctx = make_ctx(
        active_orders=[make_order("A")], recent_trades=[make_trade("t1", "A")]
    )
    result = ctx.order_events(0, *LIMITS)
    assert result is not None
    events, trades, done = result
    assert done is True
    assert [(o.id, o.status, fire) for o, fire in events] == [
        ("A", OrderStatus.New, False)
    ]
    assert [t.id for t in trades] == ["t1"]
    assert ctx.order_events(0, *LIMITS) is None
    found = ctx.find_order("A")
    assert found is not None and found.id == "A"


def test_cancel_goes_through_two_stages() -> None:
    """主动撤单: 第一段返回 Cancelled 且 done=False; 第二段里旧的在途快照被去重."""
    ctx = make_ctx(active_orders=[make_order("A")])
    assert ctx.order_events(0, *LIMITS) is not None  # A 先进账本
    ctx.cancel_order("A")
    first = ctx.order_events(0, *LIMITS)
    assert first is not None
    events, trades, done = first
    assert done is False
    assert [(o.id, o.status) for o, _ in events] == [("A", OrderStatus.Cancelled)]
    assert trades == []
    assert ctx.order_events(1, *LIMITS) is None


def test_invalid_stage_raises() -> None:
    """Stage 只能是 0 或 1."""
    with pytest.raises(ValueError, match="stage"):
        make_ctx().order_events(2, *LIMITS)


class _Recorder(Strategy):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_order(self, order: Any) -> None:
        self.events.append(f"order {order.id} {str(order.status).split('.')[-1]}")

    def on_reject(self, order: Any) -> None:
        self.events.append(f"reject {order.id}")

    def on_trade(self, trade: Any) -> None:
        self.events.append(f"trade {trade.id}")


def test_dispatch_through_real_context() -> None:
    """派发经真实上下文: on_order 在前、on_trade 在后; 同一上下文再检查不重复.

    新旧实现都必须通过(Task 3 加入时 Python 仍走旧实现)。
    """
    strategy = _Recorder()
    strategy.ctx = make_ctx(
        active_orders=[make_order("A")], recent_trades=[make_trade("t1", "A")]
    )
    strategy._check_order_events()
    strategy._check_order_events()
    assert strategy.events == ["order A New", "trade t1"]
