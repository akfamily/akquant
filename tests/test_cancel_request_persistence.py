"""撤单请求必须被持久记住, 跨拍仍生效.

``ctx.canceled_order_ids`` 只是**当拍增量**; ``ctx.active_orders`` 每次访问返回
新的 Rust 包装对象, 对其 ``status`` 的写入不落回引擎。因此撤单请求若未被策略侧
持久记录, 一旦订单在撤单后仍停留在 active_orders 超过一拍(异步撤单, 实盘常态),
它就会重新出现在 get_open_orders 里, 且 get_order 报告的状态仍是 New。
"""

from types import SimpleNamespace
from typing import Any, List

from akquant import Strategy
from akquant.akquant import OrderStatus

ORDER_ID = "ORDER-1"


class _AsyncCancelCtx:
    """还原 Rust StrategyContext 的两个关键语义.

    1. ``active_orders`` 每次返回**新**包装对象(对其写 status 不落回引擎);
    2. ``cancel_order`` 是异步的, 本拍不把 id 放进 ``canceled_order_ids``。
    """

    def __init__(self) -> None:
        """初始化空的当拍增量视图."""
        self.canceled_order_ids: List[str] = []
        self.recent_trades: List[Any] = []
        self.cancel_calls: List[str] = []

    @property
    def active_orders(self) -> List[Any]:
        """每次构造新对象, 模拟 pyclass 包装语义."""
        return [
            SimpleNamespace(
                id=ORDER_ID,
                symbol="DEMO",
                status=OrderStatus.New,
                filled_quantity=0.0,
                average_filled_price=None,
            )
        ]

    def cancel_order(self, order_id: str) -> None:
        """记录撤单调用, 但本拍不确认(不写入 canceled_order_ids)."""
        self.cancel_calls.append(order_id)


class _Plain(Strategy):
    """最简策略, 仅用于驱动交易 API."""

    def on_bar(self, bar: Any) -> None:
        """不做任何事."""


def _strategy_with_ctx() -> tuple[_Plain, _AsyncCancelCtx]:
    strategy = _Plain()
    ctx = _AsyncCancelCtx()
    strategy.ctx = ctx  # type: ignore[assignment]
    return strategy, ctx


def test_cancel_request_is_recorded_persistently() -> None:
    """撤单请求必须落到策略实例上, 而非写进一个随即丢弃的临时集合."""
    strategy, ctx = _strategy_with_ctx()

    strategy.cancel_order(ORDER_ID)

    assert ctx.cancel_calls == [ORDER_ID], "撤单未透传到引擎, 测试前提不成立"
    assert ORDER_ID in getattr(strategy, "_pending_canceled_order_ids", set())


def test_cancelled_order_stays_out_of_open_orders_across_bars() -> None:
    """已发出撤单的订单不得在后续拍重新出现在未完成订单里."""
    strategy, _ctx = _strategy_with_ctx()

    strategy.cancel_order(ORDER_ID)

    # 下一拍: 当拍增量已清空, 订单仍留在 active_orders(引擎尚未确认撤单)。
    open_ids = [getattr(order, "id", "") for order in strategy.get_open_orders()]
    assert ORDER_ID not in open_ids


def test_cancelled_order_reports_cancelled_status_across_bars() -> None:
    """已发出撤单的订单在后续拍查询时应报告 Cancelled, 而非 New."""
    strategy, _ctx = _strategy_with_ctx()

    strategy.cancel_order(ORDER_ID)

    order = strategy.get_order(ORDER_ID)
    assert order is not None, "订单查不到, 测试前提不成立"
    assert order.status == OrderStatus.Cancelled


def test_pending_cancel_records_are_bounded() -> None:
    """撤单意图集合必须有上限.

    paper 模式走 SimExecution 且可长跑数日, 无上限时该集合只增不减。
    """
    strategy, _ctx = _strategy_with_ctx()
    strategy.pending_cancel_cache_size = 2  # type: ignore[attr-defined]

    for index in range(4):
        strategy.cancel_order(f"OID-{index}")

    pending: set[str] = getattr(strategy, "_pending_canceled_order_ids", set())
    assert len(pending) == 2, f"未按上限淘汰: {sorted(pending)}"
    assert "OID-3" in pending, "最新的撤单意图应保留"
    assert "OID-0" not in pending, "最旧的撤单意图应被淘汰"
