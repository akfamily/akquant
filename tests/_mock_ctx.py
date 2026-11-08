"""伪造 StrategyContext 的测试辅助.

订单事件的比对已移到 Rust 订单账本(``StrategyContext.order_events``)。不经引擎、
直接用伪造上下文驱动策略的测试, 用 :func:`mock_strategy_context` 得到一个
"没有订单事件"的上下文; 要测派发顺序时用 :func:`queue_order_events` 指定每次
检查返回什么(格式同 ``order_events``)。
"""

from typing import Any
from unittest.mock import MagicMock

from akquant import StrategyContext


def mock_strategy_context() -> MagicMock:
    ctx = MagicMock(spec=StrategyContext)
    ctx.order_events.return_value = None
    ctx.find_order.return_value = None
    return ctx


def queue_order_events(ctx: MagicMock, *results: Any) -> None:
    """依次返回 results 中的元素, 用完后返回 None(没有事件)."""
    pending = list(results)
    ctx.order_events.side_effect = lambda *args, **kwargs: (
        pending.pop(0) if pending else None
    )
