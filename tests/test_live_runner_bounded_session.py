"""有界 live 会话: 事件放完后自行终止, 不依赖墙钟.

live 循环在 channel 空时返回 FeedAction::Wait 并无限循环
(src/data/feed.rs:358-366), 而 Engine 未向 Python 暴露 stop()。因此复用
duration 已验证的模式: 包装策略回调, 条件满足时抛 KeyboardInterrupt,
由 run() 的 except 接住(_runner.py:466)。
"""

from typing import Any, List

import pytest
from akquant.live._runner import LiveRunner


class _CountingStrategy:
    """记录框架入口调用次数的假策略."""

    def __init__(self) -> None:
        """初始化调用记录."""
        self.bar_calls: List[Any] = []
        self.tick_calls: List[Any] = []

    def _on_bar_event_and_flush(self, bar: Any, ctx: Any) -> str:
        """框架 bar 入口."""
        self.bar_calls.append(bar)
        return "plans"

    def _on_tick_event_and_flush(self, tick: Any, ctx: Any) -> str:
        """框架 tick 入口."""
        self.tick_calls.append(tick)
        return "plans"


def _runner() -> LiveRunner:
    """构造裸 LiveRunner（不走 __init__）."""
    return LiveRunner.__new__(LiveRunner)


def test_stops_after_expected_event_count() -> None:
    """第 N 个事件处理完后抛 KeyboardInterrupt."""
    strategy = _CountingStrategy()
    _runner()._apply_bounded_event_limit(strategy, 3)

    strategy._on_bar_event_and_flush("b1", None)
    strategy._on_bar_event_and_flush("b2", None)
    with pytest.raises(KeyboardInterrupt):
        strategy._on_bar_event_and_flush("b3", None)

    assert strategy.bar_calls == ["b1", "b2", "b3"], "第 N 个事件必须先处理完再停"


def test_counts_bars_and_ticks_together() -> None:
    """Bar 与 tick 共用同一计数器."""
    strategy = _CountingStrategy()
    _runner()._apply_bounded_event_limit(strategy, 2)

    strategy._on_bar_event_and_flush("b1", None)
    with pytest.raises(KeyboardInterrupt):
        strategy._on_tick_event_and_flush("t1", None)

    assert strategy.bar_calls == ["b1"]
    assert strategy.tick_calls == ["t1"]


def test_returns_original_value_before_limit() -> None:
    """未达上限时必须透传原返回值(引擎依赖它取待注册 timer)."""
    strategy = _CountingStrategy()
    _runner()._apply_bounded_event_limit(strategy, 5)

    assert strategy._on_bar_event_and_flush("b1", None) == "plans"


def test_zero_total_installs_nothing() -> None:
    """total<=0 不安装包装器(非有界会话, 例如真实柜台).

    判据用 ``__dict__``, 不用 ``is`` 比较绑定方法: ``s.m is s.m`` 恒为 False
    (每次属性访问都新建绑定方法对象), 用 ``is`` 无法区分"已包装"与"未包装"。
    包装器是通过 ``setattr`` 写进实例 ``__dict__`` 的, 因此该名是否出现在
    ``__dict__`` 里才是可靠判据。
    """
    strategy = _CountingStrategy()

    _runner()._apply_bounded_event_limit(strategy, 0)

    assert "_on_bar_event_and_flush" not in strategy.__dict__
    assert "_on_tick_event_and_flush" not in strategy.__dict__
    # 且调用行为不变(不会抛)
    assert strategy._on_bar_event_and_flush("b1", None) == "plans"


def test_missing_entry_point_is_tolerated() -> None:
    """策略缺少某个入口时不应报错(仅包装存在的那个)."""

    class _BarOnly:
        def _on_bar_event_and_flush(self, bar: Any, ctx: Any) -> None:
            return None

    strategy = _BarOnly()
    _runner()._apply_bounded_event_limit(strategy, 1)

    with pytest.raises(KeyboardInterrupt):
        strategy._on_bar_event_and_flush("b1", None)
