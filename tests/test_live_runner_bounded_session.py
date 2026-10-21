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


def test_slots_receive_the_last_event_before_stop() -> None:
    """多 slot 时, 第 N 个事件要派发完所有 slot 才结束会话.

    Rust 对每个事件按 slot 0(主策略)→ slot 1..k 顺序派发
    (src/pipeline/stages/strategy.rs)。若在主策略处理完第 N 个事件时就抛
    KeyboardInterrupt, 后面的 slot 永远收不到最后一根 bar。
    """
    main = _CountingStrategy()
    slot_a = _CountingStrategy()
    slot_b = _CountingStrategy()
    _runner()._apply_bounded_event_limit(main, 2, [slot_a, slot_b])
    for event in ("e1", "e2"):
        main._on_bar_event_and_flush(event, None)
        slot_a._on_bar_event_and_flush(event, None)
        if event == "e2":
            with pytest.raises(KeyboardInterrupt):
                slot_b._on_bar_event_and_flush(event, None)
        else:
            slot_b._on_bar_event_and_flush(event, None)
    assert main.bar_calls == slot_a.bar_calls == slot_b.bar_calls == ["e1", "e2"]


def test_replay_slots_get_every_live_bar() -> None:
    """端到端: replay 有界会话里, 指标脚本 slot 与主策略收到同样多的 bar."""
    import akquant as aq
    import pandas as pd
    from akquant import AssetType, Bar, Instrument

    seen: dict[str, list[float]] = {}

    class Main(aq.Strategy):
        def on_bar(self, bar: Any) -> None:
            seen.setdefault("main", []).append(bar.close)

    class Sub(aq.Indicator):
        def on_bar(self, bar: Any) -> None:
            seen.setdefault("sub", []).append(bar.close)

    day0 = pd.Timestamp("2026-08-25 15:00", tz="Asia/Shanghai")
    bars = [
        Bar(
            timestamp=int((day0 + pd.Timedelta(days=k)).value),
            open=20.0 + k,
            high=20.0 + k,
            low=20.0 + k,
            close=20.0 + k,
            volume=1000.0,
            symbol="600000",
        )
        for k in range(3)
    ]
    aq.run_live(
        strategy_cls=Main,
        instruments=[Instrument("600000", AssetType.Stock, 1.0, 0.01, 100)],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars, "freq": "1d"},
        cash=1e6,
        duration="30s",
        strategy_id="main",
        indicators=[Sub],
    )
    assert seen["main"] == [20.0, 21.0, 22.0]
    assert seen["sub"] == [20.0, 21.0, 22.0]
