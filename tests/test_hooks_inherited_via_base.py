"""中间基类定义的框架钩子必须被识别为"已重写".

回归场景: 用户把 on_before_trading / on_after_trading / on_pre_open 等钩子写在
一层公共基类里, 具体策略只继承不重写。此前 ``_strategy_overrides_callback``
只比较 MRO 中第一个定义该方法的基类, 于是"用户基类 vs 用户基类"自比得到
False, 钩子被 dispatch_time_hooks / collect_pre_open_timer_entries 整段跳过,
静默不触发。
"""

import pickle
from typing import Any

import pandas as pd
from akquant import Strategy, run_backtest
from akquant.akquant import Bar
from akquant.backtest.fill_mode import CurrentClose
from akquant.strategy_framework_hooks import (
    _needs_portfolio_update,
    _needs_time_hooks,
    _strategy_overrides_callback,
)

SYMBOL = "HOOKS_BASE_DEMO"


def _two_day_bars() -> list[Bar]:
    """构造两个交易日各两根 bar, 足以跨日触发 before/after 钩子."""
    stamps = [
        ("2023-01-03 09:30:00", 10.0),
        ("2023-01-03 15:00:00", 10.5),
        ("2023-01-04 09:30:00", 10.8),
        ("2023-01-04 15:00:00", 10.6),
    ]
    return [
        Bar(
            timestamp=pd.Timestamp(text, tz="Asia/Shanghai").value,
            open=close,
            high=close + 0.2,
            low=close - 0.2,
            close=close,
            volume=1000.0,
            symbol=SYMBOL,
        )
        for text, close in stamps
    ]


class HookBase(Strategy):
    """用户侧公共基类: 钩子全部定义在这里."""

    def __init__(self) -> None:
        """初始化事件记录容器."""
        self.events: list[tuple[str, object]] = []

    def on_start(self) -> None:
        """订阅测试标的."""
        self.subscribe(SYMBOL)

    def on_before_trading(self, trading_date: object, timestamp: int) -> None:
        """记录开盘前钩子."""
        self.events.append(("before", trading_date))

    def on_after_trading(self, trading_date: object, timestamp: int) -> None:
        """记录收盘后钩子."""
        self.events.append(("after", trading_date))

    def on_pre_open(self, event: dict[str, Any]) -> None:
        """记录 pre-open 钩子."""
        self.events.append(("pre_open", event.get("trading_date")))

    def on_portfolio_update(self, snapshot: dict[str, Any]) -> None:
        """记录组合更新钩子."""
        self.events.append(("portfolio", None))


class InheritedHookStrategy(HookBase):
    """具体策略: 只继承钩子, 不重写."""

    def on_bar(self, bar: Bar) -> None:
        """记录 bar, 以证明策略本身确实在跑."""
        self.events.append(("bar", int(bar.timestamp)))


def _run(strategy: Strategy) -> None:
    run_backtest(
        data=_two_day_bars(),
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=10_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )


def test_day_hooks_defined_in_intermediate_base_fire_in_backtest() -> None:
    """继承自中间基类的 on_before_trading / on_after_trading 必须触发."""
    strategy = InheritedHookStrategy()
    _run(strategy)

    kinds = [kind for kind, _ in strategy.events]
    assert "bar" in kinds, "策略未收到 bar, 测试前提不成立"
    assert "before" in kinds, "on_before_trading 未触发"
    assert "after" in kinds, "on_after_trading 未触发"


def test_pre_open_hook_defined_in_intermediate_base_fires_in_backtest() -> None:
    """继承自中间基类的 on_pre_open 必须触发."""
    strategy = InheritedHookStrategy()
    _run(strategy)

    assert any(kind == "pre_open" for kind, _ in strategy.events), "on_pre_open 未触发"


def test_portfolio_hook_defined_in_intermediate_base_fires_in_backtest() -> None:
    """继承自中间基类的 on_portfolio_update 必须触发."""
    strategy = InheritedHookStrategy()
    _run(strategy)

    assert any(kind == "portfolio" for kind, _ in strategy.events), (
        "on_portfolio_update 未触发"
    )


def test_overrides_detection_sees_intermediate_base_definition() -> None:
    """重写判定应穿透中间基类, 与框架基类 Strategy 的默认实现比较."""
    strategy = object.__new__(InheritedHookStrategy)

    for name in (
        "on_before_trading",
        "on_after_trading",
        "on_pre_open",
        "on_portfolio_update",
    ):
        assert _strategy_overrides_callback(strategy, name), f"{name} 应判定为已重写"


def test_overrides_detection_still_false_without_override() -> None:
    """未重写时仍须判定为 False, 否则性能短路失效."""

    class Plain(Strategy):
        def on_bar(self, bar: Bar) -> None:
            """仅重写 on_bar."""

    class PlainChild(Plain):
        pass

    for cls in (Plain, PlainChild):
        strategy = object.__new__(cls)
        for name in (
            "on_before_trading",
            "on_after_trading",
            "on_pre_open",
            "on_cross_section",
            "on_portfolio_update",
        ):
            assert not _strategy_overrides_callback(strategy, name), (
                f"{cls.__name__}.{name} 未重写, 不应判定为已重写"
            )


def test_hook_shortcut_cache_is_not_persisted() -> None:
    """性能短路缓存是纯派生态, 不得随 checkpoint 落盘.

    否则旧版本存档里算错的 False 会在恢复后继续生效, 钩子照样静默失效。
    """
    strategy = InheritedHookStrategy()
    # 模拟旧版本判定留下的错误缓存(框架侧动态字段, 类上无声明)。
    polluted: Any = strategy
    polluted._framework_needs_time_hooks = False
    polluted._framework_needs_portfolio_update = False

    state = strategy.__getstate__()

    assert "_framework_needs_time_hooks" not in state
    assert "_framework_needs_portfolio_update" not in state

    restored = pickle.loads(pickle.dumps(strategy))
    assert _needs_time_hooks(restored), "恢复后应重新判定出需要交易日钩子"
    assert _needs_portfolio_update(restored), "恢复后应重新判定出需要组合钩子"
