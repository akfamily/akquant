"""函数式策略未提供的框架钩子, 不得被判定为"已重写".

回归场景: ``FunctionalStrategy`` 在类体里无条件定义了全部回调转发方法
(未提供时方法体是空转), 于是 ``_strategy_overrides_callback`` 按类比较时
恒为 True, "回调未重写就跳过"的快路径整体失效——引擎为每根 Bar 注册并分发
pre_open / cross_section / time-hooks / portfolio-update 全套逻辑, 而下游
无人消费。实测 12 组合网格搜索因此从 0.9 秒涨到约 11 秒(约 13 倍)。
"""

from typing import Any

import pandas as pd
from akquant import run_backtest
from akquant.akquant import Bar
from akquant.backtest.engine import FunctionalStrategy
from akquant.backtest.fill_mode import CurrentClose
from akquant.strategy_framework_hooks import _strategy_overrides_callback

SYMBOL = "FUNC_HOOKS_DEMO"

# FunctionalStrategy 未提供对应函数时应判定为"未重写"的框架钩子
FRAMEWORK_CALLBACKS = (
    "on_before_trading",
    "on_after_trading",
    "on_portfolio_update",
    "on_pre_open",
    "on_cross_section",
    "on_timer",
)


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


def _noop_on_bar(ctx: Any, bar: Bar) -> None:
    """最小函数式策略入口."""


def test_functional_wrapper_without_callbacks_is_not_treated_as_override() -> None:
    """只提供 on_bar 时, 未提供的框架钩子必须判定为未重写, 否则快路径失效."""
    strategy = FunctionalStrategy(initialize=None, on_bar=_noop_on_bar)

    for name in FRAMEWORK_CALLBACKS:
        assert not _strategy_overrides_callback(strategy, name), (
            f"{name} 未由用户提供, 不应判定为已重写"
        )


def test_functional_wrapper_with_callbacks_is_treated_as_override() -> None:
    """用户确实提供了钩子时, 必须判定为已重写, 否则钩子静默失效."""

    def _on_pre_open(ctx: Any, event: dict[str, Any]) -> None:
        """用户提供的 pre_open 回调."""

    def _on_timer(ctx: Any, payload: str) -> None:
        """用户提供的 timer 回调."""

    strategy = FunctionalStrategy(
        initialize=None,
        on_bar=_noop_on_bar,
        on_pre_open=_on_pre_open,
        on_timer=_on_timer,
    )

    assert _strategy_overrides_callback(strategy, "on_pre_open"), (
        "on_pre_open 应判定为已重写"
    )
    assert _strategy_overrides_callback(strategy, "on_timer"), "on_timer 应判定为已重写"
    assert not _strategy_overrides_callback(strategy, "on_cross_section"), (
        "on_cross_section 未提供, 不应判定为已重写"
    )


def test_functional_subclass_override_is_still_detected() -> None:
    """继承 FunctionalStrategy 并重写钩子时, 仍须判定为已重写."""

    class CustomFunctional(FunctionalStrategy):
        def on_pre_open(self, event: dict[str, Any]) -> None:
            """子类自行重写, 与是否传入 on_pre_open 无关."""

    strategy = CustomFunctional(initialize=None, on_bar=_noop_on_bar)

    assert _strategy_overrides_callback(strategy, "on_pre_open"), (
        "子类重写的 on_pre_open 应判定为已重写"
    )


def test_functional_pre_open_callback_still_fires_in_backtest() -> None:
    """跳过判定不得误伤真实回调: 提供了 on_pre_open 就必须被触发."""
    seen: list[Any] = []

    def _on_start(ctx: Any) -> None:
        ctx.subscribe(SYMBOL)

    def _on_pre_open(ctx: Any, event: dict[str, Any]) -> None:
        seen.append(event.get("trading_date"))

    run_backtest(
        data=_two_day_bars(),
        strategy=_noop_on_bar,
        on_start=_on_start,
        on_pre_open=_on_pre_open,
        symbols=[SYMBOL],
        initial_cash=10_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    assert seen, "函数式 on_pre_open 未触发"
