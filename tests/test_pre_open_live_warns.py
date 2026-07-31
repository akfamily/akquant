"""实盘下 on_pre_open 不触发, 必须告警而非静默.

``on_pre_open`` 的框架定时器由 ``collect_pre_open_timer_entries`` 依据
``_trading_day_bounds`` 逐交易日生成, 而该字段**只在** ``backtest/engine.py``
填充; 实盘走 ``Engine()`` + ``DataFeed.create_live()``, 它恒为空 dict, 于是
collect 返回空列表、timer 一个都不注册, ``on_pre_open`` 永不触发。

更糟的是 ``register_pre_open_timers`` 在 entries 为空时**不**置
``_framework_pre_open_timers_registered``, 因此实盘每根 bar/tick 都会重试一遍
并再次失败——既无声, 也白做功。
"""

import logging
from typing import Any, Dict, List, Tuple

import pytest
from akquant import Strategy
from akquant.akquant import Bar
from akquant.strategy_framework_hooks import register_pre_open_timers


class _PreOpen(Strategy):
    """重写 on_pre_open 的策略."""

    def on_pre_open(self, event: Dict[str, Any]) -> None:
        """盘前决策."""

    def on_bar(self, bar: Bar) -> None:
        """不做任何事."""


class _NoPreOpen(Strategy):
    """未重写 on_pre_open 的策略."""

    def on_bar(self, bar: Bar) -> None:
        """不做任何事."""


class _Ctx:
    """最简上下文, 记录 schedule 调用."""

    def __init__(self) -> None:
        """初始化."""
        self.current_time = 1_700_000_000_000_000_000
        self.scheduled: List[Tuple[int, str]] = []

    def schedule(self, timestamp: int, payload: str) -> None:
        """记录被注册的 timer."""
        self.scheduled.append((timestamp, payload))


def _live_strategy() -> _PreOpen:
    """构造一个处于实盘、且重写了 on_pre_open 的策略."""
    strategy = _PreOpen()
    strategy._set_live_market_data_owner()
    strategy.ctx = _Ctx()  # type: ignore[assignment]
    return strategy


def test_pre_open_warns_in_live_mode(caplog: pytest.LogCaptureFixture) -> None:
    """实盘 + 重写 on_pre_open + 无交易日边界 → 必须告警."""
    strategy = _live_strategy()

    with caplog.at_level(logging.WARNING):
        register_pre_open_timers(strategy)

    messages = [record.getMessage() for record in caplog.records]
    assert any("on_pre_open" in message for message in messages), (
        f"实盘 on_pre_open 不触发却未告警: {messages}"
    )


def test_pre_open_live_warning_states_it_will_not_fire(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """告警须说明该回调在实盘不会触发, 而不是只报一句内部状态."""
    strategy = _live_strategy()

    with caplog.at_level(logging.WARNING):
        register_pre_open_timers(strategy)

    text = " ".join(record.getMessage() for record in caplog.records)
    assert "不会触发" in text or "不触发" in text, f"告警未说明后果: {text}"


def test_pre_open_live_warns_only_once(caplog: pytest.LogCaptureFixture) -> None:
    """register_pre_open_timers 每根 bar/tick 都会被调用, 不能逐 bar 刷屏."""
    strategy = _live_strategy()

    with caplog.at_level(logging.WARNING):
        for _ in range(10):
            register_pre_open_timers(strategy)

    warnings = [r for r in caplog.records if "on_pre_open" in r.getMessage()]
    assert len(warnings) == 1, f"应只告警一次, 实际 {len(warnings)} 次"


def test_pre_open_live_stops_retrying() -> None:
    """告警后应置 registered 标志, 停止每根 bar 的无效重试."""
    strategy = _live_strategy()

    register_pre_open_timers(strategy)

    assert getattr(strategy, "_framework_pre_open_timers_registered", False) is True, (
        "实盘下永远拿不到 bounds, 必须停止重试"
    )


def test_no_warning_when_pre_open_not_overridden(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """未重写 on_pre_open 的策略不受影响, 不应告警."""
    strategy = _NoPreOpen()
    strategy._set_live_market_data_owner()
    strategy.ctx = _Ctx()  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        register_pre_open_timers(strategy)

    assert not caplog.records, f"未重写 on_pre_open 不应告警: {caplog.records}"


def test_no_warning_in_backtest(caplog: pytest.LogCaptureFixture) -> None:
    """回测下 bounds 由引擎填充, 属正常路径, 不得告警.

    此处模拟"引擎尚未填充 bounds"的早期时刻: 回测中它随后就会被填上,
    告警会是误报。
    """
    strategy = _PreOpen()
    strategy.ctx = _Ctx()  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        register_pre_open_timers(strategy)

    assert not caplog.records, f"回测不应告警: {caplog.records}"


def test_backtest_pre_open_timers_still_registered() -> None:
    """回归: 有 bounds 时仍照常注册 timer."""
    strategy = _PreOpen()
    ctx = _Ctx()
    strategy.ctx = ctx  # type: ignore[assignment]
    day_start = ctx.current_time + 86_400_000_000_000
    strategy._trading_day_bounds = {  # type: ignore[attr-defined]
        "2023-11-15": (day_start, day_start + 3600_000_000_000)
    }

    register_pre_open_timers(strategy)

    assert len(ctx.scheduled) == 1, f"回测 timer 未注册: {ctx.scheduled}"
    assert ctx.scheduled[0][0] == day_start - 1, "pre-open timer 应排在首根 bar 前 1ns"
    assert ctx.scheduled[0][1].startswith("__framework_pre_open__|")
