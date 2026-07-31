"""实盘下 subscribe() 无效, 必须告警而非静默.

``Strategy.subscribe()`` 只把 symbol 追加进 ``_subscriptions``, 而该列表**仅**被
``backtest/engine.py`` 消费; ``LiveRunner`` 从不读它——实盘订阅集完全来自
``run_live(instruments=[...])``。于是用户在 ``on_start`` 里 subscribe 一个未列入
instruments 的标的时, 既订不到行情也收不到任何提示。
"""

import logging
from typing import Any, cast

import pytest
from akquant import Strategy
from akquant.akquant import Bar
from akquant.live._runner import LiveRunner


class _Sub(Strategy):
    """在 on_start 里订阅的最简策略."""

    def on_start(self) -> None:
        """订阅一个标的."""
        self.subscribe("600000")

    def on_bar(self, bar: Bar) -> None:
        """不做任何事."""


class _StubEngine:
    """吸收 _configure_strategy_slots 对引擎的可选 set_* 调用."""

    def __getattr__(self, name: str) -> Any:
        """任何属性都返回一个接受任意入参的空函数."""

        def _accept(*args: Any, **kwargs: Any) -> None:
            return None

        return _accept


def test_subscribe_warns_in_live_mode(caplog: pytest.LogCaptureFixture) -> None:
    """实盘标记存在时, subscribe() 应告警并指向 run_live(instruments=...)."""
    strategy = _Sub()
    strategy._set_live_market_data_owner()

    with caplog.at_level(logging.WARNING):
        strategy.subscribe("600000")

    messages = [record.getMessage() for record in caplog.records]
    assert any("subscribe" in message for message in messages), (
        f"实盘 subscribe() 未告警: {messages}"
    )
    assert any("instruments" in message for message in messages), (
        f"告警未指向 run_live(instruments=...): {messages}"
    )


def test_subscribe_silent_in_backtest(caplog: pytest.LogCaptureFixture) -> None:
    """回测下 subscribe() 是正当用法, 不得产生任何告警."""
    strategy = _Sub()

    with caplog.at_level(logging.WARNING):
        strategy.subscribe("600000")

    assert not caplog.records, f"回测 subscribe() 不应告警: {caplog.records}"
    assert strategy._subscriptions == ["600000"], "回测语义必须保持不变"


def test_subscribe_still_records_symbol_in_live_mode() -> None:
    """告警不改变行为: 仍记录 symbol, 以免破坏读取 _subscriptions 的用户代码."""
    strategy = _Sub()
    strategy._set_live_market_data_owner()

    strategy.subscribe("600000")

    assert strategy._subscriptions == ["600000"]


def test_subscribe_warns_once_per_symbol(caplog: pytest.LogCaptureFixture) -> None:
    """同一 symbol 重复 subscribe 只告警一次, 避免刷屏.

    用户可能在 on_bar 里按条件 subscribe, 逐 bar 告警会淹没日志。
    """
    strategy = _Sub()
    strategy._set_live_market_data_owner()

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            strategy.subscribe("600000")

    warnings = [r for r in caplog.records if "subscribe" in r.getMessage()]
    assert len(warnings) == 1, f"应只告警一次, 实际 {len(warnings)} 次"


def test_live_runner_marks_strategies_as_live() -> None:
    """LiveRunner 必须给主策略与各 slot 都打上实盘标记.

    否则 subscribe() 无从判断自己处于实盘。
    """
    runner = LiveRunner.__new__(LiveRunner)
    runner.trading_mode = "paper"
    # 测试替身: 这两个属性在 LiveRunner 上有具体类型注解, 此处只需装配阶段够用。
    runner.context = cast(Any, None)
    runner.instruments = []
    runner.engine = cast(Any, _StubEngine())

    primary = _Sub()
    slot = _Sub()
    runner._configure_strategy_slots(primary, {"beta": slot}, "alpha")

    for target in (primary, slot):
        assert getattr(target, "_live_market_data_owner", False) is True
