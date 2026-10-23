"""实盘下 on_pre_open 由引擎生命周期事件触发, 不再告警"不会触发".

此前 ``on_pre_open`` 的框架定时器依据只在回测填充的 ``_trading_day_bounds``
生成, 实盘恒为空, 于是框架在实盘告警"on_pre_open 不会触发"。现在盘前事件由
引擎按交易日历生成(``__akq_lifecycle__|pre_open|<交易日>``, 实盘按墙钟排程,
排程公式见 Rust ``Engine::schedule_lifecycle`` 的单测), 回测与实盘同一路径。
"""

import logging
from typing import Any, Dict, List

import akquant.strategy_framework_hooks as hooks
import pandas as pd
import pytest
from akquant import Strategy
from akquant.akquant import AssetType, Bar, Instrument


class _PreOpen(Strategy):
    """重写 on_pre_open 的策略."""

    def __init__(self) -> None:
        """初始化."""
        super().__init__()
        self.pre_open_events: List[Dict[str, Any]] = []

    def on_pre_open(self, event: Dict[str, Any]) -> None:
        """盘前决策."""
        self.pre_open_events.append(event)

    def on_bar(self, bar: Bar) -> None:
        """不做任何事."""


class _Ctx:
    """最简上下文."""

    def __init__(self) -> None:
        """初始化."""
        self.current_time = 1_700_000_000_000_000_000
        self.session = "Closed"


def test_live_pre_open_lifecycle_event_fires_on_pre_open() -> None:
    """实盘策略收到引擎的盘前生命周期事件时调用 on_pre_open, 同一交易日只一次."""
    strategy = _PreOpen()
    strategy._set_live_market_data_owner()
    strategy.ctx = _Ctx()  # type: ignore[assignment]

    payload = "__akq_lifecycle__|pre_open|2023-11-15"
    assert hooks.dispatch_lifecycle_timer(strategy, payload) is True
    assert hooks.dispatch_lifecycle_timer(strategy, payload) is True

    assert len(strategy.pre_open_events) == 1
    event = strategy.pre_open_events[0]
    assert str(event["trading_date"]) == "2023-11-15"
    assert event["timestamp"] == strategy.ctx.current_time  # type: ignore[union-attr]


def test_python_pre_open_timer_registration_is_gone() -> None:
    """Python 侧按回测边界注册盘前定时器的旧路径已删除(只剩引擎生命周期事件)."""
    assert not hasattr(hooks, "register_pre_open_timers")
    assert not hasattr(hooks, "collect_pre_open_timer_entries")


def test_live_replay_pre_open_no_longer_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """实盘 replay 跑完一段会话: 不再出现"on_pre_open 不会触发"告警."""
    from akquant import run_live

    inst = Instrument(
        symbol="600000",
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )
    idx = pd.to_datetime(["2024-01-02 10:00", "2024-01-02 14:00"]).tz_localize(
        "Asia/Shanghai"
    )
    bars = [Bar(int(t.value), 10.0, 10.0, 10.0, 10.0, 100.0, "600000") for t in idx]
    with caplog.at_level(logging.WARNING):
        run_live(
            strategy_cls=_PreOpen,
            instruments=[inst],
            broker="replay",
            trading_mode="paper",
            gateway_options={"bars": bars, "freq": "1min"},
            cash=1e6,
            show_progress=False,
            duration="3s",
        )
    text = " ".join(record.getMessage() for record in caplog.records)
    assert "不会触发" not in text, text
