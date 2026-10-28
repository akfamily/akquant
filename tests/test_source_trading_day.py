"""行情源交易日经 live feed 进入引擎：校验显式日历、修正缺省日历（交易时段重构阶段 7）.

行情源（CTP 的 ``TradingDay`` 等）报告“某时刻属于交易日 H”，经
``DataFeed.add_trading_day_hint`` 与行情同序进入引擎。引擎按当前时段表与日历求得
该时刻所属交易日 C：

- 用户显式传了 ``trading_calendar=``：以日历为准，C != H 只告警（每个 (C, H) 一次）。
- 未传（按工作日推算）且 C 尚未开日：用 H 修正缺省日历（补入 H、剔除 [C, H) 内的
  推算交易日），被剔除日的生命周期与收盘定时器删除并重排，告警一次。
- 未传但 C 已开日：只告警，不修正。

replay 用例经 ``gateway_options["trading_day_hints"]`` 把 hint 按时间戳合并进 bar
序列推送（同一时间戳 hint 在前），与真实行情源一样与行情同序到达引擎。
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import pytest
from akquant import Bar, Strategy

_TZ = "Asia/Shanghai"
_SYMBOL = "600000"


def _ns(local: str) -> int:
    return int(pd.Timestamp(local, tz=_TZ).value)


def _bars(stamps: Sequence[str]) -> List[Bar]:
    return [Bar(_ns(s), 10.0, 10.0, 10.0, 10.0, 100.0, _SYMBOL) for s in stamps]


def _instrument() -> Any:
    from akquant.akquant import AssetType, Instrument

    return Instrument(
        symbol=_SYMBOL,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


class _HookProbe(Strategy):
    """记录（钩子, 本地时刻）与每根 bar 所属交易日."""

    def __init__(self) -> None:
        super().__init__()
        self.hooks: List[Tuple[str, str]] = []
        self.bar_days: Dict[str, str] = {}

    def _record(self, name: str) -> None:
        now = self.now
        assert now is not None
        self.hooks.append((name, now.strftime("%Y-%m-%d %H:%M:%S")))

    def on_pre_open(self, event: Dict[str, Any]) -> None:
        self._record("on_pre_open")

    def on_before_trading(self, trading_date: dt.date, timestamp: int) -> None:
        self._record("on_before_trading")

    def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
        self._record("on_after_trading")

    def on_bar(self, bar: Bar) -> None:
        self._record("on_bar")
        local = pd.Timestamp(int(bar.timestamp), tz="UTC").tz_convert(_TZ)
        self.bar_days[local.strftime("%Y-%m-%d %H:%M")] = str(self.trading_day)


def _run_replay(
    monkeypatch: pytest.MonkeyPatch,
    stamps: Sequence[str],
    hints: Sequence[Tuple[int, dt.date, str]],
    **kwargs: Any,
) -> Tuple[Any, _HookProbe]:
    """run_live(broker="replay") 并在 bar 序列中穿插行情源交易日 hint."""
    from akquant import run_live
    from akquant.live import _facade

    captured: List[Any] = []

    class _Recording(_facade.LiveRunner):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)
            captured.append(self)

    monkeypatch.setattr(_facade, "LiveRunner", _Recording)
    run_live(
        strategy_cls=_HookProbe,
        instruments=[_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options={
            "bars": _bars(stamps),
            "freq": "1min",
            "trading_day_hints": list(hints),
        },
        cash=1e6,
        show_progress=False,
        duration="30s",
        day_close_delay=0,
        **kwargs,
    )
    assert len(captured) == 1
    runner = captured[0]
    strategy = runner._stop_target_strategy
    assert isinstance(strategy, _HookProbe)
    return runner, strategy


def _days(hooks: List[Tuple[str, str]], name: str) -> List[str]:
    return [at[:10] for hook, at in hooks if hook == name]


def _times(hooks: List[Tuple[str, str]], name: str) -> List[str]:
    return [at for hook, at in hooks if hook == name]


def _warnings(caplog: Any, needle: str) -> List[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and needle in r.getMessage()
    ]


# 2024 国庆：09-30（周一）为节前最后一个交易日，10-01～10-07 休市，10-08 复市。
_HOLIDAY_STAMPS = [
    "2024-09-30 10:00",
    "2024-09-30 15:00",
    "2024-10-08 10:00",
    "2024-10-08 15:00",
]


def test_hint_corrects_default_calendar_before_day_opens(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """未传日历时行情源交易日修正缺省日历.

    国庆前最后一个交易日夜间收到 hint=10-08，10-01..10-07 被剔除，
    之后 10-08 的 bar 交易日为 10-08，且 10-01 不触发 pre_open。
    """
    hint = (_ns("2024-09-30 20:00"), dt.date(2024, 10, 8), "test")
    with caplog.at_level("WARNING", logger="akquant"):
        runner, strategy = _run_replay(monkeypatch, _HOLIDAY_STAMPS, [hint])
    hooks = strategy.hooks
    # 不修正时按工作日推算，10-01～10-04、10-07 都会开日并触发 pre_open / 结算
    assert _days(hooks, "on_pre_open") == ["2024-09-30", "2024-10-08"], hooks
    assert _times(hooks, "on_after_trading") == [
        "2024-09-30 15:00:00",
        "2024-10-08 15:00:00",
    ], hooks
    assert strategy.bar_days["2024-10-08 10:00"] == "2024-10-08"
    # 修正后的日历：假期内的时刻归入 10-08
    assert runner.engine.trading_day_at(_ns("2024-10-01 10:00")) == dt.date(2024, 10, 8)
    corrected = _warnings(caplog, "修正")
    assert len(corrected) == 1, corrected
    assert "trading_calendar=" in corrected[0]
    assert "2024-10-01" in corrected[0] and "2024-10-07" in corrected[0]


def test_hint_correction_survives_checkpoint_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """修正量随存档保存：载入后重新 configure_sessions（未传日历）仍生效."""
    from akquant import Engine

    hint = (_ns("2024-09-30 20:00"), dt.date(2024, 10, 8), "test")
    runner, _ = _run_replay(monkeypatch, _HOLIDAY_STAMPS, [hint])
    state = runner.engine.get_state_bytes()

    resumed = Engine()
    resumed.load_state_bytes(state)
    # run_from_checkpoint 未传日历时按存档（None）重新下发时段配置
    resumed.configure_sessions(None, None)
    assert resumed.trading_day_at(_ns("2024-10-01 10:00")) == dt.date(2024, 10, 8)
    assert resumed.trading_day_at(_ns("2024-10-07 10:00")) == dt.date(2024, 10, 8)
    assert resumed.trading_day_at(_ns("2024-09-30 10:00")) == dt.date(2024, 9, 30)

    # 未修正的引擎按工作日推算
    fresh = Engine()
    fresh.configure_sessions(None, None)
    assert fresh.trading_day_at(_ns("2024-10-01 10:00")) == dt.date(2024, 10, 1)


def test_hint_with_explicit_calendar_only_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """显式日历与 hint 不一致只告警.

    交易日仍按日历，WARNING 含 trading_calendar 且只出现一次。
    """
    # 日历（故意）把 10-01 写成交易日；行情源两次报告 09-30 夜间属 10-08。
    calendar = ["2024-09-27", "2024-09-30", "2024-10-01", "2024-10-08", "2024-10-09"]
    hints = [
        (_ns("2024-09-30 20:00"), dt.date(2024, 10, 8), "test"),
        (_ns("2024-09-30 21:00"), dt.date(2024, 10, 8), "test"),
    ]
    with caplog.at_level("WARNING", logger="akquant"):
        runner, strategy = _run_replay(
            monkeypatch, _HOLIDAY_STAMPS, hints, trading_calendar=calendar
        )
    hooks = strategy.hooks
    assert _days(hooks, "on_pre_open") == [
        "2024-09-30",
        "2024-10-01",
        "2024-10-08",
    ], hooks
    assert runner.engine.trading_day_at(_ns("2024-10-01 10:00")) == dt.date(2024, 10, 1)
    warned = _warnings(caplog, "以 trading_calendar 为准")
    assert len(warned) == 1, warned
    assert "2024-10-08" in warned[0] and "2024-10-01" in warned[0]
    assert "test" in warned[0]


def test_hint_after_day_opened_only_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """C 已开日：不修正，只告警."""
    stamps = [
        "2024-09-30 10:00",
        "2024-09-30 15:00",
        "2024-10-01 10:00",
        "2024-10-01 15:00",
    ]
    # 09-30 盘中（当日已开）行情源报告属 10-08：修正会剔除已开的 09-30 与 10-01
    hint = (_ns("2024-09-30 11:00"), dt.date(2024, 10, 8), "test")
    with caplog.at_level("WARNING", logger="akquant"):
        runner, strategy = _run_replay(monkeypatch, stamps, [hint])
    hooks = strategy.hooks
    assert _days(hooks, "on_pre_open") == ["2024-09-30", "2024-10-01"], hooks
    assert _times(hooks, "on_after_trading") == [
        "2024-09-30 15:00:00",
        "2024-10-01 15:00:00",
    ], hooks
    assert strategy.bar_days["2024-10-01 10:00"] == "2024-10-01"
    assert runner.engine.trading_day_at(_ns("2024-10-01 10:00")) == dt.date(2024, 10, 1)
    warned = _warnings(caplog, "已开日")
    assert len(warned) == 1, warned
    assert "2024-10-08" in warned[0] and "2024-09-30" in warned[0]
    assert not _warnings(caplog, "修正缺省交易日历")


# 2024-10-11 周五、10-12 周六（假设补班）、10-14 周一。
_MAKEUP_STAMPS = [
    "2024-10-11 10:00",
    "2024-10-11 15:00",
    "2024-10-12 10:00",
    "2024-10-12 15:00",
    "2024-10-14 10:00",
    "2024-10-14 15:00",
]


def test_hint_adds_makeup_day_after_next_day_was_chained(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """补入的交易日早于已串联的下一交易日时补排其生命周期.

    周五 before_trading 已排周一；周五夜间行情源报告属周六：周六补入日历，
    并有 pre_open / before_trading（不只是结算与 after_trading）。
    """
    hint = (_ns("2024-10-11 20:00"), dt.date(2024, 10, 12), "test")
    with caplog.at_level("WARNING", logger="akquant"):
        _, strategy = _run_replay(monkeypatch, _MAKEUP_STAMPS, [hint])
    hooks = strategy.hooks
    days = ["2024-10-11", "2024-10-12", "2024-10-14"]
    assert _days(hooks, "on_pre_open") == days, hooks
    assert _days(hooks, "on_before_trading") == days, hooks
    assert _days(hooks, "on_after_trading") == days, hooks
    assert strategy.bar_days["2024-10-12 10:00"] == "2024-10-12"
    corrected = _warnings(caplog, "修正缺省交易日历")
    assert len(corrected) == 1 and "2024-10-12" in corrected[0], corrected


def test_hint_for_ended_off_calendar_day_only_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """行情源报告的交易日已收盘且不在缺省日历内：只告警，不回溯补入."""
    stamps = [s for s in _MAKEUP_STAMPS if not s.startswith("2024-10-12")]
    hint = (_ns("2024-10-12 20:00"), dt.date(2024, 10, 12), "test")
    with caplog.at_level("WARNING", logger="akquant"):
        runner, strategy = _run_replay(monkeypatch, stamps, [hint])
    hooks = strategy.hooks
    assert _days(hooks, "on_pre_open") == ["2024-10-11", "2024-10-14"], hooks
    assert _days(hooks, "on_after_trading") == ["2024-10-11", "2024-10-14"], hooks
    assert runner.engine.trading_day_at(_ns("2024-10-12 10:00")) == dt.date(
        2024, 10, 14
    )
    warned = _warnings(caplog, "不回溯修正")
    assert len(warned) == 1 and "2024-10-12" in warned[0], warned
    assert not _warnings(caplog, "修正缺省交易日历")


def test_add_trading_day_hint_rejected_on_backtest_feed() -> None:
    """回测 feed 上调用 add_trading_day_hint 抛中文 RuntimeError."""
    from akquant import DataFeed

    with pytest.raises(RuntimeError, match="实盘"):
        DataFeed().add_trading_day_hint(0, dt.date(2024, 1, 2), "test")


def test_engine_trading_day_at_follows_configured_calendar() -> None:
    """Engine.trading_day_at 在 run() 前按时段表与日历求交易日."""
    from akquant import Engine

    engine = Engine()
    engine.set_timezone_name(_TZ)
    engine.configure_sessions(None, [dt.date(2024, 1, 2), dt.date(2024, 1, 4)])
    # 15:00 收盘后的时刻属于下一交易日（日历跳过 01-03）
    assert engine.trading_day_at(_ns("2024-01-02 10:00")) == dt.date(2024, 1, 2)
    assert engine.trading_day_at(_ns("2024-01-02 20:00")) == dt.date(2024, 1, 4)
    with pytest.raises(ValueError, match="trading_calendar"):
        engine.trading_day_at(_ns("2024-01-05 10:00"))
