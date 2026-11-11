"""未传 trading_calendar 时的缺省交易日历.

- 中国时区: 内置交易所日历(覆盖范围内剔除节假日, 范围之外按工作日)。
- 回测: 数据里出现过行情的日期并入日历; 靠工作日推算的范围里, 所有标的都没有
  行情的工作日剔除。
- 只在交易日确实要靠"周一至周五"推算时告警。
"""

import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pytest
from akquant import SessionResolver, Strategy, run_backtest

_HINT = "按工作日推算"


def _frame(days: Sequence[str], symbol: str = "600000.SH") -> pd.DataFrame:
    stamps = pd.to_datetime(list(days))
    n = len(stamps)
    return pd.DataFrame(
        {
            "date": stamps,
            "open": [10.0 + i * 0.01 for i in range(n)],
            "high": [10.2 + i * 0.01 for i in range(n)],
            "low": [9.8 + i * 0.01 for i in range(n)],
            "close": [10.1 + i * 0.01 for i in range(n)],
            "volume": [1e6] * n,
            "symbol": symbol,
        }
    )


def _weekdays(start: str, end: str, skip: Sequence[str] = ()) -> List[str]:
    drop = set(pd.to_datetime(list(skip)))
    return [str(d.date()) for d in pd.bdate_range(start, end) if d not in drop]


class _Probe(Strategy):
    """记录 bar、每日定时任务、收盘后钩子各自所在的交易日."""

    def on_start(self) -> None:
        self.bar_days: List[str] = []
        self.timer_days: List[str] = []
        self.after_days: List[str] = []
        self.schedule_daily("14:50:00", "probe")

    def on_bar(self, bar: Any) -> None:
        self.bar_days.append(str(self.trading_day))

    def on_timer(self, payload: str) -> None:
        self.timer_days.append(str(self.trading_day))

    def on_after_trading(self, *args: Any) -> None:
        self.after_days.append(str(self.trading_day))


def _run(
    data: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
    **kwargs: Any,
) -> Dict[str, Any]:
    probe = _Probe()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        result = run_backtest(
            strategy=probe, data=data, initial_cash=1e5, show_progress=False, **kwargs
        )
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING"
        and (_HINT in r.getMessage() or "非交易日" in r.getMessage())
    ]
    return {"probe": probe, "result": result, "warnings": warnings}


# 2024 国庆: 09-30(周一)后休市, 10-08(周二)复市
_NATIONAL_DAY = ["2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-07"]


def test_resolver_defaults_to_builtin_exchange_calendar() -> None:
    """中国时区不传日历: 内置交易所日历, 节假日不是交易日."""
    resolver = SessionResolver()
    assert resolver.uses_fallback_calendar is True
    assert resolver.uses_builtin_calendar is True
    first, last = SessionResolver.builtin_calendar_range()
    assert first <= dt.date(2005, 1, 4) and last >= dt.date(2026, 12, 31)
    assert resolver.is_session_day(dt.date(2024, 9, 30))
    for day in _NATIONAL_DAY:
        assert not resolver.is_session_day(dt.date.fromisoformat(day)), day
    assert resolver.next_session_day(dt.date(2024, 9, 30)) == dt.date(2024, 10, 8)
    assert resolver.builtin_calendar_covers(dt.date(2024, 10, 1))
    assert not resolver.builtin_calendar_covers(last + dt.timedelta(days=1))
    # 覆盖范围之外按工作日
    after = last + dt.timedelta(days=1)
    while after.weekday() >= 5:
        after += dt.timedelta(days=1)
    assert resolver.is_session_day(after)


def test_resolver_builtin_calendar_only_for_china_timezone() -> None:
    """显式日历、全天候表、非中国时区都不用内置日历."""
    explicit = SessionResolver(trading_calendar=[dt.date(2024, 10, 1)])
    assert explicit.uses_builtin_calendar is False
    assert explicit.is_session_day(dt.date(2024, 10, 1))
    always = SessionResolver(sessions={"default": "ALWAYS_OPEN"})
    assert always.uses_builtin_calendar is False
    other = SessionResolver(
        sessions={"default": [("09:30", "16:00", "continuous")]},
        timezone="America/New_York",
    )
    assert other.uses_builtin_calendar is False
    assert other.is_session_day(dt.date(2024, 10, 1))
    assert not other.builtin_calendar_covers(dt.date(2024, 10, 1))


def test_resolver_calendar_adjustments_return_new_resolver() -> None:
    """修正量叠加在缺省日历上, 原解析器不变; 显式日历不受影响."""
    base = SessionResolver()
    adjusted = base.with_calendar_adjustments(
        [dt.date(2024, 10, 1)], [dt.date(2024, 10, 8)]
    )
    assert adjusted.is_session_day(dt.date(2024, 10, 1))
    assert not adjusted.is_session_day(dt.date(2024, 10, 8))
    assert not base.is_session_day(dt.date(2024, 10, 1))
    assert base.is_session_day(dt.date(2024, 10, 8))
    explicit = SessionResolver(trading_calendar=[dt.date(2024, 10, 8)])
    same = explicit.with_calendar_adjustments([dt.date(2024, 10, 1)], [])
    assert not same.is_session_day(dt.date(2024, 10, 1))


def test_backtest_without_calendar_skips_holidays_and_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 股日线跨国庆、不传日历: 节假日不触发定时任务, 收盘后交易日跳到节后, 不告警."""
    days = _weekdays("2024-09-25", "2024-10-11", skip=_NATIONAL_DAY)
    out = _run(_frame(days), caplog)
    probe = out["probe"]
    assert out["warnings"] == []
    assert probe.bar_days == days
    # 只有日期的日线首日盘中定时任务不触发(既有行为); 其余每个交易日一次, 节假日没有
    assert probe.timer_days == days[1:]
    assert not set(probe.timer_days) & set(_NATIONAL_DAY)
    # 09-30 收盘结算后, 当前交易日是节后首个交易日
    after = dict(zip(days, probe.after_days))
    assert after["2024-09-30"] == "2024-10-08"


def test_backtest_without_calendar_matches_explicit_calendar(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """不传日历与传入交易所日历, 回调时点逐项一致."""
    days = _weekdays("2024-09-25", "2024-10-11", skip=_NATIONAL_DAY)
    implicit = _run(_frame(days), caplog)["probe"]
    explicit = _run(_frame(days), caplog, trading_calendar=days)["probe"]
    assert implicit.bar_days == explicit.bar_days
    assert implicit.timer_days == explicit.timer_days
    # 显式日历只覆盖到数据末日, 末日收盘后查不到下一交易日; 其余逐日相同
    assert implicit.after_days[:-1] == explicit.after_days[:-1]


def test_backtest_keeps_exchange_day_when_a_symbol_has_no_bar(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """内置日历覆盖的交易日, 即使数据里没有行情(停牌)也仍是交易日."""
    days = _weekdays("2024-03-04", "2024-03-08", skip=["2024-03-06"])
    out = _run(_frame(days), caplog)
    assert out["warnings"] == []
    assert "2024-03-06" in out["probe"].timer_days


def test_backtest_data_on_exchange_holiday_becomes_trading_day(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """数据在交易所休市日有行情(合成数据、其他市场): 以数据为准, 当日照常是交易日."""
    days = _weekdays("2024-09-27", "2024-10-09")
    out = _run(_frame(days), caplog)
    probe = out["probe"]
    assert out["warnings"] == []
    assert probe.bar_days == days
    assert probe.timer_days == days[1:]
    assert set(_NATIONAL_DAY) <= set(probe.timer_days)
    assert len(out["result"].equity_curve_daily) == len(days)


@pytest.mark.parametrize(
    ("days", "timezone", "sessions", "missing"),
    [
        # 非中国时区的自定义时段表: 2024-07-04 美国独立日, 数据里没有
        (
            _weekdays("2024-07-01", "2024-07-09", skip=["2024-07-04"]),
            "America/New_York",
            {"default": [("09:30", "16:00", "continuous")]},
            "2024-07-04",
        ),
        # 中国时区、超出内置日历覆盖范围: 2030-02-06 假设休市
        (
            _weekdays("2030-02-01", "2030-02-12", skip=["2030-02-06"]),
            None,
            None,
            "2030-02-06",
        ),
    ],
)
def test_backtest_weekday_fallback_follows_data(
    caplog: pytest.LogCaptureFixture,
    days: List[str],
    timezone: Optional[str],
    sessions: Optional[Dict[str, Any]],
    missing: str,
) -> None:
    """靠工作日推算的范围里, 所有标的都没有行情的工作日不算交易日, 也不告警."""
    kwargs: Dict[str, Any] = {}
    if timezone:
        kwargs["timezone"] = timezone
    if sessions:
        kwargs["sessions"] = sessions
    out = _run(_frame(days, symbol="X"), caplog, **kwargs)
    probe = out["probe"]
    assert out["warnings"] == []
    assert probe.bar_days == days
    assert missing not in probe.timer_days
    assert set(probe.timer_days) <= set(days)


def test_backtest_explicit_calendar_still_reports_data_outside_calendar(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """传了 trading_calendar 时以用户日历为准: 数据落在日历外照旧提示."""
    days = ["2024-09-30", "2024-10-01", "2024-10-08"]
    out = _run(
        _frame(days),
        caplog,
        trading_calendar=["2024-09-27", "2024-09-30", "2024-10-08", "2024-10-09"],
    )
    assert any("非交易日 2024-10-01" in m for m in out["warnings"]), out["warnings"]


def test_night_session_after_midnight_does_not_make_a_trading_day(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """夜盘跨过午夜的 bar(周六凌晨)属于下一交易日, 不会把周六并入交易日历."""
    stamps = [
        "2024-01-05 14:59",
        "2024-01-05 15:00",
        "2024-01-05 21:01",
        "2024-01-06 01:00",
        "2024-01-08 09:01",
    ]
    frame = _frame(stamps, symbol="au2406")
    frame["date"] = pd.to_datetime(stamps).tz_localize("Asia/Shanghai")
    out = _run(frame, caplog, sessions={"au": "CN_FUTURES_NIGHT_0230"})
    assert out["probe"].bar_days == [
        "2024-01-05",
        "2024-01-05",
        "2024-01-08",
        "2024-01-08",
        "2024-01-08",
    ]
