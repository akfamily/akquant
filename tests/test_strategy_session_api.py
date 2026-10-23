"""策略的交易日与时段接口：trading_day / session_of / is_market_open."""

from __future__ import annotations

import datetime as dt
from typing import Any, List, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest

TZ = "Asia/Shanghai"


def _minute(times: List[str], symbol: str) -> pd.DataFrame:
    idx = pd.to_datetime(times).tz_localize(TZ)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )


class _Probe(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[Tuple[str, Any, str, bool, bool]] = []

    def on_bar(self, bar: Bar) -> None:
        t = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ).strftime("%m-%d %H:%M")
        self.rows.append(
            (
                t,
                self.trading_day,
                str(self.session_of()).split(".")[-1],
                self.is_market_open(),
                self.is_market_open(include_auction=True),
            )
        )


def test_trading_day_and_session_api_on_stock_minutes() -> None:
    """股票分钟线：交易日、当前时段与是否开市（收盘集合竞价只在含竞价时算开市）."""
    df = _minute(
        ["2024-01-02 09:31", "2024-01-02 14:58", "2024-01-02 15:00"], "600000.SH"
    )
    r = run_backtest(
        strategy=_Probe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    assert r.strategy.rows == [
        ("01-02 09:31", dt.date(2024, 1, 2), "Continuous", True, True),
        ("01-02 14:58", dt.date(2024, 1, 2), "ClosingAuction", False, True),
        ("01-02 15:00", dt.date(2024, 1, 2), "ClosingAuction", False, True),
    ]


def test_night_session_trading_day_is_next_day() -> None:
    """夜盘 bar 的 trading_day 是下一交易日."""
    df = _minute(["2024-01-05 21:30"], "au2406")
    cfg = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=1e6),
        instruments_config=[
            aq.InstrumentConfig(
                symbol="au2406",
                asset_type="FUTURES",
                multiplier=1000.0,
                margin_ratio=0.1,
            )
        ],
    )
    r = run_backtest(
        strategy=_Probe,
        data=df,
        config=cfg,
        show_progress=False,
        sessions={"au": "CN_FUTURES_NIGHT_0230"},
    )
    assert r.strategy is not None
    assert r.strategy.rows[0][1] == dt.date(2024, 1, 8)  # 周五夜盘属下周一


def test_session_of_other_symbol_uses_resolver_at_now() -> None:
    """股票 10:20 bar 上查询期货（日盘 10:15～10:30 休息）的时段."""

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.got: List[str] = []

        def on_bar(self, bar: Bar) -> None:
            self.got.append(str(self.session_of("rb2405")).split(".")[-1])

    df = pd.concat(
        [
            _minute(["2024-01-02 10:20"], "600000.SH"),
            _minute(["2024-01-02 10:31"], "rb2405"),
        ]
    )
    cfg = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=1e6),
        instruments_config=[
            aq.InstrumentConfig(symbol="600000.SH", asset_type="STOCK"),
            aq.InstrumentConfig(
                symbol="rb2405",
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
            ),
        ],
    )
    r = run_backtest(strategy=P, data=df, config=cfg, show_progress=False)
    assert r.strategy is not None
    assert r.strategy.got[0] == "Break"


def test_user_timer_rejects_reserved_prefix() -> None:
    """用户定时器 payload 不能以引擎保留前缀 __akq_ 开头."""

    class P(Strategy):
        def on_start(self) -> None:
            self.schedule_daily("09:35", "__akq_lifecycle__|x")

    with pytest.raises(ValueError, match="__akq_"):
        run_backtest(
            strategy=P,
            data=_minute(["2024-01-02 09:31"], "600000.SH"),
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
        )


def test_session_of_own_symbol_matches_event_session_on_daily_bars() -> None:
    """日线 bar 覆盖整个交易日：session_of(本品种) 取本时间片的区间时段.

    与 ctx.session 一致，而不是按 15:00 前 1ns 求得的收盘集合竞价.
    """

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.got: List[Tuple[str, str]] = []

        def on_bar(self, bar: Bar) -> None:
            self.got.append(
                (
                    str(self.session_of()).split(".")[-1],
                    str(self.session_of(bar.symbol)).split(".")[-1],
                )
            )

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": "600000.SH",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )
    r = run_backtest(
        strategy=P,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    assert r.strategy.got == [("Continuous", "Continuous")] * 2


_RESERVED = "__akq_lifecycle__|x"


def _sched_schedule(s: Strategy) -> None:
    s.schedule("2024-01-02 10:00", _RESERVED)


def _sched_daily(s: Strategy) -> None:
    s.schedule_daily("09:35", _RESERVED)


def _sched_weekly(s: Strategy) -> None:
    s.schedule_weekly("09:35", _RESERVED)


def _sched_monthly(s: Strategy) -> None:
    s.schedule_monthly("09:35", _RESERVED)


@pytest.mark.parametrize(
    "register",
    [_sched_schedule, _sched_daily, _sched_weekly, _sched_monthly],
    ids=["schedule", "schedule_daily", "schedule_weekly", "schedule_monthly"],
)
@pytest.mark.parametrize("hook", ["on_start", "on_bar"])
def test_all_python_timer_entry_points_reject_reserved_prefix(
    register: Any, hook: str
) -> None:
    """四个 schedule* 入口在 on_start 与运行期都拒绝 __akq_."""

    class P(Strategy):
        def on_start(self) -> None:
            if hook == "on_start":
                register(self)

        def on_bar(self, bar: Bar) -> None:
            if hook == "on_bar":
                register(self)

    with pytest.raises(ValueError, match="__akq_"):
        run_backtest(
            strategy=P,
            data=_minute(["2024-01-02 09:31", "2024-01-03 09:31"], "600000.SH"),
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
        )


def test_ctx_schedule_and_engine_add_timer_reject_reserved_prefix() -> None:
    """Rust 入口（ctx.schedule、Engine.add_timer）同样拒绝 __akq_."""

    class P(Strategy):
        def on_bar(self, bar: Bar) -> None:
            assert self.ctx is not None
            self.ctx.schedule(int(bar.timestamp) + 1, _RESERVED)

    with pytest.raises(ValueError, match="__akq_"):
        run_backtest(
            strategy=P,
            data=_minute(["2024-01-02 09:31"], "600000.SH"),
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
        )
    engine = aq.Engine()
    with pytest.raises(ValueError, match="__akq_"):
        engine.add_timer(1, _RESERVED)  # type: ignore[attr-defined]
    engine.add_timer(1, "__daily__|09:30|x")  # type: ignore[attr-defined]


def _daily(dates: List[str], symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates),
            "symbol": symbol,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )


def test_session_of_other_symbol_on_daily_bars_uses_current_bar_interval() -> None:
    """两只股票只有日期的日线：A 的 on_bar 里 B 尚未进本时间片，按 A 的区间解析.

    否则会按 15:00 前 1ns 得到收盘集合竞价，is_market_open("B") 误为 False。
    """

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.got: List[Tuple[str, bool, bool]] = []

        def on_bar(self, bar: Bar) -> None:
            self.got.append(
                (
                    bar.symbol,
                    self.is_market_open("600000.SH"),
                    self.is_market_open("000001.SZ"),
                )
            )

    dates = ["2024-01-02", "2024-01-03"]
    df = pd.concat([_daily(dates, "600000.SH"), _daily(dates, "000001.SZ")])
    r = run_backtest(
        strategy=P,
        data=df,
        symbols=["600000.SH", "000001.SZ"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    assert len(r.strategy.got) == 4
    assert all(a and b for _, a, b in r.strategy.got), r.strategy.got


def test_trading_day_on_start_and_timers_around_close() -> None:
    """on_start 无上下文为 None；15:30 与次日 09:00 定时器都属下一交易日.

    回测的首日盘前定时器不会补触发（见 test_first_day_intraday_timer_not_backfilled_
    before_first_bar），首个交易日开日前的口径由 Rust 单测
    trading_day_before_first_open_resolves_instant_and_none_without_time 覆盖。
    """

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.on_start_day: Any = "unset"
            self.timer_days: List[Tuple[str, Any]] = []

        def on_start(self) -> None:
            self.on_start_day = self.trading_day
            self.schedule_daily("09:00", "pre")
            self.schedule_daily("15:30", "post")

        def on_timer(self, payload: str) -> None:
            assert self.ctx is not None
            t = (
                pd.Timestamp(self.ctx.current_time, tz="UTC")
                .tz_convert(TZ)
                .strftime("%m-%d %H:%M")
            )
            self.timer_days.append((t, self.trading_day))

    df = _minute(["2024-01-02 09:31", "2024-01-03 09:31"], "600000.SH")
    r = run_backtest(
        strategy=P,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        trading_calendar=["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04"],
    )
    assert r.strategy is not None
    assert r.strategy.on_start_day is None
    days = dict(r.strategy.timer_days)
    assert days["01-02 15:30"] == dt.date(2024, 1, 3)
    assert days["01-03 09:00"] == dt.date(2024, 1, 3)


def test_session_resolver_trading_day_methods() -> None:
    """SessionResolver.session_trading_day / next_session_day，越界报中文 ValueError."""
    r = aq.SessionResolver(
        sessions={"au": "CN_FUTURES_NIGHT_0230"},
        trading_calendar=[
            dt.date(2024, 1, 4),
            dt.date(2024, 1, 5),
            dt.date(2024, 1, 8),
        ],
    )
    r.set_active([("au2406", aq.AssetType.Futures)])
    night = pd.Timestamp("2024-01-05 21:30", tz=TZ).value
    assert r.session_trading_day(night) == dt.date(2024, 1, 8)
    assert r.next_session_day(dt.date(2024, 1, 5)) == dt.date(2024, 1, 8)
    with pytest.raises(ValueError, match="超出 trading_calendar 覆盖范围"):
        r.next_session_day(dt.date(2024, 1, 8))
    with pytest.raises(ValueError, match="超出 trading_calendar 覆盖范围"):
        r.session_trading_day(pd.Timestamp("2024-01-08 21:30", tz=TZ).value)


def test_ctx_trading_day_calendar_queries() -> None:
    """ctx.is_trading_day / next_trading_day 走会话日历，越界返回 None."""

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.got: List[Any] = []

        def on_bar(self, bar: Bar) -> None:
            assert self.ctx is not None
            if self.got:
                return
            self.got = [
                self.ctx.is_trading_day(dt.date(2024, 1, 2)),
                self.ctx.is_trading_day(dt.date(2024, 1, 3)),
                self.ctx.next_trading_day(dt.date(2024, 1, 2)),
                self.ctx.next_trading_day(dt.date(2024, 1, 4)),
            ]

    r = run_backtest(
        strategy=P,
        data=_minute(["2024-01-02 09:31"], "600000.SH"),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        trading_calendar=["2023-12-29", "2024-01-02", "2024-01-04"],
    )
    assert r.strategy is not None
    assert r.strategy.got == [True, False, dt.date(2024, 1, 4), None]
