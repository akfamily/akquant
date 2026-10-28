"""阶段 4 终审修复（F4-1～F4-5）的回归测试."""

from __future__ import annotations

import datetime as dt
from typing import Any, List, Tuple

import akquant
import pandas as pd
from akquant import Bar, Strategy, run_backtest

TZ = "Asia/Shanghai"


def _option_daily(symbol: str, days: List[str], closes: List[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(days),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1000,
            "symbol": symbol,
        }
    )


def _run_option_expiry(strategy_cls: type) -> None:
    cfg = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=100_000.0),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol="OPT",
                asset_type="OPTION",
                multiplier=100.0,
                option_type="CALL",
                strike_price=100.0,
                expiry_date=20231201,
                underlying_symbol="UL",
            ),
            akquant.InstrumentConfig(symbol="UL", asset_type="STOCK"),
        ],
    )
    days = ["2023-11-29", "2023-11-30", "2023-12-01", "2023-12-04", "2023-12-05"]
    run_backtest(
        data={
            "OPT": _option_daily("OPT", days[:3], [1.0] * 3),
            "UL": _option_daily("UL", days, [100.0, 101.0, 120.0, 120.0, 120.0]),
        },
        strategy=strategy_cls,
        config=cfg,
        show_progress=False,
    )


class _ExpiryProbe(Strategy):
    events: List[Tuple[str, str]] = []

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol == "OPT" and str(bar.timestamp_iso)[:10] == "2023-11-29":
            self.buy(symbol="OPT", quantity=1)

    def on_expiry(self, event: Any) -> None:
        self.events.append(("on_expiry", str(event["symbol"])))

    def on_pre_open(self, event: Any) -> None:
        self.events.append(("pre_open", str(event["trading_date"])))

    def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
        self.events.append(("after", str(trading_date)))


def test_on_expiry_delivered_by_after_trading_of_expiry_settlement_day() -> None:
    """期权在 12-01 日终结算到期：on_expiry 不晚于 on_after_trading(12-01) 送达.

    此前生命周期定时器路径不调 check_expiry_events，on_expiry 要等到下一交易日
    首根 bar（晚于 12-04 的盘前与开始交易）才送达。
    """
    _ExpiryProbe.events = []
    _run_option_expiry(_ExpiryProbe)
    log = _ExpiryProbe.events
    assert log.count(("on_expiry", "OPT")) == 1, log
    expiry_idx = log.index(("on_expiry", "OPT"))
    assert expiry_idx < log.index(("after", "2023-12-01")), log
    assert expiry_idx < log.index(("pre_open", "2023-12-04")), log


class _AfterTradingSessionProbe(Strategy):
    symbol = "600000.SH"
    quantity = 100
    order_day = dt.date(2024, 1, 2)
    seen: List[Tuple[str, Any, Any, bool, bool]] = []

    def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
        self.seen.append(
            (
                str(trading_date),
                self.session_of(),
                self.session_of(self.symbol),
                self.is_market_open(include_auction=True),
                self.is_market_open(self.symbol, include_auction=True),
            )
        )
        if trading_date == self.order_day:
            self.buy(self.symbol, self.quantity)


def _assert_after_trading_closed_and_order_fills_next_day(
    result: Any, probe: type[_AfterTradingSessionProbe], next_day: str
) -> None:
    closed = akquant.TradingSession.Closed
    assert probe.seen, "on_after_trading 未触发"
    for day, own, other, open_any, open_sym in probe.seen:
        assert (own, other, open_any, open_sym) == (closed, closed, False, False), (
            day,
            own,
            other,
            open_any,
            open_sym,
        )
    orders = result.orders_df
    assert len(orders) == 1, orders
    row = orders.iloc[0]
    assert str(row["status"]).lower() == "filled", orders
    assert (
        pd.Timestamp(row["updated_at"]).tz_convert(TZ).strftime("%Y-%m-%d") == next_day
    ), orders


def test_after_trading_session_closed_stock_minute() -> None:
    """股票分钟线：on_after_trading（收盘 + 1ns）时段为闭市，其中下的单次日成交."""

    class P(_AfterTradingSessionProbe):
        seen: List[Tuple[str, Any, Any, bool, bool]] = []

    times = [
        "2024-01-02 09:31",
        "2024-01-02 15:00",
        "2024-01-03 09:31",
        "2024-01-03 15:00",
    ]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(times).tz_localize(TZ),
            "symbol": "600000.SH",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )
    result = run_backtest(
        strategy=P,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    _assert_after_trading_closed_and_order_fills_next_day(result, P, "2024-01-03")


def test_after_trading_session_closed_night_futures() -> None:
    """夜盘期货（CN_FUTURES_NIGHT_0230）：收盘后时段为闭市，其中下的单当晚夜盘成交."""

    class P(_AfterTradingSessionProbe):
        symbol = "au2406"
        quantity = 1
        order_day = dt.date(2024, 1, 5)
        seen: List[Tuple[str, Any, Any, bool, bool]] = []

    times = [
        "2024-01-04 21:30",
        "2024-01-05 10:00",
        "2024-01-05 14:59",
        "2024-01-05 21:30",
        "2024-01-08 10:00",
        "2024-01-08 15:00",
    ]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(times).tz_localize(TZ),
            "symbol": "au2406",
            "open": 500.0,
            "high": 500.0,
            "low": 500.0,
            "close": 500.0,
            "volume": 1e3,
        }
    )
    cfg = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=1e6),
        instruments_config=[
            akquant.InstrumentConfig(
                symbol="au2406",
                asset_type="FUTURES",
                multiplier=1000.0,
                margin_ratio=0.1,
            )
        ],
    )
    result = run_backtest(
        strategy=P,
        data=df,
        config=cfg,
        show_progress=False,
        sessions={"au": "CN_FUTURES_NIGHT_0230"},
        trading_calendar=[
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ],
    )
    # 01-05 收盘后下的单属于下一交易日（01-08），在当晚 21:30 夜盘成交
    _assert_after_trading_closed_and_order_fills_next_day(result, P, "2024-01-05")


_LIFECYCLE_PREFIX = "__akq_lifecycle__|"


def _count_lifecycle_dispatches(monkeypatch: Any) -> List[str]:
    """把策略的定时器入口包一层，记录派发到 Python 的生命周期事件 payload."""
    seen: List[str] = []
    original = Strategy._on_timer_event_and_flush

    def wrapped(self: Strategy, payload: str, ctx: Any) -> Any:
        if payload.startswith(_LIFECYCLE_PREFIX):
            seen.append(payload)
        return original(self, payload, ctx)

    monkeypatch.setattr(Strategy, "_on_timer_event_and_flush", wrapped)
    return seen


def _three_symbol_daily(days: int) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-03", periods=days)
    frames = [
        pd.DataFrame(
            {
                "timestamp": idx,
                "symbol": f"60000{i}.SH",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1e6,
            }
        )
        for i in range(3)
    ]
    return pd.concat(frames)


_SYMBOLS = [f"60000{i}.SH" for i in range(3)]


def test_on_bar_only_strategy_gets_no_lifecycle_dispatch(monkeypatch: Any) -> None:
    """只写 on_bar 的策略（含槽位）不产生任何生命周期派发."""

    class P(Strategy):
        def on_bar(self, bar: Bar) -> None:
            pass

    seen = _count_lifecycle_dispatches(monkeypatch)
    run_backtest(
        strategy=P,
        data=_three_symbol_daily(30),
        symbols=_SYMBOLS,
        initial_cash=1e6,
        show_progress=False,
        strategies_by_slot={"b": P},
    )
    assert seen == []


def test_after_trading_override_dispatches_once_per_day(monkeypatch: Any) -> None:
    """只重写 on_after_trading：每个交易日恰好一次收盘后派发，没有盘前/开始交易派发."""
    days = 30

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.after_days: List[dt.date] = []

        def on_bar(self, bar: Bar) -> None:
            pass

        def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
            self.after_days.append(trading_date)

    seen = _count_lifecycle_dispatches(monkeypatch)
    result = run_backtest(
        strategy=P,
        data=_three_symbol_daily(days),
        symbols=_SYMBOLS,
        initial_cash=1e6,
        show_progress=False,
    )
    assert len(seen) == days, seen[:5]
    assert all("|after_trading|" in p for p in seen), seen[:5]
    assert result.strategy is not None
    assert len(result.strategy.after_days) == days


def test_expired_and_expiry_delivered_without_lifecycle_hooks(
    monkeypatch: Any,
) -> None:
    """未重写生命周期钩子时，Expired 与 on_expiry 仍在日终结算后的收盘后事件上送达."""

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.expired_at: List[str] = []
            self.done = False

        def on_bar(self, bar: Bar) -> None:
            if not self.done:
                self.done = True
                self.buy(
                    bar.symbol, 100, price=1.0, time_in_force=akquant.TimeInForce.Day
                )

        def on_order(self, order: Any) -> None:
            if str(order.status).split(".")[-1] == "Expired":
                ts = int(self.ctx.current_time) if self.ctx is not None else 0
                self.expired_at.append(
                    pd.Timestamp(ts, tz="UTC").tz_convert(TZ).strftime("%m-%d %H:%M")
                )

    seen = _count_lifecycle_dispatches(monkeypatch)
    times = ["2024-01-02 10:00", "2024-01-02 15:00", "2024-01-03 10:00"]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(times).tz_localize(TZ),
            "symbol": "600000.SH",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )
    result = run_backtest(
        strategy=P,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert result.strategy is not None
    # 收盘后事件（15:00 + 1ns）送达，而不是等到次日 10:00 的 bar
    assert result.strategy.expired_at == ["01-02 15:00"]
    assert seen == ["__akq_lifecycle__|after_trading|2024-01-02"]

    class Q(_ExpiryProbe):
        on_pre_open = Strategy.on_pre_open  # type: ignore[assignment]
        on_after_trading = Strategy.on_after_trading  # type: ignore[assignment]

    seen.clear()
    _ExpiryProbe.events = []
    _run_option_expiry(Q)
    assert _ExpiryProbe.events == [("on_expiry", "OPT")]
    # 只有到期结算那天（12-01）的收盘后事件派发
    assert seen == ["__akq_lifecycle__|after_trading|2023-12-01"]


_NY = "America/New_York"


def _ny_weekday_bars() -> pd.DataFrame:
    idx = pd.bdate_range("2024-05-29", "2024-06-05").tz_localize(_NY) + pd.Timedelta(
        hours=10
    )
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAPL",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )


class _NyScheduleProbe(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.days: List[str] = []
        self.fired: List[str] = []

    def on_start(self) -> None:
        self.schedule_daily("11:00", "d")

    def on_bar(self, bar: Bar) -> None:
        if not self.days:
            self.days = [d.strftime("%a %m-%d") for d in self.trading_days]

    def on_timer(self, payload: str) -> None:
        ts = int(self.ctx.current_time) if self.ctx is not None else 0
        self.fired.append(
            pd.Timestamp(ts, tz="UTC").tz_convert(_NY).strftime("%a %m-%d")
        )


_NY_DATA_DAYS = [
    "Wed 05-29",
    "Thu 05-30",
    "Fri 05-31",
    "Mon 06-03",
    "Tue 06-04",
    "Wed 06-05",
]


def test_always_open_fallback_trading_days_only_days_with_data() -> None:
    """非中国时区、未传 trading_calendar（全日历兜底）：周末无数据不算交易日."""
    result = run_backtest(
        strategy=_NyScheduleProbe,
        data=_ny_weekday_bars(),
        symbols=["AAPL"],
        initial_cash=1e5,
        show_progress=False,
        timezone=_NY,
    )
    strategy = result.strategy
    assert isinstance(strategy, _NyScheduleProbe)
    assert strategy.days == _NY_DATA_DAYS
    assert strategy.fired == _NY_DATA_DAYS


def test_always_open_with_explicit_calendar_uses_calendar_days() -> None:
    """显式传 trading_calendar 时按日历：区间内无数据的日历交易日也算（周六示例）."""
    calendar = [
        "2024-05-29",
        "2024-05-30",
        "2024-05-31",
        "2024-06-01",
        "2024-06-03",
        "2024-06-04",
        "2024-06-05",
    ]
    result = run_backtest(
        strategy=_NyScheduleProbe,
        data=_ny_weekday_bars(),
        symbols=["AAPL"],
        initial_cash=1e5,
        show_progress=False,
        timezone=_NY,
        trading_calendar=calendar,
    )
    strategy = result.strategy
    assert isinstance(strategy, _NyScheduleProbe)
    expected = [
        "Wed 05-29",
        "Thu 05-30",
        "Fri 05-31",
        "Sat 06-01",
        "Mon 06-03",
        "Tue 06-04",
        "Wed 06-05",
    ]
    assert strategy.days == expected
    assert strategy.fired == expected


def _live_ctx(all_days: bool) -> Any:
    from unittest.mock import MagicMock

    from akquant.akquant import StrategyContext

    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.current_time = 0
    ctx.uses_all_days_calendar.return_value = all_days
    ctx.is_trading_day.side_effect = lambda d: True if all_days else d.weekday() < 5
    ctx.next_trading_day.side_effect = lambda d: next(
        d + dt.timedelta(days=k)
        for k in range(1, 8)
        if all_days or (d + dt.timedelta(days=k)).weekday() < 5
    )
    return ctx


def test_live_all_days_calendar_warns_once(caplog: Any) -> None:
    """实盘未传 trading_calendar 且全是 ALWAYS_OPEN：首次登记定时任务时告警一次."""
    import logging

    class P(Strategy):
        pass

    for all_days, expected in ((True, 1), (False, 0)):
        caplog.clear()
        s = P()
        s.ctx = _live_ctx(all_days)
        s._trading_days = []
        with caplog.at_level(logging.WARNING):
            s.schedule_daily("10:00", "a")
            s.schedule_weekly("10:00", "b")
            s.schedule_daily("11:00", "c")
        hits = [
            r
            for r in caplog.records
            if "ALWAYS_OPEN 每天都算交易日" in r.getMessage()
            and "美股等请传 trading_calendar" in r.getMessage()
        ]
        assert len(hits) == expected, (all_days, [r.getMessage() for r in hits])


_REMOVED_SWITCH = "enable_precise_day_boundary_hooks"


def test_removed_precise_switch_class_attribute_raises_chinese_error() -> None:
    """类属性写旧开关：定义类时报中文错误，说明开关已删除."""
    import pytest

    with pytest.raises(ValueError, match="已删除") as info:

        class P(Strategy):
            enable_precise_day_boundary_hooks = True

    assert _REMOVED_SWITCH in str(info.value)


def test_removed_precise_switch_instance_attribute_raises_chinese_error() -> None:
    """实例属性写旧开关：赋值时报中文错误."""
    import pytest

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            setattr(self, _REMOVED_SWITCH, True)

    with pytest.raises(ValueError, match="已删除") as info:
        P()
    assert _REMOVED_SWITCH in str(info.value)


def test_removed_precise_switch_runtime_config_kwarg_raises_chinese_error() -> None:
    """StrategyRuntimeConfig 构造传旧开关：与 dict 路径一致报中文 ValueError."""
    import pytest
    from akquant import StrategyRuntimeConfig

    with pytest.raises(ValueError, match="已删除") as info:
        kwargs: Any = {_REMOVED_SWITCH: True}
        StrategyRuntimeConfig(**kwargs)
    assert _REMOVED_SWITCH in str(info.value)
    # 正常字段不受影响
    assert StrategyRuntimeConfig(error_mode="continue").error_mode == "continue"


def test_schedule_bad_time_raises_chinese_value_error() -> None:
    """定时任务时刻写错：回测、实盘、on_start（上下文未就绪）都抛中文 ValueError."""
    import pytest

    class P(Strategy):
        pass

    cases = []
    pending = P()  # 上下文未就绪（on_start 中）
    cases.append(pending)
    live = P()
    live.ctx = _live_ctx(False)
    live._trading_days = []
    cases.append(live)
    backtest = P()
    backtest.ctx = _live_ctx(False)
    backtest._trading_days = [pd.Timestamp("2024-01-02", tz=TZ)]
    cases.append(backtest)
    for s in cases:
        for register in (s.schedule_daily, s.schedule_weekly, s.schedule_monthly):
            with pytest.raises(ValueError, match="无法解析定时器时刻"):
                register("25:99", "x")


def test_live_lifecycle_event_flushes_pending_schedules() -> None:
    """实盘：on_start 登记的定时任务在生命周期事件上也刷出；回测仍等首个行情事件."""

    class P(Strategy):
        pass

    payload = "__akq_lifecycle__|pre_open|2024-01-05"

    live = P()
    live._trading_days = []
    live.schedule_daily("10:00", "d")  # 上下文未就绪：暂存
    ctx = _live_ctx(False)
    live._on_timer_event(payload, ctx)
    assert [c.args[1] for c in ctx.schedule.call_args_list] == ["__daily__|10:00|d"]

    backtest = P()
    backtest._trading_days = [pd.Timestamp("2024-01-05", tz=TZ)]
    backtest.schedule_daily("10:00", "d")
    ctx = _live_ctx(False)
    backtest._on_timer_event(payload, ctx)
    assert ctx.schedule.call_count == 0
    assert backtest._pending_daily_timers == [("10:00", "d")]
