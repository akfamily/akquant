"""生命周期事件驱动的交易日钩子（回测与实盘一致）."""

from __future__ import annotations

import datetime as dt
from typing import Any, List, Tuple

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy, run_backtest

TZ = "Asia/Shanghai"


def _local(ns: int, tz: str = TZ) -> str:
    return pd.Timestamp(ns, tz="UTC").tz_convert(tz).strftime("%m-%d %H:%M:%S.%f")


def _minute(
    times: List[str], symbol: str = "600000.SH", price: float = 10.0
) -> pd.DataFrame:
    idx = pd.to_datetime(times).tz_localize(TZ)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1e6,
        }
    )


def _daily(dates: List[str], symbol: str = "600000.SH", tz: str = TZ) -> pd.DataFrame:
    idx = pd.to_datetime(dates).tz_localize(tz)
    n = len(idx)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": [10.0 + i for i in range(n)],
            "high": [10.5 + i for i in range(n)],
            "low": [9.5 + i for i in range(n)],
            "close": [10.2 + i for i in range(n)],
            "volume": 1e6,
        }
    )


class _HookLog(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.records: List[Tuple[str, Any, str]] = []
        self.raw: List[Tuple[str, Any, int]] = []

    def on_pre_open(self, event: Any) -> None:
        ts = int(event["timestamp"])
        self.records.append(("pre_open", event["trading_date"], _local(ts)))
        self.raw.append(("pre_open", event["trading_date"], ts))

    def on_before_trading(self, trading_date: dt.date, timestamp: int) -> None:
        self.records.append(("before", trading_date, _local(timestamp)))
        self.raw.append(("before", trading_date, int(timestamp)))

    def on_bar(self, bar: Bar) -> None:
        self.records.append(("bar", self.trading_day, _local(bar.timestamp)))
        self.raw.append(("bar", self.trading_day, int(bar.timestamp)))

    def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
        self.records.append(("after", trading_date, _local(timestamp)))
        self.raw.append(("after", trading_date, int(timestamp)))


def test_stock_minute_lifecycle_order_and_times() -> None:
    """股票分钟线：盘前 09:15 前 1ns、开始交易 09:30 后 1ns、收盘后在结算之后."""
    df = _minute(
        ["2024-01-02 09:31", "2024-01-02 15:00", "2024-01-03 09:31", "2024-01-03 15:00"]
    )
    r = run_backtest(
        strategy=_HookLog,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    d2, d3 = dt.date(2024, 1, 2), dt.date(2024, 1, 3)
    assert r.strategy.records == [
        ("pre_open", d2, "01-02 09:14:59.999999"),
        ("before", d2, "01-02 09:30:00.000000"),
        ("bar", d2, "01-02 09:31:00.000000"),
        ("bar", d2, "01-02 15:00:00.000000"),
        ("after", d2, "01-02 15:00:00.000000"),
        ("pre_open", d3, "01-03 09:14:59.999999"),
        ("before", d3, "01-03 09:30:00.000000"),
        ("bar", d3, "01-03 09:31:00.000000"),
        ("bar", d3, "01-03 15:00:00.000000"),
        ("after", d3, "01-03 15:00:00.000000"),
    ]


def test_night_futures_lifecycle_belongs_to_next_trading_day() -> None:
    """夜盘期货：周五夜盘属周一交易日，周一的盘前与开始交易在周五夜盘开始时."""
    df = _minute(
        ["2024-01-05 14:00", "2024-01-05 21:30", "2024-01-08 10:00"],
        symbol="au2406",
        price=500.0,
    )
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
        strategy=_HookLog,
        data=df,
        config=cfg,
        show_progress=False,
        sessions={"au": "CN_FUTURES_NIGHT_0230"},
    )
    assert r.strategy is not None
    kinds = [(k, d) for k, d, _ in r.strategy.records if k != "bar"]
    d5, d8 = dt.date(2024, 1, 5), dt.date(2024, 1, 8)
    assert kinds == [
        ("pre_open", d5),
        ("before", d5),
        ("after", d5),
        ("pre_open", d8),
        ("before", d8),
    ]
    # 周一交易日的盘前在周五 20:55 前、开始交易在周五 21:00
    times = {(k, d): t for k, d, t in r.strategy.records}
    assert times[("pre_open", d8)] == "01-05 20:54:59.999999"
    assert times[("before", d8)] == "01-05 21:00:00.000000"


def test_after_trading_sees_settled_state_and_orders_belong_to_next_day() -> None:
    """on_after_trading 在结算后：T+1 已解冻、当日 Day 单已过期；其中下的单次日成交."""

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.avail: List[float] = []
            self.bought = False

        def on_bar(self, bar: Bar) -> None:
            if not self.bought:
                self.bought = True
                self.buy(bar.symbol, 100)

        def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
            self.avail.append(float(self.get_available_position("600000.SH")))
            if trading_date == dt.date(2024, 1, 3):
                self.sell("600000.SH", 100)

    df = _minute(
        [
            "2024-01-02 09:31",
            "2024-01-02 09:32",
            "2024-01-02 15:00",
            "2024-01-03 15:00",
            "2024-01-04 09:31",
        ]
    )
    r = run_backtest(
        strategy=P,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        t_plus_one=True,
    )
    assert r.strategy is not None
    assert r.strategy.avail[0] == 100.0  # 01-02 结算后已解冻
    sells = r.orders_df[
        r.orders_df["side"].astype(str).str.lower().str.contains("sell")
    ]
    t = pd.Timestamp(sells.iloc[0]["updated_at"]).tz_convert(TZ)
    assert t.strftime("%m-%d %H:%M") == "01-04 09:31"


def test_after_trading_receives_expired_day_order_first() -> None:
    """当日过期的 Day 单在 on_after_trading 之前（或同时）回调 on_order(Expired)."""

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.events: List[str] = []
            self.done = False

        def on_bar(self, bar: Bar) -> None:
            if not self.done:
                self.done = True
                self.buy(bar.symbol, 100, price=1.0, time_in_force=aq.TimeInForce.Day)

        def on_order(self, order: Any) -> None:
            if str(order.status).split(".")[-1] == "Expired":
                self.events.append("expired")

        def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
            self.events.append(f"after:{trading_date.isoformat()}")

    df = _minute(["2024-01-02 10:00", "2024-01-02 15:00", "2024-01-03 10:00"])
    r = run_backtest(
        strategy=P,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    assert r.strategy.events[:2] == ["expired", "after:2024-01-02"]


def test_daily_before_trading_order_fills_same_day_open() -> None:
    """日线：09:30 的 on_before_trading 中下 NextOpen 单，当日开盘价成交.

    只有日期的日线改写到 15:00，单在当日这根 bar 上按开盘价成交（Ruling S4-3）。
    """

    class P(Strategy):
        def on_before_trading(self, trading_date: dt.date, timestamp: int) -> None:
            if trading_date == dt.date(2024, 1, 3):
                self.buy("600000.SH", 100)

    r = run_backtest(
        strategy=P,
        data=_daily(["2024-01-02", "2024-01-03", "2024-01-04"]),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    filled = r.orders_df[r.orders_df["filled_quantity"] > 0]
    assert float(filled.iloc[0]["avg_price"]) == 11.0  # 01-03 开盘价
    t = pd.Timestamp(filled.iloc[0]["updated_at"]).tz_convert(TZ)
    assert t.strftime("%m-%d %H:%M") == "01-03 15:00"


def test_no_lifecycle_after_data_ends_and_no_after_for_unsettled_day() -> None:
    """数据在盘中结束：当日 after_trading 不触发，之后不再有任何生命周期事件."""
    df = _minute(["2024-01-02 15:00", "2024-01-03 10:00"])
    r = run_backtest(
        strategy=_HookLog,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    kinds = [(k, d) for k, d, _ in r.strategy.records if k != "bar"]
    assert kinds[-2:] == [
        ("pre_open", dt.date(2024, 1, 3)),
        ("before", dt.date(2024, 1, 3)),
    ]


def test_each_hook_once_per_trading_day_with_multiple_symbols() -> None:
    """多标的同一交易日：每个钩子只触发一次."""
    df = pd.concat(
        [
            _minute(["2024-01-02 09:31", "2024-01-02 15:00"], "600000.SH"),
            _minute(["2024-01-02 09:31", "2024-01-02 15:00"], "000001.SZ"),
        ]
    )
    r = run_backtest(
        strategy=_HookLog,
        data=df,
        symbols=["600000.SH", "000001.SZ"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    kinds = [k for k, _, _ in r.strategy.records if k != "bar"]
    assert kinds == ["pre_open", "before", "after"]


def test_always_open_daily_pre_open_follows_previous_after_trading() -> None:
    """全天连续（纽约时区只有日期的日线）：盘前不早于上一交易日收盘之后.

    全天连续时交易日 D 的首段起点恰是 D-1 的收盘（本地 0 点），盘前若取
    “首段起点 - 1ns”会落在上一交易日结算之前；公式取 max(.., 上一交易日收盘
    定时器 + 1ns)。
    """
    tz = "America/New_York"
    r = run_backtest(
        strategy=_HookLog,
        data=_daily(["2024-01-02", "2024-01-03"], symbol="AAPL", tz=tz),
        symbols=["AAPL"],
        initial_cash=1e5,
        show_progress=False,
        timezone=tz,
    )
    assert r.strategy is not None
    raw = r.strategy.raw
    kinds = [(k, d) for k, d, _ in raw if k != "bar"]
    d2, d3 = dt.date(2024, 1, 2), dt.date(2024, 1, 3)
    assert kinds == [
        ("pre_open", d2),
        ("before", d2),
        ("after", d2),
        ("pre_open", d3),
        ("before", d3),
        ("after", d3),
    ]
    stamps = {(k, d): ts for k, d, ts in raw if k != "bar"}
    assert stamps[("pre_open", d3)] > stamps[("after", d2)]
    assert stamps[("before", d3)] > stamps[("pre_open", d3)]


def test_resume_mid_day_does_not_refire_pre_open_or_before(tmp_path: Any) -> None:
    """同一交易日内存档续跑：存档前已触发的盘前/开始交易不再触发，收盘照常触发."""
    from akquant import run_from_checkpoint, save_checkpoint

    part1 = _minute(["2024-01-02 09:31", "2024-01-02 10:00"])
    r1 = run_backtest(
        strategy=_HookLog,
        data=part1,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r1.strategy is not None
    kinds1 = [k for k, _, _ in r1.strategy.records if k != "bar"]
    assert kinds1 == ["pre_open", "before"]
    path = str(tmp_path / "mid.ckpt")
    assert r1.engine is not None
    save_checkpoint(r1.engine, r1.strategy, path)

    part2 = _minute(["2024-01-02 14:00", "2024-01-02 15:00", "2024-01-03 09:31"])
    r2 = run_from_checkpoint(
        path, data=part2, symbols=["600000.SH"], show_progress=False
    )
    assert r2.strategy is not None
    resumed = [(k, d) for k, d, _ in r2.strategy.records[len(r1.strategy.records) :]]
    d2, d3 = dt.date(2024, 1, 2), dt.date(2024, 1, 3)
    assert [x for x in resumed if x[0] != "bar"] == [
        ("after", d2),
        ("pre_open", d3),
        ("before", d3),
    ]


class _ResumeProbe(_HookLog):
    """首根 bar 买入 100 股，记录 on_before_trading 时的可用数量."""

    def __init__(self) -> None:
        super().__init__()
        self.bought = False
        self.avail_before: List[Tuple[Any, float]] = []

    def on_bar(self, bar: Bar) -> None:
        super().on_bar(bar)
        if not self.bought:
            self.bought = True
            self.buy(bar.symbol, 100)

    def on_before_trading(self, trading_date: dt.date, timestamp: int) -> None:
        super().on_before_trading(trading_date, timestamp)
        self.avail_before.append(
            (trading_date, float(self.get_available_position("600000.SH")))
        )


def test_resume_across_unsettled_day_matches_continuous_run(tmp_path: Any) -> None:
    """存档时当日未结算、续跑从下一交易日开始：钩子次序与连续跑一致.

    上一交易日先在其收盘结算并触发 on_after_trading，然后才是下一交易日的盘前与
    开始交易；开始交易时 T+1 已解冻。
    """
    from akquant import run_from_checkpoint, save_checkpoint

    part1 = ["2024-01-02 09:31", "2024-01-02 10:00"]
    part2 = ["2024-01-03 09:31", "2024-01-03 15:00"]
    full = run_backtest(
        strategy=_ResumeProbe,
        data=_minute(part1 + part2),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        t_plus_one=True,
    )
    assert full.strategy is not None

    r1 = run_backtest(
        strategy=_ResumeProbe,
        data=_minute(part1),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        t_plus_one=True,
    )
    assert r1.strategy is not None and r1.engine is not None
    path = str(tmp_path / "unsettled.ckpt")
    save_checkpoint(r1.engine, r1.strategy, path)
    r2 = run_from_checkpoint(
        path,
        data=_minute(part2),
        symbols=["600000.SH"],
        show_progress=False,
        t_plus_one=True,
    )
    assert r2.strategy is not None
    assert r2.strategy.records == full.strategy.records
    assert r2.strategy.avail_before == full.strategy.avail_before
    assert r2.strategy.avail_before[-1] == (dt.date(2024, 1, 3), 100.0)


def test_build_session_trading_days_includes_empty_days_and_night_bars() -> None:
    """交易日列表含区间内无数据的交易日；夜盘 bar 的截面归下一交易日."""
    from akquant.akquant import AssetType
    from akquant.backtest.engine import _build_session_trading_days

    resolver = aq.SessionResolver(sessions={"au": "CN_FUTURES_NIGHT_0230"}, timezone=TZ)
    resolver.set_active([("au2406", AssetType.Futures)])
    stamps = pd.to_datetime(["2024-01-02 10:00", "2024-01-04 21:30"]).tz_localize(TZ)
    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.DatetimeIndex(stamps))
    days, cross = _build_session_trading_days({"au2406": frame}, resolver, TZ)
    assert [d.date() for d in days] == [
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),  # 无数据的交易日
        dt.date(2024, 1, 4),  # 无数据的交易日（当晚夜盘归 01-05）
        dt.date(2024, 1, 5),
    ]
    assert all(str(d.tz) == TZ for d in days)
    assert cross == {
        "2024-01-02": int(stamps[0].value),
        "2024-01-05": int(stamps[1].value),
    }


def test_schedule_daily_fires_on_calendar_days_without_data() -> None:
    """数据缺 01-03（但它是交易日）：schedule_daily 仍在 01-03 触发."""

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.fired: List[str] = []

        def on_start(self) -> None:
            self.schedule_daily("10:00", "tick")

        def on_timer(self, payload: str) -> None:
            assert self.ctx is not None
            self.fired.append(
                pd.Timestamp(self.ctx.current_time, tz="UTC")
                .tz_convert(TZ)
                .strftime("%m-%d %H:%M")
            )

    df = _minute(["2024-01-02 09:31", "2024-01-04 09:31", "2024-01-04 15:00"])
    r = run_backtest(
        strategy=P,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        trading_calendar=[
            "2023-12-29",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
        ],
    )
    assert r.strategy is not None
    assert r.strategy.fired == ["01-02 10:00", "01-03 10:00", "01-04 10:00"]


def test_schedule_daily_skips_non_trading_day_in_calendar() -> None:
    """日历上不是交易日的 01-03 不触发（即使夹在两个交易日之间）."""

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.fired: List[str] = []

        def on_start(self) -> None:
            self.schedule_daily("10:00", "tick")

        def on_timer(self, payload: str) -> None:
            assert self.ctx is not None
            self.fired.append(
                pd.Timestamp(self.ctx.current_time, tz="UTC")
                .tz_convert(TZ)
                .strftime("%m-%d")
            )

    df = _minute(["2024-01-02 09:31", "2024-01-04 15:00"])
    r = run_backtest(
        strategy=P,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        trading_calendar=["2023-12-29", "2024-01-02", "2024-01-04"],
    )
    assert r.strategy is not None
    assert r.strategy.fired == ["01-02", "01-04"]


def test_schedule_weekly_monthly_registered_in_on_start_fire_in_backtest() -> None:
    """on_start 里登记的 weekly/monthly 延后到上下文就绪再登记（此前直接被忽略）."""

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.fired: List[Tuple[str, str]] = []

        def on_start(self) -> None:
            self.schedule_weekly("10:00", "wk")
            self.schedule_monthly("10:30", "mo")

        def on_timer(self, payload: str) -> None:
            assert self.ctx is not None
            self.fired.append(
                (
                    payload,
                    pd.Timestamp(self.ctx.current_time, tz="UTC")
                    .tz_convert(TZ)
                    .strftime("%m-%d %H:%M"),
                )
            )

    # 01-29 周一无数据但为交易日；02-01 为 2 月首个交易日
    df = _minute(["2024-01-25 09:31", "2024-02-02 15:00"])
    r = run_backtest(
        strategy=P,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        trading_calendar=[
            "2024-01-25",
            "2024-01-26",
            "2024-01-29",
            "2024-01-30",
            "2024-01-31",
            "2024-02-01",
            "2024-02-02",
        ],
    )
    assert r.strategy is not None
    assert r.strategy.fired == [
        ("wk", "01-25 10:00"),
        ("mo", "01-25 10:30"),
        ("wk", "01-29 10:00"),
        ("mo", "02-01 10:30"),
    ]


# 2024 年元旦：12-30/31 周末、01-01 休市；01-09～01-26 为日历空档；
# 春节：02-09～02-18 休市
_LIVE_DAYS = {
    dt.date(2023, 12, 28),
    dt.date(2023, 12, 29),
    dt.date(2024, 1, 2),
    dt.date(2024, 1, 3),
    dt.date(2024, 1, 4),
    dt.date(2024, 1, 5),
    dt.date(2024, 1, 8),
    dt.date(2024, 1, 29),
    dt.date(2024, 1, 30),
    dt.date(2024, 1, 31),
    dt.date(2024, 2, 1),
    dt.date(2024, 2, 2),
    dt.date(2024, 2, 5),
    dt.date(2024, 2, 8),
    dt.date(2024, 2, 19),
    dt.date(2024, 3, 1),
}


def _is_td(day: dt.date) -> bool:
    return day in _LIVE_DAYS


def _next_td(day: dt.date) -> Any:
    later = sorted(d for d in _LIVE_DAYS if d > day)
    return later[0] if later else None


def _next(kind: str, time_str: str, now: str) -> str:
    from akquant.strategy_scheduler import next_occurrence

    out = next_occurrence(
        kind, time_str, pd.Timestamp(now, tz=TZ), _is_td, _next_td, TZ
    )
    assert out is not None
    assert str(out.tz) == TZ
    return out.strftime("%Y-%m-%d %H:%M")


def test_next_occurrence_daily() -> None:
    """Daily：今天未到点取今天，否则取下一交易日；跳过周末与节假日."""
    # 今天未到点 → 今天
    assert _next("daily", "10:00", "2024-01-03 09:00") == "2024-01-03 10:00"
    # 今天已过（含恰好等于）→ 下一交易日
    assert _next("daily", "10:00", "2024-01-03 10:00") == "2024-01-04 10:00"
    assert _next("daily", "10:00", "2024-01-03 11:00") == "2024-01-04 10:00"
    # 周五 → 周一
    assert _next("daily", "10:00", "2024-01-05 11:00") == "2024-01-08 10:00"
    # 节假日跳过：12-29 已过 → 跳过周末与元旦到 01-02；元旦当天未到点也不触发
    assert _next("daily", "10:00", "2023-12-29 11:00") == "2024-01-02 10:00"
    assert _next("daily", "10:00", "2024-01-01 08:00") == "2024-01-02 10:00"


def test_next_occurrence_weekly() -> None:
    """Weekly：本周首个交易日已过则取下一个有交易日的周的首个交易日."""
    # 本周首个交易日 01-02（周一元旦休市）尚未到点 → 01-02
    assert _next("weekly", "10:00", "2024-01-01 08:00") == "2024-01-02 10:00"
    # 本周首个交易日已过 → 下周首个交易日
    assert _next("weekly", "10:00", "2024-01-02 10:00") == "2024-01-08 10:00"
    assert _next("weekly", "10:00", "2024-01-04 09:00") == "2024-01-08 10:00"
    # 之后两周无交易日（日历空档）→ 顺延到再下一个有交易日的周
    assert _next("weekly", "10:00", "2024-01-08 11:00") == "2024-01-29 10:00"
    # 春节周（02-12 起整周休市）跳过
    assert _next("weekly", "10:00", "2024-02-08 11:00") == "2024-02-19 10:00"


def test_next_occurrence_monthly() -> None:
    """Monthly：本月首个交易日已过则取下月首个交易日."""
    assert _next("monthly", "09:30", "2024-01-01 08:00") == "2024-01-02 09:30"
    assert _next("monthly", "09:30", "2024-01-02 09:30") == "2024-02-01 09:30"
    assert _next("monthly", "09:30", "2024-01-31 15:00") == "2024-02-01 09:30"
    assert _next("monthly", "09:30", "2024-02-01 10:00") == "2024-03-01 09:30"


def test_next_occurrence_out_of_calendar_returns_none() -> None:
    """日历尽头之后找不到下一次触发日时返回 None."""
    from akquant.strategy_scheduler import next_occurrence

    now = pd.Timestamp("2024-03-01 11:00", tz=TZ)
    for kind in ("daily", "weekly", "monthly"):
        assert next_occurrence(kind, "10:00", now, _is_td, _next_td, TZ) is None


def test_live_weekly_monthly_register_next_and_reschedule() -> None:
    """实盘（无 _trading_days）weekly/monthly 不再直接返回：登记下一次，触发后重排."""
    from unittest.mock import MagicMock

    from akquant.akquant import StrategyContext

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.fired: List[str] = []

        def on_timer(self, payload: str) -> None:
            self.fired.append(payload)

    s = P()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.current_time = 0
    ctx.is_trading_day.side_effect = lambda d: d.weekday() < 5
    ctx.next_trading_day.side_effect = lambda d: next(
        d + dt.timedelta(days=k)
        for k in range(1, 8)
        if (d + dt.timedelta(days=k)).weekday() < 5
    )
    s.ctx = ctx
    s._trading_days = []
    now = pd.Timestamp.now(tz=s.timezone)

    s.schedule_weekly("10:00", "wk")
    s.schedule_monthly("10:00", "mo")
    calls = ctx.schedule.call_args_list
    assert [c.args[1] for c in calls] == [
        "__weekly__|10:00|wk",
        "__monthly__|10:00|mo",
    ]
    for c in calls:
        target = pd.Timestamp(c.args[0], tz="UTC").tz_convert(s.timezone)
        assert target > now
        assert target.weekday() < 5 and target.strftime("%H:%M") == "10:00"

    ctx.schedule.reset_mock()
    s._on_timer_event("__weekly__|10:00|wk", ctx)
    s._on_timer_event("__monthly__|10:00|mo", ctx)
    assert s.fired == ["wk", "mo"]
    assert [c.args[1] for c in ctx.schedule.call_args_list] == [
        "__weekly__|10:00|wk",
        "__monthly__|10:00|mo",
    ]


def _all_days(day: dt.date) -> bool:
    return True


def _following(day: dt.date) -> Any:
    return day + dt.timedelta(days=1)


def test_next_occurrence_dst_gap_and_overlap_do_not_raise() -> None:
    """夏令时缺口里的时刻顺延到缺口结束，回拨重叠里的时刻取第一次出现."""
    from akquant.strategy_scheduler import next_occurrence

    ny = "America/New_York"
    # 2024-03-10 02:00 拨快到 03:00：02:30 不存在 → 03:00 EDT
    gap = next_occurrence(
        "daily",
        "02:30",
        pd.Timestamp("2024-03-10 00:00", tz=ny),
        _all_days,
        _following,
        ny,
    )
    assert gap is not None
    assert gap == pd.Timestamp("2024-03-10 03:00", tz=ny)
    assert gap.utcoffset() == dt.timedelta(hours=-4)
    # 2024-11-03 02:00 回拨到 01:00：01:30 出现两次 → 取第一次（EDT，UTC-4）
    overlap = next_occurrence(
        "daily",
        "01:30",
        pd.Timestamp("2024-11-03 00:00", tz=ny),
        _all_days,
        _following,
        ny,
    )
    assert overlap is not None
    assert overlap.strftime("%Y-%m-%d %H:%M") == "2024-11-03 01:30"
    assert overlap.utcoffset() == dt.timedelta(hours=-4)
    # 重叠时刻的“下一次”是次日，而非同日第二次 01:30
    nxt = next_occurrence("daily", "01:30", overlap, _all_days, _following, ny)
    assert nxt is not None
    assert nxt.strftime("%Y-%m-%d %H:%M") == "2024-11-04 01:30"


def test_live_daily_timer_delivered_early_reschedules_to_next_trading_day(
    monkeypatch: Any,
) -> None:
    """墙钟略早于目标、ctx.current_time 等于目标时，重排到下一交易日（不重复触发）."""
    from unittest.mock import MagicMock

    import akquant.strategy_scheduler as sched
    from akquant.akquant import StrategyContext

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.fired: List[str] = []

        def on_timer(self, payload: str) -> None:
            self.fired.append(payload)

    target = pd.Timestamp("2024-01-05 10:00", tz=TZ)  # 周五
    monkeypatch.setattr(
        sched, "_wall_clock", lambda tz: target - pd.Timedelta(milliseconds=5)
    )
    s = P()
    ctx = MagicMock(spec=StrategyContext)
    ctx.canceled_order_ids = []
    ctx.active_orders = []
    ctx.recent_trades = []
    ctx.current_time = int(target.value)
    ctx.is_trading_day.side_effect = _is_td
    ctx.next_trading_day.side_effect = _next_td
    s.ctx = ctx
    s._trading_days = []

    s._on_timer_event("__daily__|10:00|tick", ctx)

    assert s.fired == ["tick"]
    assert ctx.schedule.call_count == 1
    ts_ns, payload = ctx.schedule.call_args.args
    assert payload == "__daily__|10:00|tick"
    assert pd.Timestamp(ts_ns, tz="UTC").tz_convert(TZ) == pd.Timestamp(
        "2024-01-08 10:00", tz=TZ
    )
