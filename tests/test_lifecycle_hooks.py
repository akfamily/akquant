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
