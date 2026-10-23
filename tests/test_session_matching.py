"""交易时段重构阶段 3：真实时段与集合竞价撮合的验收测试."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest
from akquant.backtest.engine import _freq_period_ns
from akquant.data import ParquetDataCatalog

TZ = "Asia/Shanghai"


def _ns(text: str) -> int:
    return int(pd.Timestamp(text, tz=TZ).value)


def _minute_frame(
    times: List[str], prices: List[float], symbol: str = "600000.SH"
) -> pd.DataFrame:
    idx = pd.to_datetime(times).tz_localize(TZ)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 1e6,
        }
    )


class _SessionProbe(Strategy):
    """记录每根 bar 的本地时刻与 ctx.session、interval_start."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: List[Tuple[str, str, Any]] = []

    def on_bar(self, bar: Bar) -> None:
        assert self.ctx is not None
        local = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ)
        session = str(self.ctx.session).split(".")[-1]
        self.seen.append((local.strftime("%m-%d %H:%M"), session, bar.interval_start))


def _probe(result: Any) -> _SessionProbe:
    strategy = result.strategy
    assert isinstance(strategy, _SessionProbe)
    return strategy


def test_date_only_bars_get_one_day_interval() -> None:
    """只有日期的日线改写到收盘后, 区间起点为改写后时间戳前 24h."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"]).tz_localize(TZ)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "600000.SH",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    r = run_backtest(
        strategy=_SessionProbe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    seen = _probe(r).seen
    assert [s[0] for s in seen] == ["01-02 15:00", "01-03 15:00"]
    day = 86_400_000_000_000
    assert seen[0][2] == _ns("2024-01-02 15:00") - day


def test_freq_declares_bar_interval() -> None:
    """Bar 数据传 freq: 区间起点 = 时间戳 - 周期, self.freq 照常注入."""
    df = _minute_frame(["2024-01-02 14:00", "2024-01-02 15:00"], [10.0, 10.0])
    r = run_backtest(
        strategy=_SessionProbe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        freq="1h",
    )
    assert _probe(r).seen[1][2] == _ns("2024-01-02 14:00")
    assert _probe(r).freq == "1h"


def test_bar_list_interval_start_is_preserved() -> None:
    """Bar 列表自带的 interval_start 原样保留."""
    bars = [
        Bar(
            timestamp=_ns("2024-01-02 10:00"),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1.0,
            symbol="600000.SH",
            interval_start=_ns("2024-01-02 09:59"),
        )
    ]
    r = run_backtest(
        strategy=_SessionProbe,
        data=bars,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert _probe(r).seen[0][2] == _ns("2024-01-02 09:59")


_DAY_NS = 86_400_000_000_000


@pytest.mark.parametrize("freq", ["1d", "D", "1D", "day", " 2 Day "])
def test_bar_freq_parser_accepts_day_forms(freq: str) -> None:
    """Bar 周期可以是日：Nd / Nday，大小写不敏感，省略数字为 1."""
    days = 2 if "2" in freq else 1
    assert _freq_period_ns(freq) == days * _DAY_NS


def test_bar_freq_parser_minutes_hours_and_error() -> None:
    """分钟、小时照旧；无法解析时说明 bar 周期的写法，而不是 tick 聚合."""
    assert _freq_period_ns("5min") == 5 * 60 * 10**9
    assert _freq_period_ns("1H") == 3600 * 10**9
    for bad in ["30s", "1w", "0d", "abc"]:
        with pytest.raises(ValueError, match="bar 周期") as info:
            _freq_period_ns(bad)
        assert "tick" not in str(info.value)


@pytest.mark.parametrize("freq", ["1d", "D"])
def test_freq_day_on_timestamped_daily_bars(freq: str) -> None:
    """带时刻（15:00）的日线传 freq='1d'/'D'：区间起点 = 时间戳 - 24h."""
    df = _minute_frame(["2024-01-02 15:00", "2024-01-03 15:00"], [10.0, 10.0])
    r = run_backtest(
        strategy=_SessionProbe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        freq=freq,
    )
    seen = _probe(r).seen
    assert [s[0] for s in seen] == ["01-02 15:00", "01-03 15:00"]
    assert seen[1][2] == _ns("2024-01-03 15:00") - _DAY_NS


def test_freq_applies_to_catalog_input(tmp_path: Path) -> None:
    """data=None 从 catalog 加载时 freq 同样填区间起点."""
    symbol = "600000.SH"
    catalog_root = tmp_path / "catalog"
    frame = _minute_frame(["2024-01-02 14:00", "2024-01-02 15:00"], [10.0, 10.0])
    ParquetDataCatalog(root_path=str(catalog_root)).write(
        symbol, frame.set_index("timestamp")
    )
    r = run_backtest(
        strategy=_SessionProbe,
        symbols=[symbol],
        catalog_path=str(catalog_root),
        initial_cash=1e5,
        show_progress=False,
        freq="1h",
    )
    seen = _probe(r).seen
    assert [s[0] for s in seen] == ["01-02 14:00", "01-02 15:00"]
    assert seen[1][2] == _ns("2024-01-02 14:00")


def test_ctx_session_reports_real_sessions_for_minute_bars() -> None:
    """分钟线看到真实时段：14:58、15:00 两根落在收盘集合竞价."""
    df = _minute_frame(
        [
            "2024-01-02 09:31",
            "2024-01-02 11:30",
            "2024-01-02 13:01",
            "2024-01-02 14:58",
            "2024-01-02 15:00",
        ],
        [10.0] * 5,
    )
    r = run_backtest(
        strategy=_SessionProbe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert [s[1] for s in _probe(r).seen] == [
        "Continuous",
        "Continuous",
        "Continuous",
        "ClosingAuction",
        "ClosingAuction",
    ]


def test_daily_bars_report_continuous() -> None:
    """只有日期的日线区间覆盖整个交易日，按连续竞价."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"]).tz_localize(TZ)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "600000.SH",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    r = run_backtest(
        strategy=_SessionProbe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert [s[1] for s in _probe(r).seen] == ["Continuous", "Continuous"]


class _BuyOnFirstBar(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.done = False

    def on_bar(self, bar: Bar) -> None:
        if not self.done:
            self.done = True
            self.buy(bar.symbol, 100)


def _fills(result: Any) -> List[Tuple[str, float]]:
    orders = result.orders_df
    filled = orders[orders["filled_quantity"] > 0]
    return [
        (pd.Timestamp(t).tz_convert(TZ).strftime("%m-%d %H:%M"), float(p))
        for t, p in zip(filled["updated_at"], filled["avg_price"])
    ]


def test_order_queued_through_lunch_break_fills_at_afternoon_open() -> None:
    """11:30 下的 NextOpen 单遇到 12:00 的（脏）午休 bar 不成交，13:01 按开盘价成交."""
    df = _minute_frame(
        ["2024-01-02 11:30", "2024-01-02 12:00", "2024-01-02 13:01"],
        [10.0, 11.0, 12.0],
    )
    df.loc[2, "open"] = 11.5  # 13:01 开盘价与收盘价不同，区分按开盘还是收盘成交
    r = run_backtest(
        strategy=_BuyOnFirstBar,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert _fills(r) == [("01-02 13:01", 11.5)]


@pytest.mark.parametrize("tif", ["GTC", "Day"])
def test_day_order_not_expired_by_closed_session_event(tif: str) -> None:
    """闭市时段的行情不再让 Day 单立即过期：20:00 的（脏）bar 后，次日开盘成交."""
    df = _minute_frame(
        ["2024-01-02 14:30", "2024-01-02 20:00", "2024-01-03 09:31"],
        [10.0, 11.0, 12.0],
    )

    class Probe(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.done = False

        def on_bar(self, bar: Bar) -> None:
            if not self.done:
                self.done = True
                self.buy(bar.symbol, 100, time_in_force=getattr(aq.TimeInForce, tif))

    r = run_backtest(
        strategy=Probe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert _fills(r) == [("01-03 09:31", 12.0)]


def test_closed_session_data_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """期货默认日盘时段表下出现夜盘行情：每个品种告警一次，提示传 sessions=."""
    df = _minute_frame(
        ["2024-01-02 21:01", "2024-01-02 21:02", "2024-01-03 09:01"],
        [500.0] * 3,
        symbol="au2406",
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
    with caplog.at_level("WARNING", logger="akquant"):
        run_backtest(strategy=_SessionProbe, data=df, config=cfg, show_progress=False)
    hits = [
        r
        for r in caplog.records
        if "闭市时段" in r.getMessage() and "au2406" in r.getMessage()
    ]
    assert len(hits) == 1
    message = hits[0].getMessage()
    assert "sessions=" in message
    # 时区口径：DataFrame 无时区时间按 timezone= 解释；Bar/整数时间戳是 UTC 纳秒。
    assert "按 timezone= 指定的时区解释（默认 Asia/Shanghai）" in message
    assert "UTC 纳秒" in message


class _AfterTradingProbe(Strategy):
    """记录 on_after_trading 回调的交易日与触发时所处 bar."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: List[Tuple[str, str]] = []
        self.last_bar = ""

    def on_bar(self, bar: Bar) -> None:
        local = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ)
        self.last_bar = local.strftime("%m-%d %H:%M")

    def on_after_trading(self, trading_date: Any, timestamp: int) -> None:
        self.calls.append((str(trading_date), self.last_bar))


def test_lazy_after_trading_not_triggered_by_closing_auction_bar() -> None:
    """收盘竞价的 14:58、15:00 bar 不提前触发惰性日终钩子.

    与阶段 2（强制连续竞价）一致：01-02 的回调由收盘定时器在 15:00 那根 bar
    之后触发，01-03 的回调在回测结束时补发；不会在 14:58 收盘竞价 bar 上提前触发。
    """
    df = _minute_frame(
        [
            "2024-01-02 14:30",
            "2024-01-02 14:58",
            "2024-01-02 15:00",
            "2024-01-03 09:31",
        ],
        [10.0] * 4,
    )
    r = run_backtest(
        strategy=_AfterTradingProbe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    strategy = r.strategy
    assert isinstance(strategy, _AfterTradingProbe)
    assert strategy.calls == [
        ("2024-01-02", "01-02 15:00"),
        ("2024-01-03", "01-03 09:31"),
    ]


class _TimerBuyCurrentClose(Strategy):
    """在给定本地时刻的定时器里下一笔买单（成交模式由 run_backtest 给 CurrentClose）."""

    at = "2024-01-02 12:00"

    def __init__(self) -> None:
        super().__init__()
        self.done = False

    def on_start(self) -> None:
        self.schedule(pd.Timestamp(self.at, tz=TZ), "buy")

    def on_bar(self, bar: Bar) -> None:
        pass

    def on_timer(self, payload: str) -> None:
        if payload == "buy" and not self.done:
            self.done = True
            self.buy("600000.SH", 100)


@pytest.mark.parametrize("at", ["2024-01-02 12:00", "2024-01-02 20:00"])
def test_timer_after_slice_resolves_session_at_timer_instant(at: str) -> None:
    """午休/闭市的用户定时器下的 CurrentClose 单不沿用上一根 bar 的时段，当日不成交."""
    df = _minute_frame(
        ["2024-01-02 11:00", "2024-01-02 11:30", "2024-01-03 09:31"],
        [10.0, 11.0, 12.0],
    )
    df.loc[2, "open"] = 11.8

    class Probe(_TimerBuyCurrentClose):
        pass

    Probe.at = at
    r = run_backtest(
        strategy=Probe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        fill_policy=aq.CurrentClose(),
    )
    fills = _fills(r)
    assert all(not t.startswith("01-02") for t, _ in fills), fills


def test_day_order_placed_at_close_belongs_to_next_trading_day() -> None:
    """15:00（收盘竞价结束）bar 上的 Day 单属于下一交易日：当日不过期，次日成交."""
    df = _minute_frame(
        ["2024-01-02 14:30", "2024-01-02 15:00", "2024-01-03 09:31"],
        [10.0, 11.0, 12.0],
    )

    class Probe(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.done = False

        def on_bar(self, bar: Bar) -> None:
            local = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ)
            if not self.done and local.strftime("%H:%M") == "15:00":
                self.done = True
                self.buy(bar.symbol, 100, time_in_force=aq.TimeInForce.Day)

    r = run_backtest(
        strategy=Probe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        fill_policy=aq.CurrentClose(),
    )
    orders = r.orders_df
    assert list(orders["status"].astype(str).str.lower()) == ["filled"]
    assert _fills(r) == [("01-03 09:31", 12.0)]


def _fixed_time_daily(symbol: str) -> pd.DataFrame:
    return _minute_frame(
        ["2024-01-02 15:00", "2024-01-03 15:00", "2024-01-04 15:00"],
        [10.0, 10.0, 10.0],
        symbol=symbol,
    )


def _fixed_time_hits(caplog: pytest.LogCaptureFixture) -> List[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name.startswith("akquant") and 'freq="1d"' in r.getMessage()
    ]


def test_fixed_time_stock_daily_without_freq_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """打 15:00 戳的股票日线未传 freq：告警一次，提示 freq="1d"."""
    with caplog.at_level("WARNING", logger="akquant"):
        run_backtest(
            strategy=_SessionProbe,
            data=_fixed_time_daily("600000.SH"),
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
        )
    hits = _fixed_time_hits(caplog)
    assert len(hits) == 1, hits
    assert "15:00" in hits[0] and "600000.SH" in hits[0]


def test_fixed_time_stock_daily_with_freq_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """传了 freq="1d" 不告警."""
    with caplog.at_level("WARNING", logger="akquant"):
        run_backtest(
            strategy=_SessionProbe,
            data=_fixed_time_daily("600000.SH"),
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
            freq="1d",
        )
    assert _fixed_time_hits(caplog) == []


def test_fixed_time_futures_daily_at_close_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """期货日盘 15:00 收盘前 1ns 仍是连续竞价，不告警."""
    cfg = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=1e6),
        instruments_config=[
            aq.InstrumentConfig(
                symbol="rb2405",
                asset_type="FUTURES",
                multiplier=10.0,
                margin_ratio=0.1,
            )
        ],
    )
    with caplog.at_level("WARNING", logger="akquant"):
        run_backtest(
            strategy=_SessionProbe,
            data=_fixed_time_daily("rb2405"),
            config=cfg,
            show_progress=False,
        )
    assert _fixed_time_hits(caplog) == []


def test_daily_current_close_day_order_expires_at_same_day_close() -> None:
    """日线（整日区间）上未成交的 CurrentClose 限价 Day 单：当日收盘过期（同阶段 2）."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"]).tz_localize(TZ)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "600000.SH",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )

    class Probe(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.done = False

        def on_bar(self, bar: Bar) -> None:
            if not self.done:
                self.done = True
                self.buy(bar.symbol, 100, price=5.0, time_in_force=aq.TimeInForce.Day)

    r = run_backtest(
        strategy=Probe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        fill_policy=aq.CurrentClose(),
    )
    orders = r.orders_df
    assert list(orders["status"].astype(str).str.lower()) == ["expired"]
    updated = pd.Timestamp(orders["updated_at"].iloc[0]).tz_convert(TZ)
    assert updated.strftime("%m-%d %H:%M") == "01-02 15:00"


def _tick(ts: str, price: float, symbol: str = "600000.SH", volume: float = 1e5) -> Any:
    return aq.Tick(timestamp=_ns(ts), price=price, volume=volume, symbol=symbol)


class _TickBuyAt(Strategy):
    """在指定本地时刻的 tick 上下单（市价或限价）."""

    at = "09:20:00"
    limit: Any = None

    def __init__(self) -> None:
        super().__init__()
        self.done = False

    def on_tick(self, tick: Any) -> None:
        t = pd.Timestamp(tick.timestamp, tz="UTC").tz_convert(TZ).strftime("%H:%M:%S")
        if not self.done and t == self.at:
            self.done = True
            self.buy(tick.symbol, 100, price=self.limit)


def _opening_auction_ticks() -> List[Any]:
    return [
        _tick("2024-01-02 09:20:00", 9.8),
        _tick("2024-01-02 09:25:00", 10.0),
        _tick("2024-01-02 09:30:03", 10.5),
        _tick("2024-01-02 09:31:00", 10.6),
    ]


def test_opening_auction_uncross_uses_auction_tick_price() -> None:
    """09:20 下市价单, 09:25 竞价 10.0, 09:30:03 首笔连续 10.5 → 按 10.0 成交."""
    r = run_backtest(
        strategy=_TickBuyAt,
        data=_opening_auction_ticks(),
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert [(t, round(p, 4)) for t, p in _fills(r)] == [("01-02 09:30", 10.0)]


def test_opening_auction_limit_not_met_goes_to_continuous() -> None:
    """限价 9.9 低于竞价价 10.0: 竞价不成交, 转连续竞价(10.5、10.6 也不成交)."""

    class Limit(_TickBuyAt):
        limit = 9.9

    r = run_backtest(
        strategy=Limit,
        data=_opening_auction_ticks(),
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert _fills(r) == []


class _BuyOnce(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.done = False

    def on_bar(self, bar: Bar) -> None:
        if not self.done:
            self.done = True
            self.buy(bar.symbol, 100)


def test_closing_auction_on_minute_bars_fills_at_close() -> None:
    """14:57 下的 NextOpen 市价单: 14:58 收盘竞价 bar 不成交, 15:00 按收盘价成交."""
    df = _minute_frame(
        [
            "2024-01-02 14:57",
            "2024-01-02 14:58",
            "2024-01-02 14:59",
            "2024-01-02 15:00",
        ],
        [10.0, 10.1, 10.2, 10.3],
    )
    # 15:00 bar 开盘价与收盘价不同, 区分按开盘还是按收盘成交
    df.loc[3, "open"] = 10.25
    r = run_backtest(
        strategy=_BuyOnce,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert _fills(r) == [("01-02 15:00", 10.3)]


def test_order_placed_on_closing_auction_end_bar_misses_todays_auction() -> None:
    """在 15:00(收盘竞价结束)bar 上下的单不参加当日收盘竞价, 属于下一交易日."""
    df = _minute_frame(["2024-01-02 14:59", "2024-01-02 15:00"], [10.0, 10.1])

    class Probe(Strategy):
        def on_bar(self, bar: Bar) -> None:
            t = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ).strftime("%H:%M")
            if t == "15:00":
                self.buy(bar.symbol, 100, fill_mode=aq.CurrentClose())

    r = run_backtest(
        strategy=Probe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    # 收盘时刻下的单属于下一交易日(Ruling S3-T2-6): 当日不成交、也不在 01-02 收盘过期
    assert _fills(r) == []
    assert not str(r.orders_df.iloc[0]["status"]).lower().endswith("expired")


def test_closing_auction_settles_with_last_auction_price_when_close_bar_missing() -> (
    None
):
    """数据缺 15:00 bar(14:59 后直接收盘): 日终结算前按最后一个收盘竞价事件价格成交."""
    df = _minute_frame(
        [
            "2024-01-02 14:57",
            "2024-01-02 14:58",
            "2024-01-02 14:59",
            "2024-01-03 09:31",
        ],
        [10.0, 10.1, 10.2, 11.0],
    )
    r = run_backtest(
        strategy=_BuyOnce,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert _fills(r) == [("01-02 15:00", 10.2)]


def test_daily_bar_fills_unchanged() -> None:
    """日线: NextOpen 在次日开盘价、CurrentClose 在当日收盘价成交(与阶段 2 一致)."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"]).tz_localize(TZ)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "600000.SH",
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "volume": 1e6,
        }
    )

    class Probe(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def on_bar(self, bar: Bar) -> None:
            self.n += 1
            if self.n == 1:
                self.buy(bar.symbol, 100)
                self.buy(bar.symbol, 100, fill_mode=aq.CurrentClose())

    r = run_backtest(
        strategy=Probe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert sorted(_fills(r)) == [("01-02 15:00", 10.2), ("01-03 15:00", 11.0)]


def test_closing_auction_on_ticks_ends_at_close_tick() -> None:
    """tick: 15:00:00 那笔结束收盘竞价按其价成交; 缺它时日终按 14:59:59 的价成交."""

    class Probe(_TickBuyAt):
        at = "14:56:00"

    ticks = [
        _tick("2024-01-02 14:56:00", 9.8),
        _tick("2024-01-02 14:58:00", 10.0),
        _tick("2024-01-02 14:59:59", 10.1),
        _tick("2024-01-02 15:00:00", 10.2),
        _tick("2024-01-03 09:31:00", 11.0),
    ]
    kw: Any = dict(symbols=["600000.SH"], initial_cash=1e6, show_progress=False)
    r = run_backtest(strategy=Probe, data=ticks, **kw)
    assert [(t, round(p, 4)) for t, p in _fills(r)] == [("01-02 15:00", 10.2)]
    r = run_backtest(strategy=Probe, data=ticks[:3] + ticks[4:], **kw)
    assert [(t, round(p, 4)) for t, p in _fills(r)] == [("01-02 15:00", 10.1)]


def _bars(
    times: List[str],
    closes: List[float],
    opens: Any = None,
    volume: Any = 1e6,
    symbol: str = "600000.SH",
) -> pd.DataFrame:
    df = _minute_frame(times, closes, symbol=symbol)
    if opens is not None:
        df["open"] = opens
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["volume"] = volume
    return df


def test_opening_uncross_and_continuous_share_cash_projection() -> None:
    """统一撮合与本事件的常规撮合共享资金推演: 第二笔资金不足被拒, 不透支."""
    df = _bars(
        [
            "2024-01-02 09:20",
            "2024-01-02 09:25",
            "2024-01-02 09:31",
            "2024-01-02 09:32",
        ],
        [9.8, 10.0, 10.5, 10.6],
        opens=[9.8, 10.0, 9.9, 10.5],
    )
    df.loc[2, "low"] = 9.85

    class Two(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.done = False

        def on_bar(self, bar: Bar) -> None:
            if not self.done:
                self.done = True
                self.buy(bar.symbol, 100)
                self.buy(bar.symbol, 100, price=9.95)

    r = run_backtest(
        strategy=Two,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1500,
        show_progress=False,
    )
    orders = r.orders_df
    assert list(orders["status"].astype(str).str.lower()) == ["filled", "rejected"]
    assert "Insufficient margin" in str(orders["reject_reason"].iloc[1])
    assert _fills(r) == [("01-02 09:31", 10.0)]


@pytest.mark.parametrize("with_close_bar", [False, True])
def test_closing_uncross_across_symbols_shares_cash_projection(
    with_close_bar: bool,
) -> None:
    """两个品种同时收盘统一撮合(含缺 15:00 bar 的日终收尾): 资金只够一笔."""
    times = ["2024-01-02 14:58", "2024-01-02 14:59", "2024-01-03 09:31"]
    if with_close_bar:
        times = times[:2] + ["2024-01-02 15:00"] + times[2:]
    prices = [10.0] * (len(times) - 1) + [11.0]
    df = (
        pd.concat([_bars(times, prices), _bars(times, prices, symbol="600001.SH")])
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )

    class Two(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.done = False

        def on_bar(self, bar: Bar) -> None:
            if not self.done:
                self.done = True
                self.buy("600000.SH", 100)
                self.buy("600001.SH", 100)

    r = run_backtest(
        strategy=Two,
        data=df,
        symbols=["600000.SH", "600001.SH"],
        initial_cash=1500,
        show_progress=False,
    )
    orders = r.orders_df
    assert sorted(orders["status"].astype(str).str.lower()) == ["filled", "rejected"]
    assert _fills(r) == [("01-02 15:00", 10.0)]


def test_zero_volume_last_auction_tick_still_uncrosses() -> None:
    """竞价期末笔零量快照只更新价格, 统一撮合照常按 10.0 成交."""
    ticks = [
        _tick("2024-01-02 09:20:00", 9.8),
        _tick("2024-01-02 09:24:00", 9.9),
        # 开盘竞价期末笔零量快照(09:24:58)与静默期零量快照(09:29:57)
        _tick("2024-01-02 09:24:58", 10.0, volume=0),
        _tick("2024-01-02 09:29:57", 10.0, volume=0),
        _tick("2024-01-02 09:30:03", 10.5),
        _tick("2024-01-02 09:31:00", 10.6),
    ]
    r = run_backtest(
        strategy=_TickBuyAt,
        data=ticks,
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert [(t, round(p, 4)) for t, p in _fills(r)] == [("01-02 09:30", 10.0)]


def test_tick_auction_price_not_limited_by_auction_volume() -> None:
    """竞价价来自 tick 时合成量不受 volume_limit_pct 约束, 全部按竞价价成交."""
    ticks = [
        _tick("2024-01-02 09:20:00", 9.8),
        _tick("2024-01-02 09:25:00", 10.0, volume=300),
        _tick("2024-01-02 09:30:03", 10.5),
        _tick("2024-01-02 09:31:00", 10.6),
    ]
    r = run_backtest(
        strategy=_TickBuyAt,
        data=ticks,
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
        volume_limit_pct=0.1,
    )
    assert [(t, round(p, 4)) for t, p in _fills(r)] == [("01-02 09:30", 10.0)]


def test_order_placed_on_auction_price_tick_goes_to_continuous() -> None:
    """09:25:00 竞价价 tick 上下的单: 价格已知后下单, 不参加统一撮合, 按连续竞价成交."""

    class At0925(_TickBuyAt):
        at = "09:25:00"

    r = run_backtest(
        strategy=At0925,
        data=_opening_auction_ticks(),
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert [(t, round(p, 4)) for t, p in _fills(r)] == [("01-02 09:30", 10.5)]


class _BarBuyAt(Strategy):
    at = "09:25"

    def __init__(self) -> None:
        super().__init__()
        self.done = False

    def on_bar(self, bar: Bar) -> None:
        t = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ).strftime("%H:%M")
        if not self.done and t == self.at:
            self.done = True
            self.buy(bar.symbol, 100)


def test_order_on_opening_auction_bar_skips_uncross_despite_auction_hold_bar() -> None:
    """09:25 竞价 bar 上下的单不参加统一撮合; 其后 09:30 静默期 bar 不形成新竞价价."""
    df = _bars(
        [
            "2024-01-02 09:20",
            "2024-01-02 09:25",
            "2024-01-02 09:30",
            "2024-01-02 09:31",
        ],
        [9.8, 10.0, 10.0, 10.5],
        opens=[9.8, 10.0, 10.0, 10.4],
    )
    r = run_backtest(
        strategy=_BarBuyAt,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert _fills(r) == [("01-02 09:31", 10.4)]


def test_zero_volume_auction_hold_bar_is_not_suspension() -> None:
    """09:30 静默期 bar 成交量为 0 不是停牌: 09:20 下的单不被拒, 照常统一撮合."""

    class At0920(_BarBuyAt):
        at = "09:20"

    df = _bars(
        [
            "2024-01-02 09:20",
            "2024-01-02 09:25",
            "2024-01-02 09:30",
            "2024-01-02 09:31",
        ],
        [9.8, 10.0, 10.0, 10.5],
        opens=[9.8, 10.0, 10.0, 10.4],
        volume=[1e6, 1e6, 0, 1e6],
    )
    r = run_backtest(
        strategy=At0920,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert list(r.orders_df["status"].astype(str).str.lower()) == ["filled"]
    assert _fills(r) == [("01-02 09:31", 10.0)]


def test_current_close_order_on_close_tick_misses_todays_closing_auction() -> None:
    """全局 CurrentClose: 15:00 收盘 tick 上下的单不参加当日收盘统一撮合."""

    class At1500(_TickBuyAt):
        at = "15:00:00"

    ticks = [
        _tick("2024-01-02 14:58:00", 10.0),
        _tick("2024-01-02 15:00:00", 10.2),
        _tick("2024-01-03 09:31:00", 11.0),
    ]
    r = run_backtest(
        strategy=At1500,
        data=ticks,
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
        fill_policy=aq.CurrentClose(),
    )
    assert _fills(r) == [("01-03 09:31", 11.0)]


def test_zero_volume_last_auction_bar_keeps_volume_and_uncrosses() -> None:
    """分钟线: 开盘竞价末根 bar 零量只更新价格(沿用前一根的量), 照常按 10.0 统一撮合."""

    class At0920(_BarBuyAt):
        at = "09:20"

    df = _bars(
        ["2024-01-02 09:20", "2024-01-02 09:25", "2024-01-02 09:31"],
        [9.8, 10.0, 10.5],
        opens=[9.8, 10.0, 10.4],
        volume=[1e6, 0, 1e6],
    )
    r = run_backtest(
        strategy=At0920,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert _fills(r) == [("01-02 09:31", 10.0)]


def test_tick_auction_with_volume_limit_above_one_does_not_overflow() -> None:
    """volume_limit_pct > 1: tick 竞价统一撮合不做量检查, 不溢出, 按竞价价成交."""
    r = run_backtest(
        strategy=_TickBuyAt,
        data=_opening_auction_ticks(),
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
        volume_limit_pct=1.5,
    )
    assert [(t, round(p, 4)) for t, p in _fills(r)] == [("01-02 09:30", 10.0)]


def test_all_zero_volume_auction_bars_uncross_without_volume_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """分钟线竞价期 bar 成交量全为 0: 照常按竞价价统一撮合(不受量限制), 告警一次."""

    class At0920(_BarBuyAt):
        at = "09:20"

    df = _bars(
        ["2024-01-02 09:20", "2024-01-02 09:25", "2024-01-02 09:31"],
        [9.8, 10.0, 10.5],
        opens=[9.8, 10.0, 10.4],
        volume=[0, 0, 1e6],
    )
    with caplog.at_level("WARNING", logger="akquant"):
        r = run_backtest(
            strategy=At0920,
            data=df,
            symbols=["600000.SH"],
            initial_cash=1e6,
            show_progress=False,
            volume_limit_pct=0.1,
        )
    assert _fills(r) == [("01-02 09:31", 10.0)]
    assert list(r.orders_df["filled_quantity"]) == [100]
    hits = [m for m in caplog.messages if "竞价期行情成交量为 0" in m]
    assert len(hits) == 1 and "600000.SH" in hits[0]


class _CrossSectionSessionProbe(Strategy):
    """记录 on_bar 与 on_cross_section 看到的 ctx.session，并在截面里下一笔单."""

    def __init__(self) -> None:
        super().__init__()
        self.bar_sessions: List[str] = []
        self.cross_sessions: List[str] = []
        self.done = False

    def on_bar(self, bar: Bar) -> None:
        assert self.ctx is not None
        self.bar_sessions.append(str(self.ctx.session).split(".")[-1])

    def on_cross_section(self, trading_date: Any, timestamp: int) -> None:
        assert self.ctx is not None
        self.cross_sessions.append(str(self.ctx.session).split(".")[-1])
        if not self.done:
            self.done = True
            self.buy("600000.SH", 100)


def _cross_probe(result: Any) -> _CrossSectionSessionProbe:
    strategy = result.strategy
    assert isinstance(strategy, _CrossSectionSessionProbe)
    return strategy


def test_daily_cross_section_session_matches_slice() -> None:
    """只有日期的日线：截面定时器属于本时间片，ctx.session 与 on_bar 一致为连续竞价."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"]).tz_localize(TZ)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "600000.SH",
            "open": [10.0, 11.0],
            "high": [10.0, 11.0],
            "low": [10.0, 11.0],
            "close": [10.0, 11.0],
            "volume": 1e6,
        }
    )
    r = run_backtest(
        strategy=_CrossSectionSessionProbe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    probe = _cross_probe(r)
    assert probe.bar_sessions == ["Continuous", "Continuous"]
    assert probe.cross_sessions == ["Continuous", "Continuous"]
    # 截面里下的 NextOpen 单按连续竞价撮合：次日开盘价成交。
    assert _fills(r) == [("01-03 15:00", 11.0)]


def test_hourly_cross_section_session_matches_on_bar() -> None:
    """freq='1h' 的 15:00 bar：截面看到的时段与 on_bar 相同（按 bar 区间求得）."""
    # 每个交易日只有 15:00 一根，截面定时器挂在这根 bar 上。
    df = _minute_frame(["2024-01-02 15:00", "2024-01-03 15:00"], [10.0, 10.0])
    r = run_backtest(
        strategy=_CrossSectionSessionProbe,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        freq="1h",
    )
    probe = _cross_probe(r)
    assert probe.bar_sessions == ["Continuous", "Continuous"]
    assert probe.cross_sessions == probe.bar_sessions


class _OutOfCalendarTimerBuy(Strategy):
    """在交易日历之外的时刻用定时器下单，记录 ctx.session."""

    def __init__(self) -> None:
        super().__init__()
        self.sessions: List[str] = []

    def on_start(self) -> None:
        self.schedule(pd.Timestamp("2024-01-04 10:00", tz=TZ), "buy")

    def on_bar(self, bar: Bar) -> None:
        pass

    def on_timer(self, payload: str) -> None:
        assert self.ctx is not None
        self.sessions.append(str(self.ctx.session).split(".")[-1])
        self.buy("600000.SH", 100)


def test_out_of_calendar_timer_session_matches_matching() -> None:
    """定时器时刻超出交易日历：ctx.session 与撮合同样回退为连续竞价."""
    df = _minute_frame(["2024-01-02 10:00", "2024-01-03 10:00"], [10.0, 11.0])
    r = run_backtest(
        strategy=_OutOfCalendarTimerBuy,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        trading_calendar=["2024-01-02", "2024-01-03"],
        fill_policy=aq.CurrentClose(),
    )
    strategy = r.strategy
    assert isinstance(strategy, _OutOfCalendarTimerBuy)
    assert strategy.sessions == ["Continuous"]
    # 撮合按连续竞价：定时器时刻按最新价成交。
    assert _fills(r) == [("01-04 10:00", 11.0)]


def test_closed_session_warnings_capped_with_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """闭市时段告警全局最多 20 条，超出的在回测结束时汇总一条（总条数与品种数）."""
    symbols = [f"{600000 + i}.SH" for i in range(25)]
    df = pd.concat(
        [
            _minute_frame(
                ["2024-01-02 10:00", "2024-01-02 20:00", "2024-01-02 20:01"],
                [10.0] * 3,
                symbol=s,
            )
            for s in symbols
        ],
        ignore_index=True,
    )
    with caplog.at_level("WARNING", logger="akquant"):
        run_backtest(
            strategy=_SessionProbe,
            data=df,
            symbols=symbols,
            initial_cash=1e5,
            show_progress=False,
        )
    closed = [r.getMessage() for r in caplog.records if "闭市时段" in r.getMessage()]
    per_symbol = [m for m in closed if "期间不会撮合成交。请检查时间戳" in m]
    summaries = [m for m in closed if "汇总" in m]
    assert len(per_symbol) == 20
    assert len(summaries) == 1
    # 25 个品种各 2 条闭市行情
    assert "25 个标的" in summaries[0]
    assert "50 条行情" in summaries[0]
