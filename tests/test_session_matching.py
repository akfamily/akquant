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
    r = run_backtest(
        strategy=_BuyOnFirstBar,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert _fills(r) == [("01-02 13:01", 12.0)]


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
    assert "sessions=" in hits[0].getMessage()


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
