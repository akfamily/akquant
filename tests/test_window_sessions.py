"""阶段 5：多周期窗口按交易日与交易块切分."""

from __future__ import annotations

import datetime as dt
from typing import Any, List, Tuple

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy, run_backtest

TZ = "Asia/Shanghai"


def _ns(text: str) -> int:
    return int(pd.Timestamp(text, tz=TZ).value)


def test_resolver_trading_blocks_and_block_at() -> None:
    """解析器按交易块返回区间，午休返回 None."""
    r = aq.SessionResolver(timezone=TZ)
    blocks = r.trading_blocks("600000.SH", dt.date(2024, 1, 2), aq.AssetType.Stock)
    assert blocks == [
        (_ns("2024-01-02 09:15"), _ns("2024-01-02 11:30")),
        (_ns("2024-01-02 13:00"), _ns("2024-01-02 15:00")),
    ]
    assert r.block_at("600000.SH", _ns("2024-01-02 12:00"), aq.AssetType.Stock) is None
    hit = r.block_at("600000.SH", _ns("2024-01-02 10:00"), aq.AssetType.Stock)
    assert hit is not None
    day, idx, start, end = hit
    assert (day, idx, start, end) == (
        dt.date(2024, 1, 2),
        0,
        _ns("2024-01-02 09:15"),
        _ns("2024-01-02 11:30"),
    )


def _minute(
    times: List[str], symbol: str = "600000.SH", prices: Any = None
) -> pd.DataFrame:
    idx = pd.to_datetime(times).tz_localize(TZ)
    n = len(idx)
    px = prices if prices is not None else [10.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": px,
            "high": px,
            "low": px,
            "close": px,
            "volume": 1.0,
        }
    )


class _WinLog(Strategy):
    freqs: Tuple[str, ...] = ("30min",)

    def __init__(self) -> None:
        super().__init__()
        self.windows: List[Tuple[str, str, Any, float, float, float]] = []
        for f in self.freqs:
            self.subscribe_bars(f)

    def on_bar(self, bar: Bar) -> None:
        pass

    def on_window_bar(self, bar: Bar) -> None:
        local = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ)
        start = (
            pd.Timestamp(bar.interval_start, tz="UTC")
            .tz_convert(TZ)
            .strftime("%m-%d %H:%M")
            if bar.interval_start is not None
            else None
        )
        self.windows.append(
            (
                str(bar.freq),
                local.strftime("%m-%d %H:%M"),
                start,
                float(bar.open),
                float(bar.close),
                float(bar.volume),
            )
        )


def test_intraday_window_splits_at_lunch_and_clamps_label() -> None:
    """A 股 1h：11:01～11:30 的窗口标签 11:30（不是 12:00）、在 11:30 即时闭合.

    13:01 开新窗口，起点为下午块起点 13:00。
    """

    class P(_WinLog):
        freqs = ("1h",)

    times = [
        "2024-01-02 11:00",
        "2024-01-02 11:01",
        "2024-01-02 11:30",
        "2024-01-02 13:01",
        "2024-01-02 14:00",
    ]
    r = run_backtest(
        strategy=P,
        data=_minute(times),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        freq="1min",
    )
    assert r.strategy is not None
    assert [(w[1], w[2], w[5]) for w in r.strategy.windows] == [
        ("01-02 11:00", "01-02 10:00", 1.0),
        ("01-02 11:30", "01-02 11:00", 2.0),
        ("01-02 14:00", "01-02 13:00", 2.0),
    ]


def _au_config() -> aq.BacktestConfig:
    return aq.BacktestConfig(
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


def test_daily_window_groups_night_session_into_next_trading_day() -> None:
    """夜盘期货 1d 窗口：周五 21:30 与周一 10:00 属同一交易日（周一）.

    标签为周一 15:00，区间起点为上一交易日（周五）15:00。
    """

    class P(_WinLog):
        freqs = ("1d",)

    df = _minute(
        [
            "2024-01-05 14:00",
            "2024-01-05 15:00",
            "2024-01-05 21:30",
            "2024-01-08 10:00",
            "2024-01-08 15:00",
        ],
        symbol="au2406",
    )
    r = run_backtest(
        strategy=P,
        data=df,
        config=_au_config(),
        show_progress=False,
        freq="1min",
        sessions={"au": "CN_FUTURES_NIGHT_0230"},
    )
    assert r.strategy is not None
    got = [(w[1], w[2], w[3], w[4]) for w in r.strategy.windows]
    assert got == [
        ("01-05 15:00", "01-04 15:00", 10.0, 11.0),
        ("01-08 15:00", "01-05 15:00", 12.0, 14.0),
    ]


def test_tick_synthesized_bars_close_window_immediately() -> None:
    """Tick 合成的终点戳 bar（比边界早 1ns）也能让 5min 窗口在边界即时闭合."""

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.order: List[str] = []
            self.subscribe_bars("5min")

        def on_bar(self, bar: Bar) -> None:
            self.order.append(
                "bar "
                + pd.Timestamp(bar.timestamp, tz="UTC")
                .tz_convert(TZ)
                .strftime("%H:%M:%S")
            )

        def on_window_bar(self, bar: Bar) -> None:
            self.order.append(
                "win "
                + pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ).strftime("%H:%M")
            )

    ticks = [
        aq.Tick(
            timestamp=_ns(f"2024-01-02 09:3{m}:30"),
            price=10.0,
            volume=1.0,
            symbol="600000.SH",
        )
        for m in range(1, 7)
    ]
    r = run_backtest(
        strategy=P,
        data=ticks,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        freq="1min",
    )
    assert r.strategy is not None
    i = r.strategy.order.index("win 09:35")
    # 09:34～09:35 合成 bar 之后立即闭合
    assert r.strategy.order[i - 1] == "bar 09:34:59"


def test_manual_session_windows_cross_midnight() -> None:
    """手写跨午夜时段 ("21:00","02:30") 合法.

    1d 窗口在最后一段（13:30–15:00）结束时闭合。
    """

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.closed_at: List[str] = []
            self.subscribe_bars(
                "1d",
                session_windows=[
                    ("21:00", "02:30"),
                    ("09:00", "10:15"),
                    ("10:30", "11:30"),
                    ("13:30", "15:00"),
                ],
            )

        def on_bar(self, bar: Bar) -> None:
            pass

        def on_window_bar(self, bar: Bar) -> None:
            assert self.ctx is not None
            self.closed_at.append(
                pd.Timestamp(self.ctx.current_time, tz="UTC")
                .tz_convert(TZ)
                .strftime("%m-%d %H:%M")
            )

    df = _minute(
        [
            "2024-01-05 21:30",
            "2024-01-08 10:00",
            "2024-01-08 15:00",
            "2024-01-08 21:30",
        ],
        symbol="au2406",
    )
    r = run_backtest(
        strategy=P,
        data=df,
        config=_au_config(),
        show_progress=False,
        freq="1min",
        sessions={"au": "CN_FUTURES_NIGHT_0230"},
    )
    assert r.strategy is not None
    assert r.strategy.closed_at[0] == "01-08 15:00"
