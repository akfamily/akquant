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
        self.window_ts: List[int] = []
        for f in self.freqs:
            self.subscribe_bars(f)

    def on_bar(self, bar: Bar) -> None:
        pass

    def on_window_bar(self, bar: Bar) -> None:
        self.window_ts.append(int(bar.timestamp))
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


def test_always_open_daily_label_stays_on_trading_day() -> None:
    """ALWAYS_OPEN（非中国时区）1d 窗口标签取收盘前 1ns，本地日期仍是交易日."""

    class P(_WinLog):
        freqs = ("1d",)

    ny = "America/New_York"
    idx = pd.to_datetime(
        ["2024-01-02 10:00", "2024-01-02 15:00", "2024-01-03 10:00"]
    ).tz_localize(ny)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAPL",
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": 1.0,
        }
    )
    r = run_backtest(
        strategy=P,
        data=df,
        symbols=["AAPL"],
        initial_cash=1e5,
        show_progress=False,
        freq="1min",
        timezone=ny,
    )
    assert r.strategy is not None
    labels = [w[1] for w in r.strategy.windows]
    assert len(labels) == 2
    first = r.strategy.windows[0]
    assert first[3:5] == (1.0, 2.0)
    # _WinLog 按上海时区格式化；这里按纽约时区重新取完整时刻
    got = [pd.Timestamp(b, tz="UTC").tz_convert(ny) for b in r.strategy.window_ts]
    assert [g.date() for g in got] == [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    assert all(
        (g.hour, g.minute, g.second, g.nanosecond, g.microsecond)
        == (23, 59, 59, 999, 999999)
        for g in got
    )


def test_bar_list_without_symbols_uses_stock_blocks() -> None:
    """Bar 列表不传 symbols：数据里的 A 股标的也按股票时段表切块（午休截断）."""

    class P(_WinLog):
        freqs = ("1h",)

    times = [
        "2024-01-02 09:31",
        "2024-01-02 10:00",
        "2024-01-02 10:01",
        "2024-01-02 11:00",
        "2024-01-02 11:30",
        "2024-01-02 13:01",
        "2024-01-02 14:00",
        "2024-01-02 14:01",
        "2024-01-02 15:00",
    ]
    bars = [Bar(_ns(t), 10.0, 10.0, 10.0, 10.0, 1.0, "600000.SH") for t in times]
    r = run_backtest(strategy=P, data=bars, initial_cash=1e5, show_progress=False)
    assert r.strategy is not None
    assert [w[1][-5:] for w in r.strategy.windows] == [
        "10:00",
        "11:00",
        "11:30",
        "14:00",
        "15:00",
    ]


def test_bar_generator_matches_engine_windows() -> None:
    """BarGenerator 与引擎 subscribe_bars 对同一分钟数据产出相同的 1h 窗口."""
    from akquant import BarGenerator

    times = [
        "2024-01-02 09:31",
        "2024-01-02 10:00",
        "2024-01-02 11:00",
        "2024-01-02 11:01",
        "2024-01-02 11:30",
        "2024-01-02 13:01",
        "2024-01-02 14:00",
    ]
    df = _minute(times)

    class Eng(_WinLog):
        freqs = ("1h",)

        def __init__(self) -> None:
            super().__init__()
            self.closed_at: List[str] = []

        def on_window_bar(self, bar: Bar) -> None:
            super().on_window_bar(bar)
            now = self.now
            assert now is not None
            self.closed_at.append(now.tz_convert(TZ).strftime("%m-%d %H:%M"))

    r = run_backtest(
        strategy=Eng,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        freq="1min",
    )
    assert r.strategy is not None
    engine_windows = [(w[1], w[3], w[4], w[5]) for w in r.strategy.windows]

    got: List[Tuple[str, float, float, float]] = []
    # 闭合时机: 窗口在喂入哪根基础 bar 时回调(收尾 flush 记为 "flush")
    feeding = ["flush"]
    closed_at: List[str] = []

    def cb(bar: Bar) -> None:
        closed_at.append(feeding[0])
        got.append(
            (
                pd.Timestamp(bar.timestamp, tz="UTC")
                .tz_convert(TZ)
                .strftime("%m-%d %H:%M"),
                float(bar.open),
                float(bar.close),
                float(bar.volume),
            )
        )

    gen = BarGenerator(
        cb,
        window=1,
        interval="hour",
        timezone=TZ,
        asset_type=aq.AssetType.Stock,
        base_interval_min=1,
    )
    for row in df.to_dict("records"):
        ts = pd.Timestamp(row["timestamp"])
        feeding[0] = ts.strftime("%m-%d %H:%M")
        gen.update_bar(
            Bar(
                timestamp=int(ts.value),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                symbol=str(row["symbol"]),
            )
        )
    feeding[0] = "flush"
    gen.flush()
    assert got == engine_windows
    # 即时闭合与引擎同一时机: 11:30 窗口在 11:30 那根到达时闭合, 不等 13:01
    assert closed_at == r.strategy.closed_at
    assert closed_at == ["01-02 10:00", "01-02 11:00", "01-02 11:30", "01-02 14:00"]


class _MemAdapter(aq.feed_adapter.BasePandasFeedAdapter):
    """内存 DataFrame 适配器（同 test_feed_adapter.py 的 InMemoryFeedAdapter）."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def load(self, request: Any) -> pd.DataFrame:
        out = self._frame
        return out[out["symbol"] == request.symbol] if "symbol" in out.columns else out


def test_replay_session_alignment_matches_engine_windows() -> None:
    """离线 replay(align='session') 与引擎 subscribe_bars 产出相同的 1h 窗口.

    含午休截断（11:30 标签）。
    """
    times = [
        "2024-01-02 09:31",
        "2024-01-02 10:00",
        "2024-01-02 11:00",
        "2024-01-02 11:01",
        "2024-01-02 11:30",
        "2024-01-02 13:01",
        "2024-01-02 14:00",
    ]
    df = _minute(times)

    class Eng(_WinLog):
        freqs = ("1h",)

    r = run_backtest(
        strategy=Eng,
        data=df,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        freq="1min",
    )
    assert r.strategy is not None
    engine_windows = [(w[1], w[3], w[4], w[5]) for w in r.strategy.windows]

    adapter = _MemAdapter(df.set_index("timestamp")).replay(
        "1h", align="session", emit_partial=True, asset_type=aq.AssetType.Stock
    )
    out = adapter.load(aq.feed_adapter.FeedSlice(symbol="600000.SH", timezone=TZ))
    got = [
        (i.tz_convert(TZ).strftime("%m-%d %H:%M"), float(o), float(c), float(v))
        for i, o, c, v in zip(out.index, out["open"], out["close"], out["volume"])
    ]
    assert got == engine_windows
    assert [g[0] for g in got] == [
        "01-02 10:00",
        "01-02 11:00",
        "01-02 11:30",
        "01-02 14:00",
    ]


def test_replay_day_alignment_uses_trading_day_for_night_session() -> None:
    """离线 replay(align='day') 按交易日切分：周五夜盘归下周一，标签为交易日收盘."""
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
    adapter = _MemAdapter(df.set_index("timestamp")).replay(
        "1D",
        align="day",
        emit_partial=True,
        sessions={"au": "CN_FUTURES_NIGHT_0230"},
        asset_type=aq.AssetType.Futures,
    )
    out = adapter.load(aq.feed_adapter.FeedSlice(symbol="au2406", timezone=TZ))
    assert [i.tz_convert(TZ).strftime("%m-%d %H:%M") for i in out.index] == [
        "01-05 15:00",
        "01-08 15:00",
    ]
    assert list(out["open"]) == [10.0, 12.0]


def test_replay_extra_agg_columns_follow_engine_windows() -> None:
    """额外的 agg 列按聚合器给每行分配的窗口分组（午休截断后的窗口同样适用）.

    11:30 那根同时挤出未满的 11:00 窗口并让 11:30 窗口即时闭合。
    """
    times = [
        "2024-01-02 09:31",
        "2024-01-02 10:00",
        "2024-01-02 10:30",
        "2024-01-02 11:30",
        "2024-01-02 13:01",
        "2024-01-02 14:00",
    ]
    df = _minute(times)
    df["amount"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    adapter = _MemAdapter(df.set_index("timestamp")).replay(
        "1h", align="session", emit_partial=True, agg={"amount": "sum"}
    )
    out = adapter.load(aq.feed_adapter.FeedSlice(symbol="600000.SH", timezone=TZ))
    assert list(out.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "symbol",
    ]
    assert [i.tz_convert(TZ).strftime("%H:%M") for i in out.index] == [
        "10:00",
        "11:00",
        "11:30",
        "14:00",
    ]
    assert out["amount"].tolist() == [3.0, 3.0, 4.0, 11.0]
    assert out["volume"].tolist() == [2.0, 1.0, 1.0, 2.0]


def test_replay_emit_partial_false_drops_tail_of_each_trading_day() -> None:
    """emit_partial=False 只丢每个交易日末尾数据未到达标签的窗口."""
    df = _minute(
        [
            "2024-01-02 09:45",
            "2024-01-02 10:00",
            "2024-01-02 10:10",
            "2024-01-03 09:45",
            "2024-01-03 10:00",
            "2024-01-03 10:10",
            "2024-01-03 13:01",
        ]
    )
    adapter = _MemAdapter(df.set_index("timestamp")).replay(
        "30min", align="session", emit_partial=False
    )
    out = adapter.load(aq.feed_adapter.FeedSlice(symbol="600000.SH", timezone=TZ))
    assert [i.tz_convert(TZ).strftime("%m-%d %H:%M") for i in out.index] == [
        "01-02 10:00",
        "01-03 10:00",
        "01-03 10:30",
    ]


def test_replay_manual_session_windows_cross_midnight() -> None:
    """手写时段终点早于起点表示跨午夜：午夜后的部分归前一天的段."""
    from akquant.feed_adapter import _session_window_partition_keys

    index = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-02 22:00", "2024-01-03 01:00", "2024-01-03 03:00"])
    )
    keys, keep = _session_window_partition_keys(index, TZ, [("21:00", "02:30")])
    assert keep == [True, True, False]
    assert [k.strftime("%m-%d %H:%M") for k in keys[:2]] == [
        "01-02 21:00",
        "01-02 21:00",
    ]


class _OhlcvLog(Strategy):
    """按品种记录窗口 (标签, OHLCV)."""

    win_freq = "30min"

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[Tuple[str, int, float, float, float, float, float]] = []
        self.subscribe_bars(self.win_freq)

    def on_bar(self, bar: Bar) -> None:
        pass

    def on_window_bar(self, bar: Bar) -> None:
        self.rows.append(
            (
                str(bar.symbol),
                int(bar.timestamp),
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(bar.volume),
            )
        )


def _replay_rows(
    out: pd.DataFrame, symbol: str
) -> List[Tuple[str, int, float, float, float, float, float]]:
    return [
        (symbol, int(i.value), float(o), float(h), float(lo), float(c), float(v))
        for i, o, h, lo, c, v in zip(
            out.index, out["open"], out["high"], out["low"], out["close"], out["volume"]
        )
    ]


def _block_minutes(symbol: str, days: List[str], asset_type: Any) -> pd.DataFrame:
    """按解析器的交易块生成整段分钟线（右标签），价格与成交量逐根不同."""
    r = aq.SessionResolver(sessions={"au": "CN_FUTURES_NIGHT_0230"}, timezone=TZ)
    stamps: List[pd.Timestamp] = []
    for d in days:
        for start, end in r.trading_blocks(
            symbol, dt.date.fromisoformat(d), asset_type
        ):
            stamps += list(
                pd.date_range(
                    pd.Timestamp(start + 60_000_000_000, tz="UTC"),
                    pd.Timestamp(end, tz="UTC"),
                    freq="min",
                )
            )
    idx = pd.DatetimeIndex(stamps).tz_convert(TZ)
    n = len(idx)
    base = [400.0 + (i * 7 % 13) * 0.5 for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": base,
            "high": [b + 1.0 for b in base],
            "low": [b - 1.0 for b in base],
            "close": [b + 0.5 for b in base],
            "volume": [float(i % 7 + 1) for i in range(n)],
        }
    )


def test_replay_night_futures_matches_engine_windows_bar_by_bar() -> None:
    """夜盘期货 30min / 1h：replay 与引擎逐根相同（标签与 OHLCV，含跨午夜与周末）."""
    df = _block_minutes(
        "au2406",
        ["2024-01-04", "2024-01-05", "2024-01-08"],
        aq.AssetType.Futures,
    )
    for freq in ("30min", "1h"):

        class Eng(_OhlcvLog):
            win_freq = freq

        r = run_backtest(
            strategy=Eng,
            data=df,
            symbols=["au2406"],
            config=_au_config(),
            show_progress=False,
            freq="1min",
            sessions={"au": "CN_FUTURES_NIGHT_0230"},
        )
        assert r.strategy is not None
        out = (
            _MemAdapter(df.set_index("timestamp"))
            .replay(
                freq,
                align="session",
                emit_partial=True,
                sessions={"au": "CN_FUTURES_NIGHT_0230"},
                asset_type=aq.AssetType.Futures,
            )
            .load(aq.feed_adapter.FeedSlice(symbol="au2406", timezone=TZ))
        )
        assert len(r.strategy.rows) >= 30
        assert _replay_rows(out, "au2406") == r.strategy.rows


def test_replay_multi_symbol_frame_matches_engine_windows() -> None:
    """带 symbol 列的多品种 frame：按品种各自切分，资产类型可按品种映射.

    au2406 不传 sessions，靠 asset_type 取期货默认时段表（10:15 截断）。
    """
    stock = _minute(
        ["2024-01-02 09:31", "2024-01-02 10:00", "2024-01-02 11:29", "2024-01-02 13:01"]
    )
    fut = _minute(
        [
            "2024-01-02 09:01",
            "2024-01-02 10:00",
            "2024-01-02 10:14",
            "2024-01-02 10:31",
            "2024-01-02 11:00",
        ],
        symbol="au2406",
    )
    df = pd.concat([stock, fut], ignore_index=True)

    class Eng(_OhlcvLog):
        win_freq = "1h"

    r = run_backtest(
        strategy=Eng,
        data=df,
        symbols=["600000.SH", "au2406"],
        config=_au_config(),
        show_progress=False,
        freq="1min",
    )
    assert r.strategy is not None

    class WholeFrame(aq.feed_adapter.BasePandasFeedAdapter):
        def load(self, request: Any) -> pd.DataFrame:
            return df.set_index("timestamp")

    out = (
        WholeFrame()
        .replay(
            "1h",
            align="session",
            emit_partial=True,
            asset_type={"au2406": "futures"},
        )
        .load(aq.feed_adapter.FeedSlice(symbol="600000.SH", timezone=TZ))
    )
    assert set(out["symbol"]) == {"600000.SH", "au2406"}
    fut_index = pd.DatetimeIndex(out[out["symbol"] == "au2406"].index)
    fut_labels = fut_index.tz_convert(TZ)
    assert list(fut_labels.strftime("%H:%M")) == ["10:00", "10:15", "11:00"]
    for sym in ("600000.SH", "au2406"):
        engine = [row for row in r.strategy.rows if row[0] == sym]
        assert engine
        assert _replay_rows(out[out["symbol"] == sym], sym) == engine


def test_replay_without_inferable_base_uses_deferred_close() -> None:
    """推断不出基础周期（2 行 / 间隔不规则）时与引擎不传 freq 的延迟闭合口径一致."""
    cases = [
        ["2024-01-02 09:31", "2024-01-02 11:30"],
        [
            "2024-01-02 09:31:00",
            "2024-01-02 09:59:30",
            "2024-01-02 10:00:00",
            "2024-01-02 10:31:20",
            "2024-01-02 11:30:00",
            "2024-01-02 13:01:10",
        ],
        # 重复时间戳: 即时闭合会切出两根同标签 10:00 窗口, 延迟闭合合成一根
        [
            "2024-01-02 09:31:00",
            "2024-01-02 09:45:30",
            "2024-01-02 10:00:00",
            "2024-01-02 10:00:00",
            "2024-01-02 10:20:10",
        ],
    ]
    for times in cases:
        df = _minute(times)

        class Eng(_OhlcvLog):
            win_freq = "30min"

        r = run_backtest(
            strategy=Eng,
            data=df,
            symbols=["600000.SH"],
            initial_cash=1e5,
            show_progress=False,
        )
        assert r.strategy is not None
        out = (
            _MemAdapter(df.set_index("timestamp"))
            .replay("30min", align="session", emit_partial=True)
            .load(aq.feed_adapter.FeedSlice(symbol="600000.SH", timezone=TZ))
        )
        assert _replay_rows(out, "600000.SH") == r.strategy.rows


def test_replay_outside_trading_calendar_warns_once(caplog: Any) -> None:
    """数据超出 trading_calendar 时尾部判断改按本地日期，并只告警一次."""
    df = _minute(
        [
            "2024-01-02 09:45",
            "2024-01-02 10:00",
            "2024-01-10 09:45",
            "2024-01-10 10:10",
            "2024-01-10 13:01",
            "2024-01-10 13:10",
        ]
    )
    adapter = _MemAdapter(df.set_index("timestamp")).replay(
        "30min",
        align="session",
        emit_partial=False,
        trading_calendar=["2024-01-02", "2024-01-03"],
    )
    with caplog.at_level("WARNING", logger="akquant"):
        out = adapter.load(aq.feed_adapter.FeedSlice(symbol="600000.SH", timezone=TZ))
    warnings = [r for r in caplog.records if "trading_calendar" in r.getMessage()]
    assert len(warnings) == 1
    assert "600000.SH" in warnings[0].getMessage()
    assert [i.tz_convert(TZ).strftime("%m-%d %H:%M") for i in out.index] == [
        "01-02 10:00",
        "01-10 10:00",
        "01-10 10:30",
    ]


def test_bar_generator_in_strategy_inherits_engine_sessions() -> None:
    """策略内不传 asset_type/sessions 的 BarGenerator 沿用引擎的时段表与资产类型.

    rb2405 按 sessions 取 CN_FUTURES_DAY（10:15 收第一块）：继承后与引擎
    subscribe_bars 窗口相同；若按默认股票表切分会出 10:30 标签。
    """
    from akquant import BarGenerator

    times = [
        f"2024-01-02 {h:02d}:{m:02d}"
        for h, m in [(9, m) for m in range(1, 60)]
        + [(10, m) for m in range(0, 16)]
        + [(10, m) for m in range(31, 60)]
        + [(11, 0)]
    ]

    def row(bar: Bar) -> Tuple[str, float, float, float]:
        local = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ)
        return (
            local.strftime("%m-%d %H:%M"),
            float(bar.open),
            float(bar.close),
            float(bar.volume),
        )

    class S(_WinLog):
        def __init__(self) -> None:
            super().__init__()
            self.gen: Any = None
            self.gen_rows: List[Tuple[str, float, float, float]] = []
            self.lambda_gen: Any = None
            self.lambda_rows: List[Tuple[str, float, float, float]] = []

        def on_bar(self, bar: Bar) -> None:
            if self.gen is None:
                self.gen = BarGenerator(self._on_gen, window=30, base_interval_min=1)
                # 回调不是策略方法时, 按构造处的 self 找到所属策略
                self.lambda_gen = BarGenerator(
                    lambda b: self.lambda_rows.append(row(b)),
                    window=30,
                    base_interval_min=1,
                )
            self.gen.update_bar(bar)
            self.lambda_gen.update_bar(bar)

        def _on_gen(self, bar: Bar) -> None:
            self.gen_rows.append(row(bar))

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
    r = run_backtest(
        strategy=S,
        data=_minute(times, symbol="rb2405"),
        config=cfg,
        show_progress=False,
        freq="1min",
        sessions={"rb": "CN_FUTURES_DAY"},
    )
    strategy = r.strategy
    assert isinstance(strategy, S)
    engine_rows = [(w[1], w[3], w[4], w[5]) for w in strategy.windows]
    assert [e[0] for e in engine_rows] == [
        "01-02 09:30",
        "01-02 10:00",
        "01-02 10:15",
        "01-02 11:00",
    ]
    assert strategy.gen_rows == engine_rows
    assert strategy.lambda_rows == engine_rows


def test_day_close_dispatches_forming_windows_before_after_trading() -> None:
    """收盘 bar 缺失（数据止于 14:56）时，日终结算前闭合并派发当日窗口.

    顺序：窗口 bar → 结算 → on_after_trading；次日开盘不再重复派发。
    """
    times = [f"2024-01-02 14:{m:02d}" for m in range(51, 57)] + [
        f"2024-01-03 09:{m:02d}" for m in range(31, 36)
    ]

    class S(_WinLog):
        freqs = ("5min", "1d")

        def __init__(self) -> None:
            super().__init__()
            self.events: List[Tuple[str, str]] = []

        def on_window_bar(self, bar: Bar) -> None:
            super().on_window_bar(bar)
            local = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ)
            self.events.append((str(bar.freq), local.strftime("%m-%d %H:%M")))

        def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
            self.events.append(("after", trading_date.strftime("%m-%d")))

    r = run_backtest(
        strategy=S,
        data=_minute(times),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        freq="1min",
    )
    strategy = r.strategy
    assert isinstance(strategy, S)
    events = strategy.events
    after = events.index(("after", "01-02"))
    assert events[:after] == [
        ("5min", "01-02 14:55"),
        ("5min", "01-02 15:00"),
        ("1d", "01-02 15:00"),
    ]
    later = events[after + 1 :]
    assert ("5min", "01-02 15:00") not in later
    assert ("1d", "01-02 15:00") not in later
    assert ("5min", "01-03 09:35") in later


def _close_clock_run(times: List[str]) -> Tuple[List[pd.Timestamp], List[str]]:
    class S(_WinLog):
        freqs = ("5min",)

        def __init__(self) -> None:
            super().__init__()
            self.nows: List[pd.Timestamp] = []
            self.fills: List[str] = []
            self.bought = False

        def on_window_bar(self, bar: Bar) -> None:
            super().on_window_bar(bar)
            local = pd.Timestamp(bar.timestamp, tz="UTC").tz_convert(TZ)
            if local.strftime("%m-%d %H:%M") == "01-02 15:00" and not self.bought:
                self.bought = True
                now = self.now
                assert now is not None
                self.nows.append(pd.Timestamp(now).tz_convert(TZ))
                self.buy("600000.SH", 100)

        def on_trade(self, trade: Any) -> None:
            local = pd.Timestamp(trade.timestamp, tz="UTC").tz_convert(TZ)
            self.fills.append(local.strftime("%m-%d %H:%M"))

    r = run_backtest(
        strategy=S,
        data=_minute(times),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
        freq="1min",
    )
    strategy = r.strategy
    assert isinstance(strategy, S)
    return strategy.nows, strategy.fills


def test_day_close_window_dispatch_runs_at_close_clock() -> None:
    """收盘 bar 缺失时日终派发窗口: now 为窗口标签, 回调里的市价单次日开盘成交."""
    nxt = [f"2024-01-03 09:{m:02d}" for m in range(31, 36)]
    missing = [f"2024-01-02 14:{m:02d}" for m in range(51, 57)] + nxt
    present = (
        [f"2024-01-02 14:{m:02d}" for m in range(51, 57)] + ["2024-01-02 15:00"] + nxt
    )
    close = pd.Timestamp("2024-01-02 15:00", tz=TZ)
    nows_missing, fills_missing = _close_clock_run(missing)
    nows_present, fills_present = _close_clock_run(present)
    assert nows_missing == [close]
    assert nows_present == [close]
    assert fills_missing == fills_present
    assert fills_missing == ["01-03 09:31"]
