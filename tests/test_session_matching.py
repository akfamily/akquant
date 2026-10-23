"""交易时段重构阶段 3：真实时段与集合竞价撮合的验收测试."""

from __future__ import annotations

from typing import Any, List, Tuple

import pandas as pd
from akquant import Bar, Strategy, run_backtest

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
