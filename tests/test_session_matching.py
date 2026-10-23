"""交易时段重构阶段 3：真实时段与集合竞价撮合的验收测试."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

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
