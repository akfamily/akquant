"""只有日期数据的检测与收盘改写：跨标的共享收盘缓存与快速检测不改变结果."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import akquant as aq
import numpy as np
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest
from akquant.backtest import engine as bt_engine

SH = "Asia/Shanghai"
BOND = {"T": "CN_FUTURES_CFFEX_BOND_DAY"}


class _Log:
    """收集 warning 文案的 logger 替身."""

    def __init__(self) -> None:
        self.messages: List[str] = []

    def warning(self, fmt: str, *args: Any) -> None:
        self.messages.append(fmt % args)


def _rewriter(
    pairs: List[Tuple[str, Any]],
    timezone: str = SH,
    sessions: Optional[Dict[str, str]] = None,
) -> Tuple[Any, _Log]:
    resolver = aq.SessionResolver(sessions=sessions, timezone=timezone)
    log = _Log()
    rewriter = bt_engine._DateOnlyRewriter(resolver, timezone, log)
    types = dict(pairs)
    rewriter.activate(list(types), types.get, underlying_of=lambda _s: None)
    return rewriter, log


def _frame(times: Any, symbol: Optional[str] = None) -> pd.DataFrame:
    n = len(times)
    df = pd.DataFrame(
        {
            "timestamp": times,
            "open": np.arange(n) + 10.0,
            "high": np.arange(n) + 10.5,
            "low": np.arange(n) + 9.5,
            "close": np.arange(n) + 10.2,
            "volume": 1000.0,
        }
    )
    if symbol is not None:
        df["symbol"] = symbol
    return df


def _expected_close(
    pairs: List[Tuple[str, Any]], day: dt.date, sessions: Optional[Dict[str, str]]
) -> pd.Timestamp:
    """不经缓存、按会话级收盘逐日独立求值."""
    resolver = aq.SessionResolver(sessions=sessions, timezone=SH)
    resolver.set_active(pairs)
    return pd.Timestamp(bt_engine._date_only_close_ns(resolver, day, SH), tz="UTC")


_PAIRS = [
    ("600000.SH", aq.AssetType.Stock),
    ("000001.SZ", aq.AssetType.Stock),
    ("T2403", aq.AssetType.Futures),
    ("TF2403", aq.AssetType.Futures),
    ("IF2403", aq.AssetType.Futures),
    ("rb2405", aq.AssetType.Futures),
]


class _Seen(Strategy):
    """记录每根 bar 的 (标的, UTC 时间戳)."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: List[Tuple[str, int]] = []

    def on_bar(self, bar: Bar) -> None:
        self.seen.append((bar.symbol, int(bar.timestamp)))


@pytest.mark.parametrize("sessions", [BOND, None], ids=["bond_1515", "default"])
def test_mixed_sessions_dict_rows_match_uncached_close(
    sessions: Optional[Dict[str, str]],
) -> None:
    """多品种（股票 + 国债期货等）只有日期 dict：每行等于逐日独立求值的会话收盘."""
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2024-01-02", "2024-03-29")
    data: Dict[str, pd.DataFrame] = {}
    expected: List[Tuple[str, int]] = []
    for sym, _ in _PAIRS:
        pick = np.sort(rng.choice(len(days), size=30, replace=False))
        local = days[pick].tz_localize(SH)
        data[sym] = _frame(local)
        for ts in local:
            close = _expected_close(_PAIRS, ts.date(), sessions)
            expected.append((sym, int(close.value)))
    result = run_backtest(
        data=data,
        strategy=_Seen,
        symbols=[s for s, _ in _PAIRS],
        initial_cash=1e6,
        show_progress=False,
        sessions=sessions,
    )
    seen = result.strategy.seen  # type: ignore[union-attr]
    assert sorted(seen) == sorted(expected)
    local_times = {
        pd.Timestamp(ts, tz="UTC").tz_convert(SH).strftime("%H:%M") for _, ts in seen
    }
    assert local_times == ({"15:15"} if sessions else {"15:00"})


def test_close_cache_keyed_by_active_symbols() -> None:
    """共享收盘缓存随在用标的更新：加入国债期货后收盘变为 15:15，移除后回到 15:00."""
    stock = [("600000.SH", aq.AssetType.Stock)]
    rewriter, _ = _rewriter(stock, sessions=BOND)
    day = dt.date(2024, 1, 2)
    close_1500 = pd.Timestamp("2024-01-02 15:00", tz=SH).value
    close_1515 = pd.Timestamp("2024-01-02 15:15", tz=SH).value
    assert rewriter.close_ns(day) == close_1500
    rewriter.activate(
        ["600000.SH", "T2403"],
        {"600000.SH": aq.AssetType.Stock, "T2403": aq.AssetType.Futures}.get,
        underlying_of=lambda _s: None,
    )
    assert rewriter.close_ns(day) == close_1515
    rewriter.activate(
        ["600000.SH"],
        {"600000.SH": aq.AssetType.Stock}.get,
        underlying_of=lambda _s: None,
    )
    assert rewriter.close_ns(day) == close_1500


def test_shared_cache_warns_once_at_first_appearing_non_trading_day() -> None:
    """跨标的共享缓存：非交易日只告警一次，告警日期为按行顺序首个出现的非交易日."""
    rewriter, log = _rewriter(_PAIRS, sessions=BOND)
    first = pd.DatetimeIndex(["2024-01-05", "2024-01-08"]).tz_localize(SH)
    # 乱序：按行首个非交易日是 01-13（周六），排序后首个是 01-06
    second = pd.DatetimeIndex(
        ["2024-01-13", "2024-01-06", "2024-01-08", "2024-01-14"]
    ).tz_localize(SH)
    for times in (first, second, second):
        _, rewritten = bt_engine._rewrite_date_only_frame(_frame(times), rewriter)
        assert rewritten
    assert len(log.messages) == 1, log.messages
    assert "2024-01-13" in log.messages[0]


def test_multi_symbol_frame_and_bars_rows_match_uncached_close() -> None:
    """多标的单 DataFrame 与 Bar 列表：改写行等于逐日独立求值，含时刻的标的不改写."""
    rewriter, _ = _rewriter(_PAIRS, sessions=BOND)
    parts = []
    for i, (sym, _) in enumerate(_PAIRS):
        times = pd.bdate_range("2024-01-02", periods=8).tz_localize(SH)
        if i == 4:
            times = times + pd.Timedelta(hours=15)
        parts.append(_frame(times, sym))
    df = pd.concat(parts).sort_values("timestamp", kind="stable")
    df = df.reset_index(drop=True)
    out, mask = bt_engine._rewrite_date_only_multi_symbol_frame(df, rewriter)
    assert mask.tolist() == (df["symbol"] != _PAIRS[4][0]).tolist()
    for (_, row), (_, new) in zip(df.iterrows(), out.iterrows()):
        if row["symbol"] == _PAIRS[4][0]:
            assert new["timestamp"] == row["timestamp"]
        else:
            day = row["timestamp"].tz_convert(SH).date()
            assert new["timestamp"] == _expected_close(_PAIRS, day, BOND)

    stamps = [int(pd.Timestamp(t).value) for t in df["timestamp"]]
    bars = [
        Bar(int(ts), 1.0, 1.0, 1.0, 1.0, 1.0, str(sym))
        for ts, sym in zip(stamps, df["symbol"])
    ]
    new_bars = bt_engine._rewrite_date_only_bars(bars, rewriter, None)
    assert [int(b.timestamp) for b in new_bars] == [
        int(pd.Timestamp(t).value) for t in out["timestamp"]
    ]


_DAYS = pd.bdate_range("2024-01-02", periods=20)


@pytest.mark.parametrize(
    ("times", "expected"),
    [
        (_DAYS.tz_localize(SH) + pd.Timedelta(hours=15), False),
        (_DAYS.tz_localize(SH), True),
        (_DAYS.tz_localize(SH).tz_convert("UTC"), True),
        (_DAYS.tz_localize("UTC"), False),
        (_DAYS.tz_localize(SH).as_unit("s"), True),
        ((_DAYS + pd.Timedelta(hours=15)).tz_localize(SH).as_unit("us"), False),
        (_DAYS, True),
        (_DAYS + pd.Timedelta(hours=15), False),
        (
            pd.DatetimeIndex(
                [*_DAYS[:5].tz_localize(SH), pd.Timestamp("2024-01-09 10:00", tz=SH)]
            ),
            False,
        ),
        (
            pd.DatetimeIndex(
                [pd.Timestamp("2024-01-01 15:00", tz=SH), *_DAYS[:5].tz_localize(SH)]
            ),
            False,
        ),
        (pd.DatetimeIndex([pd.NaT, *_DAYS[:5].tz_localize(SH)]), False),
        (
            pd.DatetimeIndex(
                [*_DAYS[:3].tz_localize(SH), pd.NaT, *_DAYS[3:5].tz_localize(SH)]
            ),
            False,
        ),
        ([str(d.date()) for d in _DAYS], True),
        ([d.date() for d in _DAYS], True),
    ],
    ids=[
        "aware_1500",
        "aware_midnight",
        "aware_utc_of_local_midnight",
        "aware_utc_midnight",
        "aware_unit_s",
        "aware_unit_us_1500",
        "naive_midnight",
        "naive_1500",
        "midnight_then_intraday",
        "1500_then_midnight",
        "nat_first",
        "nat_middle",
        "date_strings",
        "date_objects",
    ],
)
@pytest.mark.parametrize("as_index", [False, True], ids=["column", "index"])
def test_date_only_detection_shanghai(
    times: Any, expected: bool, as_index: bool
) -> None:
    """上海时区下各类时间列/索引的“只有日期”判定（与修复前逐例一致）."""
    rewriter, _ = _rewriter([("600000.SH", aq.AssetType.Stock)])
    df = _frame(times)
    if as_index:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            pytest.skip("索引形态只测 datetime 列")
        df = df.set_index("timestamp")
    _, rewritten = bt_engine._rewrite_date_only_frame(df, rewriter)
    assert rewritten is expected


def _year(tz: Optional[str], hours: int = 0) -> Any:
    base = pd.date_range("2024-01-01", "2024-12-31", freq="D") + pd.Timedelta(
        hours=hours
    )
    if tz is None:
        return base
    return base.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")


@pytest.mark.parametrize(
    ("timezone", "aware", "hours", "expected"),
    [
        ("America/New_York", False, 0, True),
        ("America/New_York", True, 0, True),
        ("America/New_York", True, 15, False),
        ("Europe/London", False, 0, True),
        ("Europe/London", True, 15, False),
        # 03-10 本地 0 点不存在：naive 0 点前移到 01:00，整列不再是只有日期
        ("America/Havana", False, 0, False),
        ("America/Havana", True, 0, False),
    ],
)
def test_date_only_detection_dst_timezones(
    timezone: str, aware: bool, hours: int, expected: bool
) -> None:
    """夏令时时区：naive 0 点按引擎时区本地化（跳变点前移）后的判定不变."""
    rewriter, _ = _rewriter([("SPY", aq.AssetType.Stock)], timezone=timezone)
    times = _year(timezone if aware else None, hours)
    _, rewritten = bt_engine._rewrite_date_only_frame(_frame(times), rewriter)
    assert rewritten is expected


@pytest.mark.parametrize("aware", [False, True], ids=["naive", "aware"])
def test_dst_midnight_gap_day_does_not_raise(aware: bool) -> None:
    """当日 0 点因夏令时不存在（圣地亚哥 2024-09-08）：判为非只有日期，不再报错.

    修复前用 ``normalize`` 判断，含该日的数据会抛 ValueError（nonexistent time）。
    """
    tz = "America/Santiago"
    rewriter, _ = _rewriter([("SPY", aq.AssetType.Stock)], timezone=tz)
    for hours in (0, 15):
        times = _year(tz if aware else None, hours)
        _, rewritten = bt_engine._rewrite_date_only_frame(_frame(times), rewriter)
        assert rewritten is False
