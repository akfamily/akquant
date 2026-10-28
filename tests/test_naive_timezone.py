"""不带时区的 DataFrame 时间按 run_backtest(timezone=...) 本地化."""

from __future__ import annotations

from typing import List

import pandas as pd
from akquant import Bar, Strategy, run_backtest


class _Seen(Strategy):
    """记录收到的 bar 时间(UTC)."""

    def __init__(self) -> None:
        """初始化记录列表."""
        super().__init__()
        self.seen: List[pd.Timestamp] = []

    def on_bar(self, bar: Bar) -> None:
        """记录 bar 时间."""
        self.seen.append(pd.Timestamp(bar.timestamp, tz="UTC"))


def _naive_frame(times: List[str], symbol: str) -> pd.DataFrame:
    """构造不带时区的单行 OHLCV 帧."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(times),
            "symbol": symbol,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )


def test_naive_times_follow_timezone_parameter() -> None:
    """不带时区的单 DataFrame 按 timezone= 本地化."""
    r = run_backtest(
        strategy=_Seen,
        data=_naive_frame(["2024-01-02 10:00"], "AAPL"),
        symbols=["AAPL"],
        initial_cash=1e5,
        show_progress=False,
        timezone="America/New_York",
    )
    assert r.strategy is not None
    local = r.strategy.seen[0].tz_convert("America/New_York")
    assert local.strftime("%Y-%m-%d %H:%M") == "2024-01-02 10:00"


def test_naive_times_default_to_shanghai() -> None:
    """默认时区仍是上海."""
    r = run_backtest(
        strategy=_Seen,
        data=_naive_frame(["2024-01-02 10:00"], "600000.SH"),
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r.strategy is not None
    local = r.strategy.seen[0].tz_convert("Asia/Shanghai")
    assert local.strftime("%H:%M") == "10:00"


def test_naive_dict_input_follows_timezone() -> None:
    """Dict 输入同样按 timezone= 本地化."""
    data = {"AAPL": _naive_frame(["2024-01-02 10:00"], "AAPL").set_index("timestamp")}
    r = run_backtest(
        strategy=_Seen,
        data=data,
        symbols=["AAPL"],
        initial_cash=1e5,
        show_progress=False,
        timezone="America/New_York",
    )
    assert r.strategy is not None
    assert r.strategy.seen[0].tz_convert("America/New_York").hour == 10
