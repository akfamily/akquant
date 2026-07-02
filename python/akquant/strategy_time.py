from numbers import Integral
from typing import Any, Optional, cast

import pandas as pd


def current_timestamp(strategy: Any) -> Optional[int]:
    """Return the current strategy timestamp in UTC nanoseconds when available."""
    ts = None
    ctx = getattr(strategy, "ctx", None)
    if ctx is not None:
        current_time_raw = getattr(ctx, "current_time", None)
        if isinstance(current_time_raw, Integral) and not isinstance(
            current_time_raw, bool
        ):
            current_time = int(current_time_raw)
            if current_time > 0:
                ts = current_time

    if ts is None and strategy.current_bar:
        ts = strategy.current_bar.timestamp
    elif ts is None and strategy.current_tick:
        ts = strategy.current_tick.timestamp
    return ts


def to_local_time(strategy: Any, timestamp: int) -> pd.Timestamp:
    """将 UTC 纳秒时间戳转换为本地时间 (Timestamp)."""
    ts_utc = pd.to_datetime(timestamp, unit="ns", utc=True)
    return cast(pd.Timestamp, ts_utc.tz_convert(strategy.timezone))


def format_time_iso_utc(timestamp: int) -> str:
    """将 UTC 纳秒时间戳格式化为 UTC ISO 8601 字符串."""
    ts_utc = cast(pd.Timestamp, pd.to_datetime(timestamp, unit="ns", utc=True))
    iso_utc = cast(str, ts_utc.isoformat())
    return iso_utc.replace("+00:00", "Z")


def format_time(strategy: Any, timestamp: int, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """将 UTC 纳秒时间戳格式化为本地时间字符串."""
    local_time = to_local_time(strategy, timestamp)
    return cast(str, local_time.strftime(fmt))


def now(strategy: Any) -> Optional[pd.Timestamp]:
    """获取当前回测时间的本地时间表示."""
    ts = current_timestamp(strategy)
    if ts is not None:
        return to_local_time(strategy, ts)
    return None
