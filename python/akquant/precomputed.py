from collections import deque
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd


class _AsofTable:
    """一个标的的预计算序列, 摊平成数组后按整数纳秒做 asof 定位.

    语义与 ``Series.asof`` 相同: 取该时刻及之前最后一个非 NaN 值, 之前没有则 NaN。
    无时区索引按 UTC 墙上时间比较(与原先把 UTC 时间戳去掉时区再比一致)。
    索引不是有序的时间索引、或值转不成浮点时 ``usable`` 为 False, 由调用方
    退回 ``Series.asof``。
    """

    __slots__ = ("series", "usable", "_times", "_values", "_pos")

    def __init__(self, series: pd.Series) -> None:
        self.series = series
        self.usable = False
        self._pos = -1
        index = series.index
        if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
            return
        if not index.is_monotonic_increasing:
            return
        try:
            # pandas 3 的时间索引不一定是纳秒精度, 统一后再取整数。
            times = np.asarray(index.as_unit("ns").astype("int64"))
            values = pd.Series(series.to_numpy(dtype=float, na_value=np.nan)).ffill()
        except Exception:
            return
        self._times = times
        self._values = values.to_numpy()
        self.usable = True

    def value_at(self, timestamp_ns: int) -> float:
        times = self._times
        pos = self._pos
        size = len(times)
        # 逐 bar 前进时游标只需原地不动或前进一格; 其余情况(回退、跳跃)二分。
        if pos >= 0 and times[pos] > timestamp_ns:
            pos = int(np.searchsorted(times, timestamp_ns, side="right")) - 1
        elif pos + 1 < size and times[pos + 1] <= timestamp_ns:
            if pos + 2 < size and times[pos + 2] <= timestamp_ns:
                pos = int(np.searchsorted(times, timestamp_ns, side="right")) - 1
            else:
                pos += 1
        self._pos = pos
        if pos < 0:
            return float("nan")
        return float(self._values[pos])


class PrecomputedIndicator:
    """向量化预计算指标: 整段 DataFrame 算完缓存, 按时间点 asof 查询. 实盘不可用."""

    def __init__(self, name: str, fn: Callable, **kwargs: Any) -> None:
        """Initialize the PrecomputedIndicator."""
        self.name = name
        self.fn = fn
        self.kwargs = kwargs
        self._data: Dict[str, pd.Series] = {}  # symbol -> series
        self._lookup: Dict[str, _AsofTable] = {}  # symbol -> 摊平后的定位表
        self._current_value: float = float("nan")

    def update(self, value: float) -> float:
        """Update indicator value (incremental)."""
        raise NotImplementedError(
            "Incremental update not implemented for this indicator."
        )

    def __call__(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Calculate indicator on a DataFrame."""
        if symbol in self._data:
            return self._data[symbol]

        # Assume fn takes a series/df and returns a series
        # If kwargs contains column names, extract them
        # This is a simplified version of powerful DSL
        try:
            result = self.fn(df, **self.kwargs)
        except Exception:
            # Try passing column if specified in kwargs
            # e.g. rolling_mean(df['close'], window=5)
            # This part is tricky to generalize without a full DSL,
            # so we start simple: user passes a lambda or function that takes df
            result = self.fn(df)

        if not isinstance(result, pd.Series):
            # Try to convert if it's not a Series (e.g. numpy array)
            result = pd.Series(result, index=df.index)

        self._data[symbol] = result
        return result

    def get_value(self, symbol: str, timestamp: Any) -> float:
        """
        Get indicator value at specific timestamp (or latest before it).

        Uses asof lookup which is efficient for sorted time series.
        """
        if symbol not in self._data:
            return float("nan")

        series = self._data[symbol]
        if type(timestamp) is int:
            # 回测热路径: 引擎给的是整数纳秒, 直接在数组上定位, 不构造 Timestamp。
            # 旧存档还原出来的实例没有 _lookup, 用到时再补。
            lookup = self.__dict__.get("_lookup")
            if lookup is None:
                lookup = self._lookup = {}
            table = lookup.get(symbol)
            if table is None or table.series is not series:
                table = lookup[symbol] = _AsofTable(series)
            if table.usable:
                return table.value_at(timestamp)
        # Assuming series index is datetime
        try:
            # Handle integer timestamp (nanoseconds)
            ts = timestamp
            if isinstance(timestamp, (int, float)):
                ts = pd.Timestamp(timestamp, unit="ns", tz="UTC")

            # Handle Timezone Mismatch
            if isinstance(series.index, pd.DatetimeIndex):
                if series.index.tz is None and getattr(ts, "tzinfo", None) is not None:
                    ts = ts.tz_localize(None)
                elif (
                    series.index.tz is not None and getattr(ts, "tzinfo", None) is None
                ):
                    ts = ts.tz_localize("UTC").tz_convert(series.index.tz)

            return float(series.asof(ts))  # type: ignore[arg-type]
        except Exception:
            return float("nan")


class SMA(PrecomputedIndicator):
    """Simple Moving Average."""

    def __init__(self, window: int) -> None:
        """Initialize SMA."""
        super().__init__("sma", self._calc_sma)
        self.window = window
        self._cache: Dict[str, pd.Series] = {}  # symbol -> series
        self._current_value = float("nan")
        self._buffer: deque = deque()
        self._sum = 0.0

    def _calc_sma(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].rolling(self.window).mean()

    def update(self, value: float) -> float:
        """Update with new value (incremental)."""
        if pd.isna(value):
            return self._current_value

        if len(self._buffer) == self.window:
            removed = self._buffer.popleft()
            self._sum -= removed

        self._buffer.append(value)
        self._sum += value

        if len(self._buffer) == self.window:
            self._current_value = self._sum / self.window
        else:
            self._current_value = float("nan")

        return self._current_value

    def __call__(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Calculate SMA."""
        result = df["close"].rolling(self.window).mean()
        self._cache[symbol] = result
        return result

    @property
    def value(self) -> float:
        """Get current value (requires strategy context injection)."""
        # This relies on Strategy injecting itself or data
        # For now, we use a simple mechanism:
        # Strategy calls update() or sets current_bar
        return self._current_value

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle support."""
        state = self.__dict__.copy()
        # Don't save cache to save space, or save it if we want full state
        # For warm start, we might want to clear cache and re-calculate on new data
        # BUT, if we want to support 'streaming' updates later, we need state.
        # For current DataFrame-based indicator, re-calculation is fast.
        state["_cache"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Unpickle support."""
        self.__dict__.update(state)
