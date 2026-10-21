from collections import deque
from typing import Any, Callable, Dict

import pandas as pd


class PrecomputedIndicator:
    """向量化预计算指标: 整段 DataFrame 算完缓存, 按时间点 asof 查询. 实盘不可用."""

    def __init__(self, name: str, fn: Callable, **kwargs: Any) -> None:
        """Initialize the PrecomputedIndicator."""
        self.name = name
        self.fn = fn
        self.kwargs = kwargs
        self._data: Dict[str, pd.Series] = {}  # symbol -> series
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
