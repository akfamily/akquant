from typing import Any, Callable, Dict

import pandas as pd


class Indicator:
    """
    Helper class for defining and calculating indicators.

    Inspired by PyBroker's indicator system.
    """

    def __init__(self, name: str, fn: Callable, **kwargs: Any) -> None:
        """Initialize the Indicator."""
        self.name = name
        self.fn = fn
        self.kwargs = kwargs
        self._data: Dict[str, pd.Series] = {}  # symbol -> series

    def __call__(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Calculate indicator on a DataFrame."""
        if symbol in self._data:
            return self._data[symbol]

        # Assume fn takes a series/df and returns a series
        # If kwargs contains column names, extract them
        # This is a simplified version of PyBroker's powerful DSL
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
            return float(series.asof(timestamp))  # type: ignore[arg-type]
        except Exception:
            return float("nan")


class IndicatorSet:
    """Collection of indicators for easy management."""

    def __init__(self) -> None:
        """Initialize the IndicatorSet."""
        self._indicators: Dict[str, Indicator] = {}

    def add(self, name: str, fn: Callable, **kwargs: Any) -> None:
        """Add an indicator to the set."""
        self._indicators[name] = Indicator(name, fn, **kwargs)

    def get(self, name: str) -> Indicator:
        """Get an indicator by name."""
        return self._indicators[name]

    def calculate_all(self, df: pd.DataFrame, symbol: str) -> Dict[str, pd.Series]:
        """Calculate all indicators for the given dataframe."""
        results = {}
        for name, ind in self._indicators.items():
            results[name] = ind(df, symbol)
        return results
