"""Regression tests for rust-backend warm-up masking across the EMA family.

Drop these into tests/test_talib_compat.py, next to
test_talib_adx_has_warmup_nans, which asserts the same property for ADX.

Every `if backend_key == "rust":` branch in funcs.py returns early, so the
warm-up NaN masking written into the python branch never ran on the rust path.
`auto` resolves to rust, so default-backend users were getting fabricated values
for the first N bars of EMA, DEMA, TEMA, TRIX and MACD.
"""
import numpy as np
import pandas as pd
import pytest

from akquant import talib as ta


def _sample_ohlc(size: int = 80) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Deterministic OHLC, matching the helper already in test_talib_compat.py."""
    idx = pd.date_range("2024-01-01", periods=size, freq="D", tz="UTC")
    close = pd.Series(np.linspace(10.0, 25.0, size), index=idx)
    high = close + 1.0
    low = close - 1.0
    return high, low, close


# warm-up length each python branch already uses, kept in one place so the two
# paths cannot drift apart again without a test noticing
WARMUP_FORMULA = {
    "EMA": lambda period: period - 1,
    "DEMA": lambda period: 2 * period - 2,
    "TEMA": lambda period: 3 * period - 2,
    "TRIX": lambda period: 3 * period - 2,
}

PERIOD = 14
SINGLE_OUTPUT = ["EMA", "DEMA", "TEMA", "TRIX"]
MACD_WARMUP = 26 + 9 - 2  # slowperiod + signalperiod - 2


def _series_of(value):
    return np.asarray(value, dtype=float)


@pytest.mark.parametrize("name", SINGLE_OUTPUT)
@pytest.mark.parametrize("backend", ["rust", "auto"])
def test_ema_family_masks_warmup(name: str, backend: str) -> None:
    """The rust path must mask the same warm-up region the python path masks."""
    _, _, close = _sample_ohlc()
    out = _series_of(getattr(ta, name)(close, timeperiod=PERIOD, backend=backend))
    warmup = WARMUP_FORMULA[name](PERIOD)
    assert np.isnan(out[:warmup]).all(), f"{backend}/{name} warm-up not masked"
    assert np.isfinite(out[warmup:]).any(), f"{backend}/{name} never produces a value"


@pytest.mark.parametrize("name", SINGLE_OUTPUT)
def test_ema_family_rust_matches_python(name: str) -> None:
    """The rust and python backends must agree exactly, NaNs included."""
    _, _, close = _sample_ohlc()
    expected = _series_of(getattr(ta, name)(close, timeperiod=PERIOD, backend="python"))
    for backend in ("rust", "auto"):
        actual = _series_of(getattr(ta, name)(close, timeperiod=PERIOD, backend=backend))
        np.testing.assert_allclose(expected, actual, equal_nan=True)


@pytest.mark.parametrize("backend", ["rust", "python", "auto"])
def test_macd_masks_warmup(backend: str) -> None:
    """MACD must mask the warm-up region on every backend."""
    _, _, close = _sample_ohlc()
    macd, signal, hist = ta.MACD(close, backend=backend)
    for label, series in (("macd", macd), ("signal", signal), ("hist", hist)):
        arr = _series_of(series)
        assert np.isnan(arr[:MACD_WARMUP]).all(), f"{backend}/{label} warm-up not masked"
        assert np.isfinite(arr[MACD_WARMUP:]).any(), f"{backend}/{label} never produces a value"


@pytest.mark.parametrize("backend", ["rust", "auto"])
def test_macd_rust_matches_python(backend: str) -> None:
    """The rust and python MACD backends must agree exactly, NaNs included."""
    _, _, close = _sample_ohlc()
    expected = ta.MACD(close, backend="python")
    actual = ta.MACD(close, backend=backend)
    for want, got in zip(expected, actual):
        np.testing.assert_allclose(
            _series_of(want), _series_of(got), equal_nan=True
        )
