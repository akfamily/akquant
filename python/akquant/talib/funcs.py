"""TA-Lib style indicator functions."""

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

from ..akquant import ABS as RustABS
from ..akquant import ACOS as RustACOS
from ..akquant import AD as RustAD
from ..akquant import ADD as RustADD
from ..akquant import ADOSC as RustADOSC
from ..akquant import ADX as RustADX
from ..akquant import ADXR as RustADXR
from ..akquant import APO as RustAPO
from ..akquant import AROON as RustAROON
from ..akquant import AROONOSC as RustAROONOSC
from ..akquant import ASIN as RustASIN
from ..akquant import ATAN as RustATAN
from ..akquant import ATR as RustATR
from ..akquant import AVGDEV as RustAVGDEV
from ..akquant import AVGPRICE as RustAVGPRICE
from ..akquant import BETA as RustBETA
from ..akquant import BOP as RustBOP
from ..akquant import CCI as RustCCI
from ..akquant import CEIL as RustCEIL
from ..akquant import CLAMP01 as RustCLAMP01
from ..akquant import CLIP as RustCLIP
from ..akquant import CMO as RustCMO
from ..akquant import CORREL as RustCORREL
from ..akquant import COS as RustCOS
from ..akquant import COSH as RustCOSH
from ..akquant import COVAR as RustCOVAR
from ..akquant import CUBE as RustCUBE
from ..akquant import DEG2RAD as RustDEG2RAD
from ..akquant import DEMA as RustDEMA
from ..akquant import DIV as RustDIV
from ..akquant import DX as RustDX
from ..akquant import EMA as RustEMA
from ..akquant import EXP as RustEXP
from ..akquant import EXPM1 as RustEXPM1
from ..akquant import FLOOR as RustFLOOR
from ..akquant import HT_TRENDLINE as RustHT_TRENDLINE
from ..akquant import INV_SQRT as RustINV_SQRT
from ..akquant import KAMA as RustKAMA
from ..akquant import LINEARREG as RustLINEARREG
from ..akquant import LINEARREG_ANGLE as RustLINEARREG_ANGLE
from ..akquant import LINEARREG_INTERCEPT as RustLINEARREG_INTERCEPT
from ..akquant import LINEARREG_R2 as RustLINEARREG_R2
from ..akquant import LINEARREG_SLOPE as RustLINEARREG_SLOPE
from ..akquant import LN as RustLN
from ..akquant import LOG1P as RustLOG1P
from ..akquant import LOG10 as RustLOG10
from ..akquant import MACD as RustMACD
from ..akquant import MAMA as RustMAMA
from ..akquant import MAX as RustMAX
from ..akquant import MAX2 as RustMAX2
from ..akquant import MAXINDEX as RustMAXINDEX
from ..akquant import MEDPRICE as RustMEDPRICE
from ..akquant import MFI as RustMFI
from ..akquant import MIDPOINT as RustMIDPOINT
from ..akquant import MIDPRICE as RustMIDPRICE
from ..akquant import MIN as RustMIN
from ..akquant import MIN2 as RustMIN2
from ..akquant import MININDEX as RustMININDEX
from ..akquant import MINMAX as RustMINMAX
from ..akquant import MINMAXINDEX as RustMINMAXINDEX
from ..akquant import MINUS_DI as RustMINUS_DI
from ..akquant import MOD as RustMOD
from ..akquant import MOM as RustMOM
from ..akquant import MULT as RustMULT
from ..akquant import NATR as RustNATR
from ..akquant import OBV as RustOBV
from ..akquant import PLUS_DI as RustPLUS_DI
from ..akquant import POW as RustPOW
from ..akquant import PPO as RustPPO
from ..akquant import RANGE as RustRANGE
from ..akquant import RECIP as RustRECIP
from ..akquant import ROC as RustROC
from ..akquant import ROCP as RustROCP
from ..akquant import ROCR as RustROCR
from ..akquant import ROCR100 as RustROCR100
from ..akquant import ROUND as RustROUND
from ..akquant import RSI as RustRSI
from ..akquant import SAR as RustSAR
from ..akquant import SIGN as RustSIGN
from ..akquant import SIN as RustSIN
from ..akquant import SINH as RustSINH
from ..akquant import SMA as RustSMA
from ..akquant import SQ as RustSQ
from ..akquant import SQRT as RustSQRT
from ..akquant import STDDEV as RustSTDDEV
from ..akquant import STOCH as RustSTOCH
from ..akquant import SUB as RustSUB
from ..akquant import SUM as RustSUM
from ..akquant import T3 as RustT3
from ..akquant import TAN as RustTAN
from ..akquant import TANH as RustTANH
from ..akquant import TEMA as RustTEMA
from ..akquant import TRANGE as RustTRANGE
from ..akquant import TRIMA as RustTRIMA
from ..akquant import TRIX as RustTRIX
from ..akquant import TSF as RustTSF
from ..akquant import TYPPRICE as RustTYPPRICE
from ..akquant import ULTOSC as RustULTOSC
from ..akquant import VAR as RustVAR
from ..akquant import WCLPRICE as RustWCLPRICE
from ..akquant import WILLR as RustWILLR
from ..akquant import WMA as RustWMA
from ..akquant import BollingerBands as RustBollingerBands
from . import _kernels
from .backend import resolve_backend
from .core import SeriesLike, finalize_output, skip_non_finite_rows, to_series


def _ensure_period(value: int, name: str) -> int:
    """Validate integer period parameter."""
    period = int(value)
    if period <= 0:
        raise ValueError(f"{name} must be > 0")
    return period


def _rolling_mean(series: pd.Series, period: int) -> pd.Series:
    """Compute rolling mean with default pandas behavior."""
    return cast(pd.Series, series.rolling(period).mean())


def _np(series: pd.Series) -> np.ndarray:
    """内核只吃 float64 数组."""
    return series.to_numpy(dtype=float)


def _like(values: np.ndarray, template: pd.Series) -> pd.Series:
    """把内核输出包成与输入同索引的 Series."""
    return pd.Series(values, index=template.index, dtype=float)


def _batch_call(
    update_fn: Callable[..., object], method_name: str, *arrays: np.ndarray
) -> object | None:
    owner = getattr(update_fn, "__self__", None)
    if owner is None:
        return None
    method = getattr(owner, method_name, None)
    if method is None or not callable(method):
        return None
    return cast(object, method(*arrays))


def _run_rust_single_series(
    values: pd.Series,
    update_fn: Callable[[float], float | None],
) -> pd.Series:
    values_arr = values.to_numpy(dtype=float, copy=False)
    batch_out = _batch_call(update_fn, "update_many", values_arr)
    if batch_out is not None:
        arr = np.asarray(batch_out, dtype=float)
        if arr.shape[0] == values_arr.shape[0]:
            return pd.Series(arr, index=values.index, dtype=float)
    arr = np.full(len(values), np.nan, dtype=float)
    for idx, value in enumerate(values):
        out = update_fn(float(value))
        if out is not None:
            arr[idx] = float(out)
    return pd.Series(arr, index=values.index, dtype=float)


def _run_rust_hlc_series(
    high_values: pd.Series,
    low_values: pd.Series,
    close_values: pd.Series,
    update_fn: Callable[[float, float, float], float | None],
) -> pd.Series:
    highs_arr = high_values.to_numpy(dtype=float, copy=False)
    lows_arr = low_values.to_numpy(dtype=float, copy=False)
    closes_arr = close_values.to_numpy(dtype=float, copy=False)
    batch_out = _batch_call(
        update_fn, "update_many_hlc", highs_arr, lows_arr, closes_arr
    )
    if batch_out is not None:
        arr = np.asarray(batch_out, dtype=float)
        if arr.shape[0] == closes_arr.shape[0]:
            return pd.Series(arr, index=close_values.index, dtype=float)
    arr = np.full(len(close_values), np.nan, dtype=float)
    for idx, (high_v, low_v, close_v) in enumerate(
        zip(high_values, low_values, close_values)
    ):
        out = update_fn(float(high_v), float(low_v), float(close_v))
        if out is not None:
            arr[idx] = float(out)
    return pd.Series(arr, index=close_values.index, dtype=float)


def _run_rust_hlc_pair_series(
    high_values: pd.Series,
    low_values: pd.Series,
    close_values: pd.Series,
    update_fn: Callable[[float, float, float], tuple[float, float] | None],
) -> tuple[pd.Series, pd.Series]:
    highs_arr = high_values.to_numpy(dtype=float, copy=False)
    lows_arr = low_values.to_numpy(dtype=float, copy=False)
    closes_arr = close_values.to_numpy(dtype=float, copy=False)
    batch_out = _batch_call(
        update_fn, "update_many_hlc_pair", highs_arr, lows_arr, closes_arr
    )
    if isinstance(batch_out, (tuple, list)) and len(batch_out) == 2:
        first_arr = np.asarray(batch_out[0], dtype=float)
        second_arr = np.asarray(batch_out[1], dtype=float)
        if (
            first_arr.shape[0] == closes_arr.shape[0]
            and second_arr.shape[0] == closes_arr.shape[0]
        ):
            return (
                pd.Series(first_arr, index=close_values.index, dtype=float),
                pd.Series(second_arr, index=close_values.index, dtype=float),
            )
    first = np.full(len(close_values), np.nan, dtype=float)
    second = np.full(len(close_values), np.nan, dtype=float)
    for idx, (high_v, low_v, close_v) in enumerate(
        zip(high_values, low_values, close_values)
    ):
        out = update_fn(float(high_v), float(low_v), float(close_v))
        if out is not None:
            first[idx], second[idx] = float(out[0]), float(out[1])
    return (
        pd.Series(first, index=close_values.index, dtype=float),
        pd.Series(second, index=close_values.index, dtype=float),
    )


def _run_rust_dual_series(
    first_values: pd.Series,
    second_values: pd.Series,
    update_fn: Callable[[float, float], float | None],
) -> pd.Series:
    first_arr = first_values.to_numpy(dtype=float, copy=False)
    second_arr = second_values.to_numpy(dtype=float, copy=False)
    batch_out = _batch_call(update_fn, "update_many_dual", first_arr, second_arr)
    if batch_out is not None:
        arr = np.asarray(batch_out, dtype=float)
        if arr.shape[0] == first_arr.shape[0]:
            return pd.Series(arr, index=first_values.index, dtype=float)
    arr = np.full(len(first_values), np.nan, dtype=float)
    for idx, (first_v, second_v) in enumerate(zip(first_values, second_values)):
        out = update_fn(float(first_v), float(second_v))
        if out is not None:
            arr[idx] = float(out)
    return pd.Series(arr, index=first_values.index, dtype=float)


def _run_rust_hlcv_series(
    high_values: pd.Series,
    low_values: pd.Series,
    close_values: pd.Series,
    volume_values: pd.Series,
    update_fn: Callable[[float, float, float, float], float | None],
) -> pd.Series:
    highs_arr = high_values.to_numpy(dtype=float, copy=False)
    lows_arr = low_values.to_numpy(dtype=float, copy=False)
    closes_arr = close_values.to_numpy(dtype=float, copy=False)
    volumes_arr = volume_values.to_numpy(dtype=float, copy=False)
    batch_out = _batch_call(
        update_fn, "update_many_hlcv", highs_arr, lows_arr, closes_arr, volumes_arr
    )
    if batch_out is not None:
        arr = np.asarray(batch_out, dtype=float)
        if arr.shape[0] == closes_arr.shape[0]:
            return pd.Series(arr, index=close_values.index, dtype=float)
    arr = np.full(len(close_values), np.nan, dtype=float)
    for idx, (high_v, low_v, close_v, volume_v) in enumerate(
        zip(high_values, low_values, close_values, volume_values)
    ):
        out = update_fn(float(high_v), float(low_v), float(close_v), float(volume_v))
        if out is not None:
            arr[idx] = float(out)
    return pd.Series(arr, index=close_values.index, dtype=float)


def _run_rust_hl_series(
    high_values: pd.Series,
    low_values: pd.Series,
    update_fn: Callable[[float, float], float | None],
) -> pd.Series:
    highs_arr = high_values.to_numpy(dtype=float, copy=False)
    lows_arr = low_values.to_numpy(dtype=float, copy=False)
    batch_out = _batch_call(update_fn, "update_many_hl", highs_arr, lows_arr)
    if batch_out is not None:
        arr = np.asarray(batch_out, dtype=float)
        if arr.shape[0] == highs_arr.shape[0]:
            return pd.Series(arr, index=high_values.index, dtype=float)
    arr = np.full(len(high_values), np.nan, dtype=float)
    for idx, (high_v, low_v) in enumerate(zip(high_values, low_values)):
        out = update_fn(float(high_v), float(low_v))
        if out is not None:
            arr[idx] = float(out)
    return pd.Series(arr, index=high_values.index, dtype=float)


def _run_rust_hl_pair_series(
    high_values: pd.Series,
    low_values: pd.Series,
    update_fn: Callable[[float, float], tuple[float, float] | None],
) -> tuple[pd.Series, pd.Series]:
    highs_arr = high_values.to_numpy(dtype=float, copy=False)
    lows_arr = low_values.to_numpy(dtype=float, copy=False)
    batch_out = _batch_call(update_fn, "update_many_hl_pair", highs_arr, lows_arr)
    if isinstance(batch_out, (tuple, list)) and len(batch_out) == 2:
        first_arr = np.asarray(batch_out[0], dtype=float)
        second_arr = np.asarray(batch_out[1], dtype=float)
        if (
            first_arr.shape[0] == highs_arr.shape[0]
            and second_arr.shape[0] == highs_arr.shape[0]
        ):
            return (
                pd.Series(first_arr, index=high_values.index, dtype=float),
                pd.Series(second_arr, index=high_values.index, dtype=float),
            )
    first = np.full(len(high_values), np.nan, dtype=float)
    second = np.full(len(high_values), np.nan, dtype=float)
    for idx, (high_v, low_v) in enumerate(zip(high_values, low_values)):
        out = update_fn(float(high_v), float(low_v))
        if out is not None:
            first[idx], second[idx] = float(out[0]), float(out[1])
    return (
        pd.Series(first, index=high_values.index, dtype=float),
        pd.Series(second, index=high_values.index, dtype=float),
    )


def _run_rust_ohlc_series(
    open_values: pd.Series,
    high_values: pd.Series,
    low_values: pd.Series,
    close_values: pd.Series,
    update_fn: Callable[[float, float, float, float], float | None],
) -> pd.Series:
    opens_arr = open_values.to_numpy(dtype=float, copy=False)
    highs_arr = high_values.to_numpy(dtype=float, copy=False)
    lows_arr = low_values.to_numpy(dtype=float, copy=False)
    closes_arr = close_values.to_numpy(dtype=float, copy=False)
    batch_out = _batch_call(
        update_fn, "update_many_ohlc", opens_arr, highs_arr, lows_arr, closes_arr
    )
    if batch_out is not None:
        arr = np.asarray(batch_out, dtype=float)
        if arr.shape[0] == closes_arr.shape[0]:
            return pd.Series(arr, index=close_values.index, dtype=float)
    arr = np.full(len(close_values), np.nan, dtype=float)
    for idx, (open_v, high_v, low_v, close_v) in enumerate(
        zip(open_values, high_values, low_values, close_values)
    ):
        out = update_fn(float(open_v), float(high_v), float(low_v), float(close_v))
        if out is not None:
            arr[idx] = float(out)
    return pd.Series(arr, index=close_values.index, dtype=float)


def _run_rust_single_pair_series(
    values: pd.Series,
    update_fn: Callable[[float], tuple[float, float] | None],
) -> tuple[pd.Series, pd.Series]:
    values_arr = values.to_numpy(dtype=float, copy=False)
    batch_out = _batch_call(update_fn, "update_many_pair", values_arr)
    if isinstance(batch_out, (tuple, list)) and len(batch_out) == 2:
        first_arr = np.asarray(batch_out[0], dtype=float)
        second_arr = np.asarray(batch_out[1], dtype=float)
        if (
            first_arr.shape[0] == values_arr.shape[0]
            and second_arr.shape[0] == values_arr.shape[0]
        ):
            return (
                pd.Series(first_arr, index=values.index, dtype=float),
                pd.Series(second_arr, index=values.index, dtype=float),
            )
    first = np.full(len(values), np.nan, dtype=float)
    second = np.full(len(values), np.nan, dtype=float)
    for idx, value in enumerate(values):
        out = update_fn(float(value))
        if out is not None:
            first[idx], second[idx] = float(out[0]), float(out[1])
    return (
        pd.Series(first, index=values.index, dtype=float),
        pd.Series(second, index=values.index, dtype=float),
    )


@skip_non_finite_rows()
def ROC(
    close: SeriesLike,
    timeperiod: int = 10,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Rate of Change (ROC)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustROC(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    shifted = cast(pd.Series, close_series.shift(use_period))
    out = cast(pd.Series, (close_series - shifted) / shifted * 100.0)
    # 与 TA-Lib 一致: 前值为 0 时输出 0(预热期的 NaN 保留)
    out = cast(pd.Series, out.where(shifted != 0.0, 0.0))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def ROCP(
    close: SeriesLike,
    timeperiod: int = 10,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Rate of change percentage (ROCP)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustROCP(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    shifted = cast(pd.Series, close_series.shift(use_period))
    out = cast(pd.Series, (close_series - shifted) / shifted)
    # 与 TA-Lib 一致: 前值为 0 时输出 0(预热期的 NaN 保留)
    out = cast(pd.Series, out.where(shifted != 0.0, 0.0))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def ROCR(
    close: SeriesLike,
    timeperiod: int = 10,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Rate of change ratio (ROCR)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustROCR(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    shifted = cast(pd.Series, close_series.shift(use_period))
    out = cast(pd.Series, close_series / shifted)
    out = cast(pd.Series, out.where(shifted != 0.0, 0.0))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def ROCR100(
    close: SeriesLike,
    timeperiod: int = 10,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Rate of change ratio scaled by 100 (ROCR100)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustROCR100(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    shifted = cast(pd.Series, close_series.shift(use_period))
    out = cast(pd.Series, (close_series / shifted) * 100.0)
    out = cast(pd.Series, out.where(shifted != 0.0, 0.0))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def WILLR(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Williams %R."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustWILLR(use_period)
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    result = _kernels.willr(
        _np(high_series), _np(low_series), _np(close_series), use_period
    )
    return finalize_output(_like(result, high_series), as_series=as_series)


@skip_non_finite_rows()
def CCI(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    c: float = 0.015,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Commodity Channel Index (CCI)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    if c <= 0:
        raise ValueError("c must be > 0")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustCCI(use_period, float(c))
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    result = _kernels.cci(
        _np(high_series), _np(low_series), _np(close_series), use_period, float(c)
    )
    return finalize_output(_like(result, high_series), as_series=as_series)


@skip_non_finite_rows()
def ADX(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Average Directional Index (ADX)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustADX(use_period)
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(
            _kernels.adx(
                _np(high_series), _np(low_series), _np(close_series), use_period
            ),
            close_series,
        ),
        as_series=as_series,
    )


@skip_non_finite_rows()
def DX(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Directional Movement Index (DX)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustDX(use_period)
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(
            _kernels.dx(
                _np(high_series), _np(low_series), _np(close_series), use_period
            ),
            close_series,
        ),
        as_series=as_series,
    )


@skip_non_finite_rows()
def PLUS_DI(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Plus Directional Indicator (+DI)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustPLUS_DI(use_period)
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(
            _kernels.plus_minus_di(
                _np(high_series), _np(low_series), _np(close_series), use_period
            )[0],
            close_series,
        ),
        as_series=as_series,
    )


@skip_non_finite_rows()
def MINUS_DI(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Minus Directional Indicator (-DI)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustMINUS_DI(use_period)
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(
            _kernels.plus_minus_di(
                _np(high_series), _np(low_series), _np(close_series), use_period
            )[1],
            close_series,
        ),
        as_series=as_series,
    )


@skip_non_finite_rows()
def ULTOSC(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Ultimate Oscillator (ULTOSC)."""
    backend_key = resolve_backend(backend)
    p1 = _ensure_period(timeperiod1, "timeperiod1")
    p2 = _ensure_period(timeperiod2, "timeperiod2")
    p3 = _ensure_period(timeperiod3, "timeperiod3")
    if not (p1 < p2 < p3):
        raise ValueError("require timeperiod1 < timeperiod2 < timeperiod3")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustULTOSC(p1, p2, p3)
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    result = _kernels.ultosc(
        _np(high_series), _np(low_series), _np(close_series), p1, p2, p3
    )
    return finalize_output(_like(result, high_series), as_series=as_series)


@skip_non_finite_rows()
def AROON(
    high: SeriesLike,
    low: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> tuple[pd.Series | object, pd.Series | object]:
    """Aroon oscillator components returning (aroondown, aroonup)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    if backend_key == "rust":
        indicator = RustAROON(use_period)
        down_s, up_s = _run_rust_hl_pair_series(
            high_series,
            low_series,
            indicator.update,
        )
        return (
            finalize_output(down_s, as_series=as_series),
            finalize_output(up_s, as_series=as_series),
        )

    down, up = _kernels.aroon(_np(high_series), _np(low_series), use_period)
    return (
        finalize_output(_like(down, high_series), as_series=as_series),
        finalize_output(_like(up, high_series), as_series=as_series),
    )


@skip_non_finite_rows()
def AROONOSC(
    high: SeriesLike,
    low: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Aroon oscillator (aroonup - aroondown)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    if backend_key == "rust":
        indicator = RustAROONOSC(use_period)
        out = _run_rust_hl_series(high_series, low_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    down, up = _kernels.aroon(_np(high_series), _np(low_series), use_period)
    return finalize_output(_like(up - down, high_series), as_series=as_series)


@skip_non_finite_rows()
def LINEARREG(
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Linear Regression endpoint value."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustLINEARREG(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    n = float(use_period)
    sum_x = n * (n - 1.0) / 2.0
    sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0
    denom = n * sum_x2 - sum_x * sum_x

    def _linreg(window: np.ndarray) -> float:
        sum_y = float(window.sum())
        sum_xy = float(np.dot(np.arange(use_period, dtype=float), window))
        if abs(denom) <= 1e-12:
            return float("nan")
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        return float(intercept + slope * (n - 1.0))

    out = cast(pd.Series, close_series.rolling(use_period).apply(_linreg, raw=True))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def LINEARREG_SLOPE(
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Linear Regression slope."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustLINEARREG_SLOPE(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    n = float(use_period)
    sum_x = n * (n - 1.0) / 2.0
    sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0
    denom = n * sum_x2 - sum_x * sum_x

    def _slope(window: np.ndarray) -> float:
        sum_y = float(window.sum())
        sum_xy = float(np.dot(np.arange(use_period, dtype=float), window))
        if abs(denom) <= 1e-12:
            return float("nan")
        return float((n * sum_xy - sum_x * sum_y) / denom)

    out = cast(pd.Series, close_series.rolling(use_period).apply(_slope, raw=True))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def LINEARREG_R2(
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Linear Regression R-squared."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustLINEARREG_R2(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    x = pd.Series(np.arange(len(close_series), dtype=float), index=close_series.index)
    out = cast(pd.Series, close_series.rolling(use_period).corr(x) ** 2)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def LINEARREG_INTERCEPT(
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Linear Regression intercept."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustLINEARREG_INTERCEPT(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    n = float(use_period)
    sum_x = n * (n - 1.0) / 2.0
    sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0
    denom = n * sum_x2 - sum_x * sum_x

    def _intercept(window: np.ndarray) -> float:
        sum_y = float(window.sum())
        sum_xy = float(np.dot(np.arange(use_period, dtype=float), window))
        if abs(denom) <= 1e-12:
            return float("nan")
        slope = (n * sum_xy - sum_x * sum_y) / denom
        return float((sum_y - slope * sum_x) / n)

    out = cast(
        pd.Series,
        close_series.rolling(use_period).apply(_intercept, raw=True),
    )
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def LINEARREG_ANGLE(
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Linear Regression angle in degrees."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustLINEARREG_ANGLE(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    slope = cast(
        pd.Series,
        LINEARREG_SLOPE(
            close_series,
            timeperiod=use_period,
            as_series=True,
            backend="python",
        ),
    )
    out = cast(pd.Series, np.degrees(np.arctan(slope)))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def TSF(
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Time Series Forecast (TSF)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustTSF(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    n = float(use_period)
    sum_x = n * (n - 1.0) / 2.0
    sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0
    denom = n * sum_x2 - sum_x * sum_x

    def _tsf(window: np.ndarray) -> float:
        sum_y = float(window.sum())
        sum_xy = float(np.dot(np.arange(use_period, dtype=float), window))
        if abs(denom) <= 1e-12:
            return float("nan")
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        return float(intercept + slope * n)

    out = cast(pd.Series, close_series.rolling(use_period).apply(_tsf, raw=True))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def CORREL(
    real0: SeriesLike,
    real1: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Pearson correlation coefficient (CORREL)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    series0 = to_series(real0, name="real0")
    series1 = to_series(real1, name="real1")
    if backend_key == "rust":
        indicator = RustCORREL(use_period)
        out = _run_rust_dual_series(series0, series1, indicator.update)
        return finalize_output(out, as_series=as_series)
    result = _kernels.correl(_np(series0), _np(series1), use_period)
    return finalize_output(_like(result, series0), as_series=as_series)


@skip_non_finite_rows()
def BETA(
    real0: SeriesLike,
    real1: SeriesLike,
    timeperiod: int = 5,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Beta coefficient of real0 relative to real1."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    series0 = to_series(real0, name="real0")
    series1 = to_series(real1, name="real1")
    if backend_key == "rust":
        indicator = RustBETA(use_period)
        out = _run_rust_dual_series(series0, series1, indicator.update)
        return finalize_output(out, as_series=as_series)
    result = _kernels.beta(_np(series0), _np(series1), use_period)
    return finalize_output(_like(result, series0), as_series=as_series)


@skip_non_finite_rows()
def COVAR(
    real0: SeriesLike,
    real1: SeriesLike,
    timeperiod: int = 5,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return rolling covariance (COVAR)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    series0 = to_series(real0, name="real0")
    series1 = to_series(real1, name="real1")
    if backend_key == "rust":
        indicator = RustCOVAR(use_period)
        out = _run_rust_dual_series(series0, series1, indicator.update)
        return finalize_output(out, as_series=as_series)
    mean0 = cast(pd.Series, series0.rolling(use_period).mean())
    mean1 = cast(pd.Series, series1.rolling(use_period).mean())
    out = cast(
        pd.Series,
        ((series0 - mean0) * (series1 - mean1)).rolling(use_period).mean(),
    )
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def ADXR(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Average Directional Movement Index Rating (ADXR)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustADXR(use_period)
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(
            _kernels.adxr(
                _np(high_series), _np(low_series), _np(close_series), use_period
            ),
            close_series,
        ),
        as_series=as_series,
    )


@skip_non_finite_rows()
def RSI(
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Relative Strength Index (RSI)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustRSI(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(_kernels.rsi(_np(close_series), use_period), close_series),
        as_series=as_series,
    )


@skip_non_finite_rows()
def SMA(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Calculate simple moving average (SMA)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustSMA(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = _rolling_mean(close_series, use_period)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def MA(
    close: SeriesLike,
    timeperiod: int = 30,
    matype: int = 0,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return a moving average, currently limited to SMA-compatible `matype`."""
    if matype != 0:
        raise ValueError("only matype=0 (SMA) is supported")
    return SMA(
        close,
        timeperiod=timeperiod,
        period=period,
        as_series=as_series,
        backend=backend,
    )


@skip_non_finite_rows()
def EMA(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Exponential Moving Average (EMA)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustEMA(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(_kernels.ema(_np(close_series), use_period), close_series),
        as_series=as_series,
    )


@skip_non_finite_rows()
def APO(
    close: SeriesLike,
    fastperiod: int = 12,
    slowperiod: int = 26,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Absolute Price Oscillator (APO)."""
    backend_key = resolve_backend(backend)
    fast_p = _ensure_period(fastperiod, "fastperiod")
    slow_p = _ensure_period(slowperiod, "slowperiod")
    if slow_p <= fast_p:
        raise ValueError("slowperiod must be > fastperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustAPO(fast_p, slow_p)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(_kernels.apo(_np(close_series), fast_p, slow_p), close_series),
        as_series=as_series,
    )


@skip_non_finite_rows()
def PPO(
    close: SeriesLike,
    fastperiod: int = 12,
    slowperiod: int = 26,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Percentage Price Oscillator (PPO)."""
    backend_key = resolve_backend(backend)
    fast_p = _ensure_period(fastperiod, "fastperiod")
    slow_p = _ensure_period(slowperiod, "slowperiod")
    if slow_p <= fast_p:
        raise ValueError("slowperiod must be > fastperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustPPO(fast_p, slow_p)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(_kernels.ppo(_np(close_series), fast_p, slow_p), close_series),
        as_series=as_series,
    )


@skip_non_finite_rows()
def WMA(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Weighted Moving Average (WMA)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustWMA(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    weights = np.arange(1.0, use_period + 1.0, dtype=float)
    denom = float(weights.sum())
    out = cast(
        pd.Series,
        close_series.rolling(use_period).apply(
            lambda arr: float(np.dot(arr, weights) / denom),
            raw=True,
        ),
    )
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def TRIMA(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Triangular Moving Average (TRIMA)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustTRIMA(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    if use_period % 2 == 1:
        peak = use_period // 2 + 1
        weights = np.array(
            [i + 1 if i < peak else use_period - i for i in range(use_period)],
            dtype=float,
        )
    else:
        peak = use_period // 2
        weights = np.array(
            [i + 1 if i < peak else use_period - i for i in range(use_period)],
            dtype=float,
        )
    denom = float(weights.sum())
    out = cast(
        pd.Series,
        close_series.rolling(use_period).apply(
            lambda arr: float(np.dot(arr, weights) / denom),
            raw=True,
        ),
    )
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def MIDPOINT(
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """MidPoint over rolling high/low of close."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustMIDPOINT(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    roll_max = cast(pd.Series, close_series.rolling(use_period).max())
    roll_min = cast(pd.Series, close_series.rolling(use_period).min())
    out = cast(pd.Series, (roll_max + roll_min) / 2.0)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def MAX(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Highest value over rolling window (MAX)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustMAX(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, close_series.rolling(use_period).max())
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def MIN(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Lowest value over rolling window (MIN)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustMIN(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, close_series.rolling(use_period).min())
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows(index_outputs=True)
def MAXINDEX(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Absolute index of rolling maximum (MAXINDEX)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustMAXINDEX(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    result = _kernels.extreme_index(_np(close_series), use_period, True)
    return finalize_output(_like(result, close_series), as_series=as_series)


@skip_non_finite_rows(index_outputs=True)
def MININDEX(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Absolute index of rolling minimum (MININDEX)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustMININDEX(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    result = _kernels.extreme_index(_np(close_series), use_period, False)
    return finalize_output(_like(result, close_series), as_series=as_series)


@skip_non_finite_rows(index_outputs=True)
def MINMAXINDEX(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> tuple[pd.Series | object, pd.Series | object]:
    """Return rolling minimum and maximum index pair (MINMAXINDEX)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustMINMAXINDEX(use_period)
        min_s, max_s = _run_rust_single_pair_series(close_series, indicator.update)
        return (
            finalize_output(min_s, as_series=as_series),
            finalize_output(max_s, as_series=as_series),
        )
    values = _np(close_series)
    min_idx = _kernels.extreme_index(values, use_period, False)
    max_idx = _kernels.extreme_index(values, use_period, True)
    return (
        finalize_output(_like(min_idx, close_series), as_series=as_series),
        finalize_output(_like(max_idx, close_series), as_series=as_series),
    )


@skip_non_finite_rows()
def MINMAX(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> tuple[pd.Series | object, pd.Series | object]:
    """Return rolling minimum and maximum pair (MINMAX)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustMINMAX(use_period)
        min_s, max_s = _run_rust_single_pair_series(close_series, indicator.update)
        return (
            finalize_output(min_s, as_series=as_series),
            finalize_output(max_s, as_series=as_series),
        )
    min_s = cast(pd.Series, close_series.rolling(use_period).min())
    max_s = cast(pd.Series, close_series.rolling(use_period).max())
    return (
        finalize_output(min_s, as_series=as_series),
        finalize_output(max_s, as_series=as_series),
    )


@skip_non_finite_rows()
def SUM(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return rolling sum (SUM)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustSUM(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, close_series.rolling(use_period).sum())
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def AVGDEV(
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return average absolute deviation (AVGDEV)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustAVGDEV(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)

    def _avgdev(window: np.ndarray) -> float:
        mean = float(window.mean())
        return float(np.mean(np.abs(window - mean)))

    out = cast(pd.Series, close_series.rolling(use_period).apply(_avgdev, raw=True))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def RANGE(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return rolling high-low range (RANGE)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustRANGE(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    roll_max = cast(pd.Series, close_series.rolling(use_period).max())
    roll_min = cast(pd.Series, close_series.rolling(use_period).min())
    out = cast(pd.Series, roll_max - roll_min)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def LN(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return natural logarithm transform (LN)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustLN()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.log(close_series.where(close_series > 0.0, np.nan)))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def LOG10(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return base-10 logarithm transform (LOG10)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustLOG10()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.log10(close_series.where(close_series > 0.0, np.nan)))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def SQRT(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return square-root transform (SQRT)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustSQRT()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.sqrt(close_series.where(close_series >= 0.0, np.nan)))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def CEIL(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return ceiling transform (CEIL)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustCEIL()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.ceil(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def FLOOR(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return floor transform (FLOOR)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustFLOOR()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.floor(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def SIN(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return sine transform (SIN)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustSIN()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.sin(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def COS(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return cosine transform (COS)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustCOS()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.cos(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def TAN(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return tangent transform (TAN)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustTAN()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.tan(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def ASIN(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return arcsine transform (ASIN)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustASIN()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    valid = close_series.where(close_series.abs() <= 1.0, np.nan)
    out = cast(pd.Series, np.arcsin(valid))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def ACOS(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return arccosine transform (ACOS)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustACOS()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    valid = close_series.where(close_series.abs() <= 1.0, np.nan)
    out = cast(pd.Series, np.arccos(valid))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def ATAN(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return arctangent transform (ATAN)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustATAN()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.arctan(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def SINH(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return hyperbolic sine transform (SINH)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustSINH()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.sinh(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def COSH(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return hyperbolic cosine transform (COSH)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustCOSH()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.cosh(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def TANH(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return hyperbolic tangent transform (TANH)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustTANH()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.tanh(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def EXP(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return exponential transform (EXP)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustEXP()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.exp(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def ABS(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return absolute-value transform (ABS)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustABS()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.abs(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def SIGN(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return sign transform (SIGN)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustSIGN()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.sign(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def ADD(
    real0: SeriesLike,
    real1: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise addition (ADD)."""
    backend_key = resolve_backend(backend)
    series0 = to_series(real0, name="real0")
    series1 = to_series(real1, name="real1")
    if backend_key == "rust":
        indicator = RustADD()
        out = _run_rust_dual_series(series0, series1, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, series0 + series1)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def SUB(
    real0: SeriesLike,
    real1: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise subtraction (SUB)."""
    backend_key = resolve_backend(backend)
    series0 = to_series(real0, name="real0")
    series1 = to_series(real1, name="real1")
    if backend_key == "rust":
        indicator = RustSUB()
        out = _run_rust_dual_series(series0, series1, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, series0 - series1)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def MULT(
    real0: SeriesLike,
    real1: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise multiplication (MULT)."""
    backend_key = resolve_backend(backend)
    series0 = to_series(real0, name="real0")
    series1 = to_series(real1, name="real1")
    if backend_key == "rust":
        indicator = RustMULT()
        out = _run_rust_dual_series(series0, series1, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, series0 * series1)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def DIV(
    real0: SeriesLike,
    real1: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise division (DIV)."""
    backend_key = resolve_backend(backend)
    series0 = to_series(real0, name="real0")
    series1 = to_series(real1, name="real1")
    if backend_key == "rust":
        indicator = RustDIV()
        out = _run_rust_dual_series(series0, series1, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, series0 / series1)
    out = cast(pd.Series, out.where(series1.abs() > 1e-12, np.nan))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def MAX2(
    real0: SeriesLike,
    real1: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise maximum (MAX2)."""
    backend_key = resolve_backend(backend)
    series0 = to_series(real0, name="real0")
    series1 = to_series(real1, name="real1")
    if backend_key == "rust":
        indicator = RustMAX2()
        out = _run_rust_dual_series(series0, series1, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.maximum(series0, series1))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def MIN2(
    real0: SeriesLike,
    real1: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise minimum (MIN2)."""
    backend_key = resolve_backend(backend)
    series0 = to_series(real0, name="real0")
    series1 = to_series(real1, name="real1")
    if backend_key == "rust":
        indicator = RustMIN2()
        out = _run_rust_dual_series(series0, series1, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.minimum(series0, series1))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def CLIP(
    real: SeriesLike,
    min_value: SeriesLike,
    max_value: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise clipping (CLIP)."""
    backend_key = resolve_backend(backend)
    series = to_series(real, name="real")
    min_series = to_series(min_value, name="min_value")
    max_series = to_series(max_value, name="max_value")
    if backend_key == "rust":
        indicator = RustCLIP()
        values_arr = series.to_numpy(dtype=float, copy=False)
        min_arr = min_series.to_numpy(dtype=float, copy=False)
        max_arr = max_series.to_numpy(dtype=float, copy=False)
        batch_out = _batch_call(
            indicator.update, "update_many_clip", values_arr, min_arr, max_arr
        )
        if batch_out is not None:
            arr = np.asarray(batch_out, dtype=float)
            if arr.shape[0] != len(series):
                arr = np.full(len(series), np.nan, dtype=float)
        else:
            arr = np.full(len(series), np.nan, dtype=float)
            for idx, (value, min_v, max_v) in enumerate(
                zip(series, min_series, max_series)
            ):
                out_val = indicator.update(float(value), float(min_v), float(max_v))
                if out_val is not None:
                    arr[idx] = float(out_val)
        out_series = pd.Series(arr, index=series.index, dtype=float)
        return finalize_output(out_series, as_series=as_series)
    lo = cast(pd.Series, np.minimum(min_series, max_series))
    hi = cast(pd.Series, np.maximum(min_series, max_series))
    out = cast(pd.Series, np.minimum(np.maximum(series, lo), hi))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def ROUND(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise round (ROUND)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustROUND()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.round(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def POW(
    real0: SeriesLike,
    real1: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise power (POW)."""
    backend_key = resolve_backend(backend)
    series0 = to_series(real0, name="real0")
    series1 = to_series(real1, name="real1")
    if backend_key == "rust":
        indicator = RustPOW()
        out = _run_rust_dual_series(series0, series1, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.power(series0, series1))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def MOD(
    real0: SeriesLike,
    real1: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise modulo (MOD)."""
    backend_key = resolve_backend(backend)
    series0 = to_series(real0, name="real0")
    series1 = to_series(real1, name="real1")
    if backend_key == "rust":
        indicator = RustMOD()
        out = _run_rust_dual_series(series0, series1, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.mod(series0, series1))
    out = cast(pd.Series, out.where(series1.abs() > 1e-12, np.nan))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def CLAMP01(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return clamp-to-[0,1] transform (CLAMP01)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustCLAMP01()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.clip(close_series, 0.0, 1.0))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def SQ(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise square (SQ)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustSQ()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.square(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def CUBE(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return element-wise cube (CUBE)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustCUBE()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.power(close_series, 3.0))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def RECIP(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return reciprocal transform (RECIP)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustRECIP()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, 1.0 / close_series)
    out = cast(pd.Series, out.where(close_series.abs() > 1e-12, np.nan))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def INV_SQRT(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return inverse square-root transform (INV_SQRT)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustINV_SQRT()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    valid = close_series.where(close_series > 0.0, np.nan)
    out = cast(pd.Series, 1.0 / np.sqrt(valid))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def LOG1P(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return natural logarithm of one plus input (LOG1P)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustLOG1P()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.log1p(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def EXPM1(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return exponential minus one transform (EXPM1)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustEXPM1()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.expm1(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def DEG2RAD(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Return degree-to-radian transform (DEG2RAD)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustDEG2RAD()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, np.deg2rad(close_series))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def HT_TRENDLINE(
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Hilbert Transform instantaneous trendline (TA-Lib 算法)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustHT_TRENDLINE()
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    result = _kernels.ht_trendline(_np(close_series))
    return finalize_output(_like(result, close_series), as_series=as_series)


@skip_non_finite_rows()
def ATR(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Average True Range (ATR)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustATR(use_period)
        out_series = _run_rust_hlc_series(
            high_series, low_series, close_series, indicator.update
        )
        return finalize_output(out_series, as_series=as_series)
    return finalize_output(
        _like(
            _kernels.atr(
                _np(high_series), _np(low_series), _np(close_series), use_period
            ),
            close_series,
        ),
        as_series=as_series,
    )


@skip_non_finite_rows()
def TRANGE(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Calculate True Range (TRANGE)."""
    backend_key = resolve_backend(backend)
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustTRANGE()
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(
            _kernels.trange(_np(high_series), _np(low_series), _np(close_series)),
            close_series,
        ),
        as_series=as_series,
    )


@skip_non_finite_rows()
def MEDPRICE(
    high: SeriesLike,
    low: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Median Price (MEDPRICE)."""
    backend_key = resolve_backend(backend)
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    if backend_key == "rust":
        indicator = RustMEDPRICE()
        out = _run_rust_hl_series(high_series, low_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, (high_series + low_series) / 2.0)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def AVGPRICE(
    open: SeriesLike,
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Average Price (AVGPRICE)."""
    backend_key = resolve_backend(backend)
    open_series = to_series(open, name="open")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustAVGPRICE()
        out = _run_rust_ohlc_series(
            open_series,
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, (open_series + high_series + low_series + close_series) / 4.0)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def TYPPRICE(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Typical Price (TYPPRICE)."""
    backend_key = resolve_backend(backend)
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustTYPPRICE()
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, (high_series + low_series + close_series) / 3.0)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def WCLPRICE(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Weighted Close Price (WCLPRICE)."""
    backend_key = resolve_backend(backend)
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustWCLPRICE()
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, (high_series + low_series + 2.0 * close_series) / 4.0)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def MIDPRICE(
    high: SeriesLike,
    low: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """MidPrice over rolling high/low."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    if backend_key == "rust":
        indicator = RustMIDPRICE(use_period)
        out = _run_rust_hl_series(high_series, low_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    roll_high = cast(pd.Series, high_series.rolling(use_period).max())
    roll_low = cast(pd.Series, low_series.rolling(use_period).min())
    out = cast(pd.Series, (roll_high + roll_low) / 2.0)
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def MACD(
    close: SeriesLike,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> tuple[pd.Series | object, pd.Series | object, pd.Series | object]:
    """Calculate moving average convergence divergence."""
    backend_key = resolve_backend(backend)
    fast_p = _ensure_period(fastperiod, "fastperiod")
    slow_p = _ensure_period(slowperiod, "slowperiod")
    signal_p = _ensure_period(signalperiod, "signalperiod")
    close_series = to_series(close, name="close")
    if slow_p <= fast_p:
        raise ValueError("slowperiod must be > fastperiod")
    if backend_key == "rust":
        indicator = RustMACD(fast_p, slow_p, signal_p)
        close_arr = close_series.to_numpy(dtype=float, copy=False)
        batch_out = _batch_call(indicator.update, "update_many", close_arr)
        dif = np.full(len(close_series), np.nan, dtype=float)
        dea = np.full(len(close_series), np.nan, dtype=float)
        hist = np.full(len(close_series), np.nan, dtype=float)
        if isinstance(batch_out, (tuple, list)) and len(batch_out) == 3:
            dif_arr = np.asarray(batch_out[0], dtype=float)
            dea_arr = np.asarray(batch_out[1], dtype=float)
            hist_arr = np.asarray(batch_out[2], dtype=float)
            if (
                dif_arr.shape[0] == len(close_series)
                and dea_arr.shape[0] == len(close_series)
                and hist_arr.shape[0] == len(close_series)
            ):
                dif, dea, hist = dif_arr, dea_arr, hist_arr
        else:
            for idx, value in enumerate(close_series):
                out = indicator.update(float(value))
                if out is not None:
                    dif[idx], dea[idx], hist[idx] = (
                        float(out[0]),
                        float(out[1]),
                        float(out[2]),
                    )
        dif_s = pd.Series(dif, index=close_series.index, dtype=float)
        dea_s = pd.Series(dea, index=close_series.index, dtype=float)
        hist_s = pd.Series(hist, index=close_series.index, dtype=float)
        return (
            finalize_output(dif_s, as_series=as_series),
            finalize_output(dea_s, as_series=as_series),
            finalize_output(hist_s, as_series=as_series),
        )
    dif, dea, hist = _kernels.macd(_np(close_series), fast_p, slow_p, signal_p)
    return (
        finalize_output(_like(dif, close_series), as_series=as_series),
        finalize_output(_like(dea, close_series), as_series=as_series),
        finalize_output(_like(hist, close_series), as_series=as_series),
    )


@skip_non_finite_rows()
def MAMA(
    close: SeriesLike,
    fastlimit: float = 0.5,
    slowlimit: float = 0.05,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> tuple[pd.Series | object, pd.Series | object]:
    """MESA Adaptive Moving Average returning (mama, fama)."""
    backend_key = resolve_backend(backend)
    if fastlimit <= 0.0 or slowlimit <= 0.0:
        raise ValueError("fastlimit and slowlimit must be > 0")
    if fastlimit < slowlimit:
        raise ValueError("fastlimit must be >= slowlimit")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustMAMA(float(fastlimit), float(slowlimit))
        mama_s, fama_s = _run_rust_single_pair_series(close_series, indicator.update)
        return (
            finalize_output(mama_s, as_series=as_series),
            finalize_output(fama_s, as_series=as_series),
        )
    mama_v, fama_v = _kernels.mama(
        _np(close_series), float(fastlimit), float(slowlimit)
    )
    return (
        finalize_output(_like(mama_v, close_series), as_series=as_series),
        finalize_output(_like(fama_v, close_series), as_series=as_series),
    )


@skip_non_finite_rows()
def BBANDS(
    close: SeriesLike,
    timeperiod: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> tuple[pd.Series | object, pd.Series | object, pd.Series | object]:
    """Bollinger Bands returning (upper, middle, lower)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    if matype != 0:
        raise ValueError("only matype=0 (SMA) is supported")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        if abs(nbdevup - nbdevdn) > 1e-12:
            raise ValueError("rust backend currently requires nbdevup == nbdevdn")
        indicator = RustBollingerBands(use_period, float(nbdevup))
        close_arr = close_series.to_numpy(dtype=float, copy=False)
        batch_out = _batch_call(indicator.update, "update_many", close_arr)
        upper = np.full(len(close_series), np.nan, dtype=float)
        middle = np.full(len(close_series), np.nan, dtype=float)
        lower = np.full(len(close_series), np.nan, dtype=float)
        if isinstance(batch_out, (tuple, list)) and len(batch_out) == 3:
            upper_arr = np.asarray(batch_out[0], dtype=float)
            middle_arr = np.asarray(batch_out[1], dtype=float)
            lower_arr = np.asarray(batch_out[2], dtype=float)
            if (
                upper_arr.shape[0] == len(close_series)
                and middle_arr.shape[0] == len(close_series)
                and lower_arr.shape[0] == len(close_series)
            ):
                upper, middle, lower = upper_arr, middle_arr, lower_arr
        else:
            for idx, value in enumerate(close_series):
                out = indicator.update(float(value))
                if out is not None:
                    upper[idx], middle[idx], lower[idx] = (
                        float(out[0]),
                        float(out[1]),
                        float(out[2]),
                    )
        upper_s = pd.Series(upper, index=close_series.index, dtype=float)
        middle_s = pd.Series(middle, index=close_series.index, dtype=float)
        lower_s = pd.Series(lower, index=close_series.index, dtype=float)
        return (
            finalize_output(upper_s, as_series=as_series),
            finalize_output(middle_s, as_series=as_series),
            finalize_output(lower_s, as_series=as_series),
        )
    upper, middle, lower = _kernels.bbands(
        _np(close_series), use_period, float(nbdevup), float(nbdevdn)
    )
    return (
        finalize_output(_like(upper, close_series), as_series=as_series),
        finalize_output(_like(middle, close_series), as_series=as_series),
        finalize_output(_like(lower, close_series), as_series=as_series),
    )


@skip_non_finite_rows()
def STDDEV(
    close: SeriesLike,
    timeperiod: int = 5,
    nbdev: float = 1.0,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Compute standard deviation (STDDEV)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustSTDDEV(use_period, float(nbdev))
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    result = _kernels.stddev(_np(close_series), use_period, float(nbdev))
    return finalize_output(_like(result, close_series), as_series=as_series)


@skip_non_finite_rows()
def VAR(
    close: SeriesLike,
    timeperiod: int = 5,
    nbdev: float = 1.0,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Variance (VAR)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustVAR(use_period, float(nbdev))
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    result = _kernels.variance(_np(close_series), use_period)
    return finalize_output(_like(result, close_series), as_series=as_series)


@skip_non_finite_rows()
def STOCH(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_matype: int = 0,
    slowd_period: int = 3,
    slowd_matype: int = 0,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> tuple[pd.Series | object, pd.Series | object]:
    """Stochastic oscillator returning (slowk, slowd)."""
    backend_key = resolve_backend(backend)
    if slowk_matype != 0 or slowd_matype != 0:
        raise ValueError("only matype=0 (SMA) is supported")
    k_period = _ensure_period(fastk_period, "fastk_period")
    k_smooth = _ensure_period(slowk_period, "slowk_period")
    d_period = _ensure_period(slowd_period, "slowd_period")

    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustSTOCH(k_period, k_smooth, d_period)
        slow_k, slow_d = _run_rust_hlc_pair_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return (
            finalize_output(slow_k, as_series=as_series),
            finalize_output(slow_d, as_series=as_series),
        )

    k_values, d_values = _kernels.stoch(
        _np(high_series),
        _np(low_series),
        _np(close_series),
        k_period,
        k_smooth,
        d_period,
    )
    return (
        finalize_output(_like(k_values, high_series), as_series=as_series),
        finalize_output(_like(d_values, high_series), as_series=as_series),
    )


@skip_non_finite_rows()
def MOM(
    close: SeriesLike,
    timeperiod: int = 10,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Momentum (MOM)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustMOM(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    out = cast(pd.Series, close_series.diff(use_period))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def CMO(
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Chande Momentum Oscillator (CMO)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustCMO(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(_kernels.cmo(_np(close_series), use_period), close_series),
        as_series=as_series,
    )


@skip_non_finite_rows()
def OBV(
    close: SeriesLike,
    volume: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """On Balance Volume (OBV)."""
    backend_key = resolve_backend(backend)
    close_series = to_series(close, name="close")
    volume_series = to_series(volume, name="volume")
    if backend_key == "rust":
        indicator = RustOBV()
        out = _run_rust_dual_series(close_series, volume_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    result = _kernels.obv(_np(close_series), _np(volume_series))
    return finalize_output(_like(result, close_series), as_series=as_series)


@skip_non_finite_rows()
def AD(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    volume: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Chaikin A/D Line (AD)."""
    backend_key = resolve_backend(backend)
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    volume_series = to_series(volume, name="volume")
    if backend_key == "rust":
        indicator = RustAD()
        out = _run_rust_hlcv_series(
            high_series,
            low_series,
            close_series,
            volume_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    denom = cast(pd.Series, high_series - low_series)
    mfm = cast(
        pd.Series,
        ((close_series - low_series) - (high_series - close_series)) / denom,
    )
    mfm = cast(pd.Series, mfm.where(denom.abs() > 1e-12, 0.0))
    out = cast(pd.Series, (mfm * volume_series).cumsum())
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def ADOSC(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    volume: SeriesLike,
    fastperiod: int = 3,
    slowperiod: int = 10,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Chaikin A/D Oscillator (ADOSC)."""
    backend_key = resolve_backend(backend)
    fast_p = _ensure_period(fastperiod, "fastperiod")
    slow_p = _ensure_period(slowperiod, "slowperiod")
    if slow_p <= fast_p:
        raise ValueError("slowperiod must be > fastperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    volume_series = to_series(volume, name="volume")
    if backend_key == "rust":
        indicator = RustADOSC(fast_p, slow_p)
        out = _run_rust_hlcv_series(
            high_series,
            low_series,
            close_series,
            volume_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    ad_line = cast(
        pd.Series,
        AD(
            high_series,
            low_series,
            close_series,
            volume_series,
            as_series=True,
            backend="python",
        ),
    )
    fast_ema = cast(
        pd.Series,
        ad_line.ewm(alpha=2.0 / (fast_p + 1.0), adjust=False).mean(),
    )
    slow_ema = cast(
        pd.Series,
        ad_line.ewm(alpha=2.0 / (slow_p + 1.0), adjust=False).mean(),
    )
    out = cast(pd.Series, fast_ema - slow_ema)
    out.iloc[: slow_p - 1] = np.nan
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def BOP(
    open: SeriesLike,
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Balance Of Power (BOP)."""
    backend_key = resolve_backend(backend)
    open_series = to_series(open, name="open")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustBOP()
        out = _run_rust_ohlc_series(
            open_series,
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    denominator = cast(pd.Series, high_series - low_series)
    out = cast(pd.Series, (close_series - open_series) / denominator)
    out = cast(pd.Series, out.where(denominator.abs() > 1e-12, 0.0))
    return finalize_output(out, as_series=as_series)


@skip_non_finite_rows()
def DEMA(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Double Exponential Moving Average (DEMA)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustDEMA(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(_kernels.dema(_np(close_series), use_period), close_series),
        as_series=as_series,
    )


@skip_non_finite_rows()
def TRIX(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Triple Exponential Moving Average Rate of Change (TRIX)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustTRIX(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(_kernels.trix(_np(close_series), use_period), close_series),
        as_series=as_series,
    )


@skip_non_finite_rows()
def MFI(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    volume: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Money Flow Index (MFI)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    volume_series = to_series(volume, name="volume")
    if backend_key == "rust":
        indicator = RustMFI(use_period)
        out = _run_rust_hlcv_series(
            high_series,
            low_series,
            close_series,
            volume_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    result = _kernels.mfi(
        _np(high_series),
        _np(low_series),
        _np(close_series),
        _np(volume_series),
        use_period,
    )
    return finalize_output(_like(result, high_series), as_series=as_series)


@skip_non_finite_rows()
def TEMA(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Triple Exponential Moving Average (TEMA)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustTEMA(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(_kernels.tema(_np(close_series), use_period), close_series),
        as_series=as_series,
    )


@skip_non_finite_rows()
def T3(
    close: SeriesLike,
    timeperiod: int = 5,
    vfactor: float = 0.7,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Triple Exponential Moving Average T3."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustT3(use_period, float(vfactor))
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(_kernels.t3(_np(close_series), use_period, float(vfactor)), close_series),
        as_series=as_series,
    )


@skip_non_finite_rows()
def KAMA(
    close: SeriesLike,
    timeperiod: int = 30,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Kaufman Adaptive Moving Average (KAMA)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustKAMA(use_period)
        out = _run_rust_single_series(close_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(_kernels.kama(_np(close_series), use_period), close_series),
        as_series=as_series,
    )


@skip_non_finite_rows()
def NATR(
    high: SeriesLike,
    low: SeriesLike,
    close: SeriesLike,
    timeperiod: int = 14,
    *,
    period: int | None = None,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Calculate normalized average true range (NATR)."""
    backend_key = resolve_backend(backend)
    use_period = _ensure_period(period or timeperiod, "timeperiod")
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    close_series = to_series(close, name="close")
    if backend_key == "rust":
        indicator = RustNATR(use_period)
        out = _run_rust_hlc_series(
            high_series,
            low_series,
            close_series,
            indicator.update,
        )
        return finalize_output(out, as_series=as_series)
    return finalize_output(
        _like(
            _kernels.natr(
                _np(high_series), _np(low_series), _np(close_series), use_period
            ),
            close_series,
        ),
        as_series=as_series,
    )


@skip_non_finite_rows()
def SAR(
    high: SeriesLike,
    low: SeriesLike,
    acceleration: float = 0.02,
    maximum: float = 0.2,
    *,
    as_series: bool = False,
    backend: str = "auto",
) -> pd.Series | object:
    """Parabolic SAR."""
    if acceleration <= 0.0 or maximum <= 0.0:
        raise ValueError("acceleration and maximum must be > 0")
    if acceleration > maximum:
        raise ValueError("acceleration must be <= maximum")
    backend_key = resolve_backend(backend)
    high_series = to_series(high, name="high")
    low_series = to_series(low, name="low")
    if backend_key == "rust":
        indicator = RustSAR(float(acceleration), float(maximum))
        out = _run_rust_hl_series(high_series, low_series, indicator.update)
        return finalize_output(out, as_series=as_series)
    result = _kernels.sar(
        _np(high_series), _np(low_series), float(acceleration), float(maximum)
    )
    return finalize_output(_like(result, high_series), as_series=as_series)
