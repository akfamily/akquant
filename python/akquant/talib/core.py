"""TA-Lib compatibility core helpers."""

import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, Union, cast

import numpy as np
import pandas as pd

SeriesLike = Union[pd.Series, np.ndarray, Sequence[float]]


def to_series(values: SeriesLike, name: str | None = None) -> pd.Series:
    """Convert input into float64 pandas Series."""
    if isinstance(values, pd.Series):
        series = values.astype(float)
        if name:
            series = series.rename(name)
        return series
    array = np.asarray(values, dtype=float)
    return pd.Series(array, name=name, dtype=float)


def to_numpy(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert series/array to float64 numpy array."""
    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False)
    return cast(np.ndarray, values.to_numpy(dtype=float, copy=False))


def finalize_output(
    values: pd.Series | np.ndarray,
    *,
    as_series: bool,
) -> pd.Series | np.ndarray:
    """Finalize output as pandas Series or numpy array."""
    if as_series:
        if isinstance(values, pd.Series):
            return values
        return pd.Series(values, dtype=float)
    return to_numpy(values)


_F = TypeVar("_F", bound=Callable[..., Any])


def skip_non_finite_rows(*, index_outputs: bool = False) -> Callable[[_F], _F]:
    """非有限值(NaN/±inf)所在行不参与计算, 结果对应位置记 NaN.

    凡是注解为 ``SeriesLike`` 的参数都视为输入列; 只要某一行任一输入列非有限,
    这一行就整行剔除, 用剩下的行调用原函数, 再按原位置散回。效果等同于指标从
    未见过这些行——与 TA-Lib 对开头 NaN 的处理一致, 中段 NaN 则不会像 TA-Lib
    那样把之后的输出全部污染成 NaN。输入全部有限时原样调用, 不做任何拷贝。

    :param index_outputs: 输出是下标(MAXINDEX 等)时置 True, 把压缩后的下标
        换算回原序列的下标
    """

    def decorate(func: _F) -> _F:
        signature = inspect.signature(func)
        series_params = [
            name
            for name, param in signature.parameters.items()
            if param.annotation is SeriesLike
        ]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arrays: dict[str, np.ndarray] = {}
            index: pd.Index | None = None
            for name in series_params:
                raw = bound.arguments[name]
                if isinstance(raw, pd.Series):
                    if index is None:
                        index = raw.index
                    array = raw.to_numpy(dtype=float)
                else:
                    array = np.asarray(raw, dtype=float)
                if array.ndim != 1:
                    return func(*args, **kwargs)
                arrays[name] = array
            lengths = {array.shape[0] for array in arrays.values()}
            if len(lengths) != 1:
                # 长度不一致: 交给原函数按原有方式报错
                return func(*args, **kwargs)
            mask = np.logical_and.reduce(
                [np.isfinite(array) for array in arrays.values()]
            )
            if bool(mask.all()):
                return func(*args, **kwargs)
            as_series = bool(bound.arguments.get("as_series", False))
            for name, array in arrays.items():
                bound.arguments[name] = array[mask]
            bound.arguments["as_series"] = False
            result = func(*bound.args, **bound.kwargs)
            positions = np.flatnonzero(mask)

            def scatter(compact_values: object) -> pd.Series | np.ndarray:
                compact = np.array(compact_values, dtype=float)
                if index_outputs:
                    found = ~np.isnan(compact)
                    compact[found] = positions[compact[found].astype(np.int64)]
                full = np.full(mask.shape[0], np.nan, dtype=float)
                full[mask] = compact
                if as_series:
                    return finalize_output(
                        pd.Series(full, index=index, dtype=float), as_series=True
                    )
                return finalize_output(full, as_series=False)

            if isinstance(result, tuple):
                return tuple(scatter(item) for item in result)
            return scatter(result)

        return cast(_F, wrapper)

    return decorate
