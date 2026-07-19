"""B2′ 向量化列计算原语单测: 对齐 pandas/numpy 语义 (RFC §21)."""

import numpy as np
import pandas as pd
from akquant import (
    vec_cumsum,
    vec_ema,
    vec_log_returns,
    vec_returns,
    vec_rolling_max,
    vec_rolling_min,
    vec_rolling_std,
    vec_rolling_sum,
    vec_sma,
    vec_wma,
    vec_zscore,
)

_X = np.array([10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 16.0, 14.0])


def _match(a: np.ndarray, b: object) -> bool:
    """比较两数组, 忽略双方的 NaN 位置."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b.values if hasattr(b, "values") else b, dtype=float)
    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    return bool(np.allclose(arr_a[mask], arr_b[mask], atol=1e-9)) and bool(
        (np.isnan(arr_a) == np.isnan(arr_b)).all()
    )


def test_vec_sma_matches_pandas() -> None:
    """vec_sma 与 pandas rolling().mean() 一致 (含 NaN 位置)."""
    assert _match(vec_sma(_X, 3), pd.Series(_X).rolling(3).mean())


def test_vec_rolling_sum_matches_pandas() -> None:
    """vec_rolling_sum 与 pandas rolling().sum() 一致."""
    assert _match(vec_rolling_sum(_X, 4), pd.Series(_X).rolling(4).sum())


def test_vec_rolling_std_matches_pandas_ddof1() -> None:
    """vec_rolling_std 为样本标准差 (ddof=1), 与 pandas 默认一致."""
    assert _match(vec_rolling_std(_X, 3), pd.Series(_X).rolling(3).std())


def test_vec_rolling_min_max_match_pandas() -> None:
    """vec_rolling_min/max 与 pandas 一致."""
    assert _match(vec_rolling_min(_X, 3), pd.Series(_X).rolling(3).min())
    assert _match(vec_rolling_max(_X, 3), pd.Series(_X).rolling(3).max())


def test_vec_returns_matches_pandas() -> None:
    """vec_returns 与 pandas pct_change() 一致."""
    assert _match(vec_returns(_X), pd.Series(_X).pct_change())


def test_vec_log_returns_matches_numpy() -> None:
    """vec_log_returns 与 numpy 手算对数收益一致."""
    expected = np.concatenate([[np.nan], np.diff(np.log(_X))])
    assert _match(vec_log_returns(_X), expected)


def test_vec_cumsum_matches_numpy() -> None:
    """vec_cumsum 与 numpy cumsum 一致."""
    assert _match(vec_cumsum(_X), np.cumsum(_X))


def test_vec_wma_manual() -> None:
    """vec_wma 线性加权: 末窗 [a,b,c] -> (1a+2b+3c)/6."""
    out = vec_wma(np.array([1.0, 2.0, 3.0]), 3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert abs(out[2] - (1 * 1 + 2 * 2 + 3 * 3) / 6.0) < 1e-9


def test_vec_zscore_zero_mean_unit_var_window() -> None:
    """vec_zscore 末值 = (x - 窗口均值)/窗口样本std."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = vec_zscore(x, 3)
    s = pd.Series(x)
    expected = (x - s.rolling(3).mean()) / s.rolling(3).std()
    assert _match(z, expected)


def test_vec_ema_seed_and_recurrence() -> None:
    """vec_ema 以前 period 点均值为种子, 之后按 alpha=2/(p+1) 递推."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    e = vec_ema(x, 2)
    assert np.isnan(e[0])
    assert abs(e[1] - 1.5) < 1e-9  # (1+2)/2
    assert abs(e[2] - (2.0 / 3 * 3 + 1.0 / 3 * 1.5)) < 1e-9


def test_period_exceeds_length_all_nan() -> None:
    """周期大于序列长度时全为 NaN."""
    assert np.all(np.isnan(vec_sma(np.array([1.0, 2.0]), 5)))
