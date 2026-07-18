//! 列式 f64 向量化批量计算原语 (B2′).
//!
//! 直接在 `&[f64]` 切片上计算 (零拷贝、编译器可自动向量化), 与增量指标 (逐点
//! `update`) 互补: 用于对整列一次性求值 (因子/指标预计算)。可零拷贝作用于 B.1
//! 的 [`crate::data::columns::BarColumns`] 列, 无需 pandas/polars 往返。
//!
//! NaN 约定与 pandas 一致: 窗口不足 (`< period`) 输出 `NaN`; `returns[0]=NaN`。
//! 滚动标准差为**样本标准差** (ddof=1), 与 pandas `rolling(period).std()` 默认一致。

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

const NAN: f64 = f64::NAN;

/// 简单移动平均 (SMA). 滑动和, O(n)。
pub fn sma(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let inv = 1.0 / period as f64;
    let mut sum = 0.0;
    for i in 0..n {
        sum += values[i];
        if i >= period {
            sum -= values[i - period];
        }
        if i + 1 >= period {
            out[i] = sum * inv;
        }
    }
    out
}

/// 指数移动平均 (EMA). alpha = 2/(period+1); 以前 `period` 个点的 SMA 作种子。
pub fn ema(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    // 种子: 前 period 个点的均值
    let mut seed = 0.0;
    for &v in &values[..period] {
        seed += v;
    }
    let mut prev = seed / period as f64;
    out[period - 1] = prev;
    for i in period..n {
        prev = alpha * values[i] + (1.0 - alpha) * prev;
        out[i] = prev;
    }
    out
}

/// 加权移动平均 (WMA), 线性权重 1..=period。
pub fn wma(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let denom = (period * (period + 1) / 2) as f64;
    for i in (period - 1)..n {
        let mut acc = 0.0;
        for k in 0..period {
            // 先加后减, 避免 usize 中间下溢
            acc += values[i + 1 + k - period] * (k + 1) as f64;
        }
        out[i] = acc / denom;
    }
    out
}

/// 滚动求和.
pub fn rolling_sum(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![NAN; n];
    if period == 0 || period > n {
        return out;
    }
    let mut sum = 0.0;
    for i in 0..n {
        sum += values[i];
        if i >= period {
            sum -= values[i - period];
        }
        if i + 1 >= period {
            out[i] = sum;
        }
    }
    out
}

/// 滚动最小值 (朴素 O(n·period), 语义清晰).
pub fn rolling_min(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![NAN; n];
    if period == 0 || period > n {
        return out;
    }
    for i in (period - 1)..n {
        let mut m = f64::INFINITY;
        for &v in &values[i + 1 - period..=i] {
            if v < m {
                m = v;
            }
        }
        out[i] = m;
    }
    out
}

/// 滚动最大值.
pub fn rolling_max(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![NAN; n];
    if period == 0 || period > n {
        return out;
    }
    for i in (period - 1)..n {
        let mut m = f64::NEG_INFINITY;
        for &v in &values[i + 1 - period..=i] {
            if v > m {
                m = v;
            }
        }
        out[i] = m;
    }
    out
}

/// 滚动样本标准差 (ddof=1, 与 pandas 默认一致).
pub fn rolling_std(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![NAN; n];
    if period < 2 || period > n {
        return out;
    }
    let inv_n = 1.0 / period as f64;
    let inv_ddof = 1.0 / (period as f64 - 1.0);
    for i in (period - 1)..n {
        let window = &values[i + 1 - period..=i];
        let mean = window.iter().sum::<f64>() * inv_n;
        let ss = window.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>();
        out[i] = (ss * inv_ddof).sqrt();
    }
    out
}

/// 简单收益率 (pct_change): (v[i]-v[i-1])/v[i-1]; `out[0]=NaN`。
pub fn returns(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![NAN; n];
    for i in 1..n {
        let prev = values[i - 1];
        out[i] = if prev != 0.0 {
            (values[i] - prev) / prev
        } else {
            NAN
        };
    }
    out
}

/// 对数收益率 ln(v[i]/v[i-1]); `out[0]=NaN`。
pub fn log_returns(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![NAN; n];
    for i in 1..n {
        let prev = values[i - 1];
        out[i] = if prev > 0.0 && values[i] > 0.0 {
            (values[i] / prev).ln()
        } else {
            NAN
        };
    }
    out
}

/// 累积求和.
pub fn cumsum(values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    let mut acc = 0.0;
    for &v in values {
        acc += v;
        out.push(acc);
    }
    out
}

/// 滚动 z-score: (v - rolling_mean)/rolling_std(ddof=1)。
pub fn zscore(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let means = sma(values, period);
    let stds = rolling_std(values, period);
    let mut out = vec![NAN; n];
    for i in 0..n {
        let s = stds[i];
        if !means[i].is_nan() && !s.is_nan() && s != 0.0 {
            out[i] = (values[i] - means[i]) / s;
        }
    }
    out
}

// --------------------------------------------------------------------------- #
// Python 绑定 (numpy 零拷贝读入, 返回新数组)
// --------------------------------------------------------------------------- #

/// 向量化 SMA (简单移动平均).
#[pyfunction]
pub fn vec_sma<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    period: usize,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, sma(values.as_slice().unwrap_or(&[]), period))
}

/// 向量化 EMA (指数移动平均).
#[pyfunction]
pub fn vec_ema<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    period: usize,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, ema(values.as_slice().unwrap_or(&[]), period))
}

/// 向量化 WMA (加权移动平均).
#[pyfunction]
pub fn vec_wma<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    period: usize,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, wma(values.as_slice().unwrap_or(&[]), period))
}

/// 向量化滚动求和.
#[pyfunction]
pub fn vec_rolling_sum<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    period: usize,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, rolling_sum(values.as_slice().unwrap_or(&[]), period))
}

/// 向量化滚动最小值.
#[pyfunction]
pub fn vec_rolling_min<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    period: usize,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, rolling_min(values.as_slice().unwrap_or(&[]), period))
}

/// 向量化滚动最大值.
#[pyfunction]
pub fn vec_rolling_max<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    period: usize,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, rolling_max(values.as_slice().unwrap_or(&[]), period))
}

/// 向量化滚动样本标准差 (ddof=1).
#[pyfunction]
pub fn vec_rolling_std<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    period: usize,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, rolling_std(values.as_slice().unwrap_or(&[]), period))
}

/// 向量化滚动 z-score.
#[pyfunction]
pub fn vec_zscore<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    period: usize,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, zscore(values.as_slice().unwrap_or(&[]), period))
}

/// 向量化简单收益率 (pct_change).
#[pyfunction]
pub fn vec_returns<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, returns(values.as_slice().unwrap_or(&[])))
}

/// 向量化对数收益率.
#[pyfunction]
pub fn vec_log_returns<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, log_returns(values.as_slice().unwrap_or(&[])))
}

/// 向量化累积求和.
#[pyfunction]
pub fn vec_cumsum<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, cumsum(values.as_slice().unwrap_or(&[])))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn test_sma_basic() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        let out = sma(&v, 3);
        assert!(out[0].is_nan() && out[1].is_nan());
        assert!(approx(out[2], 2.0)); // (1+2+3)/3
        assert!(approx(out[3], 3.0));
        assert!(approx(out[4], 4.0));
    }

    #[test]
    fn test_rolling_sum_and_minmax() {
        let v = [4.0, 2.0, 6.0, 1.0];
        assert!(approx(rolling_sum(&v, 2)[3], 7.0)); // 6+1
        assert!(approx(rolling_min(&v, 2)[3], 1.0));
        assert!(approx(rolling_max(&v, 2)[2], 6.0));
    }

    #[test]
    fn test_rolling_std_sample() {
        let v = [2.0, 4.0, 6.0];
        // 样本 std of [2,4,6] = sqrt(((2-4)^2+(4-4)^2+(6-4)^2)/2) = sqrt(4) = 2
        assert!(approx(rolling_std(&v, 3)[2], 2.0));
    }

    #[test]
    fn test_returns() {
        let v = [10.0, 11.0, 22.0];
        let r = returns(&v);
        assert!(r[0].is_nan());
        assert!(approx(r[1], 0.1));
        assert!(approx(r[2], 1.0));
    }

    #[test]
    fn test_ema_seed_is_sma() {
        let v = [1.0, 2.0, 3.0, 4.0];
        let e = ema(&v, 2);
        // 种子 = (1+2)/2 = 1.5 在 index 1
        assert!(e[0].is_nan());
        assert!(approx(e[1], 1.5));
        // 后续: alpha=2/3; e[2]=2/3*3 + 1/3*1.5 = 2.5
        assert!(approx(e[2], 2.5));
    }

    #[test]
    fn test_period_guards() {
        let v = [1.0, 2.0];
        assert!(sma(&v, 5).iter().all(|x| x.is_nan()));
        assert!(sma(&v, 0).iter().all(|x| x.is_nan()));
        assert_eq!(cumsum(&v), vec![1.0, 3.0]);
    }
}
