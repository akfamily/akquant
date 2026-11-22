//! 期权 Black-76 批量定价、Greeks 与隐含波动率, 供 `akquant.option` 调用.
//!
//! 输入是等长的一维数组(Python 侧负责广播); 数组较长时用 rayon 并行, 计算期间
//! 释放 GIL。求解区间、容差、迭代次数由 Python 传入, 与策略侧的诊断信息同源。

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::{PI, SQRT_2};

/// 数组长度达到该值才并行; 更短时线程调度开销大于收益.
const PARALLEL_MIN_LEN: usize = 2048;

fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT_2)
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Black-76 价格; `t<=0` 或 `sigma<=0` 时为贴现内在价值.
fn price(f: f64, k: f64, t: f64, r: f64, sigma: f64, call: bool) -> f64 {
    let df = (-r * t).exp();
    if !(t > 0.0 && sigma > 0.0) {
        let intrinsic = if call { f - k } else { k - f };
        return df * intrinsic.max(0.0);
    }
    let sd = sigma * t.sqrt();
    let d1 = ((f / k).ln() + 0.5 * sd * sd) / sd;
    let d2 = d1 - sd;
    if call {
        df * (f * norm_cdf(d1) - k * norm_cdf(d2))
    } else {
        df * (k * norm_cdf(-d2) - f * norm_cdf(-d1))
    }
}

/// `[delta, gamma, vega, theta, rho]`, 单位为原始偏导数, theta 为自然时间流逝方向.
fn greeks(f: f64, k: f64, t: f64, r: f64, sigma: f64, call: bool) -> [f64; 5] {
    let df = (-r * t).exp();
    let value = price(f, k, t, r, sigma, call);
    let rho = -t * value;
    if !(t > 0.0 && sigma > 0.0) {
        let itm = if call { f > k } else { f < k };
        let delta = match (itm, call) {
            (false, _) => 0.0,
            (true, true) => df,
            (true, false) => -df,
        };
        return [delta, 0.0, 0.0, 0.0, rho];
    }
    let sqrt_t = t.sqrt();
    let sd = sigma * sqrt_t;
    let d1 = ((f / k).ln() + 0.5 * sd * sd) / sd;
    let pdf = norm_pdf(d1);
    let delta = if call {
        df * norm_cdf(d1)
    } else {
        -df * norm_cdf(-d1)
    };
    let gamma = df * pdf / (f * sd);
    let vega = df * f * pdf * sqrt_t;
    let theta = -df * f * pdf * sigma / (2.0 * sqrt_t) + r * value;
    [delta, gamma, vega, theta, rho]
}

/// 求解参数, 全部由 Python 传入.
#[derive(Clone, Copy)]
struct Solver {
    price_tol: f64,
    sigma_tol: f64,
    sigma_low: f64,
    sigma_high: f64,
    max_iter: usize,
}

/// 隐含波动率: 区间保护的牛顿法, 按波动率步长收敛; 无解或不收敛返回 NaN.
///
/// 无解的判定与 Python 侧诊断一致: 时间价值(相对 `sigma_low` 的价格)不超过
/// `price_tol`, 或价格不低于 `sigma_high` 时的价格。
fn implied_vol(p: f64, f: f64, k: f64, t: f64, r: f64, call: bool, s: Solver) -> f64 {
    if !(t > 0.0 && f > 0.0 && k > 0.0 && p.is_finite()) {
        return f64::NAN;
    }
    let (mut lo, mut hi) = (s.sigma_low, s.sigma_high);
    if !(price(f, k, t, r, lo, call) < p - s.price_tol && price(f, k, t, r, hi, call) > p) {
        return f64::NAN;
    }
    let df = (-r * t).exp();
    let sqrt_t = t.sqrt();
    let mut sigma = 0.3_f64.clamp(lo, hi);
    for _ in 0..s.max_iter {
        let diff = price(f, k, t, r, sigma, call) - p;
        if diff == 0.0 {
            return sigma;
        }
        if diff > 0.0 {
            hi = sigma;
        } else {
            lo = sigma;
        }
        let sd = sigma * sqrt_t;
        let d1 = ((f / k).ln() + 0.5 * sd * sd) / sd;
        let vega = df * f * norm_pdf(d1) * sqrt_t;
        let newton = sigma - diff / vega;
        let next = if newton.is_finite() && newton > lo && newton < hi {
            newton
        } else {
            0.5 * (lo + hi)
        };
        if (next - sigma).abs() < s.sigma_tol || hi - lo < s.sigma_tol {
            return next;
        }
        sigma = next;
    }
    f64::NAN
}

fn slice<'a>(arr: &'a PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    arr.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} 必须是连续的一维 float64 数组")))
}

fn check_len(n: usize, others: &[(&str, usize)]) -> PyResult<()> {
    for (name, len) in others {
        if *len != n {
            return Err(PyValueError::new_err(format!(
                "{name} 长度 {len} 与首个数组长度 {n} 不一致"
            )));
        }
    }
    Ok(())
}

fn map_indices<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if n >= PARALLEL_MIN_LEN {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// Black-76 批量定价(数组等长).
#[pyfunction]
pub fn _option_black76_price<'py>(
    py: Python<'py>,
    forward: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    is_call: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (f, k, tt, r, s) = (
        slice(&forward, "forward")?,
        slice(&strike, "strike")?,
        slice(&t, "t")?,
        slice(&rate, "rate")?,
        slice(&sigma, "sigma")?,
    );
    let c = is_call
        .as_slice()
        .map_err(|_| PyValueError::new_err("is_call 必须是连续的一维 bool 数组"))?;
    let n = f.len();
    check_len(
        n,
        &[
            ("strike", k.len()),
            ("t", tt.len()),
            ("rate", r.len()),
            ("sigma", s.len()),
            ("is_call", c.len()),
        ],
    )?;
    let out = py.detach(|| map_indices(n, |i| price(f[i], k[i], tt[i], r[i], s[i], c[i])));
    Ok(PyArray1::from_vec(py, out))
}

/// Black-76 批量 Greeks, 返回 `(delta, gamma, vega, theta, rho)` 五个数组.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn _option_black76_greeks<'py>(
    py: Python<'py>,
    forward: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    is_call: PyReadonlyArray1<'py, bool>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let (f, k, tt, r, s) = (
        slice(&forward, "forward")?,
        slice(&strike, "strike")?,
        slice(&t, "t")?,
        slice(&rate, "rate")?,
        slice(&sigma, "sigma")?,
    );
    let c = is_call
        .as_slice()
        .map_err(|_| PyValueError::new_err("is_call 必须是连续的一维 bool 数组"))?;
    let n = f.len();
    check_len(
        n,
        &[
            ("strike", k.len()),
            ("t", tt.len()),
            ("rate", r.len()),
            ("sigma", s.len()),
            ("is_call", c.len()),
        ],
    )?;
    let rows = py.detach(|| map_indices(n, |i| greeks(f[i], k[i], tt[i], r[i], s[i], c[i])));
    let column = |j: usize| PyArray1::from_vec(py, rows.iter().map(|g| g[j]).collect());
    Ok((column(0), column(1), column(2), column(3), column(4)))
}

/// Black-76 批量隐含波动率; 无解或不收敛为 NaN.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn _option_black76_implied_vol<'py>(
    py: Python<'py>,
    price_: PyReadonlyArray1<'py, f64>,
    forward: PyReadonlyArray1<'py, f64>,
    strike: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    is_call: PyReadonlyArray1<'py, bool>,
    price_tol: f64,
    sigma_tol: f64,
    sigma_low: f64,
    sigma_high: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (p, f, k, tt, r) = (
        slice(&price_, "price")?,
        slice(&forward, "forward")?,
        slice(&strike, "strike")?,
        slice(&t, "t")?,
        slice(&rate, "rate")?,
    );
    let c = is_call
        .as_slice()
        .map_err(|_| PyValueError::new_err("is_call 必须是连续的一维 bool 数组"))?;
    let n = p.len();
    check_len(
        n,
        &[
            ("forward", f.len()),
            ("strike", k.len()),
            ("t", tt.len()),
            ("rate", r.len()),
            ("is_call", c.len()),
        ],
    )?;
    let solver = Solver {
        price_tol,
        sigma_tol,
        sigma_low,
        sigma_high,
        max_iter,
    };
    let out = py.detach(|| {
        map_indices(n, |i| {
            implied_vol(p[i], f[i], k[i], tt[i], r[i], c[i], solver)
        })
    });
    Ok(PyArray1::from_vec(py, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOLVER: Solver = Solver {
        price_tol: 1e-8,
        sigma_tol: 1e-10,
        sigma_low: 1e-6,
        sigma_high: 5.0,
        max_iter: 100,
    };

    #[test]
    fn hull_futures_put_example() {
        // Hull: F=K=20, r=9%, T=4 个月, σ=25% 的欧式期货认沽约 1.12
        let p = price(20.0, 20.0, 4.0 / 12.0, 0.09, 0.25, false);
        assert!((p - 1.12).abs() < 0.005, "{p}");
    }

    #[test]
    fn put_call_parity_holds() {
        let (f, k, t, r, s) = (3000.0, 3100.0, 0.25, 0.02, 0.22);
        let lhs = price(f, k, t, r, s, true) - price(f, k, t, r, s, false);
        assert!((lhs - (-r * t).exp() * (f - k)).abs() < 1e-9);
    }

    #[test]
    fn implied_vol_recovers_sigma_with_tiny_vega() {
        // 深度实值短期合约 vega 约 1e-6: 按价格误差收敛会偏 0.5 个波动率点
        let (f, k, t, r, s) = (3000.0, 2200.0, 10.0 / 365.0, 0.02, 0.4);
        let p = price(f, k, t, r, s, true);
        let iv = implied_vol(p, f, k, t, r, true, SOLVER);
        assert!((iv - s).abs() < 1e-6, "{iv}");
    }

    #[test]
    fn implied_vol_nan_below_intrinsic() {
        let iv = implied_vol(50.0, 3000.0, 2900.0, 0.1, 0.02, true, SOLVER);
        assert!(iv.is_nan());
    }

    #[test]
    fn greeks_match_finite_differences() {
        let (f, k, t, r, s) = (3000.0, 3100.0, 0.25, 0.02, 0.22);
        let g = greeks(f, k, t, r, s, false);
        let h = 1e-4;
        let px = |f: f64, t: f64, r: f64, s: f64| price(f, k, t, r, s, false);
        let delta = (px(f + h, t, r, s) - px(f - h, t, r, s)) / (2.0 * h);
        let vega = (px(f, t, r, s + h) - px(f, t, r, s - h)) / (2.0 * h);
        let theta = -(px(f, t + h, r, s) - px(f, t - h, r, s)) / (2.0 * h);
        assert!((g[0] - delta).abs() < 1e-6);
        assert!((g[2] - vega).abs() / vega < 1e-6);
        assert!((g[3] - theta).abs() / theta.abs() < 1e-5);
    }
}
