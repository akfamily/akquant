//! 波动率微笑拟合: SVI(Zeliade 准显式标定)与 SABR(Hagan 2002 对数正态近似).
//!
//! 每个到期日一条微笑, 多条微笑在一次调用中并行拟合, 计算期间释放 GIL。数据按到期日
//! 连续存放, `offsets[i]..offsets[i+1]` 是第 i 条微笑的数据。搜索区间、容差、迭代次数
//! 由 Python 传入。

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::option_pricing::{check_len, slice};

/// Nelder-Mead 求最小值的结果.
struct Minimum<const N: usize> {
    x: [f64; N],
    fx: f64,
    iterations: usize,
    converged: bool,
}

/// Nelder-Mead 单纯形法(标准系数: 反射 1、扩张 2、收缩 0.5、压缩 0.5).
///
/// 单纯形各顶点到最优点的坐标差都不超过 `xtol`, 且函数值差都不超过 `ftol` 时收敛。
fn nelder_mead<const N: usize>(
    f: &impl Fn(&[f64; N]) -> f64,
    x0: [f64; N],
    step: [f64; N],
    xtol: f64,
    ftol: f64,
    max_iter: usize,
) -> Minimum<N> {
    let mut simplex: Vec<([f64; N], f64)> = Vec::with_capacity(N + 1);
    simplex.push((x0, f(&x0)));
    for i in 0..N {
        let mut x = x0;
        x[i] += step[i];
        simplex.push((x, f(&x)));
    }
    let order = |s: &mut Vec<([f64; N], f64)>| {
        s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    };
    let blend = |a: &[f64; N], b: &[f64; N], t: f64| -> [f64; N] {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = a[i] + t * (b[i] - a[i]);
        }
        out
    };
    let mut iterations = 0;
    order(&mut simplex);
    while iterations < max_iter {
        let (best, fbest) = simplex[0];
        let spread_ok = simplex[1..].iter().all(|(x, fx)| {
            (fx - fbest).abs() <= ftol && x.iter().zip(&best).all(|(a, b)| (a - b).abs() <= xtol)
        });
        if spread_ok {
            return Minimum {
                x: best,
                fx: fbest,
                iterations,
                converged: true,
            };
        }
        iterations += 1;
        let mut centroid = [0.0; N];
        for (x, _) in &simplex[..N] {
            for i in 0..N {
                centroid[i] += x[i] / N as f64;
            }
        }
        let (worst, fworst) = simplex[N];
        let reflected = blend(&centroid, &worst, -1.0);
        let fr = f(&reflected);
        if fr < simplex[0].1 {
            let expanded = blend(&centroid, &worst, -2.0);
            let fe = f(&expanded);
            simplex[N] = if fe < fr {
                (expanded, fe)
            } else {
                (reflected, fr)
            };
        } else if fr < simplex[N - 1].1 {
            simplex[N] = (reflected, fr);
        } else {
            let (target, ft) = if fr < fworst {
                (reflected, fr)
            } else {
                (worst, fworst)
            };
            let contracted = blend(&centroid, &target, 0.5);
            let fc = f(&contracted);
            if fc < ft {
                simplex[N] = (contracted, fc);
            } else {
                let best = simplex[0].0;
                for vertex in simplex.iter_mut().skip(1) {
                    let x = blend(&best, &vertex.0, 0.5);
                    *vertex = (x, f(&x));
                }
            }
        }
        order(&mut simplex);
    }
    Minimum {
        x: simplex[0].0,
        fx: simplex[0].1,
        iterations,
        converged: false,
    }
}

/// KKT 方程组的最大阶数: 3 个未知数 + 至多 3 个有效约束.
const KKT_MAX: usize = 6;

/// 解前 n 阶线性方程组(部分主元高斯消元, 栈上定长数组); 主元过小视为奇异返回 None.
fn solve_linear(
    mut m: [[f64; KKT_MAX]; KKT_MAX],
    mut rhs: [f64; KKT_MAX],
    n: usize,
) -> Option<[f64; KKT_MAX]> {
    let scale = m[..n]
        .iter()
        .flat_map(|row| row[..n].iter())
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
        .max(1e-300);
    for col in 0..n {
        let pivot = (col..n).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[pivot][col].abs() <= 1e-13 * scale {
            return None;
        }
        m.swap(col, pivot);
        rhs.swap(col, pivot);
        for row in col + 1..n {
            let factor = m[row][col] / m[col][col];
            if factor != 0.0 {
                for j in col..n {
                    m[row][j] -= factor * m[col][j];
                }
                rhs[row] -= factor * rhs[col];
            }
        }
    }
    let mut x = [0.0; KKT_MAX];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|j| m[row][j] * x[j]).sum();
        x[row] = (rhs[row] - tail) / m[row][row];
    }
    Some(x)
}

/// SVI 原始参数(总方差口径): `w(k) = a + b(ρ(k−m) + sqrt((k−m)² + σ²))`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Svi {
    a: f64,
    b: f64,
    rho: f64,
    m: f64,
    sigma: f64,
}

fn svi_total_variance(k: f64, p: &Svi) -> f64 {
    let x = k - p.m;
    p.a + p.b * (p.rho * x + (x * x + p.sigma * p.sigma).sqrt())
}

/// 固定 (m, σ) 后的内层问题: 在 Zeliade 约束域内对 (a, d, c) 做加权最小二乘.
///
/// `y = (k−m)/σ`, `w = a + d·y + c·sqrt(y²+1)`, 约束 `|d| ≤ c`、`|d| ≤ 4σ − c`(含
/// `0 ≤ c ≤ 4σ`, 即 Lee 矩条件 `b(1+|ρ|) ≤ 4`)、`0 ≤ a ≤ max w`。目标是凸二次函数,
/// 最优点必是某个有效约束集上的等式约束最优点: 枚举至多 3 个有效约束(42 种), 取可行
/// 点中目标最小者。返回 `((a, d, c), 加权均方误差)`。
fn svi_inner(k: &[f64], w: &[f64], weight: &[f64], m: f64, sigma: f64) -> ([f64; 3], f64) {
    let mut h = [[0.0; 3]; 3];
    let mut g = [0.0; 3];
    let (mut ww, mut wsum, mut wmax) = (0.0, 0.0, 0.0_f64);
    for i in 0..k.len() {
        let y = (k[i] - m) / sigma;
        let phi = [1.0, y, (y * y + 1.0).sqrt()];
        for r in 0..3 {
            g[r] += weight[i] * w[i] * phi[r];
            for c in 0..3 {
                h[r][c] += weight[i] * phi[r] * phi[c];
            }
        }
        ww += weight[i] * w[i] * w[i];
        wsum += weight[i];
        wmax = wmax.max(w[i]);
    }
    let cap = 4.0 * sigma;
    // 约束 A·x ≤ b, x = (a, d, c)
    let rows: [([f64; 3], f64); 6] = [
        ([0.0, 1.0, -1.0], 0.0),
        ([0.0, -1.0, -1.0], 0.0),
        ([0.0, 1.0, 1.0], cap),
        ([0.0, -1.0, 1.0], cap),
        ([-1.0, 0.0, 0.0], 0.0),
        ([1.0, 0.0, 0.0], wmax),
    ];
    let sse = |x: &[f64; 3]| {
        let mut quad = 0.0;
        for r in 0..3 {
            for c in 0..3 {
                quad += x[r] * h[r][c] * x[c];
            }
        }
        ww - 2.0 * (g[0] * x[0] + g[1] * x[1] + g[2] * x[2]) + quad
    };
    let feasible = |x: &[f64; 3]| {
        rows.iter().all(|(a, b)| {
            a[0] * x[0] + a[1] * x[1] + a[2] * x[2] <= b + 1e-12 * (1.0 + b.abs() + wmax)
        })
    };
    let mut best = ([0.0; 3], sse(&[0.0; 3]));
    let mut active: Vec<usize> = Vec::with_capacity(3);
    let mut consider = |active: &[usize]| {
        let n = 3 + active.len();
        let mut mat = [[0.0; KKT_MAX]; KKT_MAX];
        let mut rhs = [0.0; KKT_MAX];
        for r in 0..3 {
            mat[r][..3].copy_from_slice(&h[r]);
            rhs[r] = g[r];
        }
        for (j, &idx) in active.iter().enumerate() {
            let (a, b) = rows[idx];
            for c in 0..3 {
                mat[3 + j][c] = a[c];
                mat[c][3 + j] = a[c];
            }
            rhs[3 + j] = b;
        }
        if let Some(sol) = solve_linear(mat, rhs, n) {
            let x = [sol[0], sol[1], sol[2]];
            if feasible(&x) {
                let value = sse(&x);
                if value < best.1 {
                    best = (x, value);
                }
            }
        }
    };
    consider(&active);
    for i in 0..6 {
        active.push(i);
        consider(&active);
        for j in i + 1..6 {
            active.push(j);
            consider(&active);
            for l in j + 1..6 {
                active.push(l);
                consider(&active);
                active.pop();
            }
            active.pop();
        }
        active.pop();
    }
    (best.0, (best.1 / wsum).max(0.0))
}

/// SVI 拟合的搜索设置(由 Python 传入).
#[derive(Clone, Copy)]
struct SviSearch {
    sigma_low: f64,
    sigma_high: f64,
    m_margin: f64,
    xtol: f64,
    ftol: f64,
    max_iter: usize,
}

/// 拟合一条 SVI 微笑(k 为对数在值程度 ln(K/F), w 为总方差 iv²·t).
///
/// 外层在 `σ ∈ [sigma_low, sigma_high]`、`m ∈ [k_min − margin·跨度, k_max + margin·跨度]`
/// 内用 Nelder-Mead 搜索 (m, σ), 从 4 个起点出发取最优; 内层见 [`svi_inner`]。
fn fit_svi(k: &[f64], w: &[f64], weight: &[f64], s: SviSearch) -> (Svi, usize, bool) {
    let kmin = k.iter().cloned().fold(f64::INFINITY, f64::min);
    let kmax = k.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (kmax - kmin).max(1e-4);
    let (m_low, m_high) = (kmin - s.m_margin * span, kmax + s.m_margin * span);
    let clamp = |x: &[f64; 2]| {
        [
            x[0].clamp(m_low, m_high),
            x[1].clamp(s.sigma_low, s.sigma_high),
        ]
    };
    let objective = |x: &[f64; 2]| {
        let [m, sigma] = clamp(x);
        svi_inner(k, w, weight, m, sigma).1
    };
    let k_at_min_w = k[w
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)];
    let mut best: Option<Minimum<2>> = None;
    let mut iterations = 0;
    for m0 in [k_at_min_w, 0.5 * (kmin + kmax)] {
        for sigma0 in [0.05_f64, 0.25] {
            let sigma0 = sigma0.clamp(s.sigma_low, s.sigma_high);
            let run = nelder_mead(
                &objective,
                [m0, sigma0],
                [0.1 * span, 0.5 * sigma0],
                s.xtol,
                s.ftol,
                s.max_iter,
            );
            iterations += run.iterations;
            if best.as_ref().is_none_or(|b| run.fx < b.fx) {
                best = Some(run);
            }
        }
    }
    let best = best.expect("至少一个起点");
    let [m, sigma] = clamp(&best.x);
    let ([a, d, c], _) = svi_inner(k, w, weight, m, sigma);
    let rho = if c > 0.0 {
        (d / c).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    (
        Svi {
            a,
            b: c / sigma,
            rho,
            m,
            sigma,
        },
        iterations,
        best.converged,
    )
}

/// Hagan 等(2002) SABR 对数正态隐含波动率近似; 输入不合法时为 nan.
fn sabr_vol(f: f64, k: f64, t: f64, alpha: f64, beta: f64, rho: f64, nu: f64) -> f64 {
    if !(f > 0.0 && k > 0.0 && alpha > 0.0 && nu >= 0.0 && rho.abs() < 1.0) {
        return f64::NAN;
    }
    let omb = 1.0 - beta;
    let lfk = (f / k).ln();
    let fk_b = (f * k).powf(0.5 * omb);
    let denom = fk_b * (1.0 + omb * omb / 24.0 * lfk * lfk + omb.powi(4) / 1920.0 * lfk.powi(4));
    let z = nu / alpha * fk_b * lfk;
    let z_over_x = if z.abs() < 1e-6 {
        1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho * rho) / 12.0 * z * z
    } else {
        let x = (((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho) / (1.0 - rho)).ln();
        z / x
    };
    let correction = 1.0
        + (omb * omb / 24.0 * alpha * alpha / (fk_b * fk_b)
            + 0.25 * rho * beta * nu * alpha / fk_b
            + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu)
            * t;
    alpha / denom * z_over_x * correction
}

/// SABR 拟合的搜索设置(由 Python 传入).
#[derive(Clone, Copy)]
struct SabrSearch {
    beta: f64,
    xtol: f64,
    ftol: f64,
    max_iter: usize,
}

/// 拟合一条 SABR 微笑(β 固定), 返回 `((α, ρ, ν), 迭代次数, 是否收敛)`.
///
/// 在变换后的参数 `(ln α, atanh ρ, ln ν)` 上用 Nelder-Mead 最小化加权波动率均方误差,
/// α 的起点取最接近远期的行权价的隐含波动率 × F^(1−β), ρ 从 −0.5、0、0.5 三个起点出发。
fn fit_sabr(
    f: f64,
    t: f64,
    k: &[f64],
    vol: &[f64],
    weight: &[f64],
    s: SabrSearch,
) -> ([f64; 3], usize, bool) {
    let wsum: f64 = weight.iter().sum();
    let decode = |u: &[f64; 3]| {
        [
            u[0].exp(),
            u[1].tanh().clamp(-0.999999, 0.999999),
            u[2].exp(),
        ]
    };
    let objective = |u: &[f64; 3]| {
        let [alpha, rho, nu] = decode(u);
        let mut sse = 0.0;
        for i in 0..k.len() {
            let model = sabr_vol(f, k[i], t, alpha, s.beta, rho, nu);
            if !model.is_finite() {
                return f64::INFINITY;
            }
            sse += weight[i] * (model - vol[i]).powi(2);
        }
        sse / wsum
    };
    let atm = (0..k.len())
        .min_by(|&a, &b| (k[a] - f).abs().total_cmp(&(k[b] - f).abs()))
        .unwrap_or(0);
    let alpha0 = vol[atm] * f.powf(1.0 - s.beta);
    let mut best: Option<Minimum<3>> = None;
    let mut iterations = 0;
    for rho0 in [-0.5_f64, 0.0, 0.5] {
        let run = nelder_mead(
            &objective,
            [alpha0.ln(), rho0.atanh(), 0.5_f64.ln()],
            [0.2, 0.3, 0.5],
            s.xtol,
            s.ftol,
            s.max_iter,
        );
        iterations += run.iterations;
        if best.as_ref().is_none_or(|b| run.fx < b.fx) {
            best = Some(run);
        }
    }
    let best = best.expect("至少一个起点");
    (decode(&best.x), iterations, best.converged)
}

/// 校验 offsets 单调、首为 0、末为数据长度, 返回各段 (起, 止).
fn segments(offsets: &[i64], n: usize) -> PyResult<Vec<(usize, usize)>> {
    if offsets.first() != Some(&0) || offsets.last() != Some(&(n as i64)) {
        return Err(PyValueError::new_err(format!(
            "offsets 须以 0 开头、以数据长度 {n} 结尾"
        )));
    }
    offsets
        .windows(2)
        .map(|p| {
            if p[1] <= p[0] {
                Err(PyValueError::new_err("offsets 须严格递增"))
            } else {
                Ok((p[0] as usize, p[1] as usize))
            }
        })
        .collect()
}

type FitOutput<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<bool>>,
);

/// 批量拟合 SVI 微笑, 返回各条的 `(a, b, rho, m, sigma, 迭代次数, 是否收敛)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn _svi_fit<'py>(
    py: Python<'py>,
    k: PyReadonlyArray1<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    weight: PyReadonlyArray1<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    sigma_low: f64,
    sigma_high: f64,
    m_margin: f64,
    xtol: f64,
    ftol: f64,
    max_iter: usize,
) -> PyResult<FitOutput<'py>> {
    let (kk, ww, wt) = (slice(&k, "k")?, slice(&w, "w")?, slice(&weight, "weight")?);
    check_len(kk.len(), &[("w", ww.len()), ("weight", wt.len())])?;
    let offs = offsets
        .as_slice()
        .map_err(|_| PyValueError::new_err("offsets 必须是连续的一维 int64 数组"))?;
    let segs = segments(offs, kk.len())?;
    let search = SviSearch {
        sigma_low,
        sigma_high,
        m_margin,
        xtol,
        ftol,
        max_iter,
    };
    let fits: Vec<(Svi, usize, bool)> = py.detach(|| {
        segs.par_iter()
            .map(|&(s, e)| fit_svi(&kk[s..e], &ww[s..e], &wt[s..e], search))
            .collect()
    });
    let col = |f: fn(&Svi) -> f64| PyArray1::from_vec(py, fits.iter().map(|x| f(&x.0)).collect());
    Ok((
        col(|p| p.a),
        col(|p| p.b),
        col(|p| p.rho),
        col(|p| p.m),
        col(|p| p.sigma),
        PyArray1::from_vec(py, fits.iter().map(|x| x.1 as i64).collect()),
        PyArray1::from_vec(py, fits.iter().map(|x| x.2).collect()),
    ))
}

/// SVI 总方差(数组等长).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn _svi_total_variance<'py>(
    py: Python<'py>,
    k: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    m: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (kk, aa, bb, rr, mm, ss) = (
        slice(&k, "k")?,
        slice(&a, "a")?,
        slice(&b, "b")?,
        slice(&rho, "rho")?,
        slice(&m, "m")?,
        slice(&sigma, "sigma")?,
    );
    let n = kk.len();
    check_len(
        n,
        &[
            ("a", aa.len()),
            ("b", bb.len()),
            ("rho", rr.len()),
            ("m", mm.len()),
            ("sigma", ss.len()),
        ],
    )?;
    let out = (0..n)
        .map(|i| {
            svi_total_variance(
                kk[i],
                &Svi {
                    a: aa[i],
                    b: bb[i],
                    rho: rr[i],
                    m: mm[i],
                    sigma: ss[i],
                },
            )
        })
        .collect();
    Ok(PyArray1::from_vec(py, out))
}

type SabrOutput<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<bool>>,
);

/// 批量拟合 SABR 微笑(β 固定), 返回各条的 `(alpha, rho, nu, 迭代次数, 是否收敛)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn _sabr_fit<'py>(
    py: Python<'py>,
    forward: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    k: PyReadonlyArray1<'py, f64>,
    vol: PyReadonlyArray1<'py, f64>,
    weight: PyReadonlyArray1<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    beta: f64,
    xtol: f64,
    ftol: f64,
    max_iter: usize,
) -> PyResult<SabrOutput<'py>> {
    let (ff, tt) = (slice(&forward, "forward")?, slice(&t, "t")?);
    let (kk, vv, wt) = (
        slice(&k, "k")?,
        slice(&vol, "vol")?,
        slice(&weight, "weight")?,
    );
    check_len(kk.len(), &[("vol", vv.len()), ("weight", wt.len())])?;
    let offs = offsets
        .as_slice()
        .map_err(|_| PyValueError::new_err("offsets 必须是连续的一维 int64 数组"))?;
    let segs = segments(offs, kk.len())?;
    check_len(segs.len(), &[("forward", ff.len()), ("t", tt.len())])?;
    let search = SabrSearch {
        beta,
        xtol,
        ftol,
        max_iter,
    };
    let fits: Vec<([f64; 3], usize, bool)> = py.detach(|| {
        segs.par_iter()
            .enumerate()
            .map(|(i, &(s, e))| fit_sabr(ff[i], tt[i], &kk[s..e], &vv[s..e], &wt[s..e], search))
            .collect()
    });
    let col = |j: usize| PyArray1::from_vec(py, fits.iter().map(|x| x.0[j]).collect());
    Ok((
        col(0),
        col(1),
        col(2),
        PyArray1::from_vec(py, fits.iter().map(|x| x.1 as i64).collect()),
        PyArray1::from_vec(py, fits.iter().map(|x| x.2).collect()),
    ))
}

/// SABR 隐含波动率(数组等长; β 为标量).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn _sabr_vol<'py>(
    py: Python<'py>,
    forward: PyReadonlyArray1<'py, f64>,
    k: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
    beta: f64,
    rho: PyReadonlyArray1<'py, f64>,
    nu: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (ff, kk, tt) = (
        slice(&forward, "forward")?,
        slice(&k, "k")?,
        slice(&t, "t")?,
    );
    let (aa, rr, nn) = (
        slice(&alpha, "alpha")?,
        slice(&rho, "rho")?,
        slice(&nu, "nu")?,
    );
    let n = ff.len();
    check_len(
        n,
        &[
            ("k", kk.len()),
            ("t", tt.len()),
            ("alpha", aa.len()),
            ("rho", rr.len()),
            ("nu", nn.len()),
        ],
    )?;
    let out = (0..n)
        .map(|i| sabr_vol(ff[i], kk[i], tt[i], aa[i], beta, rr[i], nn[i]))
        .collect();
    Ok(PyArray1::from_vec(py, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SVI_SEARCH: SviSearch = SviSearch {
        sigma_low: 0.005,
        sigma_high: 1.0,
        m_margin: 1.0,
        xtol: 1e-10,
        ftol: 1e-16,
        max_iter: 2000,
    };

    #[test]
    fn nelder_mead_minimizes_rosenbrock() {
        let f = |x: &[f64; 2]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let run = nelder_mead(&f, [-1.2, 1.0], [0.1, 0.1], 1e-10, 1e-20, 5000);
        assert!(run.converged);
        assert!((run.x[0] - 1.0).abs() < 1e-6 && (run.x[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn svi_fit_recovers_parameters() {
        let truth = Svi {
            a: 0.01,
            b: 0.08,
            rho: -0.4,
            m: 0.02,
            sigma: 0.15,
        };
        let k: Vec<f64> = (0..21).map(|i| -0.4 + 0.04 * i as f64).collect();
        let w: Vec<f64> = k.iter().map(|&x| svi_total_variance(x, &truth)).collect();
        let (fit, _, converged) = fit_svi(&k, &w, &vec![1.0; k.len()], SVI_SEARCH);
        assert!(converged);
        for (got, want) in [
            (fit.a, truth.a),
            (fit.b, truth.b),
            (fit.rho, truth.rho),
            (fit.m, truth.m),
            (fit.sigma, truth.sigma),
        ] {
            assert!((got - want).abs() < 1e-6, "{fit:?}");
        }
    }

    #[test]
    fn svi_inner_respects_constraints() {
        // 单调上升的总方差: 无约束最优解会让 |d| > c, 约束解落在边界 d = c 上
        let k: Vec<f64> = (0..11).map(|i| -0.5 + 0.1 * i as f64).collect();
        let w: Vec<f64> = k.iter().map(|&x| 0.05 + 0.2 * x).collect();
        let ([a, d, c], _) = svi_inner(&k, &w, &vec![1.0; k.len()], 0.0, 0.2);
        assert!(a >= -1e-12 && d.abs() <= c + 1e-12 && d.abs() <= 0.8 - c + 1e-12);
    }

    #[test]
    fn sabr_atm_matches_hagan_expansion() {
        // ATM: σ = α/F^(1−β) · [1 + ((1−β)²α²/(24F^(2−2β)) + ρβνα/(4F^(1−β)) + (2−3ρ²)ν²/24)t]
        let (f, t, alpha, beta, rho, nu) = (100.0, 0.5, 2.0, 0.5, -0.3, 0.4);
        let fb = f64::powf(f, 1.0 - beta);
        let expected = alpha / fb
            * (1.0
                + ((1.0 - beta).powi(2) * alpha * alpha / (24.0 * fb * fb)
                    + rho * beta * nu * alpha / (4.0 * fb)
                    + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0)
                    * t);
        assert!((sabr_vol(f, f, t, alpha, beta, rho, nu) - expected).abs() < 1e-14);
        // z = 1e-6 处由级数切换为闭式, 两侧连续(ln(F/K) = z·α/(ν·F^(1−β)) 近似)
        let at = |z: f64| {
            sabr_vol(
                f,
                f * (-z * alpha / (nu * fb)).exp(),
                t,
                alpha,
                beta,
                rho,
                nu,
            )
        };
        assert!((at(0.999e-6) - at(1.001e-6)).abs() < 1e-9);
    }

    #[test]
    fn sabr_fit_recovers_parameters() {
        let (f, t, beta) = (500.0, 0.25, 1.0);
        let (alpha, rho, nu) = (0.35, -0.25, 1.2);
        let k: Vec<f64> = (0..15).map(|i| 400.0 + 15.0 * i as f64).collect();
        let vol: Vec<f64> = k
            .iter()
            .map(|&x| sabr_vol(f, x, t, alpha, beta, rho, nu))
            .collect();
        let search = SabrSearch {
            beta,
            xtol: 1e-10,
            ftol: 1e-20,
            max_iter: 5000,
        };
        let (p, _, converged) = fit_sabr(f, t, &k, &vol, &vec![1.0; k.len()], search);
        assert!(converged);
        assert!(
            (p[0] - alpha).abs() < 1e-6 && (p[1] - rho).abs() < 1e-5 && (p[2] - nu).abs() < 1e-5
        );
    }
}
