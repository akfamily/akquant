use crate::indicators::{is_zero, true_range};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

// 内部累加器, 不对 Python 暴露; +DM/-DM/TR 的 Wilder 累加沿用 TA-Lib 口径。
#[derive(Debug, Clone)]
struct DirectionalCore {
    period: usize,
    prev: Option<(f64, f64, f64)>,
    count: usize,
    plus_dm: f64,
    minus_dm: f64,
    tr: f64,
}

impl DirectionalCore {
    fn new(period: usize) -> Self {
        DirectionalCore {
            period,
            prev: None,
            count: 0,
            plus_dm: 0.0,
            minus_dm: 0.0,
            tr: 0.0,
        }
    }

    // 第 1..period-1 根只累加, 第 period 根起 `s = s - s/period + x`, 返回 (+DI, -DI)。
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        let (prev_high, prev_low, prev_close) = self.prev.replace((high, low, close))?;
        let up = high - prev_high;
        let down = prev_low - low;
        let plus = if up > 0.0 && up > down { up } else { 0.0 };
        let minus = if down > 0.0 && down > up { down } else { 0.0 };
        let tr = true_range(high, low, prev_close);
        self.count += 1;
        if self.count < self.period {
            self.plus_dm += plus;
            self.minus_dm += minus;
            self.tr += tr;
            return None;
        }
        let n = self.period as f64;
        self.plus_dm = self.plus_dm - self.plus_dm / n + plus;
        self.minus_dm = self.minus_dm - self.minus_dm / n + minus;
        self.tr = self.tr - self.tr / n + tr;
        if self.tr <= 0.0 {
            return Some((0.0, 0.0));
        }
        Some((
            100.0 * self.plus_dm / self.tr,
            100.0 * self.minus_dm / self.tr,
        ))
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct ADX {
    period: usize,
    dx: DX,
    seed_sum: f64,
    seed_count: usize,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ADX {
    #[new]
    pub fn new(period: usize) -> Self {
        ADX {
            period,
            dx: DX::new(period),
            seed_sum: 0.0,
            seed_count: 0,
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        let dx = self.dx.update(high, low, close)?;
        let n = self.period as f64;
        match self.current_value {
            Some(prev) => self.current_value = Some((prev * (n - 1.0) + dx) / n),
            None => {
                self.seed_sum += dx;
                self.seed_count += 1;
                if self.seed_count < self.period {
                    return None;
                }
                self.current_value = Some(self.seed_sum / n);
            }
        }
        self.current_value
    }

    pub fn update_many_hlc<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if highs.len() != lows.len() || highs.len() != closes.len() {
            return Err(PyValueError::new_err("highs/lows/closes length mismatch"));
        }
        let mut out = Vec::with_capacity(highs.len());
        for (high, (low, close)) in highs
            .into_iter()
            .zip(lows.into_iter().zip(closes.into_iter()))
        {
            out.push(self.update(high, low, close).unwrap_or(f64::NAN));
        }
        Ok(PyArray1::from_vec(py, out))
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct DX {
    core: DirectionalCore,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl DX {
    #[new]
    pub fn new(period: usize) -> Self {
        DX {
            core: DirectionalCore::new(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        let (plus, minus) = self.core.update(high, low, close)?;
        let sum = plus + minus;
        // 分母为 0 时沿用上一个 DX(首个之前记 0), 与 TA-Lib 一致。
        let dx = if is_zero(sum) {
            self.current_value.unwrap_or(0.0)
        } else {
            100.0 * (plus - minus).abs() / sum
        };
        self.current_value = Some(dx);
        self.current_value
    }

    pub fn update_many_hlc<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if highs.len() != lows.len() || highs.len() != closes.len() {
            return Err(PyValueError::new_err("highs/lows/closes length mismatch"));
        }
        let mut out = Vec::with_capacity(highs.len());
        for (high, (low, close)) in highs
            .into_iter()
            .zip(lows.into_iter().zip(closes.into_iter()))
        {
            out.push(self.update(high, low, close).unwrap_or(f64::NAN));
        }
        Ok(PyArray1::from_vec(py, out))
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct ADXR {
    period: usize,
    adx: ADX,
    history: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ADXR {
    #[new]
    pub fn new(period: usize) -> Self {
        ADXR {
            period,
            adx: ADX::new(period),
            history: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        let adx_now = self.adx.update(high, low, close)?;
        self.history.push_back(adx_now);
        if self.history.len() > self.period {
            self.history.pop_front();
        }
        if self.history.len() < self.period {
            return None;
        }
        // (ADX[i] + ADX[i-(period-1)]) / 2, 与 TA-Lib 一致。
        self.current_value = Some((adx_now + *self.history.front()?) / 2.0);
        self.current_value
    }

    pub fn update_many_hlc<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if highs.len() != lows.len() || highs.len() != closes.len() {
            return Err(PyValueError::new_err("highs/lows/closes length mismatch"));
        }
        let mut out = Vec::with_capacity(highs.len());
        for (high, (low, close)) in highs
            .into_iter()
            .zip(lows.into_iter().zip(closes.into_iter()))
        {
            out.push(self.update(high, low, close).unwrap_or(f64::NAN));
        }
        Ok(PyArray1::from_vec(py, out))
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct PLUS_DI {
    core: DirectionalCore,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PLUS_DI {
    #[new]
    pub fn new(period: usize) -> Self {
        PLUS_DI {
            core: DirectionalCore::new(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        let (plus, _) = self.core.update(high, low, close)?;
        self.current_value = Some(plus);
        self.current_value
    }

    pub fn update_many_hlc<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if highs.len() != lows.len() || highs.len() != closes.len() {
            return Err(PyValueError::new_err("highs/lows/closes length mismatch"));
        }
        let mut out = Vec::with_capacity(highs.len());
        for (high, (low, close)) in highs
            .into_iter()
            .zip(lows.into_iter().zip(closes.into_iter()))
        {
            out.push(self.update(high, low, close).unwrap_or(f64::NAN));
        }
        Ok(PyArray1::from_vec(py, out))
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct MINUS_DI {
    core: DirectionalCore,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MINUS_DI {
    #[new]
    pub fn new(period: usize) -> Self {
        MINUS_DI {
            core: DirectionalCore::new(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        let (_, minus) = self.core.update(high, low, close)?;
        self.current_value = Some(minus);
        self.current_value
    }

    pub fn update_many_hlc<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if highs.len() != lows.len() || highs.len() != closes.len() {
            return Err(PyValueError::new_err("highs/lows/closes length mismatch"));
        }
        let mut out = Vec::with_capacity(highs.len());
        for (high, (low, close)) in highs
            .into_iter()
            .zip(lows.into_iter().zip(closes.into_iter()))
        {
            out.push(self.update(high, low, close).unwrap_or(f64::NAN));
        }
        Ok(PyArray1::from_vec(py, out))
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ADX>()?;
    m.add_class::<DX>()?;
    m.add_class::<ADXR>()?;
    m.add_class::<PLUS_DI>()?;
    m.add_class::<MINUS_DI>()?;
    Ok(())
}
