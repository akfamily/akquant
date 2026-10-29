use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

use crate::indicators::{TA_EPSILON, is_zero};

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct LINEARREG {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl LINEARREG {
    #[new]
    pub fn new(period: usize) -> Self {
        LINEARREG {
            period,
            buffer: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period {
            self.current_value = None;
            return None;
        }
        let n = self.period as f64;
        let sum_x = n * (n - 1.0) / 2.0;
        let sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;
        let sum_y = self.buffer.iter().sum::<f64>();
        let sum_xy = self
            .buffer
            .iter()
            .enumerate()
            .map(|(i, y)| i as f64 * *y)
            .sum::<f64>();
        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() <= f64::EPSILON {
            self.current_value = Some(f64::NAN);
            return self.current_value;
        }
        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;
        self.current_value = Some(intercept + slope * (n - 1.0));
        self.current_value
    }

    pub fn update_many<'py>(
        &mut self,
        py: Python<'py>,
        values: Vec<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            out.push(self.update(value).unwrap_or(f64::NAN));
        }
        PyArray1::from_vec(py, out)
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
pub struct LINEARREG_SLOPE {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl LINEARREG_SLOPE {
    #[new]
    pub fn new(period: usize) -> Self {
        LINEARREG_SLOPE {
            period,
            buffer: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period {
            self.current_value = None;
            return None;
        }
        let n = self.period as f64;
        let sum_x = n * (n - 1.0) / 2.0;
        let sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;
        let sum_y = self.buffer.iter().sum::<f64>();
        let sum_xy = self
            .buffer
            .iter()
            .enumerate()
            .map(|(i, y)| i as f64 * *y)
            .sum::<f64>();
        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() <= f64::EPSILON {
            self.current_value = Some(f64::NAN);
            return self.current_value;
        }
        self.current_value = Some((n * sum_xy - sum_x * sum_y) / denom);
        self.current_value
    }

    pub fn update_many<'py>(
        &mut self,
        py: Python<'py>,
        values: Vec<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            out.push(self.update(value).unwrap_or(f64::NAN));
        }
        PyArray1::from_vec(py, out)
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
pub struct LINEARREG_INTERCEPT {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl LINEARREG_INTERCEPT {
    #[new]
    pub fn new(period: usize) -> Self {
        LINEARREG_INTERCEPT {
            period,
            buffer: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period {
            self.current_value = None;
            return None;
        }
        let n = self.period as f64;
        let sum_x = n * (n - 1.0) / 2.0;
        let sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;
        let sum_y = self.buffer.iter().sum::<f64>();
        let sum_xy = self
            .buffer
            .iter()
            .enumerate()
            .map(|(i, y)| i as f64 * *y)
            .sum::<f64>();
        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() <= f64::EPSILON {
            self.current_value = Some(f64::NAN);
            return self.current_value;
        }
        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        self.current_value = Some((sum_y - slope * sum_x) / n);
        self.current_value
    }

    pub fn update_many<'py>(
        &mut self,
        py: Python<'py>,
        values: Vec<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            out.push(self.update(value).unwrap_or(f64::NAN));
        }
        PyArray1::from_vec(py, out)
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
pub struct LINEARREG_ANGLE {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl LINEARREG_ANGLE {
    #[new]
    pub fn new(period: usize) -> Self {
        LINEARREG_ANGLE {
            period,
            buffer: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period {
            self.current_value = None;
            return None;
        }
        let n = self.period as f64;
        let sum_x = n * (n - 1.0) / 2.0;
        let sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;
        let sum_y = self.buffer.iter().sum::<f64>();
        let sum_xy = self
            .buffer
            .iter()
            .enumerate()
            .map(|(i, y)| i as f64 * *y)
            .sum::<f64>();
        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() <= f64::EPSILON {
            self.current_value = Some(f64::NAN);
            return self.current_value;
        }
        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        self.current_value = Some(slope.atan().to_degrees());
        self.current_value
    }

    pub fn update_many<'py>(
        &mut self,
        py: Python<'py>,
        values: Vec<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            out.push(self.update(value).unwrap_or(f64::NAN));
        }
        PyArray1::from_vec(py, out)
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct TSF {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl TSF {
    #[new]
    pub fn new(period: usize) -> Self {
        TSF {
            period,
            buffer: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period {
            self.current_value = None;
            return None;
        }
        let n = self.period as f64;
        let sum_x = n * (n - 1.0) / 2.0;
        let sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;
        let sum_y = self.buffer.iter().sum::<f64>();
        let sum_xy = self
            .buffer
            .iter()
            .enumerate()
            .map(|(i, y)| i as f64 * *y)
            .sum::<f64>();
        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() <= f64::EPSILON {
            self.current_value = Some(f64::NAN);
            return self.current_value;
        }
        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;
        self.current_value = Some(intercept + slope * n);
        self.current_value
    }

    pub fn update_many<'py>(
        &mut self,
        py: Python<'py>,
        values: Vec<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            out.push(self.update(value).unwrap_or(f64::NAN));
        }
        PyArray1::from_vec(py, out)
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct CORREL {
    period: usize,
    xs: VecDeque<f64>,
    ys: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl CORREL {
    #[new]
    pub fn new(period: usize) -> Self {
        CORREL {
            period,
            xs: VecDeque::with_capacity(period),
            ys: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, x: f64, y: f64) -> Option<f64> {
        skip_non_finite!(x, y);
        // 周期为 0 时不计算, 避免空窗口 panic
        if self.period == 0 {
            return None;
        }
        self.xs.push_back(x);
        self.ys.push_back(y);
        if self.xs.len() > self.period {
            self.xs.pop_front();
            self.ys.pop_front();
        }
        if self.xs.len() < self.period {
            self.current_value = None;
            return None;
        }
        // 窗口每个值先减去窗口首值再求和, 平盘窗口的离差平方和恰为 0
        let (shift_x, shift_y) = (self.xs[0], self.ys[0]);
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (a, b) in self.xs.iter().zip(self.ys.iter()) {
            let dx = a - shift_x;
            let dy = b - shift_y;
            sx += dx;
            sy += dy;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }
        let inv = 1.0 / self.period as f64;
        let ss_x = sxx - sx * sx * inv;
        let ss_y = syy - sy * sy * inv;
        let sp_xy = sxy - sx * sy * inv;
        // TA-Lib 0.8.1: 任一序列在窗口内没有离散度时记 0, 结果夹到 [-1, 1]
        self.current_value = Some(
            if ss_x > TA_EPSILON * sxx && ss_y > TA_EPSILON * syy && ss_x * ss_y > 0.0 {
                (sp_xy / (ss_x * ss_y).sqrt()).clamp(-1.0, 1.0)
            } else {
                0.0
            },
        );
        self.current_value
    }

    pub fn update_many_dual<'py>(
        &mut self,
        py: Python<'py>,
        xs: Vec<f64>,
        ys: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if xs.len() != ys.len() {
            return Err(PyValueError::new_err("xs/ys length mismatch"));
        }
        let mut out = Vec::with_capacity(xs.len());
        for (x, y) in xs.into_iter().zip(ys.into_iter()) {
            out.push(self.update(x, y).unwrap_or(f64::NAN));
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
pub struct BETA {
    period: usize,
    prev: Option<(f64, f64)>,
    returns: VecDeque<(f64, f64)>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl BETA {
    #[new]
    pub fn new(period: usize) -> Self {
        BETA {
            period,
            prev: None,
            returns: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, x: f64, y: f64) -> Option<f64> {
        skip_non_finite!(x, y);
        // TA-Lib BETA: 先换成单期收益率, 再算 y 收益对 x 收益的回归斜率
        let Some((prev_x, prev_y)) = self.prev.replace((x, y)) else {
            self.current_value = None;
            return None;
        };
        let rx = if is_zero(prev_x) {
            0.0
        } else {
            (x - prev_x) / prev_x
        };
        let ry = if is_zero(prev_y) {
            0.0
        } else {
            (y - prev_y) / prev_y
        };
        self.returns.push_back((rx, ry));
        if self.returns.len() > self.period {
            self.returns.pop_front();
        }
        if self.returns.len() < self.period {
            self.current_value = None;
            return None;
        }
        let (mut sxx, mut sx, mut sxy, mut sy) = (0.0, 0.0, 0.0, 0.0);
        for (a, b) in &self.returns {
            sxx += a * a;
            sx += a;
            sxy += a * b;
            sy += b;
        }
        let n = self.period as f64;
        let den = n * sxx - sx * sx;
        self.current_value = Some(if den > TA_EPSILON * n * sxx {
            (n * sxy - sx * sy) / den
        } else {
            0.0
        });
        self.current_value
    }

    pub fn update_many_dual<'py>(
        &mut self,
        py: Python<'py>,
        xs: Vec<f64>,
        ys: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if xs.len() != ys.len() {
            return Err(PyValueError::new_err("xs/ys length mismatch"));
        }
        let mut out = Vec::with_capacity(xs.len());
        for (x, y) in xs.into_iter().zip(ys.into_iter()) {
            out.push(self.update(x, y).unwrap_or(f64::NAN));
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
pub struct COVAR {
    period: usize,
    buffer_x: VecDeque<f64>,
    buffer_y: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl COVAR {
    #[new]
    pub fn new(period: usize) -> Self {
        COVAR {
            period,
            buffer_x: VecDeque::with_capacity(period),
            buffer_y: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, x: f64, y: f64) -> Option<f64> {
        skip_non_finite!(x, y);
        self.buffer_x.push_back(x);
        self.buffer_y.push_back(y);
        if self.buffer_x.len() > self.period {
            self.buffer_x.pop_front();
            self.buffer_y.pop_front();
        }
        if self.buffer_x.len() < self.period {
            self.current_value = None;
            return None;
        }
        let mean_x = self.buffer_x.iter().sum::<f64>() / self.period as f64;
        let mean_y = self.buffer_y.iter().sum::<f64>() / self.period as f64;
        let cov = self
            .buffer_x
            .iter()
            .zip(self.buffer_y.iter())
            .map(|(a, b)| (*a - mean_x) * (*b - mean_y))
            .sum::<f64>()
            / self.period as f64;
        self.current_value = Some(cov);
        self.current_value
    }

    pub fn update_many_dual<'py>(
        &mut self,
        py: Python<'py>,
        xs: Vec<f64>,
        ys: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if xs.len() != ys.len() {
            return Err(PyValueError::new_err("xs/ys length mismatch"));
        }
        let mut out = Vec::with_capacity(xs.len());
        for (x, y) in xs.into_iter().zip(ys.into_iter()) {
            out.push(self.update(x, y).unwrap_or(f64::NAN));
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
pub struct LINEARREG_R2 {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl LINEARREG_R2 {
    #[new]
    pub fn new(period: usize) -> Self {
        LINEARREG_R2 {
            period,
            buffer: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period {
            self.current_value = None;
            return None;
        }

        let n = self.period as f64;
        let sum_x = n * (n - 1.0) / 2.0;
        let sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;
        let sum_y = self.buffer.iter().sum::<f64>();
        let sum_xy = self
            .buffer
            .iter()
            .enumerate()
            .map(|(i, y)| i as f64 * *y)
            .sum::<f64>();
        let sum_y2 = self.buffer.iter().map(|v| v * v).sum::<f64>();

        let num = n * sum_xy - sum_x * sum_y;
        let den_x = n * sum_x2 - sum_x * sum_x;
        let den_y = n * sum_y2 - sum_y * sum_y;
        let den = (den_x * den_y).sqrt();
        if den <= f64::EPSILON {
            self.current_value = Some(f64::NAN);
        } else {
            let r = num / den;
            self.current_value = Some(r * r);
        }
        self.current_value
    }

    pub fn update_many<'py>(
        &mut self,
        py: Python<'py>,
        values: Vec<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            out.push(self.update(value).unwrap_or(f64::NAN));
        }
        PyArray1::from_vec(py, out)
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LINEARREG>()?;
    m.add_class::<LINEARREG_SLOPE>()?;
    m.add_class::<LINEARREG_INTERCEPT>()?;
    m.add_class::<LINEARREG_ANGLE>()?;
    m.add_class::<TSF>()?;
    m.add_class::<CORREL>()?;
    m.add_class::<BETA>()?;
    m.add_class::<COVAR>()?;
    m.add_class::<LINEARREG_R2>()?;
    Ok(())
}
