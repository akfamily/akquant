use crate::indicators::true_range;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

// 窗口内每个值减去窗口首值后的方差: 平盘窗口恰为 0, 避免大数相减的舍入残差;
// 负的舍入残差记 0。BollingerBands、STDDEV、VAR 共用。
fn window_variance(window: &VecDeque<f64>) -> f64 {
    let n = window.len() as f64;
    let shift = window[0];
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for x in window {
        let d = x - shift;
        sum += d;
        sum_sq += d * d;
    }
    let mean = sum / n;
    (sum_sq / n - mean * mean).max(0.0)
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct BollingerBands {
    period: usize,
    multiplier: f64,
    buffer: VecDeque<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl BollingerBands {
    #[new]
    pub fn new(period: usize, multiplier: f64) -> Self {
        BollingerBands {
            period,
            multiplier,
            buffer: VecDeque::with_capacity(period),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        skip_non_finite!(value);
        // 周期为 0 时不计算, 避免空窗口 panic
        if self.period == 0 {
            return None;
        }
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        self.bands()
    }

    pub fn update_many<'py>(
        &mut self,
        py: Python<'py>,
        values: Vec<f64>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        let mut upper = Vec::with_capacity(values.len());
        let mut middle = Vec::with_capacity(values.len());
        let mut lower = Vec::with_capacity(values.len());
        for value in values {
            if let Some((u, m, l)) = self.update(value) {
                upper.push(u);
                middle.push(m);
                lower.push(l);
            } else {
                upper.push(f64::NAN);
                middle.push(f64::NAN);
                lower.push(f64::NAN);
            }
        }
        (
            PyArray1::from_vec(py, upper),
            PyArray1::from_vec(py, middle),
            PyArray1::from_vec(py, lower),
        )
    }

    #[getter]
    pub fn value(&self) -> Option<(f64, f64, f64)> {
        self.bands()
    }
}

impl BollingerBands {
    // 窗口已满时按当前窗口精确重算 (upper, middle, lower)。
    fn bands(&self) -> Option<(f64, f64, f64)> {
        if self.buffer.len() < self.period {
            return None;
        }
        let middle = self.buffer.iter().sum::<f64>() / self.period as f64;
        let width = self.multiplier * window_variance(&self.buffer).sqrt();
        Some((middle + width, middle, middle - width))
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct ATR {
    period: usize,
    prev_close: Option<f64>,
    seed_sum: f64,
    seed_count: usize,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ATR {
    #[new]
    pub fn new(period: usize) -> Self {
        ATR {
            period,
            prev_close: None,
            seed_sum: 0.0,
            seed_count: 0,
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        let prev_close = self.prev_close.replace(close)?;
        let tr = true_range(high, low, prev_close);
        let n = self.period as f64;
        match self.current_value {
            Some(prev) => self.current_value = Some((prev * (n - 1.0) + tr) / n),
            None => {
                self.seed_sum += tr;
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
pub struct NATR {
    atr: ATR,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl NATR {
    #[new]
    pub fn new(period: usize) -> Self {
        NATR {
            atr: ATR::new(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        let atr = self.atr.update(high, low, close)?;
        self.current_value = Some(if close == 0.0 {
            0.0
        } else {
            atr / close * 100.0
        });
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
pub struct TRANGE {
    prev_close: Option<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl TRANGE {
    #[new]
    pub fn new() -> Self {
        TRANGE {
            prev_close: None,
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        let prev_close = self.prev_close.replace(close)?;
        self.current_value = Some(true_range(high, low, prev_close));
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
pub struct STDDEV {
    period: usize,
    nbdev: f64,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl STDDEV {
    #[new]
    pub fn new(period: usize, nbdev: f64) -> Self {
        STDDEV {
            period,
            nbdev,
            buffer: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        // 周期为 0 时不计算, 避免空窗口 panic
        if self.period == 0 {
            return None;
        }
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period {
            self.current_value = None;
            return None;
        }
        self.current_value = Some(window_variance(&self.buffer).sqrt() * self.nbdev);
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
pub struct VAR {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl VAR {
    #[new]
    pub fn new(period: usize, nbdev: f64) -> Self {
        // TA-Lib 的 VAR 不使用 nbdev, 保留参数只为不破坏构造签名
        let _ = nbdev;
        VAR {
            period,
            buffer: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        // 周期为 0 时不计算, 避免空窗口 panic
        if self.period == 0 {
            return None;
        }
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period {
            self.current_value = None;
            return None;
        }
        self.current_value = Some(window_variance(&self.buffer));
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
    m.add_class::<BollingerBands>()?;
    m.add_class::<ATR>()?;
    m.add_class::<NATR>()?;
    m.add_class::<TRANGE>()?;
    m.add_class::<STDDEV>()?;
    m.add_class::<VAR>()?;
    Ok(())
}
