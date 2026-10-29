use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

use crate::indicators::is_zero_scaled;

// RSI/CMO 共用的 Wilder 平均涨跌幅, 不对 Python 暴露。
#[derive(Debug, Clone)]
struct WilderGainLoss {
    period: usize,
    prev: Option<f64>,
    count: usize,
    gain: f64,
    loss: f64,
}

impl WilderGainLoss {
    fn new(period: usize) -> Self {
        WilderGainLoss {
            period,
            prev: None,
            count: 0,
            gain: 0.0,
            loss: 0.0,
        }
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        let prev = self.prev.replace(value)?;
        let change = value - prev;
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };
        let n = self.period as f64;
        if self.count < self.period {
            self.gain += gain;
            self.loss += loss;
            self.count += 1;
            if self.count < self.period {
                return None;
            }
            self.gain /= n;
            self.loss /= n;
        } else {
            self.gain = (self.gain * (n - 1.0) + gain) / n;
            self.loss = (self.loss * (n - 1.0) + loss) / n;
        }
        Some((self.gain, self.loss))
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct RSI {
    avg: WilderGainLoss,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl RSI {
    #[new]
    pub fn new(period: usize) -> Self {
        RSI {
            avg: WilderGainLoss::new(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        let (gain, loss) = self.avg.update(value)?;
        let sum = gain + loss;
        self.current_value = Some(if sum > 0.0 { 100.0 * gain / sum } else { 0.0 });
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
pub struct WILLR {
    period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl WILLR {
    #[new]
    pub fn new(period: usize) -> Self {
        WILLR {
            period,
            highs: VecDeque::with_capacity(period),
            lows: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        self.highs.push_back(high);
        self.lows.push_back(low);
        if self.highs.len() > self.period {
            self.highs.pop_front();
            self.lows.pop_front();
        }
        if self.highs.len() < self.period {
            self.current_value = None;
            return None;
        }
        let highest = self
            .highs
            .iter()
            .fold(f64::NEG_INFINITY, |acc, x| acc.max(*x));
        let lowest = self.lows.iter().fold(f64::INFINITY, |acc, x| acc.min(*x));
        let range = highest - lowest;
        // TA-Lib 0.8.1: 区间相对量级可忽略时记 0, 结果夹到 [-100, 0]
        self.current_value = Some(if is_zero_scaled(range, highest.abs() + lowest.abs()) {
            0.0
        } else {
            ((highest - close) / range * -100.0).clamp(-100.0, 0.0)
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
pub struct CMO {
    avg: WilderGainLoss,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl CMO {
    #[new]
    pub fn new(period: usize) -> Self {
        CMO {
            avg: WilderGainLoss::new(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, close: f64) -> Option<f64> {
        skip_non_finite!(close);
        let (gain, loss) = self.avg.update(close)?;
        let sum = gain + loss;
        self.current_value = Some(if sum > 0.0 {
            100.0 * (gain - loss) / sum
        } else {
            0.0
        });
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
    m.add_class::<RSI>()?;
    m.add_class::<WILLR>()?;
    m.add_class::<CMO>()?;
    Ok(())
}
