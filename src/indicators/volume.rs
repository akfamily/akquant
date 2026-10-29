use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

use crate::indicators::is_zero_scaled;

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct OBV {
    prev_close: Option<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl OBV {
    #[new]
    pub fn new() -> Self {
        OBV {
            prev_close: None,
            current_value: None,
        }
    }

    pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        skip_non_finite!(close, volume);
        // TA-Lib: 首根直接取该根成交量
        let next = match (self.prev_close.replace(close), self.current_value) {
            (Some(prev), Some(obv)) if close > prev => obv + volume,
            (Some(prev), Some(obv)) if close < prev => obv - volume,
            (Some(_), Some(obv)) => obv,
            _ => volume,
        };
        self.current_value = Some(next);
        self.current_value
    }

    pub fn update_many_dual<'py>(
        &mut self,
        py: Python<'py>,
        closes: Vec<f64>,
        volumes: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if closes.len() != volumes.len() {
            return Err(PyValueError::new_err("closes/volumes length mismatch"));
        }
        let mut out = Vec::with_capacity(closes.len());
        for (close, volume) in closes.into_iter().zip(volumes.into_iter()) {
            out.push(self.update(close, volume).unwrap_or(f64::NAN));
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
pub struct MFI {
    period: usize,
    prev_typical: Option<f64>,
    flows: VecDeque<(f64, f64)>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MFI {
    #[new]
    pub fn new(period: usize) -> Self {
        MFI {
            period,
            prev_typical: None,
            flows: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        skip_non_finite!(high, low, close, volume);
        let typical = (high + low + close) / 3.0;
        let Some(prev_typical) = self.prev_typical.replace(typical) else {
            self.current_value = None;
            return None;
        };
        let change = typical - prev_typical;
        // TA-Lib 0.8.1: 典型价变动相对量级可忽略时不计资金流
        let flow = if is_zero_scaled(change, typical.abs() + prev_typical.abs()) {
            0.0
        } else {
            typical * volume
        };
        self.flows.push_back(if change < 0.0 {
            (0.0, flow)
        } else {
            (flow, 0.0)
        });
        if self.flows.len() > self.period {
            self.flows.pop_front();
        }
        if self.flows.len() < self.period {
            self.current_value = None;
            return None;
        }
        // 每个窗口精确重算资金流之和
        let (mut positive, mut negative) = (0.0, 0.0);
        for (p, n) in &self.flows {
            positive += p;
            negative += n;
        }
        let total = positive + negative;
        self.current_value = Some(if total <= 0.0 {
            0.0
        } else {
            100.0 * positive.clamp(0.0, total) / total
        });
        self.current_value
    }

    pub fn update_many_hlcv<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
        volumes: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if highs.len() != lows.len() || highs.len() != closes.len() || highs.len() != volumes.len()
        {
            return Err(PyValueError::new_err(
                "highs/lows/closes/volumes length mismatch",
            ));
        }
        let mut out = Vec::with_capacity(highs.len());
        for ((high, low), (close, volume)) in highs
            .into_iter()
            .zip(lows.into_iter())
            .zip(closes.into_iter().zip(volumes.into_iter()))
        {
            out.push(self.update(high, low, close, volume).unwrap_or(f64::NAN));
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
pub struct BOP {
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl BOP {
    #[new]
    pub fn new() -> Self {
        BOP {
            current_value: None,
        }
    }

    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(open, high, low, close);
        let denom = high - low;
        let value = if denom.abs() <= f64::EPSILON {
            0.0
        } else {
            (close - open) / denom
        };
        self.current_value = Some(value);
        self.current_value
    }

    pub fn update_many_ohlc<'py>(
        &mut self,
        py: Python<'py>,
        opens: Vec<f64>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if opens.len() != highs.len() || opens.len() != lows.len() || opens.len() != closes.len() {
            return Err(PyValueError::new_err(
                "opens/highs/lows/closes length mismatch",
            ));
        }
        let mut out = Vec::with_capacity(opens.len());
        for ((open, high), (low, close)) in opens
            .into_iter()
            .zip(highs.into_iter())
            .zip(lows.into_iter().zip(closes.into_iter()))
        {
            out.push(self.update(open, high, low, close).unwrap_or(f64::NAN));
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
pub struct AD {
    current_value: Option<f64>,
    current_ad: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl AD {
    #[new]
    pub fn new() -> Self {
        AD {
            current_value: None,
            current_ad: 0.0,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        skip_non_finite!(high, low, close, volume);
        let denom = high - low;
        let mfm = if denom.abs() <= f64::EPSILON {
            0.0
        } else {
            ((close - low) - (high - close)) / denom
        };
        let mfv = mfm * volume;
        self.current_ad += mfv;
        self.current_value = Some(self.current_ad);
        self.current_value
    }

    pub fn update_many_hlcv<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
        volumes: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if highs.len() != lows.len() || highs.len() != closes.len() || highs.len() != volumes.len()
        {
            return Err(PyValueError::new_err(
                "highs/lows/closes/volumes length mismatch",
            ));
        }
        let mut out = Vec::with_capacity(highs.len());
        for ((high, low), (close, volume)) in highs
            .into_iter()
            .zip(lows.into_iter())
            .zip(closes.into_iter().zip(volumes.into_iter()))
        {
            out.push(self.update(high, low, close, volume).unwrap_or(f64::NAN));
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
pub struct ADOSC {
    slow_period: usize,
    current_ad: f64,
    fast_ema: Option<f64>,
    slow_ema: Option<f64>,
    k_fast: f64,
    k_slow: f64,
    count: usize,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ADOSC {
    #[new]
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        ADOSC {
            slow_period,
            current_ad: 0.0,
            fast_ema: None,
            slow_ema: None,
            k_fast: 2.0 / (fast_period as f64 + 1.0),
            k_slow: 2.0 / (slow_period as f64 + 1.0),
            count: 0,
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        skip_non_finite!(high, low, close, volume);
        let denom = high - low;
        let mfm = if denom.abs() <= f64::EPSILON {
            0.0
        } else {
            ((close - low) - (high - close)) / denom
        };
        let mfv = mfm * volume;
        self.current_ad += mfv;
        self.count += 1;
        self.fast_ema = Some(match self.fast_ema {
            Some(prev) => prev + self.k_fast * (self.current_ad - prev),
            None => self.current_ad,
        });
        self.slow_ema = Some(match self.slow_ema {
            Some(prev) => prev + self.k_slow * (self.current_ad - prev),
            None => self.current_ad,
        });
        if self.count < self.slow_period {
            self.current_value = None;
            return None;
        }
        self.current_value = Some(self.fast_ema? - self.slow_ema?);
        self.current_value
    }

    pub fn update_many_hlcv<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
        volumes: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if highs.len() != lows.len() || highs.len() != closes.len() || highs.len() != volumes.len()
        {
            return Err(PyValueError::new_err(
                "highs/lows/closes/volumes length mismatch",
            ));
        }
        let mut out = Vec::with_capacity(highs.len());
        for ((high, low), (close, volume)) in highs
            .into_iter()
            .zip(lows.into_iter())
            .zip(closes.into_iter().zip(volumes.into_iter()))
        {
            out.push(self.update(high, low, close, volume).unwrap_or(f64::NAN));
        }
        Ok(PyArray1::from_vec(py, out))
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OBV>()?;
    m.add_class::<MFI>()?;
    m.add_class::<BOP>()?;
    m.add_class::<AD>()?;
    m.add_class::<ADOSC>()?;
    Ok(())
}
