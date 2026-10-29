use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

use crate::indicators::is_zero_scaled;

#[path = "trend_directional.rs"]
mod trend_directional;
#[path = "trend_oscillators.rs"]
mod trend_oscillators;
#[path = "trend_regression.rs"]
mod trend_regression;

pub use trend_directional::{ADX, ADXR, DX, MINUS_DI, PLUS_DI};
pub use trend_oscillators::{AROON, AROONOSC, SAR, STOCH, ULTOSC};
pub use trend_regression::{
    BETA, CORREL, COVAR, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_R2,
    LINEARREG_SLOPE, TSF,
};

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct CCI {
    period: usize,
    c: f64,
    typical_prices: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl CCI {
    #[new]
    pub fn new(period: usize, c: f64) -> Self {
        CCI {
            period,
            c,
            typical_prices: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        let typical_price = (high + low + close) / 3.0;
        self.typical_prices.push_back(typical_price);
        if self.typical_prices.len() > self.period {
            self.typical_prices.pop_front();
        }
        if self.typical_prices.len() < self.period {
            self.current_value = None;
            return None;
        }
        // 每个窗口精确重算均值与平均偏差, 不做滑动累加
        let n = self.period as f64;
        let mean = self.typical_prices.iter().sum::<f64>() / n;
        let mean_deviation = self
            .typical_prices
            .iter()
            .map(|x| (x - mean).abs())
            .sum::<f64>()
            / n;
        let numerator = typical_price - mean;
        let divisor = self.c * mean_deviation;
        let scale = mean.abs();
        // TA-Lib 0.8.1: 分子或平均偏差相对均值可忽略(停牌、一字板)时输出 0
        self.current_value = Some(
            if is_zero_scaled(numerator, scale)
                || is_zero_scaled(mean_deviation, scale)
                || divisor == 0.0
            {
                0.0
            } else {
                numerator / divisor
            },
        );
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
    m.add_class::<CCI>()?;
    trend_directional::register_classes(m)?;
    trend_oscillators::register_classes(m)?;
    trend_regression::register_classes(m)?;
    Ok(())
}
