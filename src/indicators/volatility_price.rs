use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct MEDPRICE {
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MEDPRICE {
    #[new]
    pub fn new() -> Self {
        MEDPRICE {
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        skip_non_finite!(high, low);
        self.current_value = Some((high + low) / 2.0);
        self.current_value
    }

    pub fn update_many_hl<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if highs.len() != lows.len() {
            return Err(PyValueError::new_err("highs/lows length mismatch"));
        }
        let mut out = Vec::with_capacity(highs.len());
        for (high, low) in highs.into_iter().zip(lows.into_iter()) {
            out.push(self.update(high, low).unwrap_or(f64::NAN));
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
pub struct TYPPRICE {
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl TYPPRICE {
    #[new]
    pub fn new() -> Self {
        TYPPRICE {
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        self.current_value = Some((high + low + close) / 3.0);
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
pub struct WCLPRICE {
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl WCLPRICE {
    #[new]
    pub fn new() -> Self {
        WCLPRICE {
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        self.current_value = Some((high + low + 2.0 * close) / 4.0);
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
pub struct AVGPRICE {
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl AVGPRICE {
    #[new]
    pub fn new() -> Self {
        AVGPRICE {
            current_value: None,
        }
    }

    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(open, high, low, close);
        self.current_value = Some((open + high + low + close) / 4.0);
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
pub struct MIDPRICE {
    period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MIDPRICE {
    #[new]
    pub fn new(period: usize) -> Self {
        MIDPRICE {
            period,
            highs: VecDeque::with_capacity(period),
            lows: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        skip_non_finite!(high, low);
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
        let max_h = self.highs.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let min_l = self.lows.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        self.current_value = Some((max_h + min_l) / 2.0);
        self.current_value
    }

    pub fn update_many_hl<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if highs.len() != lows.len() {
            return Err(PyValueError::new_err("highs/lows length mismatch"));
        }
        let mut out = Vec::with_capacity(highs.len());
        for (high, low) in highs.into_iter().zip(lows.into_iter()) {
            out.push(self.update(high, low).unwrap_or(f64::NAN));
        }
        Ok(PyArray1::from_vec(py, out))
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MEDPRICE>()?;
    m.add_class::<TYPPRICE>()?;
    m.add_class::<WCLPRICE>()?;
    m.add_class::<AVGPRICE>()?;
    m.add_class::<MIDPRICE>()?;
    Ok(())
}
