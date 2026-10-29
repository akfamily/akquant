use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct ROC {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ROC {
    #[new]
    pub fn new(period: usize) -> Self {
        ROC {
            period,
            buffer: VecDeque::with_capacity(period + 1),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period + 1 {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period + 1 {
            self.current_value = None;
            return None;
        }
        let base = *self.buffer.front()?;
        // 与 TA-Lib 一致: 前值为 0 时输出 0
        if base == 0.0 {
            self.current_value = Some(0.0);
        } else {
            self.current_value = Some((value - base) / base * 100.0);
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

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct ROCP {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ROCP {
    #[new]
    pub fn new(period: usize) -> Self {
        ROCP {
            period,
            buffer: VecDeque::with_capacity(period + 1),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period + 1 {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period + 1 {
            self.current_value = None;
            return None;
        }
        let base = *self.buffer.front()?;
        // 与 TA-Lib 一致: 前值为 0 时输出 0
        if base == 0.0 {
            self.current_value = Some(0.0);
        } else {
            self.current_value = Some((value - base) / base);
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

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct ROCR {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ROCR {
    #[new]
    pub fn new(period: usize) -> Self {
        ROCR {
            period,
            buffer: VecDeque::with_capacity(period + 1),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period + 1 {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period + 1 {
            self.current_value = None;
            return None;
        }
        let base = *self.buffer.front()?;
        // 与 TA-Lib 一致: 前值为 0 时输出 0
        if base == 0.0 {
            self.current_value = Some(0.0);
        } else {
            self.current_value = Some(value / base);
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

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct ROCR100 {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ROCR100 {
    #[new]
    pub fn new(period: usize) -> Self {
        ROCR100 {
            period,
            buffer: VecDeque::with_capacity(period + 1),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period + 1 {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period + 1 {
            self.current_value = None;
            return None;
        }
        let base = *self.buffer.front()?;
        // 与 TA-Lib 一致: 前值为 0 时输出 0
        if base == 0.0 {
            self.current_value = Some(0.0);
        } else {
            self.current_value = Some((value / base) * 100.0);
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

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct MOM {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MOM {
    #[new]
    pub fn new(period: usize) -> Self {
        MOM {
            period,
            buffer: VecDeque::with_capacity(period + 1),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period + 1 {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period + 1 {
            self.current_value = None;
            return None;
        }
        let base = *self.buffer.front()?;
        self.current_value = Some(value - base);
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
    m.add_class::<ROC>()?;
    m.add_class::<ROCP>()?;
    m.add_class::<ROCR>()?;
    m.add_class::<ROCR100>()?;
    m.add_class::<MOM>()?;
    Ok(())
}
