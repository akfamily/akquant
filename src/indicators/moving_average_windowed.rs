use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

// MAXINDEX/MININDEX 的状态: 复刻 TA-Lib 的并列规则(极值滑出窗口后重扫取最早的一根,
// 新值与当前极值相等时取新的一根)。下标从首个有效值起数, 不对 Python 暴露。
#[derive(Debug, Clone)]
struct ExtremeIndex {
    period: usize,
    find_max: bool,
    window: VecDeque<f64>,
    count: usize,
    best_idx: Option<usize>,
    best: f64,
}

impl ExtremeIndex {
    fn new(period: usize, find_max: bool) -> Self {
        ExtremeIndex {
            period,
            find_max,
            window: VecDeque::with_capacity(period),
            count: 0,
            best_idx: None,
            best: 0.0,
        }
    }

    // 喂一个(已判为有限的)值; 窗口满后返回极值的绝对下标。
    fn update(&mut self, value: f64) -> Option<usize> {
        // 周期为 0 时不计算, 避免空窗口 panic
        if self.period == 0 {
            return None;
        }
        let today = self.count;
        self.count += 1;
        self.window.push_back(value);
        if self.window.len() > self.period {
            self.window.pop_front();
        }
        if self.window.len() < self.period {
            return None;
        }
        let trailing = today + 1 - self.period;
        let find_max = self.find_max;
        let better = |a: f64, b: f64, strict: bool| match (find_max, strict) {
            (true, true) => a > b,
            (true, false) => a >= b,
            (false, true) => a < b,
            (false, false) => a <= b,
        };
        match self.best_idx {
            Some(idx) if idx >= trailing => {
                if better(value, self.best, false) {
                    self.best_idx = Some(today);
                    self.best = value;
                }
            }
            _ => {
                let mut best_idx = trailing;
                let mut best = self.window[0];
                for (offset, candidate) in self.window.iter().enumerate().skip(1) {
                    if better(*candidate, best, true) {
                        best_idx = trailing + offset;
                        best = *candidate;
                    }
                }
                self.best_idx = Some(best_idx);
                self.best = best;
            }
        }
        self.best_idx
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct MIDPOINT {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MIDPOINT {
    #[new]
    pub fn new(period: usize) -> Self {
        MIDPOINT {
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
        let max_v = self.buffer.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let min_v = self.buffer.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        self.current_value = Some((max_v + min_v) / 2.0);
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
pub struct MAX {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MAX {
    #[new]
    pub fn new(period: usize) -> Self {
        MAX {
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
        let max_v = self.buffer.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        self.current_value = Some(max_v);
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
pub struct MIN {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MIN {
    #[new]
    pub fn new(period: usize) -> Self {
        MIN {
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
        let min_v = self.buffer.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        self.current_value = Some(min_v);
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
pub struct MAXINDEX {
    inner: ExtremeIndex,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MAXINDEX {
    #[new]
    pub fn new(period: usize) -> Self {
        MAXINDEX {
            inner: ExtremeIndex::new(period, true),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        let Some(index) = self.inner.update(value) else {
            self.current_value = None;
            return None;
        };
        self.current_value = Some(index as f64);
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
pub struct MININDEX {
    inner: ExtremeIndex,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MININDEX {
    #[new]
    pub fn new(period: usize) -> Self {
        MININDEX {
            inner: ExtremeIndex::new(period, false),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        let Some(index) = self.inner.update(value) else {
            self.current_value = None;
            return None;
        };
        self.current_value = Some(index as f64);
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
pub struct MINMAXINDEX {
    min: ExtremeIndex,
    max: ExtremeIndex,
    current_value: Option<(f64, f64)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MINMAXINDEX {
    #[new]
    pub fn new(period: usize) -> Self {
        MINMAXINDEX {
            min: ExtremeIndex::new(period, false),
            max: ExtremeIndex::new(period, true),
            current_value: None,
        }
    }

    // 返回 (minidx, maxidx), 与 TA-Lib 的输出顺序一致。
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        skip_non_finite!(value);
        let min = self.min.update(value);
        let max = self.max.update(value);
        let (Some(min), Some(max)) = (min, max) else {
            self.current_value = None;
            return None;
        };
        self.current_value = Some((min as f64, max as f64));
        self.current_value
    }

    pub fn update_many_pair<'py>(
        &mut self,
        py: Python<'py>,
        values: Vec<f64>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let mut first = Vec::with_capacity(values.len());
        let mut second = Vec::with_capacity(values.len());
        for value in values {
            if let Some((f, s)) = self.update(value) {
                first.push(f);
                second.push(s);
            } else {
                first.push(f64::NAN);
                second.push(f64::NAN);
            }
        }
        (
            PyArray1::from_vec(py, first),
            PyArray1::from_vec(py, second),
        )
    }

    #[getter]
    pub fn value(&self) -> Option<(f64, f64)> {
        self.current_value
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct MINMAX {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<(f64, f64)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MINMAX {
    #[new]
    pub fn new(period: usize) -> Self {
        MINMAX {
            period,
            buffer: VecDeque::with_capacity(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period {
            self.current_value = None;
            return None;
        }
        let max_v = self.buffer.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let min_v = self.buffer.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        self.current_value = Some((min_v, max_v));
        self.current_value
    }

    pub fn update_many_pair<'py>(
        &mut self,
        py: Python<'py>,
        values: Vec<f64>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let mut first = Vec::with_capacity(values.len());
        let mut second = Vec::with_capacity(values.len());
        for value in values {
            if let Some((f, s)) = self.update(value) {
                first.push(f);
                second.push(s);
            } else {
                first.push(f64::NAN);
                second.push(f64::NAN);
            }
        }
        (
            PyArray1::from_vec(py, first),
            PyArray1::from_vec(py, second),
        )
    }

    #[getter]
    pub fn value(&self) -> Option<(f64, f64)> {
        self.current_value
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct SUM {
    period: usize,
    buffer: VecDeque<f64>,
    sum: f64,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl SUM {
    #[new]
    pub fn new(period: usize) -> Self {
        SUM {
            period,
            buffer: VecDeque::with_capacity(period),
            sum: 0.0,
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.buffer.push_back(value);
        self.sum += value;
        if self.buffer.len() > self.period
            && let Some(removed) = self.buffer.pop_front()
        {
            self.sum -= removed;
        }
        if self.buffer.len() < self.period {
            self.current_value = None;
            return None;
        }
        self.current_value = Some(self.sum);
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
pub struct AVGDEV {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl AVGDEV {
    #[new]
    pub fn new(period: usize) -> Self {
        AVGDEV {
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
        let mean = self.buffer.iter().sum::<f64>() / self.period as f64;
        let avgdev =
            self.buffer.iter().map(|v| (*v - mean).abs()).sum::<f64>() / self.period as f64;
        self.current_value = Some(avgdev);
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
pub struct RANGE {
    period: usize,
    buffer: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl RANGE {
    #[new]
    pub fn new(period: usize) -> Self {
        RANGE {
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
        let max_v = self.buffer.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let min_v = self.buffer.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        self.current_value = Some(max_v - min_v);
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
pub struct LN {
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl LN {
    #[new]
    pub fn new() -> Self {
        LN {
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.current_value = if value > 0.0 {
            Some(value.ln())
        } else {
            Some(f64::NAN)
        };
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
    m.add_class::<MIDPOINT>()?;
    m.add_class::<MAX>()?;
    m.add_class::<MIN>()?;
    m.add_class::<MAXINDEX>()?;
    m.add_class::<MININDEX>()?;
    m.add_class::<MINMAXINDEX>()?;
    m.add_class::<MINMAX>()?;
    m.add_class::<SUM>()?;
    m.add_class::<AVGDEV>()?;
    m.add_class::<RANGE>()?;
    m.add_class::<LN>()?;
    Ok(())
}
