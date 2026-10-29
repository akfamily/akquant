use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

use crate::indicators::is_zero_scaled;

// 窗口最高价/最低价; 用于 STOCH。
fn high_low(highs: &VecDeque<f64>, lows: &VecDeque<f64>) -> (f64, f64) {
    let highest = highs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lowest = lows.iter().copied().fold(f64::INFINITY, f64::min);
    (highest, lowest)
}

// 简单均线窗口(STOCH 的两次平滑, 每个窗口精确重算), 不对 Python 暴露。
#[derive(Debug, Clone)]
struct SmaWindow {
    period: usize,
    buffer: VecDeque<f64>,
}

impl SmaWindow {
    fn new(period: usize) -> Self {
        SmaWindow {
            period,
            buffer: VecDeque::with_capacity(period),
        }
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer.push_back(value);
        if self.buffer.len() > self.period {
            self.buffer.pop_front();
        }
        if self.buffer.len() < self.period {
            return None;
        }
        Some(self.buffer.iter().sum::<f64>() / self.period as f64)
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct STOCH {
    fastk_period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    slowk: SmaWindow,
    slowd: SmaWindow,
    current_value: Option<(f64, f64)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl STOCH {
    #[new]
    pub fn new(fastk_period: usize, slowk_period: usize, slowd_period: usize) -> Self {
        STOCH {
            fastk_period,
            highs: VecDeque::with_capacity(fastk_period),
            lows: VecDeque::with_capacity(fastk_period),
            slowk: SmaWindow::new(slowk_period),
            slowd: SmaWindow::new(slowd_period),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        skip_non_finite!(high, low, close);
        self.highs.push_back(high);
        self.lows.push_back(low);
        if self.highs.len() > self.fastk_period {
            self.highs.pop_front();
            self.lows.pop_front();
        }
        if self.highs.len() < self.fastk_period {
            self.current_value = None;
            return None;
        }

        let (highest, lowest) = high_low(&self.highs, &self.lows);
        let range = highest - lowest;
        // TA-Lib 0.8.1: 区间相对量级可忽略时 fastk 记 0
        let fastk = if is_zero_scaled(range, highest.abs() + lowest.abs()) {
            0.0
        } else {
            (close - lowest) / range * 100.0
        };
        let Some(slowk) = self.slowk.update(fastk) else {
            self.current_value = None;
            return None;
        };
        let Some(slowd) = self.slowd.update(slowk) else {
            self.current_value = None;
            return None;
        };
        self.current_value = Some((slowk, slowd));
        self.current_value
    }

    pub fn update_many_hlc_pair<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        if highs.len() != lows.len() || highs.len() != closes.len() {
            return Err(PyValueError::new_err("highs/lows/closes length mismatch"));
        }
        let mut out_first = Vec::with_capacity(highs.len());
        let mut out_second = Vec::with_capacity(highs.len());
        for (high, (low, close)) in highs
            .into_iter()
            .zip(lows.into_iter().zip(closes.into_iter()))
        {
            if let Some((first, second)) = self.update(high, low, close) {
                out_first.push(first);
                out_second.push(second);
            } else {
                out_first.push(f64::NAN);
                out_second.push(f64::NAN);
            }
        }
        Ok((
            PyArray1::from_vec(py, out_first),
            PyArray1::from_vec(py, out_second),
        ))
    }

    #[getter]
    pub fn value(&self) -> Option<(f64, f64)> {
        self.current_value
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct AROON {
    period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    current_value: Option<(f64, f64)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl AROON {
    #[new]
    pub fn new(period: usize) -> Self {
        AROON {
            period,
            highs: VecDeque::with_capacity(period + 1),
            lows: VecDeque::with_capacity(period + 1),
            current_value: None,
        }
    }

    // 返回 (aroondown, aroonup)。窗口是 period+1 根, 并列极值取最近的一根。
    pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        skip_non_finite!(high, low);
        self.highs.push_back(high);
        self.lows.push_back(low);
        if self.highs.len() > self.period + 1 {
            self.highs.pop_front();
            self.lows.pop_front();
        }
        if self.highs.len() <= self.period {
            self.current_value = None;
            return None;
        }
        let (mut high_ago, mut high_val) = (0usize, f64::NEG_INFINITY);
        let (mut low_ago, mut low_val) = (0usize, f64::INFINITY);
        for (ago, (h, l)) in self
            .highs
            .iter()
            .rev()
            .zip(self.lows.iter().rev())
            .enumerate()
        {
            if *h > high_val {
                high_val = *h;
                high_ago = ago;
            }
            if *l < low_val {
                low_val = *l;
                low_ago = ago;
            }
        }
        let n = self.period as f64;
        let factor = 100.0 / n;
        let up = factor * (n - high_ago as f64);
        let down = factor * (n - low_ago as f64);
        self.current_value = Some((down, up));
        self.current_value
    }

    pub fn update_many_hl_pair<'py>(
        &mut self,
        py: Python<'py>,
        highs: Vec<f64>,
        lows: Vec<f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        if highs.len() != lows.len() {
            return Err(PyValueError::new_err("highs/lows length mismatch"));
        }
        let mut first = Vec::with_capacity(highs.len());
        let mut second = Vec::with_capacity(highs.len());
        for (high, low) in highs.into_iter().zip(lows.into_iter()) {
            if let Some((f, s)) = self.update(high, low) {
                first.push(f);
                second.push(s);
            } else {
                first.push(f64::NAN);
                second.push(f64::NAN);
            }
        }
        Ok((
            PyArray1::from_vec(py, first),
            PyArray1::from_vec(py, second),
        ))
    }

    #[getter]
    pub fn value(&self) -> Option<(f64, f64)> {
        self.current_value
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct AROONOSC {
    inner: AROON,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl AROONOSC {
    #[new]
    pub fn new(period: usize) -> Self {
        AROONOSC {
            inner: AROON::new(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        skip_non_finite!(high, low);
        let (down, up) = self.inner.update(high, low)?;
        self.current_value = Some(up - down);
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
pub struct SAR {
    acceleration: f64,
    maximum: f64,
    first: Option<(f64, f64)>,
    started: bool,
    is_long: bool,
    af: f64,
    ep: f64,
    sar: f64,
    last_high: f64,
    last_low: f64,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl SAR {
    #[new]
    pub fn new(acceleration: f64, maximum: f64) -> Self {
        // TA-Lib: 加速因子大于上限时两者都取上限
        let acceleration = acceleration.min(maximum);
        SAR {
            acceleration,
            maximum,
            first: None,
            started: false,
            is_long: true,
            af: acceleration,
            ep: 0.0,
            sar: 0.0,
            last_high: 0.0,
            last_low: 0.0,
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        skip_non_finite!(high, low);
        if !self.started {
            let Some((first_high, first_low)) = self.first else {
                self.first = Some((high, low));
                return None;
            };
            // 方向由前两根的 -DM 决定: 有正的 -DM 就做空, 否则做多
            let up = high - first_high;
            let down = first_low - low;
            self.is_long = !(down > 0.0 && up < down);
            if self.is_long {
                self.ep = high;
                self.sar = first_low;
            } else {
                self.ep = low;
                self.sar = first_high;
            }
            self.last_high = high;
            self.last_low = low;
            self.started = true;
        }
        let (prev_high, prev_low) = (self.last_high, self.last_low);
        self.last_high = high;
        self.last_low = low;
        let out;
        if self.is_long {
            if low <= self.sar {
                self.is_long = false;
                self.sar = self.ep.max(prev_high).max(high);
                out = self.sar;
                self.af = self.acceleration;
                self.ep = low;
                self.sar = (self.sar + self.af * (self.ep - self.sar))
                    .max(prev_high)
                    .max(high);
            } else {
                out = self.sar;
                if high > self.ep {
                    self.ep = high;
                    self.af = (self.af + self.acceleration).min(self.maximum);
                }
                self.sar = (self.sar + self.af * (self.ep - self.sar))
                    .min(prev_low)
                    .min(low);
            }
        } else if high >= self.sar {
            self.is_long = true;
            self.sar = self.ep.min(prev_low).min(low);
            out = self.sar;
            self.af = self.acceleration;
            self.ep = high;
            self.sar = (self.sar + self.af * (self.ep - self.sar))
                .min(prev_low)
                .min(low);
        } else {
            out = self.sar;
            if low < self.ep {
                self.ep = low;
                self.af = (self.af + self.acceleration).min(self.maximum);
            }
            self.sar = (self.sar + self.af * (self.ep - self.sar))
                .max(prev_high)
                .max(high);
        }
        self.current_value = Some(out);
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
pub struct ULTOSC {
    periods: [usize; 3],
    prev_close: Option<f64>,
    bp_values: VecDeque<f64>,
    tr_values: VecDeque<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ULTOSC {
    #[new]
    pub fn new(period1: usize, period2: usize, period3: usize) -> Self {
        // TA-Lib: 三个周期先排序
        let mut periods = [period1, period2, period3];
        periods.sort_unstable();
        ULTOSC {
            periods,
            prev_close: None,
            bp_values: VecDeque::with_capacity(periods[2]),
            tr_values: VecDeque::with_capacity(periods[2]),
            current_value: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        skip_non_finite!(high, low, close);
        // 需要前收盘, 首个输出在下标 max(period)
        let prev_close = self.prev_close.replace(close)?;
        let true_low = low.min(prev_close);
        self.bp_values.push_back(close - true_low);
        self.tr_values.push_back(high.max(prev_close) - true_low);
        let longest = self.periods[2];
        if self.bp_values.len() > longest {
            self.bp_values.pop_front();
            self.tr_values.pop_front();
        }
        if self.bp_values.len() < longest {
            self.current_value = None;
            return None;
        }
        let mut total = 0.0;
        for (period, weight) in self.periods.iter().zip([4.0, 2.0, 1.0]) {
            let bp: f64 = self.bp_values.iter().rev().take(*period).sum();
            let tr: f64 = self.tr_values.iter().rev().take(*period).sum();
            if tr > 0.0 {
                total += weight * (bp / tr);
            }
        }
        self.current_value = Some(100.0 * total / 7.0);
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
    m.add_class::<AROON>()?;
    m.add_class::<AROONOSC>()?;
    m.add_class::<STOCH>()?;
    m.add_class::<SAR>()?;
    m.add_class::<ULTOSC>()?;
    Ok(())
}
