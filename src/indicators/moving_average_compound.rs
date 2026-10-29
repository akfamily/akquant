use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

use super::moving_average_core::EMA;
use crate::indicators::is_zero;

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
/// 平滑异同移动平均线：快慢两条 EMA 同在第 slow 根起种子，输出（DIF、DEA、柱），信号线就绪后才有值。
// TA-Lib MACD: 快线种子取最后 fast 根的均值; 信号线是 MACD 线的 EMA, 三个输出都要等
// 信号线就绪才给.
pub struct MACD {
    fast_period: usize,
    slow_period: usize,
    k_fast: f64,
    k_slow: f64,
    warmup: Vec<f64>,
    fast: f64,
    slow: f64,
    seeded: bool,
    signal_ema: EMA,
    current_value: Option<(f64, f64, f64)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl MACD {
    #[new]
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        // TA-Lib: 慢线短于快线时两者互换
        let (fast_period, slow_period) = if slow_period < fast_period {
            (slow_period, fast_period)
        } else {
            (fast_period, slow_period)
        };
        MACD {
            fast_period,
            slow_period,
            k_fast: 2.0 / (fast_period as f64 + 1.0),
            k_slow: 2.0 / (slow_period as f64 + 1.0),
            warmup: Vec::with_capacity(slow_period),
            fast: 0.0,
            slow: 0.0,
            seeded: false,
            signal_ema: EMA::new(signal_period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        skip_non_finite!(value);
        if self.seeded {
            self.fast = (value - self.fast) * self.k_fast + self.fast;
            self.slow = (value - self.slow) * self.k_slow + self.slow;
        } else {
            self.warmup.push(value);
            if self.warmup.len() < self.slow_period {
                return None;
            }
            // 两条线同在第 slow 根起种子; 快线种子取最后 fast 根的均值
            let tail = &self.warmup[self.slow_period - self.fast_period..];
            self.fast = tail.iter().sum::<f64>() / self.fast_period as f64;
            self.slow = self.warmup.iter().sum::<f64>() / self.slow_period as f64;
            self.seeded = true;
            self.warmup = Vec::new();
        }
        let macd_line = self.fast - self.slow;
        let signal_line = self.signal_ema.update(macd_line)?;
        self.current_value = Some((macd_line, signal_line, macd_line - signal_line));
        self.current_value
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
        let mut macd = Vec::with_capacity(values.len());
        let mut signal = Vec::with_capacity(values.len());
        let mut hist = Vec::with_capacity(values.len());
        for value in values {
            let out = self.update(value);
            if let Some((m, s, h)) = out {
                macd.push(m);
                signal.push(s);
                hist.push(h);
            } else {
                macd.push(f64::NAN);
                signal.push(f64::NAN);
                hist.push(f64::NAN);
            }
        }
        (
            PyArray1::from_vec(py, macd),
            PyArray1::from_vec(py, signal),
            PyArray1::from_vec(py, hist),
        )
    }

    #[getter]
    pub fn value(&self) -> Option<(f64, f64, f64)> {
        self.current_value
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct DEMA {
    ema1: EMA,
    ema2: EMA,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl DEMA {
    #[new]
    pub fn new(period: usize) -> Self {
        DEMA {
            ema1: EMA::new(period),
            ema2: EMA::new(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        let ema1 = self.ema1.update(value)?;
        let ema2 = self.ema2.update(ema1)?;
        self.current_value = Some(2.0 * ema1 - ema2);
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
pub struct TRIX {
    ema1: EMA,
    ema2: EMA,
    ema3: EMA,
    prev_ema3: Option<f64>,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl TRIX {
    #[new]
    pub fn new(period: usize) -> Self {
        TRIX {
            ema1: EMA::new(period),
            ema2: EMA::new(period),
            ema3: EMA::new(period),
            prev_ema3: None,
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        let ema1 = self.ema1.update(value)?;
        let ema2 = self.ema2.update(ema1)?;
        let ema3 = self.ema3.update(ema2)?;
        let prev = self.prev_ema3.replace(ema3)?;
        // TA-Lib TA_INT_ROC: 前值为 0 时输出 0
        self.current_value = Some(if prev == 0.0 {
            0.0
        } else {
            (ema3 - prev) / prev * 100.0
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

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct TEMA {
    ema1: EMA,
    ema2: EMA,
    ema3: EMA,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl TEMA {
    #[new]
    pub fn new(period: usize) -> Self {
        TEMA {
            ema1: EMA::new(period),
            ema2: EMA::new(period),
            ema3: EMA::new(period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        let ema1 = self.ema1.update(value)?;
        let ema2 = self.ema2.update(ema1)?;
        let ema3 = self.ema3.update(ema2)?;
        self.current_value = Some(3.0 * ema1 - 3.0 * ema2 + ema3);
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
pub struct KAMA {
    period: usize,
    prices: VecDeque<f64>,
    fast_sc: f64,
    slow_sc: f64,
    prev: f64,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl KAMA {
    #[new]
    pub fn new(period: usize) -> Self {
        KAMA {
            period,
            prices: VecDeque::with_capacity(period + 1),
            fast_sc: 2.0 / 3.0,
            slow_sc: 2.0 / 31.0,
            prev: 0.0,
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        self.prices.push_back(value);
        if self.prices.len() > self.period + 1 {
            self.prices.pop_front();
        }
        if self.prices.len() <= self.period {
            // TA-Lib 以第 period 根(下标 period-1)的价格作为 KAMA 递推起点
            self.prev = value;
            return None;
        }
        let first = *self.prices.front()?;
        let change = value - first;
        let volatility = self
            .prices
            .iter()
            .zip(self.prices.iter().skip(1))
            .map(|(a, b)| (b - a).abs())
            .sum::<f64>();
        // TA-Lib: 波动和不为正, 或不大于区间涨幅(浮点误差)时效率比取 1
        let er = if volatility <= 0.0 || volatility <= change {
            1.0
        } else {
            (change / volatility).abs()
        };
        let sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc).powi(2);
        self.prev += (value - self.prev) * sc;
        self.current_value = Some(self.prev);
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
/// 绝对价格振荡器：快慢两条 EMA 各自以前 n 根均值起算后相减。
// APO / PPO: 两条 EMA 各自起种子(与 MACD 不同, 这是 TA-Lib 的实际行为).
// 两条线都必须先 update 再判就绪, 否则慢线会漏掉快线预热期的数据.
pub struct APO {
    fast_ema: EMA,
    slow_ema: EMA,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl APO {
    #[new]
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        // TA-Lib: 慢线短于快线时两者互换
        let (fast_period, slow_period) = if slow_period < fast_period {
            (slow_period, fast_period)
        } else {
            (fast_period, slow_period)
        };
        APO {
            fast_ema: EMA::new(fast_period),
            slow_ema: EMA::new(slow_period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        // 不能用 `?` 链: 快线先就绪时慢线也要吃到这个值
        let fast = self.fast_ema.update(value);
        let slow = self.slow_ema.update(value);
        let (Some(fast), Some(slow)) = (fast, slow) else {
            return None;
        };
        self.current_value = Some(fast - slow);
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
/// 百分比价格振荡器：（快线 - 慢线）/ 慢线 × 100，两条 EMA 各自以前 n 根均值起算。
pub struct PPO {
    fast_ema: EMA,
    slow_ema: EMA,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PPO {
    #[new]
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        // TA-Lib: 慢线短于快线时两者互换
        let (fast_period, slow_period) = if slow_period < fast_period {
            (slow_period, fast_period)
        } else {
            (fast_period, slow_period)
        };
        PPO {
            fast_ema: EMA::new(fast_period),
            slow_ema: EMA::new(slow_period),
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        // 不能用 `?` 链: 快线先就绪时慢线也要吃到这个值
        let fast = self.fast_ema.update(value);
        let slow = self.slow_ema.update(value);
        let (Some(fast), Some(slow)) = (fast, slow) else {
            return None;
        };
        // TA-Lib: 慢线为 0(TA_IS_ZERO)时输出 0
        self.current_value = Some(if is_zero(slow) {
            0.0
        } else {
            (fast - slow) / slow * 100.0
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

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct T3 {
    ema1: EMA,
    ema2: EMA,
    ema3: EMA,
    ema4: EMA,
    ema5: EMA,
    ema6: EMA,
    vfactor: f64,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl T3 {
    #[new]
    pub fn new(period: usize, vfactor: f64) -> Self {
        T3 {
            ema1: EMA::new(period),
            ema2: EMA::new(period),
            ema3: EMA::new(period),
            ema4: EMA::new(period),
            ema5: EMA::new(period),
            ema6: EMA::new(period),
            vfactor,
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        let e1 = self.ema1.update(value)?;
        let e2 = self.ema2.update(e1)?;
        let e3 = self.ema3.update(e2)?;
        let e4 = self.ema4.update(e3)?;
        let e5 = self.ema5.update(e4)?;
        let e6 = self.ema6.update(e5)?;

        let a = self.vfactor;
        let c1 = -a * a * a;
        let c2 = 3.0 * a * a + 3.0 * a * a * a;
        let c3 = -6.0 * a * a - 3.0 * a - 3.0 * a * a * a;
        let c4 = 1.0 + 3.0 * a + a * a * a + 3.0 * a * a;
        self.current_value = Some(c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3);
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
    m.add_class::<MACD>()?;
    m.add_class::<DEMA>()?;
    m.add_class::<TRIX>()?;
    m.add_class::<TEMA>()?;
    m.add_class::<KAMA>()?;
    m.add_class::<APO>()?;
    m.add_class::<PPO>()?;
    m.add_class::<T3>()?;
    Ok(())
}
