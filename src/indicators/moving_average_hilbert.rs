// TA-Lib Hilbert-transform kernels (MAMA / HT_TRENDLINE), streaming form.
// 逐步复刻 TA-Lib `ta_MAMA.c` / `ta_HT_TRENDLINE.c` 的增量算术顺序, 浮点结果才对得上。
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

const HILBERT_A: f64 = 0.0962;
const HILBERT_B: f64 = 0.5769;

#[inline]
fn rad2deg() -> f64 {
    45.0 / 1.0_f64.atan()
}

// TA-Lib `DO_HILBERT_ODD/EVEN` 的一路状态。
#[derive(Debug, Clone, Default)]
struct HilbertLeg {
    buffer: [f64; 3],
    prev: f64,
    prev_input: f64,
}

impl HilbertLeg {
    fn step(&mut self, input: f64, idx: usize, adjusted_period: f64) -> f64 {
        let hilbert_temp = HILBERT_A * input;
        let mut value = -self.buffer[idx];
        self.buffer[idx] = hilbert_temp;
        value += hilbert_temp;
        value -= self.prev;
        self.prev = HILBERT_B * self.prev_input;
        value += self.prev;
        self.prev_input = input;
        value * adjusted_period
    }
}

// 四路 Hilbert(detrender / Q1 / jI / jQ)各分奇偶两套 + 周期自适应。
#[derive(Debug, Clone, Default)]
struct HilbertCore {
    odd: [HilbertLeg; 4],
    even: [HilbertLeg; 4],
    idx: usize,
    period: f64,
    prev_i2: f64,
    prev_q2: f64,
    re: f64,
    im: f64,
    i1_odd_prev3: f64,
    i1_odd_prev2: f64,
    i1_even_prev3: f64,
    i1_even_prev2: f64,
}

impl HilbertCore {
    // 返回 (Q1, I1, Q2, I2)。`even` 取当前 bar 在整个序列里的下标奇偶。
    fn step(&mut self, smoothed: f64, even: bool) -> (f64, f64, f64, f64) {
        let adjusted = 0.075 * self.period + 0.54;
        let idx = self.idx;
        let (q1, i1, ji, jq);
        if even {
            let detrender = self.even[0].step(smoothed, idx, adjusted);
            q1 = self.even[1].step(detrender, idx, adjusted);
            ji = self.even[2].step(self.i1_even_prev3, idx, adjusted);
            jq = self.even[3].step(q1, idx, adjusted);
            i1 = self.i1_even_prev3;
            self.idx = (self.idx + 1) % 3;
            self.i1_odd_prev3 = self.i1_odd_prev2;
            self.i1_odd_prev2 = detrender;
        } else {
            let detrender = self.odd[0].step(smoothed, idx, adjusted);
            q1 = self.odd[1].step(detrender, idx, adjusted);
            ji = self.odd[2].step(self.i1_odd_prev3, idx, adjusted);
            jq = self.odd[3].step(q1, idx, adjusted);
            i1 = self.i1_odd_prev3;
            self.i1_even_prev3 = self.i1_even_prev2;
            self.i1_even_prev2 = detrender;
        }
        let q2 = 0.2 * (q1 + ji) + 0.8 * self.prev_q2;
        let i2 = 0.2 * (i1 - jq) + 0.8 * self.prev_i2;
        (q1, i1, q2, i2)
    }

    fn adapt_period(&mut self, q2: f64, i2: f64) {
        self.re = 0.2 * (i2 * self.prev_i2 + q2 * self.prev_q2) + 0.8 * self.re;
        self.im = 0.2 * (i2 * self.prev_q2 - q2 * self.prev_i2) + 0.8 * self.im;
        self.prev_q2 = q2;
        self.prev_i2 = i2;
        let old = self.period;
        if self.im != 0.0 && self.re != 0.0 {
            self.period = 360.0 / ((self.im / self.re).atan() * rad2deg());
        }
        self.period = self.period.min(1.5 * old);
        self.period = self.period.max(0.67 * old);
        self.period = self.period.clamp(6.0, 50.0);
        self.period = 0.2 * self.period + 0.8 * old;
    }
}

// TA-Lib `DO_PRICE_WMA`: (4,3,2,1)/10 加权, 增量形式。前 3 根只做初始化。
#[derive(Debug, Clone, Default)]
struct PriceWma {
    recent: VecDeque<f64>,
    sub: f64,
    sum: f64,
}

impl PriceWma {
    // 喂第 `index` 根(从 0 起)价格; 第 0..=2 根返回 None。
    fn push(&mut self, index: usize, price: f64) -> Option<f64> {
        self.recent.push_back(price);
        if self.recent.len() > 5 {
            self.recent.pop_front();
        }
        match index {
            0 | 1 => None,
            2 => {
                let p0 = self.recent[0];
                let p1 = self.recent[1];
                self.sub = p0 + p1 + price;
                self.sum = p0 + p1 * 2.0 + price * 3.0;
                None
            }
            _ => {
                let trailing = if self.recent.len() == 5 {
                    self.recent[0]
                } else {
                    0.0
                };
                self.sub += price;
                self.sub -= trailing;
                self.sum += price * 4.0;
                let out = self.sum * 0.1;
                self.sum -= self.sub;
                Some(out)
            }
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct MAMA {
    fast_limit: f64,
    slow_limit: f64,
    index: usize,
    wma: PriceWma,
    core: HilbertCore,
    mama: f64,
    fama: f64,
    prev_phase: f64,
    current_value: Option<(f64, f64)>,
}

impl MAMA {
    const LOOKBACK: usize = 32;
    const FIRST_HILBERT: usize = 12;
}

#[gen_stub_pymethods]
#[pymethods]
impl MAMA {
    #[new]
    pub fn new(fast_limit: f64, slow_limit: f64) -> Self {
        MAMA {
            fast_limit,
            slow_limit,
            index: 0,
            wma: PriceWma::default(),
            core: HilbertCore::default(),
            mama: 0.0,
            fama: 0.0,
            prev_phase: 0.0,
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        skip_non_finite!(value);
        let today = self.index;
        self.index += 1;
        let smoothed = self.wma.push(today, value)?;
        if today < Self::FIRST_HILBERT {
            return None;
        }
        let (q1, i1, q2, i2) = self.core.step(smoothed, today.is_multiple_of(2));
        let phase = if i1 != 0.0 {
            (q1 / i1).atan() * rad2deg()
        } else {
            0.0
        };
        let mut delta = self.prev_phase - phase;
        self.prev_phase = phase;
        if delta < 1.0 {
            delta = 1.0;
        }
        let mut alpha = if delta > 1.0 {
            (self.fast_limit / delta).max(self.slow_limit)
        } else {
            self.fast_limit
        };
        self.mama = alpha * value + (1.0 - alpha) * self.mama;
        alpha *= 0.5;
        self.fama = alpha * self.mama + (1.0 - alpha) * self.fama;
        self.core.adapt_period(q2, i2);
        if today < Self::LOOKBACK {
            return None;
        }
        self.current_value = Some((self.mama, self.fama));
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
#[allow(non_camel_case_types)]
pub struct HT_TRENDLINE {
    index: usize,
    wma: PriceWma,
    core: HilbertCore,
    history: VecDeque<f64>,
    smooth_period: f64,
    trendline: [f64; 3],
    current_value: Option<f64>,
}

impl HT_TRENDLINE {
    const LOOKBACK: usize = 63;
    const FIRST_HILBERT: usize = 37;
    const MAX_CYCLE: usize = 50;
}

#[gen_stub_pymethods]
#[pymethods]
impl HT_TRENDLINE {
    #[new]
    pub fn new() -> Self {
        HT_TRENDLINE {
            index: 0,
            wma: PriceWma::default(),
            core: HilbertCore::default(),
            history: VecDeque::with_capacity(Self::MAX_CYCLE),
            smooth_period: 0.0,
            trendline: [0.0; 3],
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        skip_non_finite!(value);
        let today = self.index;
        self.index += 1;
        self.history.push_back(value);
        if self.history.len() > Self::MAX_CYCLE {
            self.history.pop_front();
        }
        let smoothed = self.wma.push(today, value)?;
        if today < Self::FIRST_HILBERT {
            return None;
        }
        let (_, _, q2, i2) = self.core.step(smoothed, today.is_multiple_of(2));
        self.core.adapt_period(q2, i2);
        self.smooth_period = 0.33 * self.core.period + 0.67 * self.smooth_period;
        let dc_period = (self.smooth_period + 0.5) as usize;
        let mut total = 0.0;
        for price in self.history.iter().rev().take(dc_period) {
            total += price;
        }
        if dc_period > 0 {
            total /= dc_period as f64;
        }
        let [it1, it2, it3] = self.trendline;
        let trendline = (4.0 * total + 3.0 * it1 + 2.0 * it2 + it3) / 10.0;
        self.trendline = [total, it1, it2];
        if today < Self::LOOKBACK {
            return None;
        }
        self.current_value = Some(trendline);
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
    m.add_class::<MAMA>()?;
    m.add_class::<HT_TRENDLINE>()?;
    Ok(())
}
