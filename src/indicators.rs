use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::VecDeque;

/// 简单移动平均线 (SMA).
#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone)]
pub struct SMA {
    period: usize,
    buffer: VecDeque<f64>,
    sum: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl SMA {
    /// 创建 SMA 指标.
    ///
    /// :param period: 周期
    #[new]
    pub fn new(period: usize) -> Self {
        SMA {
            period,
            buffer: VecDeque::with_capacity(period),
            sum: 0.0,
        }
    }

    /// 更新指标值.
    ///
    /// :param value: 新数据点
    /// :return: 当前 SMA 值 (如果数据不足则返回 None)
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer.push_back(value);
        self.sum += value;

        if self.buffer.len() > self.period {
            if let Some(removed) = self.buffer.pop_front() {
                self.sum -= removed;
            }
        }

        if self.buffer.len() == self.period {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }

    /// 获取当前指标值.
    ///
    /// :return: 当前 SMA 值
    #[getter]
    pub fn value(&self) -> Option<f64> {
        if self.buffer.len() == self.period {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }

    /// 检查指标是否就绪.
    ///
    /// :return: 是否已收集足够数据
    #[getter]
    pub fn is_ready(&self) -> bool {
        self.buffer.len() == self.period
    }
}

/// 指数移动平均线 (EMA).
#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone)]
pub struct EMA {
    period: usize,
    k: f64,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl EMA {
    /// 创建 EMA 指标.
    ///
    /// :param period: 周期
    #[new]
    pub fn new(period: usize) -> Self {
        EMA {
            period,
            k: 2.0 / (period as f64 + 1.0),
            current_value: None,
        }
    }

    /// 更新指标值.
    ///
    /// :param value: 新数据点
    /// :return: 当前 EMA 值
    pub fn update(&mut self, value: f64) -> Option<f64> {
        match self.current_value {
            Some(prev) => {
                let next = (value - prev) * self.k + prev;
                self.current_value = Some(next);
            }
            None => {
                self.current_value = Some(value);
            }
        }
        self.current_value
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }

    #[getter]
    pub fn is_ready(&self) -> bool {
        self.current_value.is_some()
    }

    #[getter]
    pub fn period(&self) -> usize {
        self.period
    }
}

/// 平滑异同移动平均线 (MACD).
#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone)]
pub struct MACD {
    fast_ema: EMA,
    slow_ema: EMA,
    signal_ema: EMA,
}

#[gen_stub_pymethods]
#[pymethods]
impl MACD {
    /// 创建 MACD 指标.
    ///
    /// :param fast_period: 快线周期 (通常 12)
    /// :param slow_period: 慢线周期 (通常 26)
    /// :param signal_period: 信号线周期 (通常 9)
    #[new]
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        MACD {
            fast_ema: EMA::new(fast_period),
            slow_ema: EMA::new(slow_period),
            signal_ema: EMA::new(signal_period),
        }
    }

    /// 更新指标值.
    ///
    /// :param value: 新数据点
    /// :return: (DIF, DEA, MACD柱)
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        let fast = self.fast_ema.update(value)?;
        let slow = self.slow_ema.update(value)?;

        let macd_line = fast - slow;
        let signal_line = self.signal_ema.update(macd_line)?;
        let histogram = macd_line - signal_line;

        Some((macd_line, signal_line, histogram))
    }

    /// 获取当前指标值.
    ///
    /// :return: (DIF, DEA, MACD柱)
    #[getter]
    pub fn value(&self) -> Option<(f64, f64, f64)> {
        let fast = self.fast_ema.value()?;
        let slow = self.slow_ema.value()?;
        let macd_line = fast - slow;
        let signal_line = self.signal_ema.value()?;
        let histogram = macd_line - signal_line;
        Some((macd_line, signal_line, histogram))
    }
}

/// 相对强弱指数 (RSI).
#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone)]
pub struct RSI {
    period: usize,
    prev_price: Option<f64>,
    avg_gain: f64,
    avg_loss: f64,
    count: usize,
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl RSI {
    /// 创建 RSI 指标.
    ///
    /// :param period: 周期 (通常 14)
    #[new]
    pub fn new(period: usize) -> Self {
        RSI {
            period,
            prev_price: None,
            avg_gain: 0.0,
            avg_loss: 0.0,
            count: 0,
            current_value: None,
        }
    }

    /// 更新指标值.
    ///
    /// :param value: 新数据点 (通常是收盘价)
    /// :return: 当前 RSI 值
    pub fn update(&mut self, value: f64) -> Option<f64> {
        match self.prev_price {
            Some(prev) => {
                let change = value - prev;
                let gain = if change > 0.0 { change } else { 0.0 };
                let loss = if change < 0.0 { -change } else { 0.0 };

                if self.count < self.period {
                    // Initial accumulation phase
                    self.avg_gain += gain;
                    self.avg_loss += loss;
                    self.count += 1;

                    if self.count == self.period {
                        // Calculate initial average
                        self.avg_gain /= self.period as f64;
                        self.avg_loss /= self.period as f64;
                    }
                } else {
                    // Wilder's Smoothing
                    self.avg_gain =
                        (self.avg_gain * (self.period as f64 - 1.0) + gain) / self.period as f64;
                    self.avg_loss =
                        (self.avg_loss * (self.period as f64 - 1.0) + loss) / self.period as f64;
                }
            }
            None => {}
        }

        self.prev_price = Some(value);

        if self.count < self.period {
            return None;
        }

        let rs = if self.avg_loss == 0.0 {
            100.0
        } else {
            self.avg_gain / self.avg_loss
        };

        let rsi = 100.0 - (100.0 / (1.0 + rs));
        self.current_value = Some(rsi);
        Some(rsi)
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

/// 布林带 (Bollinger Bands).
#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone)]
pub struct BollingerBands {
    period: usize,
    multiplier: f64,
    buffer: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl BollingerBands {
    /// 创建布林带指标.
    ///
    /// :param period: 周期 (通常 20)
    /// :param multiplier: 标准差倍数 (通常 2.0)
    #[new]
    pub fn new(period: usize, multiplier: f64) -> Self {
        BollingerBands {
            period,
            multiplier,
            buffer: VecDeque::with_capacity(period),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// 更新指标值.
    ///
    /// :param value: 新数据点
    /// :return: (上轨, 中轨, 下轨)
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.buffer.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;

        if self.buffer.len() > self.period {
            if let Some(removed) = self.buffer.pop_front() {
                self.sum -= removed;
                self.sum_sq -= removed * removed;
            }
        }

        if self.buffer.len() == self.period {
            let mean = self.sum / self.period as f64;
            // Variance = E[X^2] - (E[X])^2
            // Use max(0.0) to avoid negative variance due to floating point errors
            let variance = (self.sum_sq / self.period as f64 - mean * mean).max(0.0);
            let std_dev = variance.sqrt();

            let upper = mean + self.multiplier * std_dev;
            let lower = mean - self.multiplier * std_dev;

            Some((upper, mean, lower))
        } else {
            None
        }
    }

    /// 获取当前指标值.
    ///
    /// :return: (上轨, 中轨, 下轨)
    #[getter]
    pub fn value(&self) -> Option<(f64, f64, f64)> {
        if self.buffer.len() == self.period {
            let mean = self.sum / self.period as f64;
            let variance = (self.sum_sq / self.period as f64 - mean * mean).max(0.0);
            let std_dev = variance.sqrt();

            let upper = mean + self.multiplier * std_dev;
            let lower = mean - self.multiplier * std_dev;

            Some((upper, mean, lower))
        } else {
            None
        }
    }
}

/// 平均真实波幅 (ATR).
#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone)]
pub struct ATR {
    period: usize,
    prev_close: Option<f64>,
    smoothed_tr: f64,
    count: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl ATR {
    /// 创建 ATR 指标.
    ///
    /// :param period: 周期 (通常 14)
    #[new]
    pub fn new(period: usize) -> Self {
        ATR {
            period,
            prev_close: None,
            smoothed_tr: 0.0,
            count: 0,
        }
    }

    /// 更新指标值.
    ///
    /// :param high: 最高价
    /// :param low: 最低价
    /// :param close: 收盘价
    /// :return: 当前 ATR 值
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = match self.prev_close {
            Some(pc) => {
                let hl = high - low;
                let hpc = (high - pc).abs();
                let lpc = (low - pc).abs();
                hl.max(hpc).max(lpc)
            }
            None => high - low,
        };

        self.prev_close = Some(close);

        if self.count < self.period {
            self.smoothed_tr += tr;
            self.count += 1;

            if self.count == self.period {
                self.smoothed_tr /= self.period as f64;
                return Some(self.smoothed_tr);
            } else {
                return None;
            }
        }

        // Wilder's Smoothing
        self.smoothed_tr =
            (self.smoothed_tr * (self.period as f64 - 1.0) + tr) / self.period as f64;
        Some(self.smoothed_tr)
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        if self.count >= self.period {
            Some(self.smoothed_tr)
        } else {
            None
        }
    }
}
