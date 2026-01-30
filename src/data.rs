use crate::model::{Bar, Tick};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::{HashMap, VecDeque};

#[inline]
fn normalize_timestamp(ts: i64) -> i64 {
    let abs_ts = ts.abs();
    if abs_ts < 100_000_000_000 {
        // Seconds (< 10^11, up to year ~5138)
        ts * 1_000_000_000
    } else if abs_ts < 100_000_000_000_000 {
        // Milliseconds (< 10^14, up to year ~5138)
        ts * 1_000_000
    } else if abs_ts < 100_000_000_000_000_000 {
        // Microseconds (< 10^17, up to year ~5138)
        ts * 1_000
    } else {
        // Nanoseconds
        ts
    }
}

/// 从数组批量创建 Bar 列表 (Python 优化用)
#[gen_stub_pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn from_arrays(
    timestamps: Vec<i64>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    symbol: Option<String>,
    symbols: Option<Vec<String>>,
    extra: Option<HashMap<String, Vec<f64>>>,
) -> PyResult<Vec<Bar>> {
    let len = timestamps.len();
    if opens.len() != len
        || highs.len() != len
        || lows.len() != len
        || closes.len() != len
        || volumes.len() != len
    {
        return Err(PyValueError::new_err(
            "All arrays must have the same length",
        ));
    }

    if let Some(ref syms) = symbols
        && syms.len() != len
    {
        return Err(PyValueError::new_err(
            "symbols array must have the same length as other arrays",
        ));
    }

    // Check extra arrays length
    if let Some(ref extra_data) = extra {
        for (key, val) in extra_data {
            if val.len() != len {
                 return Err(PyValueError::new_err(format!(
                    "Extra array '{}' must have the same length as other arrays",
                    key
                )));
            }
        }
    }

    let mut bars = Vec::with_capacity(len);

    // Determine the symbol source
    // If `symbols` (vector) is provided, use it.
    // Else if `symbol` (scalar) is provided, use it repeatedly.
    // Else default to "UNKNOWN".

    // Note: To avoid cloning strings unnecessarily, we can optimize,
    // but for now cloning is safe and much faster than Python loops.

    for i in 0..len {
        let sym = if let Some(ref syms) = symbols {
            syms[i].clone()
        } else if let Some(ref s) = symbol {
            s.clone()
        } else {
            "UNKNOWN".to_string()
        };

        let ts = timestamps[i];
        let normalized_ts = normalize_timestamp(ts);

        let mut bar_extra = HashMap::new();
        if let Some(ref extra_data) = extra {
            for (k, v) in extra_data {
                bar_extra.insert(k.clone(), v[i]);
            }
        }

        bars.push(Bar {
            timestamp: normalized_ts,
            open: Decimal::from_f64(opens[i]).unwrap_or(Decimal::ZERO),
            high: Decimal::from_f64(highs[i]).unwrap_or(Decimal::ZERO),
            low: Decimal::from_f64(lows[i]).unwrap_or(Decimal::ZERO),
            close: Decimal::from_f64(closes[i]).unwrap_or(Decimal::ZERO),
            volume: Decimal::from_f64(volumes[i]).unwrap_or(Decimal::ZERO),
            symbol: sym,
            extra: bar_extra,
        });
    }

    Ok(bars)
}

/// 事件类型枚举
#[derive(Debug, Clone)]
pub enum Event {
    Bar(Bar),
    Tick(Tick),
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
pub struct DataFeed {
    // 按时间戳排序的事件队列
    // 使用 VecDeque 存储，方便从头部取出
    pub events: VecDeque<Event>,
}

#[pymethods]
impl DataFeed {
    #[new]
    pub fn new() -> Self {
        DataFeed {
            events: VecDeque::new(),
        }
    }

    /// 添加 Bar 数据
    pub fn add_bar(&mut self, bar: Bar) {
        // 简单插入，假设用户按顺序添加。
        // 实际生产中可能需要在这里进行排序或合并。
        self.events.push_back(Event::Bar(bar));
    }

    /// 批量添加 Bar 数据 (优化)
    pub fn add_bars(&mut self, bars: Vec<Bar>) {
        for bar in bars {
            self.events.push_back(Event::Bar(bar));
        }
    }

    /// 从数组批量添加 Bar 数据 (高性能优化)
    #[allow(clippy::too_many_arguments)]
    pub fn add_arrays(
        &mut self,
        timestamps: Vec<i64>,
        opens: Vec<f64>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
        volumes: Vec<f64>,
        symbol: Option<String>,
        symbols: Option<Vec<String>>,
        extra: Option<HashMap<String, Vec<f64>>>,
    ) -> PyResult<()> {
        let len = timestamps.len();
        if opens.len() != len
            || highs.len() != len
            || lows.len() != len
            || closes.len() != len
            || volumes.len() != len
        {
            return Err(PyValueError::new_err(
                "All arrays must have the same length",
            ));
        }

        if let Some(ref syms) = symbols
            && syms.len() != len
        {
            return Err(PyValueError::new_err(
                "symbols array must have the same length as other arrays",
            ));
        }

        // Check extra arrays length
        if let Some(ref extra_data) = extra {
            for (key, val) in extra_data {
                if val.len() != len {
                     return Err(PyValueError::new_err(format!(
                        "Extra array '{}' must have the same length as other arrays",
                        key
                    )));
                }
            }
        }

        // Reserve space
        self.events.reserve(len);

        for i in 0..len {
            let sym = if let Some(ref syms) = symbols {
                syms[i].clone()
            } else if let Some(ref s) = symbol {
                s.clone()
            } else {
                "UNKNOWN".to_string()
            };

            let ts = timestamps[i];
            let normalized_ts = normalize_timestamp(ts);

            let mut bar_extra = HashMap::new();
            if let Some(ref extra_data) = extra {
                for (k, v) in extra_data {
                    bar_extra.insert(k.clone(), v[i]);
                }
            }

            self.events.push_back(Event::Bar(Bar {
                timestamp: normalized_ts,
                open: Decimal::from_f64(opens[i]).unwrap_or(Decimal::ZERO),
                high: Decimal::from_f64(highs[i]).unwrap_or(Decimal::ZERO),
                low: Decimal::from_f64(lows[i]).unwrap_or(Decimal::ZERO),
                close: Decimal::from_f64(closes[i]).unwrap_or(Decimal::ZERO),
                volume: Decimal::from_f64(volumes[i]).unwrap_or(Decimal::ZERO),
                symbol: sym,
                extra: bar_extra,
            }));
        }

        Ok(())
    }

    /// 对事件队列按时间戳进行排序
    /// 在多标的批量加载后必须调用，以确保回测时序正确
    pub fn sort(&mut self) {
        self.events.make_contiguous().sort_by_key(|e| match e {
            Event::Bar(b) => b.timestamp,
            Event::Tick(t) => t.timestamp,
        });
    }

    /// 添加 Tick 数据
    pub fn add_tick(&mut self, tick: Tick) {
        self.events.push_back(Event::Tick(tick));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_arrays_valid() {
        let ts = vec![1625097600]; // 2021-07-01 00:00:00 UTC
        let opens = vec![100.0];
        let highs = vec![105.0];
        let lows = vec![95.0];
        let closes = vec![102.0];
        let volumes = vec![1000.0];
        let symbol = Some("AAPL".to_string());

        let result = from_arrays(ts, opens, highs, lows, closes, volumes, symbol, None, None);
        assert!(result.is_ok());

        let bars = result.unwrap();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].symbol, "AAPL");
        assert_eq!(bars[0].open, Decimal::from(100));
        assert_eq!(bars[0].timestamp, 1625097600000000000); // Normalized to nanoseconds
    }

    #[test]
    fn test_from_arrays_mismatch_length() {
        let ts = vec![1625097600];
        let opens = vec![100.0, 101.0]; // Mismatch
        let highs = vec![105.0];
        let lows = vec![95.0];
        let closes = vec![102.0];
        let volumes = vec![1000.0];

        let result = from_arrays(ts, opens, highs, lows, closes, volumes, Some("AAPL".to_string()), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_arrays_individual_symbols() {
        let ts = vec![1, 2];
        let opens = vec![100.0, 200.0];
        let highs = vec![100.0, 200.0];
        let lows = vec![100.0, 200.0];
        let closes = vec![100.0, 200.0];
        let volumes = vec![100.0, 200.0];
        let symbols = vec!["AAPL".to_string(), "MSFT".to_string()];

        let result = from_arrays(ts, opens, highs, lows, closes, volumes, None, Some(symbols), None);
        assert!(result.is_ok());

        let bars = result.unwrap();
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].symbol, "AAPL");
        assert_eq!(bars[1].symbol, "MSFT");
    }

    #[test]
    fn test_data_feed() {
        let mut feed = DataFeed::new();
        let bar = Bar {
            timestamp: 1000,
            open: Decimal::ZERO,
            high: Decimal::ZERO,
            low: Decimal::ZERO,
            close: Decimal::ZERO,
            volume: Decimal::ZERO,
            symbol: "TEST".to_string(),
            extra: std::collections::HashMap::new(),
        };

        feed.add_bar(bar.clone());
        assert_eq!(feed.events.len(), 1);

        if let Some(Event::Bar(b)) = feed.events.front() {
            assert_eq!(b.symbol, "TEST");
        } else {
            panic!("Expected Bar event");
        }
    }
}
