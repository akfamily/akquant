use std::collections::HashMap;
use chrono::{NaiveDate, NaiveDateTime, TimeZone, Utc, FixedOffset};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// K线数据结构.
///
/// :ivar timestamp: Unix 时间戳 (纳秒)
/// :ivar open: 开盘价
/// :ivar high: 最高价
/// :ivar low: 最低价
/// :ivar close: 收盘价
/// :ivar volume: 成交量
/// :ivar symbol: 标的代码
pub struct Bar {
    #[pyo3(get, set)]
    pub timestamp: i64, // Unix 时间戳 (纳秒)
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    #[pyo3(get, set)]
    pub symbol: String, // 标的代码
    #[pyo3(get, set)]
    pub extra: HashMap<String, f64>, // 自定义字段
}

#[gen_stub_pymethods]
#[pymethods]
impl Bar {
    /// 创建 Bar 对象.
    ///
    /// :param timestamp: Unix 时间戳 (纳秒)
    /// :param open: 开盘价
    /// :param high: 最高价
    /// :param low: 最低价
    /// :param close: 收盘价
    /// :param volume: 成交量
    /// :param symbol: 标的代码
    /// :param extra: 自定义字段 (可选)
    #[new]
    #[pyo3(signature = (timestamp, open, high, low, close, volume, symbol, extra=None))]
    pub fn new(
        timestamp: &Bound<'_, PyAny>,
        open: &Bound<'_, PyAny>,
        high: &Bound<'_, PyAny>,
        low: &Bound<'_, PyAny>,
        close: &Bound<'_, PyAny>,
        volume: &Bound<'_, PyAny>,
        symbol: String,
        extra: Option<HashMap<String, f64>>,
    ) -> PyResult<Self> {
        let ts_val = extract_timestamp(timestamp)?;
        let open_val = extract_decimal(open)?;
        let high_val = extract_decimal(high)?;
        let low_val = extract_decimal(low)?;
        let close_val = extract_decimal(close)?;
        let volume_val = extract_decimal(volume)?;

        Ok(Bar {
            timestamp: ts_val,
            open: open_val,
            high: high_val,
            low: low_val,
            close: close_val,
            volume: volume_val,
            symbol,
            extra: extra.unwrap_or_default(),
        })
    }

    #[getter]
    fn get_open(&self) -> f64 {
        self.open.to_f64().unwrap_or_default()
    }

    #[setter]
    fn set_open(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.open = extract_decimal(value)?;
        Ok(())
    }

    #[getter]
    fn get_high(&self) -> f64 {
        self.high.to_f64().unwrap_or_default()
    }

    #[setter]
    fn set_high(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.high = extract_decimal(value)?;
        Ok(())
    }

    #[getter]
    fn get_low(&self) -> f64 {
        self.low.to_f64().unwrap_or_default()
    }

    #[setter]
    fn set_low(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.low = extract_decimal(value)?;
        Ok(())
    }

    #[getter]
    fn get_close(&self) -> f64 {
        self.close.to_f64().unwrap_or_default()
    }

    #[setter]
    fn set_close(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.close = extract_decimal(value)?;
        Ok(())
    }

    #[getter]
    fn get_volume(&self) -> f64 {
        self.volume.to_f64().unwrap_or_default()
    }

    #[setter]
    fn set_volume(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.volume = extract_decimal(value)?;
        Ok(())
    }

    /// 获取格式化的时间字符串 (Asia/Shanghai).
    /// 格式: YYYY-MM-DD HH:MM:SS
    #[getter]
    pub fn timestamp_str(&self) -> String {
        let secs = self.timestamp.div_euclid(1_000_000_000);
        let nanos = self.timestamp.rem_euclid(1_000_000_000) as u32;

        if let Some(dt) = Utc.timestamp_opt(secs, nanos).single() {
            // Default to Asia/Shanghai (UTC+8)
            let tz = FixedOffset::east_opt(8 * 3600).unwrap();
            dt.with_timezone(&tz).format("%Y-%m-%d %H:%M:%S").to_string()
        } else {
            self.timestamp.to_string()
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Bar(symbol={}, time={}, close={})",
            self.symbol, self.timestamp, self.close
        )
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Tick 数据结构.
///
/// :ivar timestamp: Unix 时间戳 (纳秒)
/// :ivar price: 最新价
/// :ivar volume: 成交量
/// :ivar symbol: 标的代码
pub struct Tick {
    #[pyo3(get, set)]
    pub timestamp: i64,
    pub price: Decimal,
    pub volume: Decimal,
    #[pyo3(get, set)]
    pub symbol: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl Tick {
    /// 创建 Tick 对象.
    ///
    /// :param timestamp: Unix 时间戳 (纳秒, int) 或 日期字符串 (str, "YYYY-MM-DD" / "YYYY-MM-DD HH:MM:SS")
    /// :param price: 最新价
    /// :param volume: 成交量
    /// :param symbol: 标的代码
    #[new]
    pub fn new(
        timestamp: &Bound<'_, PyAny>,
        price: &Bound<'_, PyAny>,
        volume: &Bound<'_, PyAny>,
        symbol: String,
    ) -> PyResult<Self> {
        let ts_val = extract_timestamp(timestamp)?;
        let price_val = extract_decimal(price)?;
        let volume_val = extract_decimal(volume)?;

        Ok(Tick {
            timestamp: ts_val,
            price: price_val,
            volume: volume_val,
            symbol,
        })
    }

    #[getter]
    fn get_price(&self) -> f64 {
        self.price.to_f64().unwrap_or_default()
    }

    #[setter]
    fn set_price(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.price = extract_decimal(value)?;
        Ok(())
    }

    #[getter]
    fn get_volume(&self) -> f64 {
        self.volume.to_f64().unwrap_or_default()
    }

    #[setter]
    fn set_volume(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.volume = extract_decimal(value)?;
        Ok(())
    }

    /// 获取格式化的时间字符串 (Asia/Shanghai).
    /// 格式: YYYY-MM-DD HH:MM:SS
    #[getter]
    pub fn timestamp_str(&self) -> String {
        let secs = self.timestamp.div_euclid(1_000_000_000);
        let nanos = self.timestamp.rem_euclid(1_000_000_000) as u32;

        if let Some(dt) = Utc.timestamp_opt(secs, nanos).single() {
            // Default to Asia/Shanghai (UTC+8)
            let tz = FixedOffset::east_opt(8 * 3600).unwrap();
            dt.with_timezone(&tz).format("%Y-%m-%d %H:%M:%S").to_string()
        } else {
            self.timestamp.to_string()
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Tick(symbol={}, time={}, price={})",
            self.symbol, self.timestamp, self.price
        )
    }
}

// Helper functions
pub fn extract_timestamp(timestamp: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(val) = timestamp.extract::<i64>() {
        let normalized = if val.abs() < 1_000_000_000_000 {
            val * 1_000_000_000
        } else {
            val
        };
        Ok(normalized)
    } else if let Ok(s) = timestamp.extract::<String>() {
        // Try YYYY-MM-DD HH:MM:SS
        if let Ok(dt) = NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S") {
            Ok(dt
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap_or(dt.and_utc().timestamp() * 1_000_000_000))
        } else if let Ok(date) = NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
            // Default to 00:00:00 UTC
            let dt = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
            Ok(dt
                .timestamp_nanos_opt()
                .unwrap_or(dt.timestamp() * 1_000_000_000))
        } else {
            Err(PyValueError::new_err(format!(
                "Invalid date format: {}. Expected YYYY-MM-DD or YYYY-MM-DD HH:MM:SS",
                s
            )))
        }
    } else {
        Err(PyTypeError::new_err(
            "timestamp must be int (unix timestamp) or str (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
        ))
    }
}

pub fn extract_decimal(value: &Bound<'_, PyAny>) -> PyResult<Decimal> {
    if let Ok(f) = value.extract::<f64>() {
        Decimal::from_f64_retain(f).ok_or_else(|| PyValueError::new_err("Invalid float"))
    } else if let Ok(s) = value.extract::<String>() {
        Decimal::from_str(&s).map_err(|e| PyValueError::new_err(e.to_string()))
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(Decimal::from(i))
    } else {
        Err(PyTypeError::new_err("Value must be float, str or int"))
    }
}
