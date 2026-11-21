use chrono::{NaiveDate, NaiveDateTime, SecondsFormat, TimeZone, Utc};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn format_timestamp_iso(timestamp: i64) -> String {
    let secs = timestamp.div_euclid(1_000_000_000);
    let nanos = timestamp.rem_euclid(1_000_000_000) as u32;

    if let Some(dt) = Utc.timestamp_opt(secs, nanos).single() {
        dt.to_rfc3339_opts(SecondsFormat::AutoSi, true)
    } else {
        timestamp.to_string()
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// K 线数据结构.
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
    /// 该 bar 所属周期。基础输入 bar 为 `None`; 引擎 `WindowAggregator` 聚合出的
    /// 窗口 bar 带 `"5min"` / `"1h"` / `"1d"` 之类的标签, 供共用回调分流。
    #[pyo3(get, set)]
    #[serde(default)]
    pub freq: Option<String>,
    /// 区间起点（UTC 纳秒，不含）：bar 覆盖 `(interval_start, timestamp]`。
    /// `None` 表示未知，按“时间戳前 1ns”判断时段。由数据入口填写。
    #[pyo3(get, set)]
    #[serde(default)]
    pub interval_start: Option<i64>,
}

/// pickle 还原函数: 由 `Bar.__reduce__` 引用.
#[pyfunction]
pub fn _restore_bar(data: &[u8]) -> PyResult<Bar> {
    rmp_serde::from_slice(data).map_err(|e| PyValueError::new_err(format!("Bar 反序列化失败: {e}")))
}

#[gen_stub_pymethods]
#[pymethods]
impl Bar {
    /// pickle 支持: 以具名 MessagePack 编码还原.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (Py<PyBytes>,))> {
        let data = rmp_serde::to_vec_named(self)
            .map_err(|e| PyValueError::new_err(format!("Bar 序列化失败: {e}")))?;
        let restore = py
            .import("akquant.akquant")?
            .getattr("_restore_bar")?
            .unbind();
        Ok((restore, (PyBytes::new(py, &data).unbind(),)))
    }

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
    /// :param freq: 周期标签 (可选, 引擎聚合出的窗口 bar 才带)
    /// :param interval_start: 区间起点 (UTC 纳秒, 可选)
    #[new]
    #[pyo3(signature = (timestamp, open, high, low, close, volume, symbol, extra=None, freq=None, interval_start=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        timestamp: &Bound<'_, PyAny>,
        open: &Bound<'_, PyAny>,
        high: &Bound<'_, PyAny>,
        low: &Bound<'_, PyAny>,
        close: &Bound<'_, PyAny>,
        volume: &Bound<'_, PyAny>,
        symbol: String,
        extra: Option<HashMap<String, f64>>,
        freq: Option<String>,
        interval_start: Option<i64>,
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
            freq,
            interval_start,
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

    /// 获取格式化的时间 ISO 8601 字符串 (UTC).
    #[getter]
    pub fn timestamp_iso(&self) -> String {
        format_timestamp_iso(self.timestamp)
    }

    pub fn __repr__(&self) -> String {
        match &self.freq {
            Some(freq) => format!(
                "Bar(symbol={}, time={}, close={}, freq={})",
                self.symbol, self.timestamp, self.close, freq
            ),
            None => format!(
                "Bar(symbol={}, time={}, close={})",
                self.symbol, self.timestamp, self.close
            ),
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Tick 数据结构.
///
/// :ivar timestamp: Unix 时间戳 (纳秒)
/// :ivar price: 最新价
/// :ivar volume: 成交量
/// :ivar symbol: 标的代码
/// :ivar bid_price: 买一价, 无报价为 None
/// :ivar ask_price: 卖一价, 无报价为 None
/// :ivar bid_volume: 买一量, 无报价为 None
/// :ivar ask_volume: 卖一量, 无报价为 None
pub struct Tick {
    #[pyo3(get, set)]
    pub timestamp: i64,
    pub price: Decimal,
    pub volume: Decimal,
    #[pyo3(get, set)]
    pub symbol: String,
    /// 一档盘口; 旧版序列化数据没有该字段, 还原为无报价
    #[serde(default)]
    pub book: TopOfBook,
}

/// 一档盘口: 买一、卖一的价格与挂单量, 无报价的一侧为 None.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TopOfBook {
    pub bid_price: Option<Decimal>,
    pub ask_price: Option<Decimal>,
    pub bid_volume: Option<Decimal>,
    pub ask_volume: Option<Decimal>,
}

fn optional_decimal(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Decimal>> {
    match value {
        Some(v) if !v.is_none() => extract_decimal(v).map(Some),
        _ => Ok(None),
    }
}

fn optional_f64(value: Option<Decimal>) -> Option<f64> {
    value.and_then(|v| v.to_f64())
}

/// pickle 还原函数: 由 `Tick.__reduce__` 引用.
#[pyfunction]
pub fn _restore_tick(data: &[u8]) -> PyResult<Tick> {
    rmp_serde::from_slice(data)
        .map_err(|e| PyValueError::new_err(format!("Tick 反序列化失败: {e}")))
}

#[gen_stub_pymethods]
#[pymethods]
impl Tick {
    /// pickle 支持: 以具名 MessagePack 编码还原.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (Py<PyBytes>,))> {
        let data = rmp_serde::to_vec_named(self)
            .map_err(|e| PyValueError::new_err(format!("Tick 序列化失败: {e}")))?;
        let restore = py
            .import("akquant.akquant")?
            .getattr("_restore_tick")?
            .unbind();
        Ok((restore, (PyBytes::new(py, &data).unbind(),)))
    }

    /// 创建 Tick 对象.
    ///
    /// :param timestamp: Unix 时间戳 (纳秒, int) 或 日期字符串 (str, "YYYY-MM-DD" / "YYYY-MM-DD HH:MM:SS")
    /// :param price: 最新价
    /// :param volume: 成交量
    /// :param symbol: 标的代码
    /// :param bid_price: 买一价 (可选, 无报价不传或传 None)
    /// :param ask_price: 卖一价 (可选)
    /// :param bid_volume: 买一量 (可选)
    /// :param ask_volume: 卖一量 (可选)
    #[new]
    #[pyo3(signature = (timestamp, price, volume, symbol, bid_price=None, ask_price=None, bid_volume=None, ask_volume=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        timestamp: &Bound<'_, PyAny>,
        price: &Bound<'_, PyAny>,
        volume: &Bound<'_, PyAny>,
        symbol: String,
        bid_price: Option<&Bound<'_, PyAny>>,
        ask_price: Option<&Bound<'_, PyAny>>,
        bid_volume: Option<&Bound<'_, PyAny>>,
        ask_volume: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let ts_val = extract_timestamp(timestamp)?;
        let price_val = extract_decimal(price)?;
        let volume_val = extract_decimal(volume)?;

        Ok(Tick {
            timestamp: ts_val,
            price: price_val,
            volume: volume_val,
            symbol,
            book: TopOfBook {
                bid_price: optional_decimal(bid_price)?,
                ask_price: optional_decimal(ask_price)?,
                bid_volume: optional_decimal(bid_volume)?,
                ask_volume: optional_decimal(ask_volume)?,
            },
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

    /// 获取格式化的时间 ISO 8601 字符串 (UTC).
    #[getter]
    pub fn timestamp_iso(&self) -> String {
        format_timestamp_iso(self.timestamp)
    }

    /// 买一价, 无报价为 None.
    #[getter]
    fn get_bid_price(&self) -> Option<f64> {
        optional_f64(self.book.bid_price)
    }

    #[setter]
    fn set_bid_price(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.book.bid_price = optional_decimal(value)?;
        Ok(())
    }

    /// 卖一价, 无报价为 None.
    #[getter]
    fn get_ask_price(&self) -> Option<f64> {
        optional_f64(self.book.ask_price)
    }

    #[setter]
    fn set_ask_price(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.book.ask_price = optional_decimal(value)?;
        Ok(())
    }

    /// 买一量, 无报价为 None.
    #[getter]
    fn get_bid_volume(&self) -> Option<f64> {
        optional_f64(self.book.bid_volume)
    }

    #[setter]
    fn set_bid_volume(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.book.bid_volume = optional_decimal(value)?;
        Ok(())
    }

    /// 卖一量, 无报价为 None.
    #[getter]
    fn get_ask_volume(&self) -> Option<f64> {
        optional_f64(self.book.ask_volume)
    }

    #[setter]
    fn set_ask_volume(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.book.ask_volume = optional_decimal(value)?;
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 加入一档盘口之前的 Tick 序列化格式.
    #[derive(Serialize)]
    struct TickBeforeBook {
        timestamp: i64,
        price: Decimal,
        volume: Decimal,
        symbol: String,
    }

    #[test]
    fn tick_serialized_before_book_restores_without_quotes() {
        let old = TickBeforeBook {
            timestamp: 1,
            price: Decimal::from(10),
            volume: Decimal::from(2),
            symbol: "X".to_string(),
        };
        let bytes = rmp_serde::to_vec_named(&old).unwrap();
        let tick: Tick = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(tick.price, Decimal::from(10));
        assert_eq!(tick.book, TopOfBook::default());
    }
}
