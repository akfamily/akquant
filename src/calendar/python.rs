//! `SessionResolver` 的 Python 绑定。

use super::clock::LocalClock;
use super::resolver::{SessionResolver, effective_sessions};
use super::schedule::{TEMPLATE_NAMES, template};
use super::trading_calendar::{CalendarError, TradingCalendar};
use crate::model::{AssetType, TradingSession};
use chrono::NaiveDate;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::HashMap;

fn calendar_err(e: CalendarError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// 交易时段解析器：把时间点或 bar 区间解析为 (交易日, 时段)。
///
/// :param sessions: 代码前缀 → 时段模板名；``"default"`` 为全局默认。
///     未匹配时按资产类型取默认（股票/基金 CN_STOCK、期货 CN_FUTURES_DAY、
///     期权 CN_OPTION、其他 ALWAYS_OPEN）。
/// :param trading_calendar: 交易日列表；不传时按工作日推算（节假日会算错）。
/// :param timezone: IANA 时区名，默认 ``"Asia/Shanghai"``。
#[gen_stub_pyclass]
#[pyclass(name = "SessionResolver", module = "akquant")]
pub struct PySessionResolver {
    inner: SessionResolver,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySessionResolver {
    #[new]
    #[pyo3(signature = (sessions=None, trading_calendar=None, timezone="Asia/Shanghai"))]
    fn new(
        sessions: Option<HashMap<String, String>>,
        trading_calendar: Option<Vec<NaiveDate>>,
        timezone: &str,
    ) -> PyResult<Self> {
        let clock = LocalClock::new(Some(timezone), 0);
        if !clock.has_valid_tz() {
            return Err(PyValueError::new_err(format!(
                "无法识别的时区 {timezone:?}，请使用 IANA 时区名，如 \"Asia/Shanghai\""
            )));
        }
        let sessions_given = sessions.is_some();
        let mut pairs = Vec::new();
        for (key, name) in sessions.unwrap_or_default() {
            let schedule = template(&name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "未知时段模板 {name:?}（键 {key:?}），可选：{}",
                    TEMPLATE_NAMES.join(" / ")
                ))
            })?;
            pairs.push((key, schedule));
        }
        let user = if sessions_given { Some(pairs) } else { None };
        let pairs = effective_sessions(user, Some(timezone), 0);
        let calendar = match trading_calendar {
            Some(days) => Some(TradingCalendar::explicit(days).map_err(calendar_err)?),
            None => None,
        };
        let inner = SessionResolver::new(clock, calendar, pairs).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// 解析时间点（tick）。
    #[pyo3(signature = (symbol, timestamp, asset_type=None))]
    fn resolve_instant(
        &self,
        symbol: &str,
        timestamp: i64,
        asset_type: Option<AssetType>,
    ) -> PyResult<(NaiveDate, TradingSession)> {
        let r = self
            .inner
            .resolve_instant(symbol, asset_type, timestamp)
            .map_err(calendar_err)?;
        Ok((r.trading_day, r.session))
    }

    /// 解析 bar：区间 ``(interval_start, timestamp]``；``interval_start`` 为 None 时按时间戳前 1ns。
    #[pyo3(signature = (symbol, timestamp, interval_start=None, asset_type=None))]
    fn resolve_bar(
        &self,
        symbol: &str,
        timestamp: i64,
        interval_start: Option<i64>,
        asset_type: Option<AssetType>,
    ) -> PyResult<(NaiveDate, TradingSession)> {
        let r = self
            .inner
            .resolve_bar(symbol, asset_type, interval_start, timestamp)
            .map_err(calendar_err)?;
        Ok((r.trading_day, r.session))
    }

    /// 交易日窗口 ``(第一个时段开始, 最后一个时段结束)``，UTC 纳秒。
    #[pyo3(signature = (symbol, trading_day, asset_type=None))]
    fn day_bounds(
        &self,
        symbol: &str,
        trading_day: NaiveDate,
        asset_type: Option<AssetType>,
    ) -> PyResult<(i64, i64)> {
        self.inner
            .day_bounds(symbol, asset_type, trading_day)
            .map_err(calendar_err)
    }

    /// 设置在用标的 ``[(symbol, asset_type), ...]``：会话级收盘取它们时段表收盘的最大值。
    ///
    /// 不调用（或传空列表）时按全局默认表（``"default"``，缺省 CN_STOCK）计算。
    fn set_active(&mut self, symbols: Vec<(String, Option<AssetType>)>) {
        self.inner.set_active(&symbols);
    }

    /// 会话级交易日收盘时刻（UTC 纳秒）：所有在用时段表收盘的最大值。
    fn session_day_end(&self, trading_day: NaiveDate) -> PyResult<i64> {
        self.inner
            .session_day_end(trading_day)
            .map_err(calendar_err)
    }

    /// ``day`` 是否为会话交易日历上的交易日（用户日历；否则按在用时段表的缺省日历）。
    fn is_session_day(&self, day: NaiveDate) -> bool {
        self.inner.is_session_day(day)
    }

    /// 是否在使用缺省日历（未传 ``trading_calendar``）。
    #[getter]
    fn uses_fallback_calendar(&self) -> bool {
        self.inner.uses_fallback_calendar()
    }

    /// 内置时段模板名。
    #[staticmethod]
    fn templates() -> Vec<String> {
        TEMPLATE_NAMES.iter().map(|s| s.to_string()).collect()
    }
}
