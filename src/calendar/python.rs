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
use std::sync::Arc;

fn calendar_err(e: CalendarError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// 交易时段解析器：把时间点或 bar 区间解析为 (交易日, 时段)。
///
/// :param sessions: 代码前缀 → 时段模板名；``"default"`` 为全局默认。
///     未匹配时按资产类型取默认（股票/基金 CN_STOCK、期货 CN_FUTURES_DAY、
///     期权 CN_OPTION、其他 ALWAYS_OPEN）。
/// :param trading_calendar: 交易日列表；不传时中国市场按工作日推算（节假日会算错），ALWAYS_OPEN 每天都算。
/// :param timezone: IANA 时区名，默认 ``"Asia/Shanghai"``。
#[gen_stub_pyclass]
#[pyclass(name = "SessionResolver", module = "akquant")]
pub struct PySessionResolver {
    inner: Arc<SessionResolver>,
}

impl PySessionResolver {
    /// 供 Rust 侧（窗口聚合器等）共享同一解析器及其缓存。
    pub(crate) fn arc(&self) -> Arc<SessionResolver> {
        self.inner.clone()
    }
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
        Ok(Self {
            inner: Arc::new(inner),
        })
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
        // 克隆一份新解析器替换 Arc：已通过 arc() 共享出去的持有者不受影响
        let mut next = self.inner.as_ref().clone();
        next.set_active(&symbols);
        self.inner = Arc::new(next);
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

    /// 时刻 ``timestamp``（UTC 纳秒）所属的会话交易日：第一个收盘不早于它的交易日
    /// （夜盘属下一交易日）。
    fn session_trading_day(&self, timestamp: i64) -> PyResult<NaiveDate> {
        self.inner
            .session_trading_day(timestamp)
            .map_err(calendar_err)
    }

    /// 会话交易日历上 ``day`` 之后的下一个交易日；超出日历范围时报 ValueError。
    fn next_session_day(&self, day: NaiveDate) -> PyResult<NaiveDate> {
        self.inner.next_session_day(day).map_err(calendar_err)
    }

    /// 是否在使用缺省日历（未传 ``trading_calendar``）。
    #[getter]
    fn uses_fallback_calendar(&self) -> bool {
        self.inner.uses_fallback_calendar()
    }

    /// 会话日历是否为全日历兜底（未传 ``trading_calendar``，且在用时段表全是
    /// ALWAYS_OPEN 这类每天都开的表）：此时周末也算交易日。
    #[getter]
    fn uses_all_days_fallback(&self) -> bool {
        self.inner.uses_all_days_fallback()
    }

    /// 品种在交易日 ``trading_day`` 的交易块 ``[(起点, 终点), ...]``（UTC 纳秒，半开区间）。
    ///
    /// 交易块是首尾相接的非休市/非闭市时段合并成的连续交易区间，如 A 股为
    /// 09:15-11:30 与 13:00-15:00 两块。
    #[pyo3(signature = (symbol, trading_day, asset_type=None))]
    fn trading_blocks(
        &self,
        symbol: &str,
        trading_day: NaiveDate,
        asset_type: Option<AssetType>,
    ) -> PyResult<Vec<(i64, i64)>> {
        let blocks = self
            .inner
            .trading_blocks(symbol, asset_type, trading_day)
            .map_err(calendar_err)?;
        Ok(blocks.iter().map(|b| (b.start, b.end)).collect())
    }

    /// 时刻 ``timestamp`` 所在的交易块 ``(交易日, 序号, 起点, 终点)``；落在块外
    /// （午休、闭市）返回 None。块为半开区间，终点戳事件请传区间内一点（ts-1）。
    #[pyo3(signature = (symbol, timestamp, asset_type=None))]
    fn block_at(
        &self,
        symbol: &str,
        timestamp: i64,
        asset_type: Option<AssetType>,
    ) -> PyResult<Option<(NaiveDate, usize, i64, i64)>> {
        let b = self
            .inner
            .block_at(symbol, asset_type, timestamp)
            .map_err(calendar_err)?;
        Ok(b.map(|b| (b.trading_day, b.index, b.start, b.end)))
    }

    /// 内置时段模板名。
    #[staticmethod]
    fn templates() -> Vec<String> {
        TEMPLATE_NAMES.iter().map(|s| s.to_string()).collect()
    }
}
