//! 窗口聚合器的 Python 绑定: 策略侧 `BarGenerator` 与引擎 `subscribe_bars` 共用
//! 同一套窗口切分规则(交易日、交易块、即时闭合)。

use super::window::{WindowAggregator, WindowSubscription};
use crate::calendar::clock::LocalClock;
use crate::calendar::python::PySessionResolver;
use crate::calendar::resolver::{SessionResolver, effective_sessions};
use crate::model::{AssetType, Bar};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::collections::HashMap;
use std::sync::Arc;

/// 单周期窗口聚合器: 把基础 bar 流聚合成 ``freq`` 周期的窗口 bar, 规则与引擎
/// ``subscribe_bars`` 相同。
///
/// :param freq: 窗口周期, ``"Nmin"`` / ``"Nh"`` / ``"1d"``。
/// :param session_windows: 手写本地交易时段 ``[(start, end), ...]``(``"HH:MM"``),
///     给定则覆盖时段表; ``end`` 早于 ``start`` 表示跨午夜。
/// :param base_interval_min: 基础 bar 周期(分钟)。给定(或 bar 带 ``interval_start``)
///     时, bar 区间终点到达窗口标签即闭合; 否则等下一根落入新窗口的 bar 才闭合。
/// :param resolver: 会话解析器; 不传时按 ``timezone`` 新建默认解析器(中国时区按
///     资产类型取时段表, 其他时区全天连续)。应与 ``timezone`` 用同一时区。
/// :param timezone: IANA 时区名, 默认 ``"Asia/Shanghai"``。
/// :param asset_types: 品种 → 资产类型; 未登记的品种按解析器的前缀/默认表取时段。
#[gen_stub_pyclass]
#[pyclass(name = "WindowAggregator", module = "akquant")]
pub struct PyWindowAggregator {
    inner: WindowAggregator,
    freq_label: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWindowAggregator {
    #[new]
    #[pyo3(signature = (
        freq,
        session_windows=None,
        base_interval_min=None,
        resolver=None,
        timezone="Asia/Shanghai",
        asset_types=None
    ))]
    fn new(
        freq: &str,
        session_windows: Option<Vec<(String, String)>>,
        base_interval_min: Option<u32>,
        resolver: Option<PyRef<'_, PySessionResolver>>,
        timezone: &str,
        asset_types: Option<HashMap<String, AssetType>>,
    ) -> PyResult<Self> {
        let clock = LocalClock::new(Some(timezone), 0);
        if !clock.has_valid_tz() {
            return Err(PyValueError::new_err(format!(
                "无法识别的时区 {timezone:?}，请使用 IANA 时区名，如 \"Asia/Shanghai\""
            )));
        }
        if base_interval_min == Some(0) {
            return Err(PyValueError::new_err(
                "base_interval_min 必须为正整数（基础 bar 的分钟数），不知道时不传",
            ));
        }
        let sub = WindowSubscription::parse(None, freq, session_windows)
            .map_err(PyValueError::new_err)?;
        let freq_label = sub.freq_label.clone();
        let resolver = match resolver {
            Some(r) => r.arc(),
            None => {
                let sessions = effective_sessions(None, Some(timezone), 0);
                Arc::new(
                    SessionResolver::new(clock, None, sessions).map_err(PyValueError::new_err)?,
                )
            }
        };
        let mut inner = WindowAggregator::with_clock(Some(timezone), 0);
        inner
            .configure(vec![sub], base_interval_min)
            .map_err(PyValueError::new_err)?;
        inner.set_resolver(Some(resolver), asset_types.unwrap_or_default());
        Ok(Self { inner, freq_label })
    }

    /// 窗口周期标签(规范化小写, 如 ``"5min"``), 即窗口 bar 的 ``freq``。
    #[getter]
    fn freq(&self) -> String {
        self.freq_label.clone()
    }

    /// 登记(或改写)品种的资产类型, 决定其时段表。
    fn set_asset_type(&mut self, symbol: &str, asset_type: AssetType) {
        self.inner.set_asset_type(symbol, asset_type);
    }

    /// 品种是否已登记资产类型。
    fn has_asset_type(&self, symbol: &str) -> bool {
        self.inner.has_asset_type(symbol)
    }

    /// 喂入一根基础 bar(时间戳非递减), 返回本步闭合的窗口 bar。
    fn update(&mut self, bar: PyRef<'_, Bar>) -> Vec<Bar> {
        self.inner.update(&bar)
    }

    /// 闭合全部在形成窗口(数据结束/收尾), 按品种排序返回。
    fn flush(&mut self) -> Vec<Bar> {
        self.inner.flush()
    }

    /// 品种正在形成(未闭合)的窗口 bar 快照; 没有则 None。
    fn current(&self, symbol: &str) -> Option<Bar> {
        self.inner.current(symbol, &self.freq_label)
    }
}
