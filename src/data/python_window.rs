//! 窗口聚合器的 Python 绑定: 策略侧 `BarGenerator` 与引擎 `subscribe_bars` 共用
//! 同一套窗口切分规则(交易日、交易块、即时闭合)。

use super::window::{WindowAggregator, WindowSubscription};
use crate::calendar::clock::LocalClock;
use crate::calendar::python::PySessionResolver;
use crate::calendar::resolver::{SessionResolver, effective_sessions};
use crate::model::{AssetType, Bar};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use std::collections::HashMap;
use std::sync::Arc;

/// 单周期窗口聚合器: 把基础 bar 流聚合成 ``freq`` 周期的窗口 bar, 规则与引擎
/// ``subscribe_bars`` 相同。bar 落在品种时段表全部交易块之外时(如期货误用股票时段表),
/// 每个品种告警一次。
///
/// :param freq: 窗口周期, ``"Nmin"`` / ``"Nh"`` / ``"1d"``。
/// :param session_windows: 手写本地交易时段 ``[(start, end), ...]``(``"HH:MM"``),
///     给定则覆盖时段表; ``end`` 早于 ``start`` 表示跨午夜。
/// :param base_interval_min: 基础 bar 周期(分钟)。给定(或 bar 带 ``interval_start``)
///     时, bar 区间终点到达窗口标签即闭合; 否则等下一根落入新窗口的 bar 才闭合。
/// :param resolver: 会话解析器; 不传时按 ``timezone`` 新建默认解析器(中国时区按
///     资产类型取时段表, 其他时区全天连续)。
/// :param timezone: IANA 时区名。不传时: 有 ``resolver`` 用其时区, 否则
///     ``"Asia/Shanghai"``。与 ``resolver`` 的时区不一致时报错。
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
        timezone=None,
        asset_types=None
    ))]
    fn new(
        freq: &str,
        session_windows: Option<Vec<(String, String)>>,
        base_interval_min: Option<u32>,
        resolver: Option<PyRef<'_, PySessionResolver>>,
        timezone: Option<&str>,
        asset_types: Option<HashMap<String, AssetType>>,
    ) -> PyResult<Self> {
        let named = match timezone {
            Some(tz) => {
                let clock = LocalClock::new(Some(tz), 0);
                if !clock.has_valid_tz() {
                    return Err(PyValueError::new_err(format!(
                        "无法识别的时区 {tz:?}，请使用 IANA 时区名，如 \"Asia/Shanghai\""
                    )));
                }
                Some(clock)
            }
            None => None,
        };
        if base_interval_min == Some(0) {
            return Err(PyValueError::new_err(
                "base_interval_min 必须为正整数（基础 bar 的分钟数），不知道时不传",
            ));
        }
        let sub = WindowSubscription::parse(None, freq, session_windows)
            .map_err(PyValueError::new_err)?;
        let freq_label = sub.freq_label.clone();
        let resolver = match resolver {
            Some(r) => {
                let r = r.arc();
                if let Some(clock) = &named
                    && !clock.same_zone(r.clock())
                {
                    return Err(PyValueError::new_err(format!(
                        "resolver 的时区（{}）与 timezone={:?} 不一致，窗口会按两个时区错位切分；请让两者一致，或不传 timezone（沿用 resolver 的时区）",
                        r.clock().describe(),
                        timezone.unwrap_or_default()
                    )));
                }
                r
            }
            None => {
                let tz = timezone.unwrap_or("Asia/Shanghai");
                let clock = named.unwrap_or_else(|| LocalClock::new(Some(tz), 0));
                let sessions = effective_sessions(None, Some(tz), 0);
                Arc::new(
                    SessionResolver::new(clock, None, sessions).map_err(PyValueError::new_err)?,
                )
            }
        };
        let mut inner = WindowAggregator::with_local_clock(resolver.clock().clone());
        inner
            .configure(vec![sub], base_interval_min)
            .map_err(PyValueError::new_err)?;
        inner.set_resolver(Some(resolver), asset_types.unwrap_or_default());
        inner.set_warn_out_of_block(true);
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

    /// 批量喂入单品种的基础 bar 列(时间戳非递减), 返回 ``(窗口 bar 列表, 行→窗口序号)``。
    ///
    /// 窗口按闭合顺序排列; 第二个返回值是与输入等长的 int64 数组, 第 i 行归属的
    /// 窗口在列表中的下标, 未闭合(``flush=False`` 时的尾部窗口)为 -1。
    /// ``flush=True`` 时最后闭合全部在形成窗口并计入列表——包括本聚合器里**其他品种**
    /// 的在形成窗口, 与 ``update()`` 混用时请对本品种单独建聚合器。数组须 C 连续;
    /// 价格 NaN/Inf 按 0 处理, 调用方应先剔除。
    ///
    /// :param timestamps: UTC 纳秒时间戳(int64 数组)
    /// :param opens: 开盘价(float64 数组), 其余价格/成交量列同
    /// :param symbol: 品种代码
    /// :param interval_starts: bar 区间起点(int64 数组, ``i64`` 最小值表示未知), 可选
    /// :param flush: 是否在末尾闭合全部在形成窗口
    #[pyo3(signature = (
        timestamps, opens, highs, lows, closes, volumes, symbol, interval_starts=None, flush=true
    ))]
    #[allow(clippy::too_many_arguments)]
    fn update_arrays<'py>(
        &mut self,
        py: Python<'py>,
        timestamps: &Bound<'py, PyAny>,
        opens: &Bound<'py, PyAny>,
        highs: &Bound<'py, PyAny>,
        lows: &Bound<'py, PyAny>,
        closes: &Bound<'py, PyAny>,
        volumes: &Bound<'py, PyAny>,
        symbol: &str,
        interval_starts: Option<&Bound<'py, PyAny>>,
        flush: bool,
    ) -> PyResult<(Vec<Bar>, Bound<'py, PyArray1<i64>>)> {
        let ts: PyReadonlyArray1<i64> = timestamps.extract()?;
        let o: PyReadonlyArray1<f64> = opens.extract()?;
        let h: PyReadonlyArray1<f64> = highs.extract()?;
        let l: PyReadonlyArray1<f64> = lows.extract()?;
        let c: PyReadonlyArray1<f64> = closes.extract()?;
        let v: PyReadonlyArray1<f64> = volumes.extract()?;
        let starts: Option<PyReadonlyArray1<i64>> =
            interval_starts.map(|a| a.extract()).transpose()?;
        let ts = ts.as_slice()?;
        let (o, h, l, c, v) = (
            o.as_slice()?,
            h.as_slice()?,
            l.as_slice()?,
            c.as_slice()?,
            v.as_slice()?,
        );
        let starts = match &starts {
            Some(a) => Some(a.as_slice()?),
            None => None,
        };
        let n = ts.len();
        if [o.len(), h.len(), l.len(), c.len(), v.len()]
            .iter()
            .any(|&len| len != n)
            || starts.is_some_and(|s| s.len() != n)
        {
            return Err(PyValueError::new_err(
                "update_arrays 的各列长度必须相同，请检查传入的时间戳与 OHLCV 数组",
            ));
        }
        let dec = |x: f64| Decimal::from_f64(x).unwrap_or(Decimal::ZERO);
        let mut windows: Vec<Bar> = Vec::new();
        let mut row_window: Vec<i64> = vec![-1; n];
        // 尚未闭合窗口里的行(闭合时回填下标)
        let mut pending: Vec<usize> = Vec::new();
        for i in 0..n {
            let bar = Bar {
                timestamp: ts[i],
                open: dec(o[i]),
                high: dec(h[i]),
                low: dec(l[i]),
                close: dec(c[i]),
                volume: dec(v[i]),
                symbol: symbol.to_string(),
                extra: HashMap::new(),
                freq: None,
                interval_start: starts.and_then(|s| (s[i] != i64::MIN).then_some(s[i])),
            };
            let closed = self.inner.update(&bar);
            let forming = self.inner.is_forming(symbol, &self.freq_label);
            // 单周期单品种: 至多两根, 先是被本行挤出的旧窗口(不含本行), 后是本行所在
            // 且即时闭合的窗口; 本行仍在形成中的窗口里时留待闭合时回填。
            let includes_row = !closed.is_empty() && !forming;
            let evicted = closed.len() - usize::from(includes_row);
            let base = windows.len();
            windows.extend(closed);
            if evicted > 0 {
                for r in pending.drain(..) {
                    row_window[r] = base as i64;
                }
            }
            if includes_row {
                let idx = (base + evicted) as i64;
                for r in pending.drain(..) {
                    row_window[r] = idx;
                }
                row_window[i] = idx;
            } else if forming {
                pending.push(i);
            }
        }
        if flush {
            let base = windows.len();
            let flushed = self.inner.flush();
            // flush 闭合全部品种; 回填只认本品种那一根
            if let Some(pos) = flushed.iter().position(|b| b.symbol == symbol) {
                for r in pending.drain(..) {
                    row_window[r] = (base + pos) as i64;
                }
            }
            windows.extend(flushed);
        }
        Ok((windows, PyArray1::from_vec(py, row_window)))
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
