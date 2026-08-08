use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

impl pyo3_stub_gen::PyStubType for AssetType {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.AssetType", "akquant".into())
    }
}

impl pyo3_stub_gen::PyStubType for OptionType {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.OptionType", "akquant".into())
    }
}

impl pyo3_stub_gen::PyStubType for OptionMarginModel {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.OptionMarginModel", "akquant".into())
    }
}

impl pyo3_stub_gen::PyStubType for OrderType {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.OrderType", "akquant".into())
    }
}

impl pyo3_stub_gen::PyStubType for OrderRole {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.OrderRole", "akquant".into())
    }
}

impl pyo3_stub_gen::PyStubType for OrderSide {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.OrderSide", "akquant".into())
    }
}

impl pyo3_stub_gen::PyStubType for PositionEffect {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.PositionEffect", "akquant".into())
    }
}

impl pyo3_stub_gen::PyStubType for OrderStatus {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.OrderStatus", "akquant".into())
    }
}

impl pyo3_stub_gen::PyStubType for TimeInForce {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.TimeInForce", "akquant".into())
    }
}

impl pyo3_stub_gen::PyStubType for ExecutionMode {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.ExecutionMode", "akquant".into())
    }
}

impl pyo3_stub_gen::PyStubType for TradingSession {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.TradingSession", "akquant".into())
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 结算方式
pub enum SettlementType {
    Physical, // 实物交割
    Cash,     // 现金交割
    ForceClose,
}

#[pymethods]
impl SettlementType {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl pyo3_stub_gen::PyStubType for SettlementType {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.SettlementType", "akquant".into())
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 资产类型
pub enum AssetType {
    Stock,
    Fund,
    Futures,
    Option,
    Crypto,
    Forex,
}

#[pymethods]
impl AssetType {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 期权类型
pub enum OptionType {
    Call,
    Put,
}

#[pymethods]
impl OptionType {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 期权保证金模型
pub enum OptionMarginModel {
    Ratio,
    #[default]
    ChinaSingleLeg,
    USBrokerSingleLeg,
    USBrokerSingleLegVolAdjusted,
}

#[pymethods]
impl OptionMarginModel {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 订单类型
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
    OCO,
    Bracket,
    StopTrail,
    StopTrailLimit,
}

#[pymethods]
impl OrderType {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 复杂订单节点角色
pub enum OrderRole {
    #[default]
    Standalone,
    Entry,
    StopLoss,
    TakeProfit,
    TrailStop,
    TrailStopLimit,
}

#[pymethods]
impl OrderRole {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 交易方向
pub enum OrderSide {
    Buy,
    Sell,
}

#[pymethods]
impl OrderSide {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 开平语义
pub enum PositionEffect {
    #[default]
    Auto,
    Open,
    Close,
    CloseToday,
    CloseYesterday,
}

impl PositionEffect {
    /// 导出用的规范词表,与 Python 入参词表一致(``auto`` / ``open`` / ``close``
    /// / ``close_today`` / ``close_yesterday``)。
    ///
    /// 不能用 `format!("{:?}").to_lowercase()`:那会把 `CloseToday` 变成
    /// `closetoday`,而 API 接受的是 `close_today`,导致 DataFrame 按规范词筛选
    /// 静默匹配不到。`position_effect` 与只读的 `status` 不同,它是可回传值,
    /// 入参与导出必须同一套词表。
    pub fn as_canonical_str(&self) -> &'static str {
        match self {
            PositionEffect::Auto => "auto",
            PositionEffect::Open => "open",
            PositionEffect::Close => "close",
            PositionEffect::CloseToday => "close_today",
            PositionEffect::CloseYesterday => "close_yesterday",
        }
    }
}

#[pymethods]
impl PositionEffect {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 订单状态
pub enum OrderStatus {
    New,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

#[pymethods]
impl OrderStatus {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 订单有效期
#[allow(clippy::upper_case_acronyms)]
pub enum TimeInForce {
    GTC, // Good Till Cancelled
    IOC, // Immediate Or Cancel
    FOK, // Fill Or Kill
    Day, // Good for Day
}

#[pymethods]
impl TimeInForce {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 撮合执行模式
pub enum ExecutionMode {
    CurrentClose,   // 当前Bar收盘价成交 (Cheat-on-Close)
    NextOpen,       // 下一根Bar开盘价成交 (Real-world)
    NextClose,      // 下一根Bar收盘价成交
    NextAverage,    // 下一根Bar均价成交 (TWAP/VWAP 模拟)
    NextHighLowMid, // 下一根Bar最高价和最低价的中间价成交
}

#[pymethods]
impl ExecutionMode {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PriceBasis {
    Open,
    Close,
    Ohlc4,
    Hl2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporalPolicy {
    SameCycle,
    NextEvent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutionPolicyCore {
    pub price_basis: PriceBasis,
    pub bar_offset: u8,
    pub temporal: TemporalPolicy,
}

impl Default for ExecutionPolicyCore {
    fn default() -> Self {
        Self {
            price_basis: PriceBasis::Open,
            bar_offset: 1,
            temporal: TemporalPolicy::SameCycle,
        }
    }
}

impl ExecutionPolicyCore {
    pub fn from_legacy(mode: ExecutionMode, timer_execution_policy: &str) -> Self {
        let temporal = if timer_execution_policy
            .trim()
            .eq_ignore_ascii_case("next_event")
        {
            TemporalPolicy::NextEvent
        } else {
            TemporalPolicy::SameCycle
        };
        match mode {
            ExecutionMode::CurrentClose => Self {
                price_basis: PriceBasis::Close,
                bar_offset: 0,
                temporal,
            },
            ExecutionMode::NextOpen => Self {
                price_basis: PriceBasis::Open,
                bar_offset: 1,
                temporal,
            },
            ExecutionMode::NextClose => Self {
                price_basis: PriceBasis::Close,
                bar_offset: 1,
                temporal,
            },
            ExecutionMode::NextAverage => Self {
                price_basis: PriceBasis::Ohlc4,
                bar_offset: 1,
                temporal,
            },
            ExecutionMode::NextHighLowMid => Self {
                price_basis: PriceBasis::Hl2,
                bar_offset: 1,
                temporal,
            },
        }
    }

    pub fn to_legacy_mode(self) -> ExecutionMode {
        match (self.price_basis, self.bar_offset) {
            (PriceBasis::Open, 1) => ExecutionMode::NextOpen,
            (PriceBasis::Close, 0) => ExecutionMode::CurrentClose,
            (PriceBasis::Close, 1) => ExecutionMode::NextClose,
            (PriceBasis::Ohlc4, 1) => ExecutionMode::NextAverage,
            (PriceBasis::Hl2, 1) => ExecutionMode::NextHighLowMid,
            (PriceBasis::Open, _) => ExecutionMode::NextOpen,
            (PriceBasis::Close, _) => ExecutionMode::CurrentClose,
            (PriceBasis::Ohlc4, _) => ExecutionMode::NextAverage,
            (PriceBasis::Hl2, _) => ExecutionMode::NextHighLowMid,
        }
    }

    pub fn temporal_as_str(self) -> &'static str {
        match self.temporal {
            TemporalPolicy::SameCycle => "same_cycle",
            TemporalPolicy::NextEvent => "next_event",
        }
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// 交易时段状态
pub enum TradingSession {
    PreOpen,     // 盘前 (如集合竞价)
    Continuous,  // 连续竞价
    CallAuction, // 集合竞价 (开盘或收盘)
    Break,       // 休市 (如午休)
    Closed,      // 闭市
    PostClose,   // 盘后
}

#[pymethods]
impl TradingSession {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}
