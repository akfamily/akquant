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

impl pyo3_stub_gen::PyStubType for OrderType {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.OrderType", "akquant".into())
    }
}

impl pyo3_stub_gen::PyStubType for OrderSide {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.OrderSide", "akquant".into())
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

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 资产类型
pub enum AssetType {
    Stock,
    Fund,
    Futures,
    Option,
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 期权类型
pub enum OptionType {
    Call,
    Put,
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 订单类型
pub enum OrderType {
    Market,
    Limit,
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 交易方向
pub enum OrderSide {
    Buy,
    Sell,
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 订单状态
pub enum OrderStatus {
    New,
    Submitted,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 订单有效期
#[allow(clippy::upper_case_acronyms)]
pub enum TimeInForce {
    GTC, // Good Till Cancelled
    IOC, // Immediate Or Cancel
    FOK, // Fill Or Kill
    Day, // Good for Day
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 撮合执行模式
pub enum ExecutionMode {
    CurrentClose, // 当前Bar收盘价成交 (Cheat-on-Close)
    NextOpen,     // 下一根Bar开盘价成交 (Real-world)
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 交易时段状态
pub enum TradingSession {
    PreOpen,     // 盘前 (如集合竞价)
    Continuous,  // 连续竞价
    CallAuction, // 集合竞价 (开盘或收盘)
    Break,       // 休市 (如午休)
    Closed,      // 闭市
    PostClose,   // 盘后
}
