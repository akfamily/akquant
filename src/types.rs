use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 资产类型
pub enum AssetType {
    Stock,
    Future,
}

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 交易标的
///
/// :ivar symbol: 代码
/// :ivar asset_type: 资产类型
/// :ivar multiplier: 合约乘数
/// :ivar margin_ratio: 保证金比率
/// :ivar tick_size: 最小变动价位
pub struct Instrument {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub asset_type: AssetType,
    #[pyo3(get)]
    pub multiplier: f64,     // 合约乘数 (股票为1.0)
    #[pyo3(get)]
    pub margin_ratio: f64,   // 保证金比率 (股票为1.0)
    #[pyo3(get)]
    pub tick_size: f64,      // 最小变动价位
}

#[pymethods]
impl Instrument {
    /// 创建交易标的
    ///
    /// :param symbol: 代码
    /// :param asset_type: 资产类型
    /// :param multiplier: 合约乘数
    /// :param margin_ratio: 保证金比率
    /// :param tick_size: 最小变动价位
    #[new]
    pub fn new(symbol: String, asset_type: AssetType, multiplier: f64, margin_ratio: f64, tick_size: f64) -> Self {
        Instrument {
            symbol,
            asset_type,
            multiplier,
            margin_ratio,
            tick_size,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 订单类型
pub enum OrderType {
    Market,
    Limit,
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 交易方向
pub enum OrderSide {
    Buy,
    Sell,
}

#[pyclass]
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

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// 订单有效期
pub enum TimeInForce {
    GTC, // Good Till Cancelled
    IOC, // Immediate Or Cancel
    FOK, // Fill Or Kill
    Day, // Good for Day
}
