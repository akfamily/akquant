use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use crate::types::{OrderSide, OrderType, OrderStatus, TimeInForce};

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 订单
///
/// :ivar id: 订单ID
/// :ivar symbol: 标的代码
/// :ivar side: 交易方向
/// :ivar order_type: 订单类型
/// :ivar quantity: 数量
/// :ivar price: 价格 (限价单有效)
/// :ivar time_in_force: 订单有效期
/// :ivar trigger_price: 触发价格 (止损/止盈单)
/// :ivar status: 订单状态
/// :ivar filled_quantity: 已成交数量
/// :ivar average_filled_price: 成交均价
pub struct Order {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: OrderSide,
    #[pyo3(get)]
    pub order_type: OrderType,
    #[pyo3(get)]
    pub quantity: f64,
    #[pyo3(get)]
    pub price: Option<f64>,
    #[pyo3(get)]
    pub time_in_force: TimeInForce,
    #[pyo3(get)]
    pub trigger_price: Option<f64>,
    #[pyo3(get, set)]
    pub status: OrderStatus,
    #[pyo3(get, set)]
    pub filled_quantity: f64,
    #[pyo3(get, set)]
    pub average_filled_price: Option<f64>,
}

#[pymethods]
impl Order {
    /// 创建订单
    ///
    /// :param id: 订单ID
    /// :param symbol: 标的代码
    /// :param side: 交易方向
    /// :param order_type: 订单类型
    /// :param quantity: 数量
    /// :param price: 价格
    /// :param time_in_force: 订单有效期 (可选，默认 Day)
    /// :param trigger_price: 触发价格 (可选)
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        quantity: f64,
        price: Option<f64>,
        time_in_force: Option<TimeInForce>,
        trigger_price: Option<f64>,
    ) -> Self {
        Order {
            id,
            symbol,
            side,
            order_type,
            quantity,
            price,
            time_in_force: time_in_force.unwrap_or(TimeInForce::Day),
            trigger_price,
            status: OrderStatus::New,
            filled_quantity: 0.0,
            average_filled_price: None,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Order(id={}, symbol={}, side={:?}, type={:?}, qty={}, price={:?}, tif={:?}, status={:?})",
            self.id, self.symbol, self.side, self.order_type, self.quantity, self.price, self.time_in_force, self.status
        )
    }
}

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 成交记录
///
/// :ivar id: 成交ID
/// :ivar order_id: 订单ID
/// :ivar symbol: 标的代码
/// :ivar side: 交易方向
/// :ivar quantity: 成交数量
/// :ivar price: 成交价格
/// :ivar commission: 佣金
/// :ivar timestamp: 成交时间戳
pub struct Trade {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub order_id: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: OrderSide,
    #[pyo3(get)]
    pub quantity: f64,
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub commission: f64,
    #[pyo3(get)]
    pub timestamp: i64,
}

#[pymethods]
impl Trade {
    /// 创建成交记录
    ///
    /// :param id: 成交ID
    /// :param order_id: 订单ID
    /// :param symbol: 标的代码
    /// :param side: 交易方向
    /// :param quantity: 成交数量
    /// :param price: 成交价格
    /// :param commission: 佣金
    /// :param timestamp: 成交时间戳
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        order_id: String,
        symbol: String,
        side: OrderSide,
        quantity: f64,
        price: f64,
        commission: f64,
        timestamp: i64,
    ) -> Self {
        Trade {
            id,
            order_id,
            symbol,
            side,
            quantity,
            price,
            commission,
            timestamp,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Trade(id={}, order_id={}, symbol={}, side={:?}, qty={}, price={}, time={})",
            self.id, self.order_id, self.symbol, self.side, self.quantity, self.price, self.timestamp
        )
    }
}
