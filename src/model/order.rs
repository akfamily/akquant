use super::market_data::extract_decimal;
use super::types::{OrderSide, OrderStatus, OrderType, TimeInForce};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 订单.
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
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    #[pyo3(get)]
    pub time_in_force: TimeInForce,
    pub trigger_price: Option<Decimal>,
    #[pyo3(get, set)]
    pub status: OrderStatus,
    pub filled_quantity: Decimal,
    pub average_filled_price: Option<Decimal>,
    #[pyo3(get)]
    pub created_at: i64,
    #[pyo3(get)]
    pub updated_at: i64,
    pub commission: Decimal,
    #[pyo3(get, set)]
    pub tag: String,
    #[pyo3(get)]
    pub reject_reason: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl Order {
    /// 创建订单.
    ///
    /// :param id: 订单ID
    /// :param symbol: 标的代码
    /// :param side: 交易方向
    /// :param order_type: 订单类型
    /// :param quantity: 数量
    /// :param price: 价格
    /// :param time_in_force: 订单有效期 (可选，默认 Day)
    /// :param trigger_price: 触发价格 (可选)
    /// :param created_at: 创建时间戳 (可选，默认 0)
    /// :param tag: 订单标签 (可选，默认 "")
    #[new]
    #[pyo3(signature = (id, symbol, side, order_type, quantity, price=None, time_in_force=None, trigger_price=None, created_at=None, tag=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        quantity: &Bound<'_, PyAny>,
        price: Option<&Bound<'_, PyAny>>,
        time_in_force: Option<TimeInForce>,
        trigger_price: Option<&Bound<'_, PyAny>>,
        created_at: Option<i64>,
        tag: Option<String>,
    ) -> PyResult<Self> {
        let created_at_ts = created_at.unwrap_or(0);
        Ok(Order {
            id,
            symbol,
            side,
            order_type,
            quantity: extract_decimal(quantity)?,
            price: match price {
                Some(p) => Some(extract_decimal(p)?),
                None => None,
            },
            time_in_force: time_in_force.unwrap_or(TimeInForce::Day),
            trigger_price: match trigger_price {
                Some(p) => Some(extract_decimal(p)?),
                None => None,
            },
            status: OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: created_at_ts,
            updated_at: created_at_ts,
            commission: Decimal::ZERO,
            tag: tag.unwrap_or_default(),
            reject_reason: String::new(),
        })
    }

    #[getter]
    /// 获取手续费.
    /// :return: 手续费
    fn get_commission(&self) -> f64 {
        self.commission.to_f64().unwrap_or_default()
    }

    #[getter]
    /// 获取订单数量.
    /// :return: 订单数量
    fn get_quantity(&self) -> f64 {
        self.quantity.to_f64().unwrap_or_default()
    }

    #[getter]
    /// 获取订单价格.
    /// :return: 订单价格 (如果为市价单则返回 None)
    fn get_price(&self) -> Option<f64> {
        self.price.map(|d| d.to_f64().unwrap_or_default())
    }

    #[getter]
    /// 获取触发价格.
    /// :return: 触发价格 (如果未设置则返回 None)
    fn get_trigger_price(&self) -> Option<f64> {
        self.trigger_price.map(|d| d.to_f64().unwrap_or_default())
    }

    #[getter]
    /// 获取已成交数量.
    /// :return: 已成交数量
    fn get_filled_quantity(&self) -> f64 {
        self.filled_quantity.to_f64().unwrap_or_default()
    }

    #[getter]
    /// 获取成交均价.
    /// :return: 成交均价 (如果未成交则返回 None)
    fn get_average_filled_price(&self) -> Option<f64> {
        self.average_filled_price
            .map(|d| d.to_f64().unwrap_or_default())
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Order(id={}, symbol={}, side={:?}, type={:?}, qty={}, price={:?}, tif={:?}, status={:?}, tag={}, reject_reason={})",
            self.id,
            self.symbol,
            self.side,
            self.order_type,
            self.quantity,
            self.price,
            self.time_in_force,
            self.status,
            self.tag,
            self.reject_reason
        )
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 成交记录.
///
/// :ivar id: 成交ID
/// :ivar order_id: 订单ID
/// :ivar symbol: 标的代码
/// :ivar side: 交易方向
/// :ivar quantity: 成交数量
/// :ivar price: 成交价格
/// :ivar commission: 手续费
/// :ivar timestamp: Unix 时间戳 (纳秒)
pub struct Trade {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub order_id: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: OrderSide,
    pub quantity: Decimal,
    pub price: Decimal,
    pub commission: Decimal,
    #[pyo3(get)]
    pub timestamp: i64,
    #[pyo3(get)]
    pub bar_index: usize,
}

#[gen_stub_pymethods]
#[pymethods]
impl Trade {
    /// 创建成交记录.
    ///
    /// :param id: 成交ID
    /// :param order_id: 订单ID
    /// :param symbol: 标的代码
    /// :param side: 交易方向
    /// :param quantity: 成交数量
    /// :param price: 成交价格
    /// :param commission: 手续费
    /// :param timestamp: Unix 时间戳 (纳秒)
    /// :param bar_index: K线索引
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        order_id: String,
        symbol: String,
        side: OrderSide,
        quantity: &Bound<'_, PyAny>,
        price: &Bound<'_, PyAny>,
        commission: &Bound<'_, PyAny>,
        timestamp: i64,
        bar_index: usize,
    ) -> PyResult<Self> {
        Ok(Trade {
            id,
            order_id,
            symbol,
            side,
            quantity: extract_decimal(quantity)?,
            price: extract_decimal(price)?,
            commission: extract_decimal(commission)?,
            timestamp,
            bar_index,
        })
    }

    #[getter]
    /// 获取成交数量.
    /// :return: 成交数量
    fn get_quantity(&self) -> f64 {
        self.quantity.to_f64().unwrap_or_default()
    }

    #[getter]
    /// 获取成交价格.
    /// :return: 成交价格
    fn get_price(&self) -> f64 {
        self.price.to_f64().unwrap_or_default()
    }

    #[getter]
    /// 获取手续费.
    /// :return: 手续费
    fn get_commission(&self) -> f64 {
        self.commission.to_f64().unwrap_or_default()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Trade(id={}, order_id={}, symbol={}, side={:?}, qty={}, price={}, time={}, bar={})",
            self.id,
            self.order_id,
            self.symbol,
            self.side,
            self.quantity,
            self.price,
            self.timestamp,
            self.bar_index
        )
    }
}
