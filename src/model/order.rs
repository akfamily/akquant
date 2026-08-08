use super::market_data::extract_decimal;
use super::types::{
    ExecutionPolicyCore, OrderRole, OrderSide, OrderStatus, OrderType, PositionEffect, TimeInForce,
};
use chrono::{SecondsFormat, TimeZone, Utc};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};

const ORDER_VALIDATION_PREFIX: &str = "AKQ-ORDER-VALIDATION";

fn validation_err(message: &str) -> PyErr {
    PyValueError::new_err(format!("[{ORDER_VALIDATION_PREFIX}] {message}"))
}

fn ensure_positive(value: Decimal, field: &str) -> PyResult<()> {
    if value <= Decimal::ZERO {
        return Err(validation_err(&format!("{field} must be > 0")));
    }
    Ok(())
}

fn ensure_non_negative(value: Decimal, field: &str) -> PyResult<()> {
    if value < Decimal::ZERO {
        return Err(validation_err(&format!("{field} must be >= 0")));
    }
    Ok(())
}

fn validate_order_inputs(
    quantity: Decimal,
    price: Option<Decimal>,
    trigger_price: Option<Decimal>,
    trail_offset: Option<Decimal>,
    trail_reference_price: Option<Decimal>,
) -> PyResult<()> {
    ensure_positive(quantity, "quantity")?;
    if let Some(v) = price {
        ensure_positive(v, "price")?;
    }
    if let Some(v) = trigger_price {
        ensure_positive(v, "trigger_price")?;
    }
    if let Some(v) = trail_offset {
        ensure_non_negative(v, "trail_offset")?;
    }
    if let Some(v) = trail_reference_price {
        ensure_positive(v, "trail_reference_price")?;
    }
    Ok(())
}

pub(crate) fn format_timestamp_iso(timestamp: i64) -> String {
    let secs = timestamp.div_euclid(1_000_000_000);
    let nanos = timestamp.rem_euclid(1_000_000_000) as u32;

    if let Some(dt) = Utc.timestamp_opt(secs, nanos).single() {
        dt.to_rfc3339_opts(SecondsFormat::AutoSi, true)
    } else {
        timestamp.to_string()
    }
}

pub fn project_position_after(
    side: OrderSide,
    position_effect: PositionEffect,
    current_pos: Decimal,
    quantity: Decimal,
) -> Decimal {
    match position_effect {
        PositionEffect::Close | PositionEffect::CloseToday | PositionEffect::CloseYesterday => {
            match side {
                OrderSide::Buy => {
                    if current_pos < Decimal::ZERO {
                        current_pos + quantity.min(-current_pos)
                    } else {
                        current_pos
                    }
                }
                OrderSide::Sell => {
                    if current_pos > Decimal::ZERO {
                        current_pos - quantity.min(current_pos)
                    } else {
                        current_pos
                    }
                }
            }
        }
        PositionEffect::Open | PositionEffect::Auto => match side {
            OrderSide::Buy => current_pos + quantity,
            OrderSide::Sell => current_pos - quantity,
        },
    }
}

pub fn position_delta(
    side: OrderSide,
    position_effect: PositionEffect,
    current_pos: Decimal,
    quantity: Decimal,
) -> Decimal {
    project_position_after(side, position_effect, current_pos, quantity) - current_pos
}

pub fn reduction_priority_rank(side: OrderSide, position_effect: PositionEffect) -> u8 {
    match position_effect {
        PositionEffect::Close | PositionEffect::CloseToday | PositionEffect::CloseYesterday => 0,
        PositionEffect::Open => 1,
        PositionEffect::Auto => {
            if side == OrderSide::Sell {
                0
            } else {
                1
            }
        }
    }
}

pub fn is_reduce_first_order(side: OrderSide, position_effect: PositionEffect) -> bool {
    reduction_priority_rank(side, position_effect) == 0
}

/// 在途订单是否会减少持仓敞口(用于可平仓投影)。
///
/// 与 [`is_reduce_first_order`] 的区别:后者判定撮合排序优先级,把
/// `(Sell, Auto)` 也算作 reduce;此处只认**显式**平仓语义与 `reduce_only`,
/// 避免把 `Auto` 单误算成减仓,从而低估可开量。
pub fn is_position_reducing_order(order: &Order) -> bool {
    if order.reduce_only {
        return true;
    }
    matches!(
        order.position_effect,
        PositionEffect::Close | PositionEffect::CloseToday | PositionEffect::CloseYesterday
    )
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
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
/// :ivar trail_offset: 跟踪止损偏移量
/// :ivar trail_reference_price: 跟踪止损参考价
/// :ivar graph_id: 复杂订单图 ID
/// :ivar parent_order_id: 父订单 ID
/// :ivar order_role: 复杂订单节点角色
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
    #[serde(default)]
    pub trail_offset: Option<Decimal>,
    #[serde(default)]
    pub trail_reference_price: Option<Decimal>,
    #[serde(default)]
    pub fill_policy_override: Option<ExecutionPolicyCore>,
    #[serde(default)]
    pub slippage_type_override: Option<String>,
    #[serde(default)]
    pub slippage_value_override: Option<Decimal>,
    #[serde(default)]
    pub commission_type_override: Option<String>,
    #[serde(default)]
    pub commission_value_override: Option<Decimal>,
    #[pyo3(get)]
    #[serde(default)]
    pub graph_id: Option<String>,
    #[pyo3(get)]
    #[serde(default)]
    pub parent_order_id: Option<String>,
    #[pyo3(get)]
    #[serde(default)]
    pub order_role: OrderRole,
    #[pyo3(get)]
    #[serde(default)]
    pub position_effect: PositionEffect,
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
    #[pyo3(get)]
    #[serde(default)]
    pub owner_strategy_id: Option<String>,
    #[serde(default)]
    pub allow_quantity_auto_resize: bool,
    #[pyo3(get)]
    #[serde(default)]
    pub reduce_only: bool,
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
    #[pyo3(signature = (id, symbol, side, order_type, quantity, price=None, time_in_force=None, trigger_price=None, created_at=None, tag=None, owner_strategy_id=None, graph_id=None, parent_order_id=None, order_role=None, trail_offset=None, trail_reference_price=None, position_effect=None, reduce_only=false))]
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
        owner_strategy_id: Option<String>,
        graph_id: Option<String>,
        parent_order_id: Option<String>,
        order_role: Option<OrderRole>,
        trail_offset: Option<&Bound<'_, PyAny>>,
        trail_reference_price: Option<&Bound<'_, PyAny>>,
        position_effect: Option<PositionEffect>,
        reduce_only: bool,
    ) -> PyResult<Self> {
        let created_at_ts = created_at.unwrap_or(0);
        let quantity_dec = extract_decimal(quantity)?;
        let price_dec = match price {
            Some(p) => Some(extract_decimal(p)?),
            None => None,
        };
        let trigger_dec = match trigger_price {
            Some(p) => Some(extract_decimal(p)?),
            None => None,
        };
        let trail_offset_dec = match trail_offset {
            Some(v) => Some(extract_decimal(v)?),
            None => None,
        };
        let trail_reference_dec = match trail_reference_price {
            Some(v) => Some(extract_decimal(v)?),
            None => None,
        };
        validate_order_inputs(
            quantity_dec,
            price_dec,
            trigger_dec,
            trail_offset_dec,
            trail_reference_dec,
        )?;
        Ok(Order {
            id,
            symbol,
            side,
            order_type,
            quantity: quantity_dec,
            price: price_dec,
            time_in_force: time_in_force.unwrap_or(TimeInForce::Day),
            trigger_price: trigger_dec,
            trail_offset: trail_offset_dec,
            trail_reference_price: trail_reference_dec,
            fill_policy_override: None,
            slippage_type_override: None,
            slippage_value_override: None,
            commission_type_override: None,
            commission_value_override: None,
            graph_id,
            parent_order_id,
            order_role: order_role.unwrap_or_default(),
            position_effect: position_effect.unwrap_or_default(),
            status: OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: created_at_ts,
            updated_at: created_at_ts,
            commission: Decimal::ZERO,
            tag: tag.unwrap_or_default(),
            reject_reason: String::new(),
            owner_strategy_id,
            allow_quantity_auto_resize: false,
            reduce_only,
        })
    }

    #[getter]
    /// 获取手续费.
    /// :return: 手续费
    fn get_commission(&self) -> f64 {
        self.commission.to_f64().unwrap_or_default()
    }

    #[getter]
    /// 获取创建时间 ISO 8601 字符串 (UTC).
    /// :return: 格式化时间字符串 YYYY-MM-DDTHH:MM:SS[.nnnnnnnnn]Z
    fn created_at_iso(&self) -> String {
        format_timestamp_iso(self.created_at)
    }

    #[getter]
    /// 获取更新时间 ISO 8601 字符串 (UTC).
    /// :return: 格式化时间字符串 YYYY-MM-DDTHH:MM:SS[.nnnnnnnnn]Z
    fn updated_at_iso(&self) -> String {
        format_timestamp_iso(self.updated_at)
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
    /// 获取跟踪止损偏移量.
    /// :return: 跟踪止损偏移量 (如果未设置则返回 None)
    fn get_trail_offset(&self) -> Option<f64> {
        self.trail_offset.map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
    fn set_trail_offset(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        if let Some(v) = value {
            let dec = extract_decimal(v)?;
            ensure_non_negative(dec, "trail_offset")?;
            self.trail_offset = Some(dec);
        } else {
            self.trail_offset = None;
        }
        Ok(())
    }

    #[getter]
    /// 获取跟踪止损参考价.
    /// :return: 跟踪止损参考价 (如果未设置则返回 None)
    fn get_trail_reference_price(&self) -> Option<f64> {
        self.trail_reference_price
            .map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
    fn set_trail_reference_price(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        if let Some(v) = value {
            let dec = extract_decimal(v)?;
            ensure_positive(dec, "trail_reference_price")?;
            self.trail_reference_price = Some(dec);
        } else {
            self.trail_reference_price = None;
        }
        Ok(())
    }

    #[getter]
    /// 获取已成交数量.
    /// :return: 已成交数量
    fn get_filled_quantity(&self) -> f64 {
        self.filled_quantity.to_f64().unwrap_or_default()
    }

    #[setter]
    fn set_filled_quantity(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let dec = extract_decimal(value)?;
        ensure_non_negative(dec, "filled_quantity")?;
        self.filled_quantity = dec;
        Ok(())
    }

    #[getter]
    /// 获取成交均价.
    /// :return: 成交均价 (如果未成交则返回 None)
    fn get_average_filled_price(&self) -> Option<f64> {
        self.average_filled_price
            .map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
    fn set_average_filled_price(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        if let Some(v) = value {
            let dec = extract_decimal(v)?;
            ensure_positive(dec, "average_filled_price")?;
            self.average_filled_price = Some(dec);
        } else {
            self.average_filled_price = None;
        }
        Ok(())
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Order(id={}, symbol={}, side={:?}, type={:?}, qty={}, price={:?}, trigger={:?}, trail_offset={:?}, trail_ref={:?}, graph_id={:?}, parent_order_id={:?}, role={:?}, effect={:?}, reduce_only={}, tif={:?}, status={:?}, tag={}, reject_reason={})",
            self.id,
            self.symbol,
            self.side,
            self.order_type,
            self.quantity,
            self.price,
            self.trigger_price,
            self.trail_offset,
            self.trail_reference_price,
            self.graph_id,
            self.parent_order_id,
            self.order_role,
            self.position_effect,
            self.reduce_only,
            self.time_in_force,
            self.status,
            self.tag,
            self.reject_reason
        )
    }
}

impl Order {
    #[cfg(test)]
    pub fn test_new(
        id: &str,
        symbol: &str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
    ) -> Self {
        Order {
            id: id.to_string(),
            symbol: symbol.to_string(),
            side,
            order_type,
            quantity,
            price: None,
            time_in_force: TimeInForce::Day,
            trigger_price: None,
            trail_offset: None,
            trail_reference_price: None,
            fill_policy_override: None,
            slippage_type_override: None,
            slippage_value_override: None,
            commission_type_override: None,
            commission_value_override: None,
            graph_id: None,
            parent_order_id: None,
            order_role: OrderRole::Standalone,
            position_effect: PositionEffect::Auto,
            status: OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: 0,
            updated_at: 0,
            commission: Decimal::ZERO,
            tag: String::new(),
            reject_reason: String::new(),
            owner_strategy_id: None,
            allow_quantity_auto_resize: false,
            reduce_only: false,
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
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
    #[pyo3(get)]
    #[serde(default)]
    pub position_effect: PositionEffect,
    pub quantity: Decimal,
    pub price: Decimal,
    pub commission: Decimal,
    #[pyo3(get)]
    pub timestamp: i64,
    #[pyo3(get)]
    pub bar_index: usize,
    #[pyo3(get)]
    #[serde(default)]
    pub owner_strategy_id: Option<String>,
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
    #[pyo3(signature = (id, order_id, symbol, side, quantity, price, commission, timestamp, bar_index, owner_strategy_id=None, position_effect=None))]
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
        owner_strategy_id: Option<String>,
        position_effect: Option<PositionEffect>,
    ) -> PyResult<Self> {
        Ok(Trade {
            id,
            order_id,
            symbol,
            side,
            position_effect: position_effect.unwrap_or_default(),
            quantity: extract_decimal(quantity)?,
            price: extract_decimal(price)?,
            commission: extract_decimal(commission)?,
            timestamp,
            bar_index,
            owner_strategy_id,
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

    #[getter]
    /// 获取格式化的成交时间 ISO 8601 字符串 (UTC).
    /// :return: 格式化时间字符串 YYYY-MM-DDTHH:MM:SS[.nnnnnnnnn]Z
    fn timestamp_iso(&self) -> String {
        format_timestamp_iso(self.timestamp)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct LegacyOrder {
        id: String,
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Option<Decimal>,
        time_in_force: TimeInForce,
        trigger_price: Option<Decimal>,
        status: OrderStatus,
        filled_quantity: Decimal,
        average_filled_price: Option<Decimal>,
        created_at: i64,
        updated_at: i64,
        commission: Decimal,
        tag: String,
        reject_reason: String,
    }

    #[derive(Serialize)]
    struct LegacyTrade {
        id: String,
        order_id: String,
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
        timestamp: i64,
        bar_index: usize,
    }

    #[test]
    fn test_order_deserialize_legacy_without_owner_strategy_id() {
        let legacy = LegacyOrder {
            id: "o1".to_string(),
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: Decimal::from(10),
            price: Some(Decimal::from(100)),
            time_in_force: TimeInForce::Day,
            trigger_price: None,
            status: OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: 1,
            updated_at: 1,
            commission: Decimal::ZERO,
            tag: String::new(),
            reject_reason: String::new(),
        };
        let bytes = rmp_serde::to_vec_named(&legacy).expect("serialize legacy order");
        let order: Order =
            rmp_serde::from_slice(&bytes).expect("deserialize order from legacy payload");
        assert!(order.owner_strategy_id.is_none());
        assert!(order.trail_offset.is_none());
        assert!(order.trail_reference_price.is_none());
        assert!(order.graph_id.is_none());
        assert!(order.parent_order_id.is_none());
        assert_eq!(order.order_role, OrderRole::Standalone);
    }

    #[test]
    fn test_trade_deserialize_legacy_without_owner_strategy_id() {
        let legacy = LegacyTrade {
            id: "t1".to_string(),
            order_id: "o1".to_string(),
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(10),
            price: Decimal::from(100),
            commission: Decimal::ZERO,
            timestamp: 1,
            bar_index: 0,
        };
        let bytes = rmp_serde::to_vec_named(&legacy).expect("serialize legacy trade");
        let trade: Trade =
            rmp_serde::from_slice(&bytes).expect("deserialize trade from legacy payload");
        assert!(trade.owner_strategy_id.is_none());
    }

    #[test]
    fn test_order_new_rejects_non_positive_quantity() {
        let result = validate_order_inputs(Decimal::ZERO, None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_order_new_rejects_non_positive_price() {
        let result = validate_order_inputs(Decimal::from(1), Some(Decimal::ZERO), None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_timestamp_iso_uses_utc() {
        let timestamp = 1_735_801_200_000_000_000_i64; // 2025-01-02 15:00:00+08:00
        assert_eq!(format_timestamp_iso(timestamp), "2025-01-02T07:00:00Z");
    }

    #[test]
    fn test_order_time_iso_getters() {
        let mut order = Order::test_new(
            "o1",
            "AAPL",
            OrderSide::Buy,
            OrderType::Limit,
            Decimal::from(10),
        );
        let timestamp = 1_735_801_200_000_000_000_i64;
        order.created_at = timestamp;
        order.updated_at = timestamp;

        assert_eq!(order.created_at_iso(), "2025-01-02T07:00:00Z");
        assert_eq!(order.updated_at_iso(), "2025-01-02T07:00:00Z");
    }

    #[test]
    fn test_trade_timestamp_iso_getter() {
        let trade = Trade {
            id: "t1".to_string(),
            order_id: "o1".to_string(),
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            position_effect: PositionEffect::Auto,
            quantity: Decimal::from(10),
            price: Decimal::from(100),
            commission: Decimal::ZERO,
            timestamp: 1_735_801_200_000_000_000_i64,
            bar_index: 0,
            owner_strategy_id: None,
        };

        assert_eq!(trade.timestamp_iso(), "2025-01-02T07:00:00Z");
    }
}
