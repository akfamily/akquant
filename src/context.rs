use crate::analysis::ClosedTrade;
use crate::event::Event;
use crate::history::HistoryBuffer;
use crate::model::market_data::extract_decimal;
use crate::model::{Order, OrderSide, OrderType, TimeInForce, Timer, Trade, TradingSession};
use crate::risk::RiskConfig;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, mpsc::Sender};
use uuid::Uuid;

#[gen_stub_pyclass]
#[pyclass]
/// 策略上下文.
///
/// :ivar orders: 订单列表 (内部使用)
/// :ivar cash: 当前现金
/// :ivar positions: 当前持仓
/// :ivar available_positions: 可用持仓
/// :ivar session: 当前交易时段
pub struct StrategyContext {
    #[pyo3(get)]
    pub orders: Vec<Order>, // Accumulated orders (new)
    #[pyo3(get)]
    pub canceled_order_ids: Vec<String>, // Accumulated cancellations
    #[pyo3(get)]
    pub active_orders: Vec<Order>, // Existing pending orders
    pub timers: Vec<Timer>, // Accumulated timers
    pub cash: Decimal,
    pub positions: HashMap<String, Decimal>,
    pub available_positions: HashMap<String, Decimal>,
    #[pyo3(get)]
    pub session: TradingSession,
    #[pyo3(get)]
    pub current_time: i64,
    // Do NOT expose closed_trades as a direct getter to avoid expensive cloning on every access
    pub closed_trades: Arc<Vec<ClosedTrade>>,
    // Recent trades generated in the last step
    #[pyo3(get)]
    pub recent_trades: Vec<Trade>,
    // History Buffer (Shared with Engine)
    pub history_buffer: Option<Arc<RwLock<HistoryBuffer>>>,
    // Event Channel (Optional, for async order submission)
    pub event_tx: Option<Sender<Event>>,
    #[pyo3(get)]
    pub risk_config: RiskConfig,
}

impl StrategyContext {
    pub fn new(
        cash: Decimal,
        positions: HashMap<String, Decimal>,
        available_positions: HashMap<String, Decimal>,
        session: TradingSession,
        current_time: i64,
        active_orders: Vec<Order>,
        closed_trades: Arc<Vec<ClosedTrade>>,
        recent_trades: Vec<Trade>,
        history_buffer: Option<Arc<RwLock<HistoryBuffer>>>,
        event_tx: Option<Sender<Event>>,
        risk_config: RiskConfig,
    ) -> Self {
        StrategyContext {
            orders: Vec::new(),
            canceled_order_ids: Vec::new(),
            active_orders,
            timers: Vec::new(),
            cash,
            positions,
            available_positions,
            session,
            current_time,
            closed_trades,
            recent_trades,
            history_buffer,
            event_tx,
            risk_config,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl StrategyContext {
    /// 从 Python 端创建 StrategyContext (通常由内部调用).
    ///
    /// :param cash: 初始资金
    /// :param positions: 初始持仓 {symbol: quantity}
    /// :param available_positions: 初始可用持仓 {symbol: quantity}
    /// :param session: 当前交易时段
    /// :param current_time: 当前时间戳 (纳秒)
    /// :param active_orders: 当前活跃订单列表
    /// :param closed_trades: 已平仓交易列表
    /// :param recent_trades: 最近成交列表
    /// :param risk_config: 风控配置
    #[new]
    pub fn py_new(
        cash: &Bound<'_, PyAny>,
        positions: HashMap<String, f64>,
        available_positions: HashMap<String, f64>,
        session: Option<TradingSession>,
        current_time: Option<i64>,
        active_orders: Option<Vec<Order>>,
        closed_trades: Option<Vec<ClosedTrade>>,
        recent_trades: Option<Vec<Trade>>,
        risk_config: Option<RiskConfig>,
    ) -> PyResult<Self> {
        let pos_dec: HashMap<String, Decimal> = positions
            .into_iter()
            .map(|(k, v)| (k, Decimal::from_f64(v).unwrap_or(Decimal::ZERO)))
            .collect();
        let avail_dec: HashMap<String, Decimal> = available_positions
            .into_iter()
            .map(|(k, v)| (k, Decimal::from_f64(v).unwrap_or(Decimal::ZERO)))
            .collect();

        Ok(StrategyContext {
            orders: Vec::new(),
            canceled_order_ids: Vec::new(),
            active_orders: active_orders.unwrap_or_default(),
            timers: Vec::new(),
            cash: extract_decimal(cash)?,
            positions: pos_dec,
            available_positions: avail_dec,
            session: session.unwrap_or(TradingSession::Continuous),
            current_time: current_time.unwrap_or(0),
            closed_trades: Arc::new(closed_trades.unwrap_or_default()),
            recent_trades: recent_trades.unwrap_or_default(),
            history_buffer: None,
            event_tx: None,
            risk_config: risk_config.unwrap_or_else(RiskConfig::new),
        })
    }

    /// 获取历史数据.
    ///
    /// :param symbol: 标的代码
    /// :param field: 字段名 (open, high, low, close, volume)
    /// :param count: 获取的数据长度
    /// :return: numpy array or None
    fn history<'py>(
        &self,
        py: Python<'py>,
        symbol: String,
        field: String,
        count: usize,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        if let Some(ref buffer_lock) = self.history_buffer {
            let buffer = buffer_lock.read().unwrap();
            if let Some(history) = buffer.get_history(&symbol) {
                let len = history.timestamps.len();
                if len == 0 {
                    return Ok(None);
                }

                let start = if len > count { len - count } else { 0 };
                let py_array = match field.as_str() {
                    "open" => PyArray1::from_iter(py, history.opens.iter().skip(start).cloned()),
                    "high" => PyArray1::from_iter(py, history.highs.iter().skip(start).cloned()),
                    "low" => PyArray1::from_iter(py, history.lows.iter().skip(start).cloned()),
                    "close" => PyArray1::from_iter(py, history.closes.iter().skip(start).cloned()),
                    "volume" => PyArray1::from_iter(py, history.volumes.iter().skip(start).cloned()),
                    _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid field")),
                };

                return Ok(Some(py_array));
            }
        }
        Ok(None)
    }

    #[getter]
    fn get_last_closed_trade(&self) -> Option<ClosedTrade> {
        self.closed_trades.last().cloned()
    }

    #[getter]
    fn get_closed_trades(&self) -> Vec<ClosedTrade> {
        self.closed_trades.to_vec()
    }

    #[getter]
    fn get_cash(&self) -> f64 {
        self.cash.to_f64().unwrap_or_default()
    }

    #[getter]
    fn get_positions(&self) -> HashMap<String, f64> {
        self.positions
            .iter()
            .map(|(k, v)| (k.clone(), v.to_f64().unwrap_or_default()))
            .collect()
    }

    #[getter]
    fn get_available_positions(&self) -> HashMap<String, f64> {
        self.available_positions
            .iter()
            .map(|(k, v)| (k.clone(), v.to_f64().unwrap_or_default()))
            .collect()
    }

    /// 注册定时器.
    ///
    /// :param timestamp: 触发时间戳 (纳秒)
    /// :param payload: 携带的数据 (如回调函数名)
    fn schedule(&mut self, timestamp: i64, payload: String) {
        let normalized = if timestamp.abs() < 1_000_000_000_000 {
            timestamp * 1_000_000_000
        } else {
            timestamp
        };
        self.timers.push(Timer {
            timestamp: normalized,
            payload,
        });
    }

    /// 取消订单.
    ///
    /// :param order_id: 订单 ID
    fn cancel_order(&mut self, order_id: String) {
        self.canceled_order_ids.push(order_id);
    }

    /// 买入下单.
    ///
    /// :param symbol: 标的代码
    /// :param quantity: 买入数量 (正数)
    /// :param price: 限价 (可选, 默认为 Market 单)
    /// :param time_in_force: 订单有效期 (可选, 默认 GTC)
    /// :param trigger_price: 触发价格 (可选, 用于止损/止盈单)
    /// :param tag: 订单标签 (可选)
    /// :return: 订单 ID
    #[pyo3(signature = (symbol, quantity, price=None, time_in_force=None, trigger_price=None, tag=None))]
    fn buy(
        &mut self,
        symbol: String,
        quantity: &Bound<'_, PyAny>,
        price: Option<&Bound<'_, PyAny>>,
        time_in_force: Option<TimeInForce>,
        trigger_price: Option<&Bound<'_, PyAny>>,
        tag: Option<String>,
    ) -> PyResult<String> {
        let qty_decimal = extract_decimal(quantity)?;
        let price_decimal = if let Some(p) = price {
            Some(extract_decimal(p)?)
        } else {
            None
        };
        let trigger_decimal = if let Some(t) = trigger_price {
            Some(extract_decimal(t)?)
        } else {
            None
        };

        let id = Uuid::new_v4().to_string();
        let order = Order {
            id: id.clone(),
            symbol,
            side: OrderSide::Buy,
            order_type: match (price.is_some(), trigger_price.is_some()) {
                (true, true) => OrderType::StopLimit,
                (false, true) => OrderType::StopMarket,
                (true, false) => OrderType::Limit,
                (false, false) => OrderType::Market,
            },
            quantity: qty_decimal,
            price: price_decimal,
            time_in_force: time_in_force.unwrap_or(TimeInForce::GTC),
            trigger_price: trigger_decimal,
            status: crate::model::OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: self.current_time,
            updated_at: self.current_time,
            commission: Decimal::ZERO,
            tag: tag.unwrap_or_default(),
            reject_reason: String::new(),
        };
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(Event::OrderRequest(order));
        } else {
            self.orders.push(order);
        }
        Ok(id)
    }

    /// 卖出下单.
    ///
    /// :param symbol: 标的代码
    /// :param quantity: 卖出数量 (正数)
    /// :param price: 限价 (可选, 默认为 Market 单)
    /// :param time_in_force: 订单有效期 (可选, 默认 GTC)
    /// :param trigger_price: 触发价格 (可选, 用于止损/止盈单)
    /// :param tag: 订单标签 (可选)
    /// :return: 订单 ID
    #[pyo3(signature = (symbol, quantity, price=None, time_in_force=None, trigger_price=None, tag=None))]
    fn sell(
        &mut self,
        symbol: String,
        quantity: &Bound<'_, PyAny>,
        price: Option<&Bound<'_, PyAny>>,
        time_in_force: Option<TimeInForce>,
        trigger_price: Option<&Bound<'_, PyAny>>,
        tag: Option<String>,
    ) -> PyResult<String> {
        let qty_decimal = extract_decimal(quantity)?;
        let price_decimal = if let Some(p) = price {
            Some(extract_decimal(p)?)
        } else {
            None
        };
        let trigger_decimal = if let Some(t) = trigger_price {
            Some(extract_decimal(t)?)
        } else {
            None
        };

        let id = Uuid::new_v4().to_string();
        let order = Order {
            id: id.clone(),
            symbol,
            side: OrderSide::Sell,
            order_type: match (price.is_some(), trigger_price.is_some()) {
                (true, true) => OrderType::StopLimit,
                (false, true) => OrderType::StopMarket,
                (true, false) => OrderType::Limit,
                (false, false) => OrderType::Market,
            },
            quantity: qty_decimal,
            price: price_decimal,
            time_in_force: time_in_force.unwrap_or(TimeInForce::GTC),
            trigger_price: trigger_decimal,
            status: crate::model::OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: self.current_time,
            updated_at: self.current_time,
            commission: Decimal::ZERO,
            tag: tag.unwrap_or_default(),
            reject_reason: String::new(),
        };
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(Event::OrderRequest(order));
        } else {
            self.orders.push(order);
        }
        Ok(id)
    }

    /// 获取当前持仓数量.
    ///
    /// :param symbol: 标的代码
    /// :return: 持仓数量 (Long为正, Short为负)
    fn get_position(&self, symbol: String) -> f64 {
        self.positions
            .get(&symbol)
            .unwrap_or(&Decimal::ZERO)
            .to_f64()
            .unwrap_or_default()
    }

    /// 获取当前可用持仓数量.
    ///
    /// :param symbol: 标的代码
    /// :return: 可用持仓数量
    fn get_available_position(&self, symbol: String) -> f64 {
        self.available_positions
            .get(&symbol)
            .unwrap_or(&Decimal::ZERO)
            .to_f64()
            .unwrap_or_default()
    }
}
