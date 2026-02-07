use crate::model::market_data::extract_decimal;
use crate::model::{Order, OrderSide, OrderType, TimeInForce, Timer, TradingSession, Trade};
use crate::event::Event;
use crate::analysis::ClosedTrade;
use crate::history::HistoryBuffer;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, mpsc::Sender};
use uuid::Uuid;
use numpy::PyArray1;

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
    // Do NOT expose closed_trades as a direct getter to avoid expensive cloning on every access
    pub closed_trades: Arc<Vec<ClosedTrade>>,
    // Recent trades generated in the last step
    #[pyo3(get)]
    pub recent_trades: Vec<Trade>,
    // History Buffer (Shared with Engine)
    pub history_buffer: Option<Arc<RwLock<HistoryBuffer>>>,
    // Event Channel (Optional, for async order submission)
    pub event_tx: Option<Sender<Event>>,
}

impl StrategyContext {
    pub fn new(
        cash: Decimal,
        positions: HashMap<String, Decimal>,
        available_positions: HashMap<String, Decimal>,
        session: TradingSession,
        active_orders: Vec<Order>,
        closed_trades: Arc<Vec<ClosedTrade>>,
        recent_trades: Vec<Trade>,
        history_buffer: Option<Arc<RwLock<HistoryBuffer>>>,
        event_tx: Option<Sender<Event>>,
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
            closed_trades,
            recent_trades,
            history_buffer,
            event_tx,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl StrategyContext {
    #[new]
    pub fn py_new(
        cash: &Bound<'_, PyAny>,
        positions: HashMap<String, f64>,
        available_positions: HashMap<String, f64>,
        session: Option<TradingSession>,
        active_orders: Option<Vec<Order>>,
        closed_trades: Option<Vec<ClosedTrade>>,
        recent_trades: Option<Vec<Trade>>,
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
            closed_trades: Arc::new(closed_trades.unwrap_or_default()),
            recent_trades: recent_trades.unwrap_or_default(),
            history_buffer: None,
            event_tx: None,
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
                let data_slice = match field.as_str() {
                    "open" => &history.opens[start..],
                    "high" => &history.highs[start..],
                    "low" => &history.lows[start..],
                    "close" => &history.closes[start..],
                    "volume" => &history.volumes[start..],
                    _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid field")),
                };

                let py_array = PyArray1::from_slice(py, data_slice);
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

    #[pyo3(signature = (symbol, quantity, price=None, time_in_force=None, trigger_price=None))]
    fn buy(
        &mut self,
        symbol: String,
        quantity: &Bound<'_, PyAny>,
        price: Option<&Bound<'_, PyAny>>,
        time_in_force: Option<TimeInForce>,
        trigger_price: Option<&Bound<'_, PyAny>>,
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
        };
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(Event::OrderRequest(order));
        } else {
            self.orders.push(order);
        }
        Ok(id)
    }

    #[pyo3(signature = (symbol, quantity, price=None, time_in_force=None, trigger_price=None))]
    fn sell(
        &mut self,
        symbol: String,
        quantity: &Bound<'_, PyAny>,
        price: Option<&Bound<'_, PyAny>>,
        time_in_force: Option<TimeInForce>,
        trigger_price: Option<&Bound<'_, PyAny>>,
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
        };
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(Event::OrderRequest(order));
        } else {
            self.orders.push(order);
        }
        Ok(id)
    }

    fn get_position(&self, symbol: String) -> f64 {
        self.positions
            .get(&symbol)
            .unwrap_or(&Decimal::ZERO)
            .to_f64()
            .unwrap_or_default()
    }

    fn get_available_position(&self, symbol: String) -> f64 {
        self.available_positions
            .get(&symbol)
            .unwrap_or(&Decimal::ZERO)
            .to_f64()
            .unwrap_or_default()
    }
}
