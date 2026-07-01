use crate::analysis::{ClosedTrade, TradeTracker};
use crate::event::Event;
use crate::history::HistoryBuffer;
use crate::market::MarketModel;
use crate::model::market_data::extract_decimal;
use crate::model::{
    AssetType, ExecutionPolicyCore, Instrument, Order, OrderSide, OrderType, PositionEffect,
    PriceBasis, TemporalPolicy, TimeInForce, Timer, Trade, TradingSession,
};
use crate::portfolio::Portfolio;
use crate::risk::RiskConfig;
use crossbeam_channel::Sender;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// 引擎上下文 (Engine Context)
/// 用于在 Rust 内部组件之间传递共享状态
pub struct EngineContext<'a> {
    pub instruments: &'a HashMap<String, Instrument>,
    pub portfolio: &'a Portfolio,
    pub last_prices: &'a HashMap<String, Decimal>,
    pub trade_tracker: &'a TradeTracker,
    pub market_model: &'a dyn MarketModel,
    pub execution_policy_core: ExecutionPolicyCore,
    pub bar_index: usize,
    pub current_time: i64,
    pub session: TradingSession,
    pub active_orders: &'a [Order],
    pub risk_config: &'a RiskConfig,
    pub timezone_name: Option<&'a str>,
    pub timezone_offset: i32,
}

pub struct ContextInit {
    pub cash: Decimal,
    pub previous_cash: Decimal,
    pub positions: Arc<HashMap<String, Decimal>>,
    pub available_positions: Arc<HashMap<String, Decimal>>,
    pub position_entry_prices: Arc<HashMap<String, Decimal>>,
    pub session: TradingSession,
    pub current_time: i64,
    pub active_orders: Arc<Vec<Order>>,
    pub closed_trades: Arc<Vec<ClosedTrade>>,
    pub recent_trades: Vec<Trade>,
    pub recent_rejected_orders: Vec<Order>,
    pub recent_expiry_events: Vec<ExpiryEvent>,
    pub history_buffer: Option<Arc<RwLock<HistoryBuffer>>>,
    pub event_tx: Option<Sender<Event>>,
    pub risk_config: RiskConfig,
    pub strategy_id: Option<String>,
    pub account_equity: f64,
    pub account_market_value: f64,
    pub account_notional_value: f64,
    pub account_used_margin: f64,
    pub account_unrealized_pnl: f64,
    pub account_maintenance_ratio: f64,
    pub account_short_market_value: f64,
    pub account_frozen_cash: f64,
    pub previous_account_equity: f64,
    pub previous_account_market_value: f64,
    pub previous_account_notional_value: f64,
    pub previous_account_used_margin: f64,
    pub previous_account_unrealized_pnl: f64,
    pub previous_account_maintenance_ratio: f64,
    pub margin_accrued_interest: f64,
    pub margin_daily_interest: f64,
}

pub struct ContextUpdate {
    pub cash: Decimal,
    pub previous_cash: Decimal,
    pub positions: Arc<HashMap<String, Decimal>>,
    pub available_positions: Arc<HashMap<String, Decimal>>,
    pub position_entry_prices: Arc<HashMap<String, Decimal>>,
    pub session: TradingSession,
    pub current_time: i64,
    pub active_orders: Arc<Vec<Order>>,
    pub closed_trades: Arc<Vec<ClosedTrade>>,
    pub recent_trades: Vec<Trade>,
    pub recent_rejected_orders: Vec<Order>,
    pub recent_expiry_events: Vec<ExpiryEvent>,
    pub account_equity: f64,
    pub account_market_value: f64,
    pub account_notional_value: f64,
    pub account_used_margin: f64,
    pub account_unrealized_pnl: f64,
    pub account_maintenance_ratio: f64,
    pub account_short_market_value: f64,
    pub account_frozen_cash: f64,
    pub previous_account_equity: f64,
    pub previous_account_market_value: f64,
    pub previous_account_notional_value: f64,
    pub previous_account_used_margin: f64,
    pub previous_account_unrealized_pnl: f64,
    pub previous_account_maintenance_ratio: f64,
    pub margin_accrued_interest: f64,
    pub margin_daily_interest: f64,
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct ExpiryEvent {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub asset_type: AssetType,
    #[pyo3(get)]
    pub trading_date: String,
    #[pyo3(get)]
    pub expiry_date: Option<u32>,
    #[pyo3(get)]
    pub quantity_before: f64,
    #[pyo3(get)]
    pub quantity_closed: f64,
    #[pyo3(get)]
    pub cash_flow: f64,
    #[pyo3(get)]
    pub settlement_type: Option<String>,
    #[pyo3(get)]
    pub settlement_price: Option<f64>,
    #[pyo3(get)]
    pub reason: String,
    #[pyo3(get)]
    pub description: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl ExpiryEvent {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        symbol: String,
        asset_type: AssetType,
        trading_date: String,
        expiry_date: Option<u32>,
        quantity_before: f64,
        quantity_closed: f64,
        cash_flow: f64,
        settlement_type: Option<String>,
        settlement_price: Option<f64>,
        reason: String,
        description: String,
    ) -> Self {
        Self {
            symbol,
            asset_type,
            trading_date,
            expiry_date,
            quantity_before,
            quantity_closed,
            cash_flow,
            settlement_type,
            settlement_price,
            reason,
            description,
        }
    }
}

fn parse_order_fill_policy_override(
    fill_price_basis: Option<String>,
    fill_bar_offset: Option<u8>,
    fill_temporal: Option<String>,
) -> PyResult<Option<ExecutionPolicyCore>> {
    if fill_price_basis.is_none() && fill_bar_offset.is_none() && fill_temporal.is_none() {
        return Ok(None);
    }
    let raw_basis = fill_price_basis.unwrap_or_else(|| "open".to_string());
    let raw_basis = raw_basis.trim().to_ascii_lowercase();
    let basis = match raw_basis.as_str() {
        "open" => PriceBasis::Open,
        "close" => PriceBasis::Close,
        "ohlc4" => PriceBasis::Ohlc4,
        "hl2" => PriceBasis::Hl2,
        _ => {
            return Err(PyValueError::new_err(
                "fill_policy.price_basis must be one of: open, close, ohlc4, hl2",
            ));
        }
    };
    let temporal = match fill_temporal
        .unwrap_or_else(|| "same_cycle".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "same_cycle" => TemporalPolicy::SameCycle,
        "next_event" => TemporalPolicy::NextEvent,
        _ => {
            return Err(PyValueError::new_err(
                "fill_policy.temporal must be one of: same_cycle, next_event",
            ));
        }
    };
    let bar_offset = fill_bar_offset.unwrap_or(match basis {
        PriceBasis::Close => 0,
        _ => 1,
    });
    if bar_offset > 1 {
        return Err(PyValueError::new_err(
            "fill_policy.bar_offset must be 0 or 1",
        ));
    }
    match basis {
        PriceBasis::Open if bar_offset != 1 => {
            return Err(PyValueError::new_err(
                "fill_policy(open) requires bar_offset=1",
            ));
        }
        PriceBasis::Ohlc4 if bar_offset != 1 => {
            return Err(PyValueError::new_err(
                "fill_policy(ohlc4) requires bar_offset=1",
            ));
        }
        PriceBasis::Hl2 if bar_offset != 1 => {
            return Err(PyValueError::new_err(
                "fill_policy(hl2) requires bar_offset=1",
            ));
        }
        _ => {}
    }
    Ok(Some(ExecutionPolicyCore {
        price_basis: basis,
        bar_offset,
        temporal,
    }))
}

fn parse_order_slippage_override(
    slippage_type: Option<String>,
    slippage_value: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Option<String>, Option<Decimal>)> {
    if slippage_type.is_none() && slippage_value.is_none() {
        return Ok((None, None));
    }
    let raw_type = slippage_type
        .unwrap_or_else(|| "percent".to_string())
        .trim()
        .to_ascii_lowercase();
    if raw_type != "percent" && raw_type != "fixed" {
        return Err(PyValueError::new_err(
            "slippage.type must be one of: percent, fixed",
        ));
    }
    let value = match slippage_value {
        Some(v) => extract_decimal(v)?,
        None => Decimal::ZERO,
    };
    if value < Decimal::ZERO {
        return Err(PyValueError::new_err("slippage.value must be >= 0"));
    }
    Ok((Some(raw_type), Some(value)))
}

fn parse_order_commission_override(
    commission_type: Option<String>,
    commission_value: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Option<String>, Option<Decimal>)> {
    if commission_type.is_none() && commission_value.is_none() {
        return Ok((None, None));
    }
    let raw_type = commission_type
        .unwrap_or_else(|| "percent".to_string())
        .trim()
        .to_ascii_lowercase();
    if raw_type != "percent" && raw_type != "fixed" && raw_type != "per_unit" {
        return Err(PyValueError::new_err(
            "commission.type must be one of: percent, fixed, per_unit",
        ));
    }
    let value = match commission_value {
        Some(v) => extract_decimal(v)?,
        None => Decimal::ZERO,
    };
    if value < Decimal::ZERO {
        return Err(PyValueError::new_err("commission.value must be >= 0"));
    }
    Ok((Some(raw_type), Some(value)))
}

impl StrategyContext {
    pub fn update_state(&mut self, update: ContextUpdate) {
        self.cash = update.cash;
        self.previous_cash = update.previous_cash;
        self.positions = update.positions;
        self.available_positions = update.available_positions;
        self.position_entry_prices = update.position_entry_prices;
        self.session = update.session;
        self.current_time = update.current_time;
        self.active_orders_arc = update.active_orders.clone();
        self.closed_trades = update.closed_trades;

        // Lazy update: clear the vector but don't fill it yet.
        // We will rely on a getter to populate it if accessed, or just update it here if needed.
        // For true zero-copy, we need to implement a custom getter for active_orders.
        // But PyO3 #[pyo3(get)] generates a simple field access.
        // To fix this properly, we should rename the field active_orders -> _active_orders_cache
        // and expose a getter method active_orders() that populates it on demand.
        //
        // HOWEVER, for this "Zero-Copy" optimization step, let's just avoid the clone if the list is empty.

        if update.active_orders.is_empty() {
            self.active_orders.clear();
        } else {
            // Still copying for now to maintain API compatibility without breaking changes
            // Optimization: reuse capacity
            self.active_orders.clear();
            self.active_orders.extend_from_slice(&update.active_orders);
        }

        self.recent_trades = update.recent_trades;
        self.recent_rejected_orders = update.recent_rejected_orders;
        self.recent_expiry_events = update.recent_expiry_events;
        self.account_equity = update.account_equity;
        self.account_market_value = update.account_market_value;
        self.account_notional_value = update.account_notional_value;
        self.account_used_margin = update.account_used_margin;
        self.account_unrealized_pnl = update.account_unrealized_pnl;
        self.account_maintenance_ratio = update.account_maintenance_ratio;
        self.account_short_market_value = update.account_short_market_value;
        self.account_frozen_cash = update.account_frozen_cash;
        self.previous_account_equity = update.previous_account_equity;
        self.previous_account_market_value = update.previous_account_market_value;
        self.previous_account_notional_value = update.previous_account_notional_value;
        self.previous_account_used_margin = update.previous_account_used_margin;
        self.previous_account_unrealized_pnl = update.previous_account_unrealized_pnl;
        self.previous_account_maintenance_ratio = update.previous_account_maintenance_ratio;
        self.margin_accrued_interest = update.margin_accrued_interest;
        self.margin_daily_interest = update.margin_daily_interest;

        // Reset accumulators
        self.orders.clear();
        self.canceled_order_ids.clear();
        self.timers.clear();

        // Reset Arc accumulators (internal mutability)
        if let Ok(mut orders) = self.orders_arc.write() {
            orders.clear();
        }
        if let Ok(mut canceled) = self.canceled_order_ids_arc.write() {
            canceled.clear();
        }
        if let Ok(mut timers) = self.timers_arc.write() {
            timers.clear();
        }
    }
}

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
    #[pyo3(get)]
    pub timers: Vec<Timer>, // Accumulated timers

    // Internal thread-safe storage
    pub orders_arc: Arc<RwLock<Vec<Order>>>,
    pub canceled_order_ids_arc: Arc<RwLock<Vec<String>>>,
    pub active_orders_arc: Arc<Vec<Order>>,
    pub timers_arc: Arc<RwLock<Vec<Timer>>>,

    pub cash: Decimal,
    pub previous_cash: Decimal,
    pub positions: Arc<HashMap<String, Decimal>>,
    pub available_positions: Arc<HashMap<String, Decimal>>,
    pub position_entry_prices: Arc<HashMap<String, Decimal>>,
    #[pyo3(get)]
    pub session: TradingSession,
    #[pyo3(get)]
    pub current_time: i64,
    // Do NOT expose closed_trades as a direct getter to avoid expensive cloning on every access
    pub closed_trades: Arc<Vec<ClosedTrade>>,
    // Recent trades generated in the last step
    #[pyo3(get)]
    pub recent_trades: Vec<Trade>,
    // Recent rejected orders generated in the last step
    #[pyo3(get)]
    pub recent_rejected_orders: Vec<Order>,
    // Recent expiry events generated in the last step
    #[pyo3(get)]
    pub recent_expiry_events: Vec<ExpiryEvent>,
    // History Buffer (Shared with Engine)
    pub history_buffer: Option<Arc<RwLock<HistoryBuffer>>>,
    // Event Channel (Optional, for async order submission)
    pub event_tx: Option<Sender<Event>>,
    #[pyo3(get)]
    pub risk_config: RiskConfig,
    #[pyo3(get)]
    pub strategy_id: Option<String>,
    #[pyo3(get)]
    pub account_equity: f64,
    #[pyo3(get)]
    pub account_market_value: f64,
    #[pyo3(get)]
    pub account_notional_value: f64,
    #[pyo3(get)]
    pub account_used_margin: f64,
    #[pyo3(get)]
    pub account_unrealized_pnl: f64,
    #[pyo3(get)]
    pub account_maintenance_ratio: f64,
    #[pyo3(get)]
    pub account_short_market_value: f64,
    #[pyo3(get)]
    pub account_frozen_cash: f64,
    #[pyo3(get)]
    pub previous_account_equity: f64,
    #[pyo3(get)]
    pub previous_account_market_value: f64,
    #[pyo3(get)]
    pub previous_account_notional_value: f64,
    #[pyo3(get)]
    pub previous_account_used_margin: f64,
    #[pyo3(get)]
    pub previous_account_unrealized_pnl: f64,
    #[pyo3(get)]
    pub previous_account_maintenance_ratio: f64,
    #[pyo3(get)]
    pub margin_accrued_interest: f64,
    #[pyo3(get)]
    pub margin_daily_interest: f64,
}

impl StrategyContext {
    pub fn new(init: ContextInit) -> Self {
        StrategyContext {
            orders: Vec::new(),
            canceled_order_ids: Vec::new(),
            active_orders: init.active_orders.as_ref().clone(),
            timers: Vec::new(),
            orders_arc: Arc::new(RwLock::new(Vec::new())),
            canceled_order_ids_arc: Arc::new(RwLock::new(Vec::new())),
            active_orders_arc: init.active_orders,
            timers_arc: Arc::new(RwLock::new(Vec::new())),
            cash: init.cash,
            previous_cash: init.previous_cash,
            positions: init.positions,
            available_positions: init.available_positions,
            position_entry_prices: init.position_entry_prices,
            session: init.session,
            current_time: init.current_time,
            closed_trades: init.closed_trades,
            recent_trades: init.recent_trades,
            recent_rejected_orders: init.recent_rejected_orders,
            recent_expiry_events: init.recent_expiry_events,
            history_buffer: init.history_buffer,
            event_tx: init.event_tx,
            risk_config: init.risk_config,
            strategy_id: init.strategy_id,
            account_equity: init.account_equity,
            account_market_value: init.account_market_value,
            account_notional_value: init.account_notional_value,
            account_used_margin: init.account_used_margin,
            account_unrealized_pnl: init.account_unrealized_pnl,
            account_maintenance_ratio: init.account_maintenance_ratio,
            account_short_market_value: init.account_short_market_value,
            account_frozen_cash: init.account_frozen_cash,
            previous_account_equity: init.previous_account_equity,
            previous_account_market_value: init.previous_account_market_value,
            previous_account_notional_value: init.previous_account_notional_value,
            previous_account_used_margin: init.previous_account_used_margin,
            previous_account_unrealized_pnl: init.previous_account_unrealized_pnl,
            previous_account_maintenance_ratio: init.previous_account_maintenance_ratio,
            margin_accrued_interest: init.margin_accrued_interest,
            margin_daily_interest: init.margin_daily_interest,
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
    /// :param position_entry_prices: 初始持仓均价 {symbol: entry_price}
    /// :param session: 当前交易时段
    /// :param current_time: 当前时间戳 (纳秒)
    /// :param active_orders: 当前活跃订单列表
    /// :param closed_trades: 已平仓交易列表
    /// :param recent_trades: 最近成交列表
    /// :param recent_expiry_events: 最近到期事件列表
    /// :param risk_config: 风控配置
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        cash: &Bound<'_, PyAny>,
        previous_cash: Option<&Bound<'_, PyAny>>,
        positions: HashMap<String, f64>,
        available_positions: HashMap<String, f64>,
        position_entry_prices: Option<HashMap<String, f64>>,
        session: Option<TradingSession>,
        current_time: Option<i64>,
        active_orders: Option<Vec<Order>>,
        closed_trades: Option<Vec<ClosedTrade>>,
        recent_trades: Option<Vec<Trade>>,
        recent_expiry_events: Option<Vec<ExpiryEvent>>,
        risk_config: Option<RiskConfig>,
        strategy_id: Option<String>,
        account_equity: Option<f64>,
        account_market_value: Option<f64>,
        account_notional_value: Option<f64>,
        account_used_margin: Option<f64>,
        account_unrealized_pnl: Option<f64>,
        account_maintenance_ratio: Option<f64>,
        previous_account_equity: Option<f64>,
        previous_account_market_value: Option<f64>,
        previous_account_notional_value: Option<f64>,
        previous_account_used_margin: Option<f64>,
        previous_account_unrealized_pnl: Option<f64>,
        previous_account_maintenance_ratio: Option<f64>,
        margin_accrued_interest: Option<f64>,
        margin_daily_interest: Option<f64>,
    ) -> PyResult<Self> {
        let pos_dec: HashMap<String, Decimal> = positions
            .into_iter()
            .map(|(k, v)| (k, Decimal::from_f64(v).unwrap_or(Decimal::ZERO)))
            .collect();
        let avail_dec: HashMap<String, Decimal> = available_positions
            .into_iter()
            .map(|(k, v)| (k, Decimal::from_f64(v).unwrap_or(Decimal::ZERO)))
            .collect();
        let entry_price_dec: HashMap<String, Decimal> = position_entry_prices
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (k, Decimal::from_f64(v).unwrap_or(Decimal::ZERO)))
            .collect();

        Ok(StrategyContext {
            orders: Vec::new(),
            canceled_order_ids: Vec::new(),
            active_orders: active_orders.clone().unwrap_or_default(),
            timers: Vec::new(),
            orders_arc: Arc::new(RwLock::new(Vec::new())),
            canceled_order_ids_arc: Arc::new(RwLock::new(Vec::new())),
            active_orders_arc: Arc::new(active_orders.unwrap_or_default()),
            timers_arc: Arc::new(RwLock::new(Vec::new())),
            cash: extract_decimal(cash)?,
            previous_cash: extract_decimal(previous_cash.unwrap_or(cash))?,
            positions: Arc::new(pos_dec),
            available_positions: Arc::new(avail_dec),
            position_entry_prices: Arc::new(entry_price_dec),
            session: session.unwrap_or(TradingSession::Continuous),
            current_time: current_time.unwrap_or(0),
            closed_trades: Arc::new(closed_trades.unwrap_or_default()),
            recent_trades: recent_trades.unwrap_or_default(),
            recent_rejected_orders: Vec::new(),
            recent_expiry_events: recent_expiry_events.unwrap_or_default(),
            history_buffer: None,
            event_tx: None,
            risk_config: risk_config.unwrap_or_default(),
            strategy_id,
            account_equity: account_equity.unwrap_or(0.0),
            account_market_value: account_market_value.unwrap_or(0.0),
            account_notional_value: account_notional_value.unwrap_or(0.0),
            account_used_margin: account_used_margin.unwrap_or(0.0),
            account_unrealized_pnl: account_unrealized_pnl.unwrap_or(0.0),
            account_maintenance_ratio: account_maintenance_ratio.unwrap_or(0.0),
            account_short_market_value: 0.0,
            account_frozen_cash: 0.0,
            previous_account_equity: previous_account_equity.unwrap_or(0.0),
            previous_account_market_value: previous_account_market_value.unwrap_or(0.0),
            previous_account_notional_value: previous_account_notional_value.unwrap_or(0.0),
            previous_account_used_margin: previous_account_used_margin.unwrap_or(0.0),
            previous_account_unrealized_pnl: previous_account_unrealized_pnl.unwrap_or(0.0),
            previous_account_maintenance_ratio: previous_account_maintenance_ratio.unwrap_or(0.0),
            margin_accrued_interest: margin_accrued_interest.unwrap_or(0.0),
            margin_daily_interest: margin_daily_interest.unwrap_or(0.0),
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
        end_before_ns: Option<i64>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        if let Some(ref buffer_lock) = self.history_buffer {
            let buffer = buffer_lock.read().unwrap();
            let history = match (buffer.get_history(&symbol), end_before_ns) {
                (Some(history), Some(cutoff))
                    if history
                        .timestamps
                        .back()
                        .is_some_and(|timestamp| *timestamp >= cutoff) =>
                {
                    buffer.get_previous_history(&symbol).unwrap_or(history)
                }
                (Some(history), _) => history,
                (None, _) => return Ok(None),
            };
            {
                let len = match end_before_ns {
                    Some(cutoff) => history
                        .timestamps
                        .iter()
                        .take_while(|timestamp| **timestamp < cutoff)
                        .count(),
                    None => history.timestamps.len(),
                };
                if len == 0 {
                    return Ok(None);
                }

                let start = len.saturating_sub(count);
                let py_array = match field.as_str() {
                    "open" => PyArray1::from_iter(py, history.opens.iter().skip(start).cloned()),
                    "high" => PyArray1::from_iter(py, history.highs.iter().skip(start).cloned()),
                    "low" => PyArray1::from_iter(py, history.lows.iter().skip(start).cloned()),
                    "close" => PyArray1::from_iter(py, history.closes.iter().skip(start).cloned()),
                    "volume" => {
                        PyArray1::from_iter(py, history.volumes.iter().skip(start).cloned())
                    }
                    _ => {
                        if let Some(series) = history.extras.get(&field) {
                            PyArray1::from_iter(py, series.iter().skip(start).cloned())
                        } else {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Invalid field: '{}'. Available extra fields: {:?}",
                                field,
                                history.extras.keys()
                            )));
                        }
                    }
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
    fn get_previous_cash(&self) -> f64 {
        self.previous_cash.to_f64().unwrap_or_default()
    }

    #[getter]
    fn get_positions(&self) -> HashMap<String, f64> {
        self.positions
            .iter()
            .filter(|(_, v)| !v.is_zero())
            .map(|(k, v)| (k.clone(), v.to_f64().unwrap_or_default()))
            .collect()
    }

    #[getter]
    fn get_available_positions(&self) -> HashMap<String, f64> {
        self.available_positions
            .iter()
            .filter(|(_, v)| !v.is_zero())
            .map(|(k, v)| (k.clone(), v.to_f64().unwrap_or_default()))
            .collect()
    }

    #[getter]
    fn get_position_entry_prices(&self) -> HashMap<String, f64> {
        self.position_entry_prices
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
        if let Ok(mut timers) = self.timers_arc.write() {
            timers.push(Timer {
                timestamp: normalized,
                payload,
            });
        }
    }

    /// 取消订单.
    ///
    /// :param order_id: 订单 ID
    fn cancel_order(&mut self, order_id: String) {
        if !self
            .canceled_order_ids
            .iter()
            .any(|existing| existing == &order_id)
        {
            self.canceled_order_ids.push(order_id.clone());
        }
        if let Ok(mut canceled) = self.canceled_order_ids_arc.write() {
            if !canceled.iter().any(|existing| existing == &order_id) {
                canceled.push(order_id);
            }
        }
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
    #[pyo3(signature = (symbol, quantity, price=None, time_in_force=None, trigger_price=None, tag=None, order_type=None, trail_offset=None, trail_reference_price=None, fill_price_basis=None, fill_bar_offset=None, fill_temporal=None, fill_slippage_type=None, fill_slippage_value=None, fill_commission_type=None, fill_commission_value=None, allow_quantity_auto_resize=false, position_effect=None, reduce_only=false))]
    #[allow(clippy::too_many_arguments)]
    fn buy(
        &mut self,
        symbol: String,
        quantity: &Bound<'_, PyAny>,
        price: Option<&Bound<'_, PyAny>>,
        time_in_force: Option<TimeInForce>,
        trigger_price: Option<&Bound<'_, PyAny>>,
        tag: Option<String>,
        order_type: Option<OrderType>,
        trail_offset: Option<&Bound<'_, PyAny>>,
        trail_reference_price: Option<&Bound<'_, PyAny>>,
        fill_price_basis: Option<String>,
        fill_bar_offset: Option<u8>,
        fill_temporal: Option<String>,
        fill_slippage_type: Option<String>,
        fill_slippage_value: Option<&Bound<'_, PyAny>>,
        fill_commission_type: Option<String>,
        fill_commission_value: Option<&Bound<'_, PyAny>>,
        allow_quantity_auto_resize: bool,
        position_effect: Option<PositionEffect>,
        reduce_only: bool,
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
        let trail_offset_decimal = if let Some(v) = trail_offset {
            Some(extract_decimal(v)?)
        } else {
            None
        };
        let trail_reference_decimal = if let Some(v) = trail_reference_price {
            Some(extract_decimal(v)?)
        } else {
            None
        };
        let resolved_order_type =
            order_type.unwrap_or(match (price.is_some(), trigger_price.is_some()) {
                (true, true) => OrderType::StopLimit,
                (false, true) => OrderType::StopMarket,
                (true, false) => OrderType::Limit,
                (false, false) => OrderType::Market,
            });
        let fill_policy_override =
            parse_order_fill_policy_override(fill_price_basis, fill_bar_offset, fill_temporal)?;
        let (slippage_type_override, slippage_value_override) =
            parse_order_slippage_override(fill_slippage_type, fill_slippage_value)?;
        let (commission_type_override, commission_value_override) =
            parse_order_commission_override(fill_commission_type, fill_commission_value)?;

        let id = Uuid::new_v4().to_string();
        let order = Order {
            id: id.clone(),
            symbol,
            side: OrderSide::Buy,
            order_type: resolved_order_type,
            quantity: qty_decimal,
            price: price_decimal,
            time_in_force: time_in_force.unwrap_or(TimeInForce::GTC),
            trigger_price: trigger_decimal,
            trail_offset: trail_offset_decimal,
            trail_reference_price: trail_reference_decimal,
            fill_policy_override,
            slippage_type_override,
            slippage_value_override,
            commission_type_override,
            commission_value_override,
            graph_id: None,
            parent_order_id: None,
            order_role: crate::model::OrderRole::Standalone,
            position_effect: position_effect.unwrap_or_default(),
            status: crate::model::OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: self.current_time,
            updated_at: self.current_time,
            commission: Decimal::ZERO,
            tag: tag.unwrap_or_default(),
            reject_reason: String::new(),
            owner_strategy_id: self.strategy_id.clone(),
            allow_quantity_auto_resize,
            reduce_only,
        };
        self.orders.push(order.clone());
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(Event::OrderRequest(order));
        } else if let Ok(mut orders) = self.orders_arc.write() {
            orders.push(order);
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
    #[pyo3(signature = (symbol, quantity, price=None, time_in_force=None, trigger_price=None, tag=None, order_type=None, trail_offset=None, trail_reference_price=None, fill_price_basis=None, fill_bar_offset=None, fill_temporal=None, fill_slippage_type=None, fill_slippage_value=None, fill_commission_type=None, fill_commission_value=None, position_effect=None, reduce_only=false))]
    #[allow(clippy::too_many_arguments)]
    fn sell(
        &mut self,
        symbol: String,
        quantity: &Bound<'_, PyAny>,
        price: Option<&Bound<'_, PyAny>>,
        time_in_force: Option<TimeInForce>,
        trigger_price: Option<&Bound<'_, PyAny>>,
        tag: Option<String>,
        order_type: Option<OrderType>,
        trail_offset: Option<&Bound<'_, PyAny>>,
        trail_reference_price: Option<&Bound<'_, PyAny>>,
        fill_price_basis: Option<String>,
        fill_bar_offset: Option<u8>,
        fill_temporal: Option<String>,
        fill_slippage_type: Option<String>,
        fill_slippage_value: Option<&Bound<'_, PyAny>>,
        fill_commission_type: Option<String>,
        fill_commission_value: Option<&Bound<'_, PyAny>>,
        position_effect: Option<PositionEffect>,
        reduce_only: bool,
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
        let trail_offset_decimal = if let Some(v) = trail_offset {
            Some(extract_decimal(v)?)
        } else {
            None
        };
        let trail_reference_decimal = if let Some(v) = trail_reference_price {
            Some(extract_decimal(v)?)
        } else {
            None
        };
        let resolved_order_type =
            order_type.unwrap_or(match (price.is_some(), trigger_price.is_some()) {
                (true, true) => OrderType::StopLimit,
                (false, true) => OrderType::StopMarket,
                (true, false) => OrderType::Limit,
                (false, false) => OrderType::Market,
            });
        let fill_policy_override =
            parse_order_fill_policy_override(fill_price_basis, fill_bar_offset, fill_temporal)?;
        let (slippage_type_override, slippage_value_override) =
            parse_order_slippage_override(fill_slippage_type, fill_slippage_value)?;
        let (commission_type_override, commission_value_override) =
            parse_order_commission_override(fill_commission_type, fill_commission_value)?;

        let id = Uuid::new_v4().to_string();
        let order = Order {
            id: id.clone(),
            symbol,
            side: OrderSide::Sell,
            order_type: resolved_order_type,
            quantity: qty_decimal,
            price: price_decimal,
            time_in_force: time_in_force.unwrap_or(TimeInForce::GTC),
            trigger_price: trigger_decimal,
            trail_offset: trail_offset_decimal,
            trail_reference_price: trail_reference_decimal,
            fill_policy_override,
            slippage_type_override,
            slippage_value_override,
            commission_type_override,
            commission_value_override,
            graph_id: None,
            parent_order_id: None,
            order_role: crate::model::OrderRole::Standalone,
            position_effect: position_effect.unwrap_or_default(),
            status: crate::model::OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: self.current_time,
            updated_at: self.current_time,
            commission: Decimal::ZERO,
            tag: tag.unwrap_or_default(),
            reject_reason: String::new(),
            owner_strategy_id: self.strategy_id.clone(),
            allow_quantity_auto_resize: false,
            reduce_only,
        };
        self.orders.push(order.clone());
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(Event::OrderRequest(order));
        } else if let Ok(mut orders) = self.orders_arc.write() {
            orders.push(order);
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

    /// 获取当前持仓均价.
    ///
    /// :param symbol: 标的代码
    /// :return: 持仓均价
    fn get_position_entry_price(&self, symbol: String) -> f64 {
        self.position_entry_prices
            .get(&symbol)
            .unwrap_or(&Decimal::ZERO)
            .to_f64()
            .unwrap_or_default()
    }
}
