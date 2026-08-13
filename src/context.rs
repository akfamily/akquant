use crate::analysis::{ClosedTrade, TradeTracker};
use crate::event::Event;
use crate::history::HistoryBuffer;
use crate::market::MarketModel;
use crate::model::market_data::extract_decimal;
use crate::model::{
    AssetType, ExecutionMode, ExecutionPolicyCore, Instrument, Order, OrderSide, OrderType,
    PositionEffect, TimeInForce, Timer, Trade, TradingSession,
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
    pub instruments: Arc<HashMap<String, Instrument>>,
    pub last_prices: Arc<RwLock<HashMap<String, Decimal>>>,
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
    pub last_prices: Arc<RwLock<HashMap<String, Decimal>>>,
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
    fill_mode: Option<ExecutionMode>,
    fill_timer_timing: Option<String>,
) -> PyResult<Option<ExecutionPolicyCore>> {
    let Some(mode) = fill_mode else {
        return Ok(None);
    };
    let timing = fill_timer_timing.unwrap_or_else(|| "same_cycle".to_string());
    let timing = timing.trim().to_ascii_lowercase();
    if timing != "same_cycle" && timing != "next_event" {
        return Err(PyValueError::new_err(
            "fill_timer_timing must be one of: same_cycle, next_event",
        ));
    }
    Ok(Some(ExecutionPolicyCore::from_legacy(mode, &timing)))
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
        self.last_prices = update.last_prices;
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

    // Snapshots for buying-power computation (set at construction / each bar).
    pub instruments: Arc<HashMap<String, Instrument>>,
    pub last_prices: Arc<RwLock<HashMap<String, Decimal>>>,

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
            instruments: init.instruments,
            last_prices: init.last_prices,
        }
    }

    /// 把在途订单投影进持仓,返回预期持仓。
    ///
    /// 合并 `active_orders`(往期挂单)与 `orders`(本回调已提交),口径对齐
    /// `EngineCore::current_frozen_cash`:只认非终态(`New`/`Submitted`/
    /// `PartiallyFilled`),按 `remaining = quantity - filled_quantity` 折算。
    ///
    /// `reducing_only=true` 只投影**减仓**方向的在途单(平仓/`reduce_only`),
    /// 用于 auto 拆腿判定"还能平多少";`false` 投影全部在途单(按各自
    /// `position_effect`),用于目标仓位算 delta 判定"仓位将落在哪"。
    fn project_pending_position(&self, symbol: &str, reducing_only: bool) -> Decimal {
        let mut projected = self
            .positions
            .get(symbol)
            .copied()
            .unwrap_or(Decimal::ZERO);
        for order in self.active_orders.iter().chain(self.orders.iter()) {
            if order.symbol != symbol {
                continue;
            }
            if !matches!(
                order.status,
                crate::model::OrderStatus::New
                    | crate::model::OrderStatus::Submitted
                    | crate::model::OrderStatus::PartiallyFilled
            ) {
                continue;
            }
            let reducing = crate::model::is_position_reducing_order(order);
            if reducing_only && !reducing {
                continue;
            }
            let remaining = order.quantity - order.filled_quantity;
            if remaining <= Decimal::ZERO {
                continue;
            }
            // 减仓单统一按 Close 语义投影:`project_position_after` 已对平仓量
            // 做 `min(remaining, |pos|)` 封顶,等价于 vn.py 的 `min(frozen, pos)`。
            // 净仓模型下 CloseToday / CloseYesterday 与 Close 同效。
            let effect = if reducing {
                crate::model::PositionEffect::Close
            } else {
                order.position_effect
            };
            projected =
                crate::model::project_position_after(order.side, effect, projected, remaining);
        }
        projected
    }
}

/// 按 `freq` 决定 `history`/`history_multi` 该走 bar 还是 tick 容器.
///
/// `freq` 缺省(`None`)时若同一 symbol 的 bar 与 tick 历史序列同时非空,说明策略
/// 同时挂了 `on_bar` 与 `on_tick`,取哪一条无法从调用点本身推断——这里选择显式
/// 报错而不是悄悄挑一条(比如固定优先 tick 或优先 bar):静默选错会让
/// `get_history` 的返回值悄悄混入另一条流的数据且完全没有报错信号,比报错危险
/// 得多。未识别的 `freq` 取值同样显式报错,为将来的 "1min"/"5min" 等取值留出
/// 空间,不兜底成 bar。
fn resolve_use_tick_history(
    buffer: &HistoryBuffer,
    symbol: &str,
    freq: Option<&str>,
) -> PyResult<bool> {
    match freq {
        Some("tick") => Ok(true),
        Some("bar") => Ok(false),
        None => {
            let has_bar = buffer.has_bar_history(symbol);
            let has_tick = buffer.has_tick_history(symbol);
            if has_bar && has_tick {
                return Err(PyValueError::new_err(format!(
                    "symbol {symbol} 同时存在 bar 与 tick 历史序列, get_history 无法判断该取哪条; \
                     请显式指定 freq='bar' 或 freq='tick'"
                )));
            }
            Ok(has_tick)
        }
        Some(other) => Err(PyValueError::new_err(format!(
            "不支持的 freq={other:?}; 当前支持 'tick' / 'bar' / None"
        ))),
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
            instruments: Arc::new(HashMap::new()),
            last_prices: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// 获取历史数据.
    ///
    /// :param symbol: 标的代码
    /// :param field: 字段名 (open, high, low, close, volume)
    /// :param count: 获取的数据长度
    /// :param freq: 序列来源,`"tick"` / `"bar"` / `None`(缺省时若双流序列并存则报错)
    /// :return: numpy array or None
    fn history<'py>(
        &self,
        py: Python<'py>,
        symbol: String,
        field: String,
        count: usize,
        end_before_ns: Option<i64>,
        freq: Option<&str>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        if let Some(ref buffer_lock) = self.history_buffer {
            let buffer = buffer_lock.read().unwrap();
            let use_tick = resolve_use_tick_history(&buffer, &symbol, freq)?;
            let current = if use_tick {
                buffer.get_tick_history(&symbol)
            } else {
                buffer.get_history(&symbol)
            };
            let history = match (current, end_before_ns) {
                (Some(history), Some(cutoff))
                    if history
                        .timestamps
                        .back()
                        .is_some_and(|timestamp| *timestamp >= cutoff) =>
                {
                    let previous = if use_tick {
                        buffer.get_previous_tick_history(&symbol)
                    } else {
                        buffer.get_previous_history(&symbol)
                    };
                    previous.unwrap_or(history)
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

    /// 批量获取多个字段的历史数据 (一次跨界返回).
    ///
    /// 语义与逐字段调用 `history` 完全一致(同一份 cutoff/上一交易日快照解析),
    /// 但只锁一次缓冲、只跨一次 FFI 边界,供 `get_history_df` / `get_rolling_data`
    /// 复用以减少调用开销。左侧不足的填充交由 Python 层处理(与 `history` 相同)。
    ///
    /// :param symbol: 标的代码
    /// :param fields: 字段名列表 (open/high/low/close/volume 或额外数值字段)
    /// :param count: 获取的数据长度
    /// :param end_before_ns: 可选,历史可见性截断时间戳 (纳秒)
    /// :param freq: 序列来源,`"tick"` / `"bar"` / `None`(缺省时若双流序列并存则报错)
    /// :return: {field: numpy array} 或 None
    fn history_multi<'py>(
        &self,
        py: Python<'py>,
        symbol: String,
        fields: Vec<String>,
        count: usize,
        end_before_ns: Option<i64>,
        freq: Option<&str>,
    ) -> PyResult<Option<HashMap<String, Bound<'py, PyArray1<f64>>>>> {
        if let Some(ref buffer_lock) = self.history_buffer {
            let buffer = buffer_lock.read().unwrap();
            let use_tick = resolve_use_tick_history(&buffer, &symbol, freq)?;
            let current = if use_tick {
                buffer.get_tick_history(&symbol)
            } else {
                buffer.get_history(&symbol)
            };
            let history = match (current, end_before_ns) {
                (Some(history), Some(cutoff))
                    if history
                        .timestamps
                        .back()
                        .is_some_and(|timestamp| *timestamp >= cutoff) =>
                {
                    let previous = if use_tick {
                        buffer.get_previous_tick_history(&symbol)
                    } else {
                        buffer.get_previous_history(&symbol)
                    };
                    previous.unwrap_or(history)
                }
                (Some(history), _) => history,
                (None, _) => return Ok(None),
            };

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
            let mut out: HashMap<String, Bound<'py, PyArray1<f64>>> =
                HashMap::with_capacity(fields.len());
            for field in fields {
                let series = match field.as_str() {
                    "open" => &history.opens,
                    "high" => &history.highs,
                    "low" => &history.lows,
                    "close" => &history.closes,
                    "volume" => &history.volumes,
                    _ => {
                        if let Some(series) = history.extras.get(&field) {
                            series
                        } else {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Invalid field: '{}'. Available extra fields: {:?}",
                                field,
                                history.extras.keys()
                            )));
                        }
                    }
                };
                let py_array = PyArray1::from_iter(py, series.iter().skip(start).cloned());
                out.insert(field, py_array);
            }
            return Ok(Some(out));
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

    /// 可用买入力:现金 + 本回调及既有挂单的净预期回笼,投影后按风控口径
    /// 计算 free_margin × (1 - safety_margin)。用于在同一 on_bar 内、卖出后
    /// 为买单定量(卖出资金当日可复用)。
    #[getter]
    fn get_buying_power(&self) -> f64 {
        let portfolio = crate::portfolio::Portfolio {
            cash: self.cash,
            positions: self.positions.clone(),
            available_positions: self.available_positions.clone(),
        };
        let mut pending: Vec<Order> =
            Vec::with_capacity(self.active_orders.len() + self.orders.len());
        pending.extend(self.active_orders.iter().cloned());
        pending.extend(self.orders.iter().cloned());
        let prices = self.last_prices.read().expect("last_prices 读锁被污染");
        let projected = crate::risk::common::project_active_orders_into(
            &portfolio,
            &pending,
            &prices,
            &self.instruments,
        );
        let stock_ratio_override = if self.risk_config.is_margin_account() {
            Some(self.risk_config.stock_initial_margin_ratio())
        } else {
            None
        };
        let free_margin = projected.calculate_free_margin_with_stock_ratio(
            &prices,
            &self.instruments,
            stock_ratio_override,
        );
        let safety = Decimal::from_f64(self.risk_config.safety_margin).unwrap_or(Decimal::ZERO);
        let factor = (Decimal::ONE - safety).max(Decimal::ZERO);
        free_margin
            .checked_mul(factor)
            .unwrap_or(Decimal::ZERO)
            .max(Decimal::ZERO)
            .to_f64()
            .unwrap_or_default()
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
    #[pyo3(signature = (symbol, quantity, price=None, time_in_force=None, trigger_price=None, tag=None, order_type=None, trail_offset=None, trail_reference_price=None, fill_mode=None, fill_timer_timing=None, fill_slippage_type=None, fill_slippage_value=None, fill_commission_type=None, fill_commission_value=None, allow_quantity_auto_resize=false, position_effect=None, reduce_only=false))]
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
        fill_mode: Option<ExecutionMode>,
        fill_timer_timing: Option<String>,
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
            parse_order_fill_policy_override(fill_mode, fill_timer_timing)?;
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
    #[pyo3(signature = (symbol, quantity, price=None, time_in_force=None, trigger_price=None, tag=None, order_type=None, trail_offset=None, trail_reference_price=None, fill_mode=None, fill_timer_timing=None, fill_slippage_type=None, fill_slippage_value=None, fill_commission_type=None, fill_commission_value=None, position_effect=None, reduce_only=false))]
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
        fill_mode: Option<ExecutionMode>,
        fill_timer_timing: Option<String>,
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
            parse_order_fill_policy_override(fill_mode, fill_timer_timing)?;
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

    /// 获取可平持仓:结算仓扣除在途**平仓/减仓**单占用后的剩余可平量.
    ///
    /// 用于 `buy()` / `sell()` 在 ``position_effect="auto"`` 下拆开平腿。结算仓
    /// 不含同一 on_bar 内已提交、未成交的在途单,直接用它拆腿会把"先平后开"的
    /// 反手第二腿误判成平仓(issue #361)。本方法是 :meth:`get_buying_power`
    /// 的持仓镜像:同样合并 ``active_orders``(往期挂单)+ ``orders``(本回调已提交)。
    ///
    /// **只**投影减仓方向的在途单,不投影在途开仓单——与 vn.py
    /// ``OffsetConverter`` / RQAlpha ``closable`` 一致。在途开仓单未必成交,不投
    /// 影它可使 auto 拆腿偏向判为开仓,即偏向**多预留**保证金(安全侧)。
    ///
    /// :param symbol: 标的代码
    /// :return: 可平方向的剩余持仓 (Long为正, Short为负)
    fn get_closable_position(&self, symbol: String) -> f64 {
        self.project_pending_position(&symbol, true)
            .to_f64()
            .unwrap_or_default()
    }

    /// 获取投影持仓:结算仓叠加**全部**在途单的预期效果.
    ///
    /// 用于 ``order_target*`` / ``close_position`` 等目标仓位语义算 delta:它们问
    /// 的是"仓位最终会落在哪",故开仓与平仓在途单都要计入,否则同一 on_bar 内
    /// 连续调用会按同一个结算仓重复下单(如先 ``close_position`` 再
    /// ``order_target_percent`` 会在全平单之外再补一笔卖单,形成超卖)。
    ///
    /// 与 :meth:`get_closable_position` 的取舍不同:那里刻意不投影在途开仓单以
    /// 偏向安全侧;这里必须投影,否则目标仓位不收敛。
    ///
    /// :param symbol: 标的代码
    /// :return: 投影后的持仓数量 (Long为正, Short为负)
    fn get_projected_position(&self, symbol: String) -> f64 {
        self.project_pending_position(&symbol, false)
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
