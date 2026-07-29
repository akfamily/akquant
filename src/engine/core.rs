use crate::account::{AccountMetrics, calculate_account_metrics};
use chrono::{DateTime, NaiveDate, NaiveTime, TimeZone, Utc};
use chrono_tz::Tz;
use indicatif::ProgressBar;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

// use crate::analysis::{BacktestResult, PositionSnapshot};
use crate::clock::Clock;
use crate::context::ExpiryEvent;
use crate::context::StrategyContext;
use crate::event::Event;
use crate::event_manager::EventManager;
use crate::execution::ExecutionClient;
use crate::history::HistoryBuffer;
use crate::market::corporate_action::CorporateActionManager;
use crate::market::manager::MarketManager;
use crate::model::{ExecutionPolicyCore, Instrument, Order, OrderSide, Timer, Trade};
use crate::pipeline::PipelineRunner;
use crate::pipeline::stages::{
    ChannelProcessor, CleanupProcessor, DataProcessor, ExecutionPhase, ExecutionProcessor,
    StatisticsProcessor, StrategyProcessor,
};
use crate::risk::RiskManager;
use crate::settlement::SettlementManager;
use crate::statistics::StatisticsManager;

use super::state::SharedState;

#[derive(Debug, Clone)]
pub struct StrategySlot {
    pub(crate) strategy_id: String,
}

/// 主回测引擎.
///
/// :ivar feed: 数据源
/// :ivar portfolio: 投资组合
/// :ivar orders: 订单列表
/// :ivar trades: 成交列表
#[gen_stub_pyclass]
#[pyclass]
pub struct Engine {
    pub(crate) state: SharedState,
    pub(crate) last_prices: HashMap<String, Decimal>,
    pub(crate) instruments: HashMap<String, Instrument>,
    pub(crate) current_date: Option<NaiveDate>,
    pub(crate) market_manager: MarketManager,
    pub(crate) corporate_action_manager: CorporateActionManager,
    pub(crate) execution_model: Box<dyn ExecutionClient>,
    pub(crate) execution_policy_core_state: ExecutionPolicyCore,
    pub(crate) clock: Clock,
    pub(crate) timers: BinaryHeap<Timer>, // Min-Heap via Timer's Ord implementation
    pub(crate) force_session_continuous: bool,
    #[pyo3(get, set)]
    pub risk_manager: RiskManager,
    pub(crate) timezone_offset: i32,
    pub(crate) timezone_name: Option<String>,
    /// 年化天数因子。传统市场 252，数字货币 24/7 用 365。
    pub(crate) days_per_year: f64,
    /// 年化无风险利率，默认 0.0。
    pub(crate) risk_free_rate: f64,
    pub(crate) history_buffer: Arc<RwLock<HistoryBuffer>>,
    pub(crate) initial_cash: Decimal,
    #[pyo3(get, set)]
    pub active_start_time_ns: Option<i64>,
    // Components
    pub(crate) event_manager: EventManager,
    pub(crate) statistics_manager: StatisticsManager,
    pub(crate) settlement_manager: SettlementManager,
    // Pipeline state
    pub(crate) current_event: Option<Event>,
    pub(crate) bar_count: usize,
    pub(crate) progress_total_steps: usize,
    pub(crate) progress_bar: Option<ProgressBar>,
    pub(crate) strategy_contexts: Vec<Option<Py<StrategyContext>>>,
    pub(crate) strategy_slot_strategies: Vec<Option<Py<PyAny>>>,
    pub(crate) strategy_slots: Vec<StrategySlot>,
    pub(crate) active_strategy_slot: usize,
    pub(crate) default_strategy_id: Option<String>,
    pub(crate) strategy_priorities: HashMap<String, i32>,
    pub(crate) strategy_risk_budget_limits: HashMap<String, Decimal>,
    pub(crate) portfolio_risk_budget_limit: Option<Decimal>,
    pub(crate) strategy_risk_budget_used: HashMap<String, Decimal>,
    pub(crate) portfolio_risk_budget_used: Decimal,
    pub(crate) risk_budget_mode: String,
    pub(crate) risk_budget_reset_daily: bool,
    pub(crate) risk_budget_usage_day: Option<NaiveDate>,
    pub(crate) strategy_max_order_value_limits: HashMap<String, Decimal>,
    pub(crate) strategy_max_order_size_limits: HashMap<String, Decimal>,
    pub(crate) strategy_max_position_size_limits: HashMap<String, Decimal>,
    pub(crate) strategy_max_daily_loss_limits: HashMap<String, Decimal>,
    pub(crate) strategy_max_drawdown_limits: HashMap<String, Decimal>,
    pub(crate) strategy_risk_cooldown_bars: HashMap<String, usize>,
    pub(crate) strategy_risk_cooldown_until_bar: HashMap<String, usize>,
    pub(crate) strategy_reduce_only_after_risk: HashSet<String>,
    pub(crate) strategy_positions: HashMap<String, HashMap<String, Decimal>>,
    pub(crate) strategy_cashflows: HashMap<String, Decimal>,
    pub(crate) strategy_daily_loss_day: HashMap<String, NaiveDate>,
    pub(crate) strategy_daily_loss_baseline_pnl: HashMap<String, Decimal>,
    pub(crate) strategy_last_pnl: HashMap<String, Decimal>,
    pub(crate) strategy_peak_pnl: HashMap<String, Decimal>,
    pub(crate) strategy_reduce_only_active: HashSet<String>,
    pub(crate) margin_accrued_interest: Decimal,
    pub(crate) margin_daily_interest: Decimal,
    pub(crate) recent_expiry_events: Vec<ExpiryEvent>,
    pub(crate) snapshot_time: i64,
    pub(crate) stream_callback: Option<Py<PyAny>>,
    pub(crate) stream_run_id: Option<String>,
    pub(crate) stream_seq: u64,
    pub(crate) stream_progress_interval: usize,
    pub(crate) stream_equity_interval: usize,
    pub(crate) stream_batch_size: usize,
    pub(crate) stream_max_buffer: usize,
    pub(crate) stream_buffer: Vec<PendingStreamEvent>,
    pub(crate) stream_fail_fast_on_callback_error: bool,
    pub(crate) stream_callback_error_count: u64,
    pub(crate) stream_callback_last_error: Option<String>,
    pub(crate) stream_fatal_error: Option<String>,
    pub(crate) stream_dropped_event_count: u64,
    pub(crate) stream_dropped_event_count_by_type: HashMap<String, u64>,
    pub(crate) stream_mode: String,
    pub(crate) stream_sampling_enabled: bool,
    pub(crate) stream_drop_non_critical: bool,
}

pub(crate) struct PendingStreamEvent {
    pub(crate) event_type: String,
    pub(crate) symbol: Option<String>,
    pub(crate) level: String,
    pub(crate) payload: Vec<(String, String)>,
}

/// 策略回调返回的待注册引擎级计划(OCO 组 / bracket 计划)。
/// 常态为 None,避免每 bar 对策略对象做两次 getattr。
pub(crate) struct PendingEnginePlans {
    pub oco_groups: Vec<(String, String, String)>,
    pub bracket_plans: Vec<(
        String,
        Option<f64>,
        Option<f64>,
        Option<crate::model::TimeInForce>,
        Option<String>,
        Option<String>,
    )>,
}

// Internal implementation of Engine (not exposed to Python)
impl Engine {
    fn current_account_metrics(&self) -> AccountMetrics {
        calculate_account_metrics(
            &self.state.portfolio,
            &self.last_prices,
            &self.instruments,
            &self.state.order_manager.trade_tracker,
            &self.risk_manager.config,
        )
    }

    /// Cash/margin reserved by open (non-terminal) orders: the incremental used
    /// margin they would consume beyond the current portfolio. Reported as the
    /// account's `frozen_cash`, computed in Rust so it cannot drift from the
    /// engine's own margin accounting (the Python re-implementation was removed).
    fn current_frozen_cash(&self, active_orders: &[Order]) -> Decimal {
        let stock_ratio_override = if self.risk_manager.config.is_margin_account() {
            Some(self.risk_manager.config.stock_initial_margin_ratio())
        } else {
            None
        };
        let base_used = self.state.portfolio.calculate_used_margin_with_stock_ratio(
            &self.last_prices,
            &self.instruments,
            stock_ratio_override,
        );

        // Project every open order's position onto a clone and diff used margin.
        let mut projected = self.state.portfolio.clone();
        for order in active_orders {
            if !matches!(
                order.status,
                crate::model::OrderStatus::New
                    | crate::model::OrderStatus::Submitted
                    | crate::model::OrderStatus::PartiallyFilled
            ) {
                continue;
            }
            let remaining = order.quantity - order.filled_quantity;
            if remaining <= Decimal::ZERO {
                continue;
            }
            let current_pos = projected
                .positions
                .get(&order.symbol)
                .copied()
                .unwrap_or(Decimal::ZERO);
            let next_pos = crate::model::project_position_after(
                order.side,
                order.position_effect,
                current_pos,
                remaining,
            );
            projected.adjust_position(&order.symbol, next_pos - current_pos);
        }
        let projected_used = projected.calculate_used_margin_with_stock_ratio(
            &self.last_prices,
            &self.instruments,
            stock_ratio_override,
        );
        (projected_used - base_used).max(Decimal::ZERO)
    }

    fn build_position_entry_prices(&self) -> Arc<HashMap<String, Decimal>> {
        let mut entry_prices = HashMap::new();
        for (symbol, quantity) in self.state.portfolio.positions.iter() {
            if quantity.is_zero() {
                continue;
            }
            let average_price = self
                .state
                .order_manager
                .trade_tracker
                .get_average_price(symbol);
            entry_prices.insert(symbol.clone(), average_price);
        }
        Arc::new(entry_prices)
    }

    pub(crate) fn is_active_timestamp(&self, timestamp: i64) -> bool {
        self.active_start_time_ns
            .is_none_or(|start_ns| timestamp >= start_ns)
    }

    pub(crate) fn current_event_timestamp(&self) -> Option<i64> {
        match self.current_event.as_ref() {
            Some(Event::Bar(bar)) => Some(bar.timestamp),
            Some(Event::Tick(tick)) => Some(tick.timestamp),
            Some(Event::Timer(timer)) => Some(timer.timestamp),
            _ => None,
        }
    }

    fn effective_terminal_timestamp_for_timer(timer: &Timer) -> Option<i64> {
        if let Some(source_ts) = timer
            .payload
            .strip_prefix("__framework_cross_section__|")
            .and_then(|payload| payload.rsplit('|').next())
            .and_then(|value| value.parse::<i64>().ok())
        {
            return Some(source_ts);
        }
        if timer.payload.starts_with("__framework_boundary__|after|") {
            return Some(timer.timestamp.saturating_sub(1));
        }
        if timer.payload.starts_with("__framework_boundary__|before|") {
            return None;
        }
        Some(timer.timestamp)
    }

    pub(crate) fn terminal_result_timestamp(&self) -> Option<i64> {
        match self.current_event.as_ref() {
            Some(Event::Bar(bar)) => Some(bar.timestamp),
            Some(Event::Tick(tick)) => Some(tick.timestamp),
            Some(Event::Timer(timer)) => Self::effective_terminal_timestamp_for_timer(timer),
            _ => self.clock.timestamp(),
        }
    }

    pub(crate) fn context_timestamp(&self) -> i64 {
        self.terminal_result_timestamp()
            .or_else(|| self.clock.timestamp())
            .unwrap_or(0)
    }

    pub(crate) fn execution_policy_core(&self) -> ExecutionPolicyCore {
        self.execution_policy_core_state
    }

    pub(crate) fn set_execution_policy_core(&mut self, policy: ExecutionPolicyCore) {
        self.execution_policy_core_state = policy;
    }

    pub(crate) fn normalized_order_strategy_id(order: &Order) -> Option<String> {
        order
            .owner_strategy_id
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    }

    pub(crate) fn ensure_strategy_slot_exists(&mut self) {
        if self.strategy_slots.is_empty() {
            let strategy_id = self
                .default_strategy_id
                .clone()
                .unwrap_or_else(|| "_default".to_string());
            self.strategy_slots.push(StrategySlot { strategy_id });
        }
        if self.active_strategy_slot >= self.strategy_slots.len() {
            self.active_strategy_slot = 0;
        }
    }

    pub(crate) fn strategy_priority_for_order(&self, order: &Order) -> i32 {
        let strategy_id =
            Self::normalized_order_strategy_id(order).unwrap_or_else(|| "_default".to_string());
        self.strategy_priorities
            .get(&strategy_id)
            .copied()
            .unwrap_or(0)
    }

    pub(crate) fn order_estimated_notional(&self, order: &Order) -> Option<Decimal> {
        let price = if let Some(p) = order.price {
            Some(p)
        } else {
            self.last_prices.get(&order.symbol).copied()
        }?;
        Some((price * order.quantity).abs())
    }

    pub(crate) fn trade_notional(&self, trade: &Trade) -> Decimal {
        (trade.price * trade.quantity).abs()
    }

    pub(crate) fn maybe_reset_risk_budget_usage(&mut self, current_time: i64) {
        if !self.risk_budget_reset_daily {
            return;
        }
        let current_day = self.local_date_from_ns(current_time);
        if self.risk_budget_usage_day == Some(current_day) {
            return;
        }
        self.strategy_risk_budget_used.clear();
        self.portfolio_risk_budget_used = Decimal::ZERO;
        self.risk_budget_usage_day = Some(current_day);
    }

    pub(crate) fn risk_budget_use_trade_mode(&self) -> bool {
        self.risk_budget_mode == "trade_notional"
    }

    pub(crate) fn check_strategy_risk_budget_limit(&self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let budget = self
            .strategy_risk_budget_limits
            .get(&strategy_id)
            .copied()?;
        let used = self
            .strategy_risk_budget_used
            .get(&strategy_id)
            .copied()
            .unwrap_or(Decimal::ZERO);
        let projected = used + self.order_estimated_notional(order)?;
        if projected > budget {
            return Some(format!(
                "Risk: Strategy {} risk budget {} exceeds strategy budget {}",
                strategy_id, projected, budget
            ));
        }
        None
    }

    pub(crate) fn check_portfolio_risk_budget_limit(&self, order: &Order) -> Option<String> {
        let budget = self.portfolio_risk_budget_limit?;
        let projected = self.portfolio_risk_budget_used + self.order_estimated_notional(order)?;
        if projected > budget {
            return Some(format!(
                "Risk: Portfolio risk budget {} exceeds portfolio budget {}",
                projected, budget
            ));
        }
        None
    }

    pub(crate) fn apply_risk_budget_usage(&mut self, order: &Order) {
        let Some(notional) = self.order_estimated_notional(order) else {
            return;
        };
        if let Some(strategy_id) = Self::normalized_order_strategy_id(order)
            && self.strategy_risk_budget_limits.contains_key(&strategy_id)
        {
            let entry = self
                .strategy_risk_budget_used
                .entry(strategy_id)
                .or_insert(Decimal::ZERO);
            *entry += notional;
        }
        if self.portfolio_risk_budget_limit.is_some() {
            self.portfolio_risk_budget_used += notional;
        }
    }

    pub(crate) fn apply_risk_budget_usage_from_trade(&mut self, trade: &Trade) {
        let notional = self.trade_notional(trade);
        if let Some(strategy_id) = trade
            .owner_strategy_id
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            && self.strategy_risk_budget_limits.contains_key(&strategy_id)
        {
            let entry = self
                .strategy_risk_budget_used
                .entry(strategy_id)
                .or_insert(Decimal::ZERO);
            *entry += notional;
        }
        if self.portfolio_risk_budget_limit.is_some() {
            self.portfolio_risk_budget_used += notional;
        }
    }

    pub(crate) fn check_strategy_order_value_limit(&self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let max_value = self.strategy_max_order_value_limits.get(&strategy_id)?;
        let price = if let Some(p) = order.price {
            Some(p)
        } else {
            self.last_prices.get(&order.symbol).copied()
        }?;
        let value = price * order.quantity;
        if value > *max_value {
            return Some(format!(
                "Risk: Strategy {} order value {} exceeds strategy limit {}",
                strategy_id, value, max_value
            ));
        }
        None
    }

    pub(crate) fn check_strategy_order_size_limit(&self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let max_size = self.strategy_max_order_size_limits.get(&strategy_id)?;
        if order.quantity > *max_size {
            return Some(format!(
                "Risk: Strategy {} order quantity {} exceeds strategy limit {}",
                strategy_id, order.quantity, max_size
            ));
        }
        None
    }

    pub(crate) fn check_strategy_position_size_limit(&self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let max_size = self.strategy_max_position_size_limits.get(&strategy_id)?;
        let strategy_positions = self.strategy_positions.get(&strategy_id);
        let current_qty = strategy_positions
            .and_then(|positions| positions.get(&order.symbol).copied())
            .unwrap_or(Decimal::ZERO);
        let delta = match order.side {
            OrderSide::Buy => order.quantity,
            OrderSide::Sell => -order.quantity,
        };
        let projected_qty = current_qty + delta;
        if projected_qty.abs() > *max_size {
            return Some(format!(
                "Risk: Strategy {} projected position {} exceeds strategy position limit {}",
                strategy_id, projected_qty, max_size
            ));
        }
        None
    }

    pub(crate) fn apply_strategy_trade_position(&mut self, trade: &Trade) {
        let strategy_id = trade
            .owner_strategy_id
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let Some(strategy_key) = strategy_id else {
            return;
        };
        let strategy_key_for_position = strategy_key.clone();
        let entry = self
            .strategy_positions
            .entry(strategy_key_for_position)
            .or_default()
            .entry(trade.symbol.clone())
            .or_insert(Decimal::ZERO);
        match trade.side {
            OrderSide::Buy => *entry += trade.quantity,
            OrderSide::Sell => *entry -= trade.quantity,
        }
        let cashflow_entry = self
            .strategy_cashflows
            .entry(strategy_key)
            .or_insert(Decimal::ZERO);
        match trade.side {
            OrderSide::Buy => *cashflow_entry -= trade.price * trade.quantity,
            OrderSide::Sell => *cashflow_entry += trade.price * trade.quantity,
        }
        *cashflow_entry -= trade.commission;
    }

    pub(crate) fn current_strategy_pnl(&self, strategy_id: &str) -> Decimal {
        let cashflow = self
            .strategy_cashflows
            .get(strategy_id)
            .copied()
            .unwrap_or(Decimal::ZERO);
        let mark_to_market = self
            .strategy_positions
            .get(strategy_id)
            .map(|positions| {
                positions
                    .iter()
                    .map(|(symbol, qty)| {
                        let price = self
                            .last_prices
                            .get(symbol)
                            .copied()
                            .unwrap_or(Decimal::ZERO);
                        *qty * price
                    })
                    .sum::<Decimal>()
            })
            .unwrap_or(Decimal::ZERO);
        cashflow + mark_to_market
    }

    pub(crate) fn check_strategy_daily_loss_limit(
        &mut self,
        order: &Order,
        current_time: i64,
    ) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let max_daily_loss = self
            .strategy_max_daily_loss_limits
            .get(&strategy_id)
            .copied()?;
        let current_day = self.local_date_from_ns(current_time);
        let current_pnl = self.current_strategy_pnl(&strategy_id);
        let needs_reset = self
            .strategy_daily_loss_day
            .get(&strategy_id)
            .map(|day| *day != current_day)
            .unwrap_or(true);
        if needs_reset {
            let reset_baseline = if self.strategy_daily_loss_day.contains_key(&strategy_id) {
                self.strategy_last_pnl
                    .get(&strategy_id)
                    .copied()
                    .unwrap_or(current_pnl)
            } else {
                current_pnl
            };
            self.strategy_daily_loss_day
                .insert(strategy_id.clone(), current_day);
            self.strategy_daily_loss_baseline_pnl
                .insert(strategy_id.clone(), reset_baseline);
        }
        let baseline = self
            .strategy_daily_loss_baseline_pnl
            .get(&strategy_id)
            .copied()
            .unwrap_or(current_pnl);
        self.strategy_last_pnl
            .insert(strategy_id.clone(), current_pnl);
        let loss = baseline - current_pnl;
        if loss > max_daily_loss {
            return Some(format!(
                "Risk: Strategy {} daily loss {} exceeds strategy limit {}",
                strategy_id, loss, max_daily_loss
            ));
        }
        None
    }

    pub(crate) fn check_strategy_drawdown_limit(&mut self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let max_drawdown = self
            .strategy_max_drawdown_limits
            .get(&strategy_id)
            .copied()?;
        let current_pnl = self.current_strategy_pnl(&strategy_id);
        let peak = self
            .strategy_peak_pnl
            .get(&strategy_id)
            .copied()
            .unwrap_or(current_pnl);
        let updated_peak = if current_pnl > peak {
            current_pnl
        } else {
            peak
        };
        self.strategy_peak_pnl
            .insert(strategy_id.clone(), updated_peak);
        let drawdown = updated_peak - current_pnl;
        if drawdown > max_drawdown {
            return Some(format!(
                "Risk: Strategy {} drawdown {} exceeds strategy limit {}",
                strategy_id, drawdown, max_drawdown
            ));
        }
        None
    }

    pub(crate) fn activate_strategy_reduce_only_if_configured(&mut self, order: &Order) {
        if let Some(strategy_id) = Self::normalized_order_strategy_id(order)
            && self.strategy_reduce_only_after_risk.contains(&strategy_id)
        {
            self.strategy_reduce_only_active.insert(strategy_id);
        }
    }

    pub(crate) fn check_strategy_reduce_only_mode(&self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        if !self.strategy_reduce_only_active.contains(&strategy_id) {
            return None;
        }
        let current_qty = self
            .strategy_positions
            .get(&strategy_id)
            .and_then(|positions| positions.get(&order.symbol).copied())
            .unwrap_or(Decimal::ZERO);
        let allowed = if current_qty > Decimal::ZERO {
            order.side == OrderSide::Sell && order.quantity <= current_qty
        } else if current_qty < Decimal::ZERO {
            order.side == OrderSide::Buy && order.quantity <= current_qty.abs()
        } else {
            false
        };
        if !allowed {
            return Some(format!(
                "Risk: Strategy {} is in reduce_only mode and cannot open/increase position",
                strategy_id
            ));
        }
        None
    }

    pub(crate) fn activate_strategy_risk_cooldown_if_configured(&mut self, order: &Order) {
        let Some(strategy_id) = Self::normalized_order_strategy_id(order) else {
            return;
        };
        let Some(cooldown_bars) = self.strategy_risk_cooldown_bars.get(&strategy_id).copied()
        else {
            return;
        };
        if cooldown_bars == 0 {
            return;
        }
        let until_bar = self.bar_count.saturating_add(cooldown_bars);
        self.strategy_risk_cooldown_until_bar
            .insert(strategy_id, until_bar);
    }

    pub(crate) fn check_strategy_risk_cooldown_mode(&mut self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let until_bar = self
            .strategy_risk_cooldown_until_bar
            .get(&strategy_id)
            .copied()?;
        if self.bar_count > until_bar {
            self.strategy_risk_cooldown_until_bar.remove(&strategy_id);
            return None;
        }
        let bars_remaining = until_bar.saturating_sub(self.bar_count);
        Some(format!(
            "Risk: Strategy {} is in cooldown for {} more bar(s)",
            strategy_id, bars_remaining
        ))
    }

    pub(crate) fn ensure_strategy_context_capacity(&mut self) {
        self.ensure_strategy_slot_exists();
        while self.strategy_contexts.len() < self.strategy_slots.len() {
            self.strategy_contexts.push(None);
        }
        while self.strategy_slot_strategies.len() < self.strategy_slots.len() {
            self.strategy_slot_strategies.push(None);
        }
    }

    pub(crate) fn get_or_create_strategy_context(
        &mut self,
        slot_index: usize,
        active_orders: Arc<Vec<Order>>,
        step_trades: Vec<Trade>,
        step_rejected_orders: Vec<Order>,
        previous_cash: Decimal,
        previous_account_metrics: AccountMetrics,
    ) -> PyResult<Py<StrategyContext>> {
        self.ensure_strategy_context_capacity();
        let account_metrics = self.current_account_metrics();
        let position_entry_prices = self.build_position_entry_prices();
        let frozen_cash = self
            .current_frozen_cash(&active_orders)
            .to_f64()
            .unwrap_or_default();
        if let Some(existing_ctx) = self
            .strategy_contexts
            .get(slot_index)
            .and_then(|ctx| ctx.as_ref())
        {
            return Python::attach(|py| {
                let py_ctx = existing_ctx.clone_ref(py);
                {
                    let mut ctx_mut = py_ctx.borrow_mut(py);
                    ctx_mut.update_state(crate::context::ContextUpdate {
                        cash: self.state.portfolio.cash,
                        previous_cash,
                        positions: self.state.portfolio.positions.clone(),
                        available_positions: self.state.portfolio.available_positions.clone(),
                        position_entry_prices,
                        session: self.clock.session,
                        current_time: self.context_timestamp(),
                        active_orders,
                        closed_trades: self.state.order_manager.trade_tracker.closed_trades.clone(),
                        recent_trades: step_trades,
                        recent_rejected_orders: step_rejected_orders,
                        recent_expiry_events: self.recent_expiry_events.clone(),
                        account_equity: account_metrics.equity.to_f64().unwrap_or_default(),
                        account_market_value: account_metrics
                            .market_value
                            .to_f64()
                            .unwrap_or_default(),
                        account_notional_value: account_metrics
                            .notional_value
                            .to_f64()
                            .unwrap_or_default(),
                        account_used_margin: account_metrics
                            .used_margin
                            .to_f64()
                            .unwrap_or_default(),
                        account_unrealized_pnl: account_metrics
                            .unrealized_pnl
                            .to_f64()
                            .unwrap_or_default(),
                        account_maintenance_ratio: account_metrics
                            .maintenance_ratio
                            .to_f64()
                            .unwrap_or_default(),
                        account_short_market_value: account_metrics
                            .short_market_value
                            .to_f64()
                            .unwrap_or_default(),
                        account_frozen_cash: frozen_cash,
                        previous_account_equity: previous_account_metrics
                            .equity
                            .to_f64()
                            .unwrap_or_default(),
                        previous_account_market_value: previous_account_metrics
                            .market_value
                            .to_f64()
                            .unwrap_or_default(),
                        previous_account_notional_value: previous_account_metrics
                            .notional_value
                            .to_f64()
                            .unwrap_or_default(),
                        previous_account_used_margin: previous_account_metrics
                            .used_margin
                            .to_f64()
                            .unwrap_or_default(),
                        previous_account_unrealized_pnl: previous_account_metrics
                            .unrealized_pnl
                            .to_f64()
                            .unwrap_or_default(),
                        previous_account_maintenance_ratio: previous_account_metrics
                            .maintenance_ratio
                            .to_f64()
                            .unwrap_or_default(),
                        margin_accrued_interest: self
                            .margin_accrued_interest
                            .to_f64()
                            .unwrap_or_default(),
                        margin_daily_interest: self
                            .margin_daily_interest
                            .to_f64()
                            .unwrap_or_default(),
                        last_prices: Arc::new(self.last_prices.clone()),
                    });
                }
                Ok::<_, PyErr>(py_ctx)
            });
        }

        let strategy_id = self
            .strategy_slots
            .get(slot_index)
            .map(|slot| slot.strategy_id.clone());
        let ctx = self.create_context(
            active_orders,
            step_trades,
            step_rejected_orders,
            strategy_id,
            previous_cash,
            previous_account_metrics,
        );
        let (py_ctx, persistent_ref) = Python::attach(|py| {
            let py_ctx = Py::new(py, ctx).unwrap();
            Ok::<_, PyErr>((py_ctx.clone_ref(py), py_ctx.clone_ref(py)))
        })?;
        if let Some(slot_ctx) = self.strategy_contexts.get_mut(slot_index) {
            *slot_ctx = Some(persistent_ref);
        }
        Ok(py_ctx)
    }

    pub(crate) fn set_stream_callback_internal(&mut self, callback: Option<Py<PyAny>>) {
        self.stream_callback = callback;
        self.stream_run_id = None;
        self.stream_seq = 0;
        self.stream_buffer.clear();
        self.stream_callback_error_count = 0;
        self.stream_callback_last_error = None;
        self.stream_fatal_error = None;
        self.stream_dropped_event_count = 0;
        self.stream_dropped_event_count_by_type.clear();
    }

    pub(crate) fn set_stream_options_internal(
        &mut self,
        progress_interval: usize,
        equity_interval: usize,
        batch_size: usize,
        max_buffer: usize,
        error_mode: &str,
        stream_mode: &str,
    ) {
        self.stream_progress_interval = progress_interval.max(1);
        self.stream_equity_interval = equity_interval.max(1);
        self.stream_batch_size = batch_size.max(1);
        self.stream_max_buffer = max_buffer.max(1);
        self.stream_fail_fast_on_callback_error = error_mode == "fail_fast";
        if stream_mode == "audit" {
            self.stream_mode = "audit".to_string();
            self.stream_sampling_enabled = false;
            self.stream_drop_non_critical = false;
        } else {
            self.stream_mode = "observability".to_string();
            self.stream_sampling_enabled = true;
            self.stream_drop_non_critical = true;
        }
    }

    pub(crate) fn start_stream_run(&mut self, py: Python<'_>, total_events: Option<usize>) {
        if self.stream_callback.is_none() {
            self.stream_run_id = None;
            self.stream_seq = 0;
            self.stream_buffer.clear();
            self.stream_callback_error_count = 0;
            self.stream_callback_last_error = None;
            self.stream_fatal_error = None;
            self.stream_dropped_event_count = 0;
            self.stream_dropped_event_count_by_type.clear();
            return;
        }
        self.stream_seq = 0;
        self.stream_run_id = Some(Uuid::new_v4().to_string());
        self.stream_callback_error_count = 0;
        self.stream_callback_last_error = None;
        self.stream_fatal_error = None;
        self.stream_dropped_event_count = 0;
        self.stream_dropped_event_count_by_type.clear();

        let mut payload = HashMap::new();
        payload.insert("total_events", total_events.unwrap_or(0).to_string());
        payload.insert(
            "fill_policy_price_basis",
            format!("{:?}", self.execution_policy_core_state.price_basis),
        );
        payload.insert(
            "fill_policy_bar_offset",
            self.execution_policy_core_state.bar_offset.to_string(),
        );
        payload.insert(
            "fill_policy_temporal",
            self.execution_policy_core_state
                .temporal_as_str()
                .to_string(),
        );
        self.emit_stream_event(py, "started", None, "info", payload);
    }

    pub(crate) fn emit_stream_event(
        &mut self,
        py: Python<'_>,
        event_type: &str,
        symbol: Option<&str>,
        level: &str,
        payload: HashMap<&str, String>,
    ) {
        let (Some(_), Some(_)) = (&self.stream_callback, &self.stream_run_id) else {
            return;
        };

        if self.stream_sampling_enabled
            && event_type == "progress"
            && !self.bar_count.is_multiple_of(self.stream_progress_interval)
        {
            return;
        }
        if self.stream_sampling_enabled
            && event_type == "equity"
            && !self.bar_count.is_multiple_of(self.stream_equity_interval)
        {
            return;
        }

        let is_critical = matches!(
            event_type,
            "started" | "order" | "trade" | "risk" | "error" | "finished"
        );
        if self.stream_buffer.len() >= self.stream_max_buffer {
            if is_critical || !self.stream_drop_non_critical {
                self.flush_stream_events(py);
            } else {
                self.record_dropped_stream_event(event_type);
                return;
            }
        }

        let mut payload_vec = Vec::new();
        for (k, v) in payload {
            payload_vec.push((k.to_string(), v));
        }
        self.stream_buffer.push(PendingStreamEvent {
            event_type: event_type.to_string(),
            symbol: symbol.map(str::to_string),
            level: level.to_string(),
            payload: payload_vec,
        });

        if is_critical || self.stream_buffer.len() >= self.stream_batch_size {
            self.flush_stream_events(py);
        }
    }

    pub(crate) fn emit_stream_event_owned(
        &mut self,
        py: Python<'_>,
        event_type: &str,
        symbol: Option<String>,
        level: &str,
        payload: HashMap<String, String>,
    ) {
        let payload_ref: HashMap<&str, String> = payload
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        self.emit_stream_event(py, event_type, symbol.as_deref(), level, payload_ref);
    }

    pub(crate) fn flush_stream_events(&mut self, py: Python<'_>) {
        let (Some(callback_ref), Some(run_id_ref)) = (&self.stream_callback, &self.stream_run_id)
        else {
            self.stream_buffer.clear();
            return;
        };
        let callback = callback_ref.clone_ref(py);
        let run_id = run_id_ref.clone();
        if self.stream_buffer.is_empty() {
            return;
        }

        for pending in self.stream_buffer.drain(..) {
            self.stream_seq = self.stream_seq.saturating_add(1);
            let event_dict = PyDict::new(py);
            if event_dict.set_item("run_id", run_id.as_str()).is_err()
                || event_dict.set_item("seq", self.stream_seq).is_err()
                || event_dict
                    .set_item("ts", self.clock.timestamp().unwrap_or(0))
                    .is_err()
                || event_dict
                    .set_item("event_type", pending.event_type.as_str())
                    .is_err()
                || event_dict.set_item("symbol", pending.symbol).is_err()
                || event_dict
                    .set_item("level", pending.level.as_str())
                    .is_err()
            {
                continue;
            }

            let payload_dict = PyDict::new(py);
            for (k, v) in pending.payload {
                if payload_dict.set_item(k, v).is_err() {
                    continue;
                }
            }
            if event_dict.set_item("payload", payload_dict).is_err() {
                continue;
            }
            if let Err(err) = callback.bind(py).call1((event_dict,)) {
                self.stream_callback_error_count =
                    self.stream_callback_error_count.saturating_add(1);
                let message = err.to_string();
                self.stream_callback_last_error = Some(message.clone());
                if self.stream_fail_fast_on_callback_error {
                    self.stream_fatal_error = Some(format!(
                        "stream callback failed in fail_fast mode: {}",
                        message
                    ));
                    break;
                }
            }
        }
    }

    pub(crate) fn finish_stream_run(&mut self, py: Python<'_>, status: &str, reason: Option<&str>) {
        if self.stream_callback.is_none() {
            self.stream_run_id = None;
            self.stream_seq = 0;
            self.stream_buffer.clear();
            self.stream_callback_error_count = 0;
            self.stream_callback_last_error = None;
            self.stream_fatal_error = None;
            self.stream_dropped_event_count = 0;
            self.stream_dropped_event_count_by_type.clear();
            return;
        }

        let mut payload = HashMap::new();
        payload.insert("status", status.to_string());
        payload.insert("processed_events", self.bar_count.to_string());
        payload.insert(
            "total_trades",
            self.state.order_manager.trades.len().to_string(),
        );
        payload.insert(
            "callback_error_count",
            self.stream_callback_error_count.to_string(),
        );
        payload.insert(
            "dropped_event_count",
            self.stream_dropped_event_count.to_string(),
        );
        payload.insert(
            "dropped_event_count_by_type",
            self.format_stream_dropped_event_counts(),
        );
        payload.insert("stream_mode", self.stream_mode.clone());
        payload.insert("sampling_enabled", self.stream_sampling_enabled.to_string());
        payload.insert(
            "backpressure_policy",
            if self.stream_drop_non_critical {
                "drop_non_critical".to_string()
            } else {
                "block".to_string()
            },
        );
        if let Some(message) = reason {
            payload.insert("reason", message.to_string());
        }
        if let Some(message) = &self.stream_callback_last_error {
            payload.insert("last_callback_error", message.clone());
        }
        self.emit_stream_event(py, "finished", None, "info", payload);
        self.flush_stream_events(py);
        self.stream_run_id = None;
        self.stream_buffer.clear();
    }

    pub(crate) fn record_dropped_stream_event(&mut self, event_type: &str) {
        self.stream_dropped_event_count = self.stream_dropped_event_count.saturating_add(1);
        let counter = self
            .stream_dropped_event_count_by_type
            .entry(event_type.to_string())
            .or_insert(0);
        *counter = counter.saturating_add(1);
    }

    pub(crate) fn format_stream_dropped_event_counts(&self) -> String {
        if self.stream_dropped_event_count_by_type.is_empty() {
            return String::new();
        }
        let mut entries: Vec<(&String, &u64)> =
            self.stream_dropped_event_count_by_type.iter().collect();
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));
        entries
            .into_iter()
            .map(|(event_type, count)| format!("{}={}", event_type, count))
            .collect::<Vec<String>>()
            .join(",")
    }

    pub(crate) fn raise_stream_fatal_error_if_any(&mut self) -> PyResult<()> {
        if let Some(message) = self.stream_fatal_error.take() {
            return Err(PyRuntimeError::new_err(message));
        }
        Ok(())
    }

    pub(crate) fn create_context(
        &self,
        active_orders: Arc<Vec<Order>>,
        step_trades: Vec<Trade>,
        step_rejected_orders: Vec<Order>,
        strategy_id: Option<String>,
        previous_cash: Decimal,
        previous_account_metrics: AccountMetrics,
    ) -> StrategyContext {
        let account_metrics = self.current_account_metrics();
        let position_entry_prices = self.build_position_entry_prices();
        let frozen_cash = self
            .current_frozen_cash(&active_orders)
            .to_f64()
            .unwrap_or_default();
        // Create a temporary context for the strategy to use
        StrategyContext::new(crate::context::ContextInit {
            cash: self.state.portfolio.cash,
            previous_cash,
            positions: self.state.portfolio.positions.clone(),
            available_positions: self.state.portfolio.available_positions.clone(),
            position_entry_prices,
            session: self.clock.session,
            current_time: self.context_timestamp(),
            active_orders,
            closed_trades: self.state.order_manager.trade_tracker.closed_trades.clone(),
            recent_trades: step_trades,
            recent_rejected_orders: step_rejected_orders,
            recent_expiry_events: self.recent_expiry_events.clone(),
            history_buffer: Some(self.history_buffer.clone()),
            event_tx: Some(self.event_manager.sender()),
            risk_config: self.risk_manager.config.clone(),
            strategy_id,
            account_equity: account_metrics.equity.to_f64().unwrap_or_default(),
            account_market_value: account_metrics.market_value.to_f64().unwrap_or_default(),
            account_notional_value: account_metrics.notional_value.to_f64().unwrap_or_default(),
            account_used_margin: account_metrics.used_margin.to_f64().unwrap_or_default(),
            account_unrealized_pnl: account_metrics.unrealized_pnl.to_f64().unwrap_or_default(),
            account_maintenance_ratio: account_metrics
                .maintenance_ratio
                .to_f64()
                .unwrap_or_default(),
            account_short_market_value: account_metrics
                .short_market_value
                .to_f64()
                .unwrap_or_default(),
            account_frozen_cash: frozen_cash,
            previous_account_equity: previous_account_metrics.equity.to_f64().unwrap_or_default(),
            previous_account_market_value: previous_account_metrics
                .market_value
                .to_f64()
                .unwrap_or_default(),
            previous_account_notional_value: previous_account_metrics
                .notional_value
                .to_f64()
                .unwrap_or_default(),
            previous_account_used_margin: previous_account_metrics
                .used_margin
                .to_f64()
                .unwrap_or_default(),
            previous_account_unrealized_pnl: previous_account_metrics
                .unrealized_pnl
                .to_f64()
                .unwrap_or_default(),
            previous_account_maintenance_ratio: previous_account_metrics
                .maintenance_ratio
                .to_f64()
                .unwrap_or_default(),
            margin_accrued_interest: self.margin_accrued_interest.to_f64().unwrap_or_default(),
            margin_daily_interest: self.margin_daily_interest.to_f64().unwrap_or_default(),
            instruments: Arc::new(self.instruments.clone()),
            last_prices: Arc::new(self.last_prices.clone()),
        })
    }

    pub(crate) fn datetime_from_ns(timestamp: i64) -> DateTime<Utc> {
        let secs = timestamp.div_euclid(1_000_000_000);
        let nanos = timestamp.rem_euclid(1_000_000_000) as u32;
        Utc.timestamp_opt(secs, nanos)
            .single()
            .expect("Invalid timestamp")
    }

    fn parsed_timezone(&self) -> Option<Tz> {
        self.timezone_name
            .as_deref()
            .and_then(|name| name.parse::<Tz>().ok())
    }

    pub(crate) fn local_time_from_ns(&self, timestamp: i64) -> NaiveTime {
        let utc_dt = Self::datetime_from_ns(timestamp);
        if let Some(tz) = self.parsed_timezone() {
            return utc_dt.with_timezone(&tz).time();
        }
        Self::local_datetime_from_ns(timestamp, self.timezone_offset).time()
    }

    pub(crate) fn local_date_from_ns(&self, timestamp: i64) -> NaiveDate {
        let utc_dt = Self::datetime_from_ns(timestamp);
        if let Some(tz) = self.parsed_timezone() {
            return utc_dt.with_timezone(&tz).date_naive();
        }
        Self::local_datetime_from_ns(timestamp, self.timezone_offset).date_naive()
    }

    pub(crate) fn local_datetime_from_ns(timestamp: i64, offset_secs: i32) -> DateTime<Utc> {
        let offset_ns = i64::from(offset_secs) * 1_000_000_000;
        Self::datetime_from_ns(timestamp + offset_ns)
    }

    pub(crate) fn parse_time_string(value: &str) -> PyResult<NaiveTime> {
        if let Ok(t) = NaiveTime::parse_from_str(value, "%H:%M:%S") {
            return Ok(t);
        }
        if let Ok(t) = NaiveTime::parse_from_str(value, "%H:%M") {
            return Ok(t);
        }
        Err(PyValueError::new_err(format!(
            "Invalid time format: {}",
            value
        )))
    }

    pub(crate) fn call_strategy_for_slot(
        &mut self,
        strategy: &Bound<'_, PyAny>,
        event: &Event,
        slot_index: usize,
        active_orders: Arc<Vec<Order>>,
        step_trades: Vec<Trade>,
        step_rejected_orders: Vec<Order>,
    ) -> PyResult<(Vec<Order>, Vec<Timer>, Vec<String>, Option<PendingEnginePlans>)> {
        self.active_strategy_slot = slot_index;
        match event {
            Event::Bar(b) => {
                let previous_cash = self.state.portfolio.cash;
                let previous_account_metrics = self.current_account_metrics();
                self.last_prices.insert(b.symbol.clone(), b.close);
                let py_ctx = self.get_or_create_strategy_context(
                    slot_index,
                    active_orders,
                    step_trades,
                    step_rejected_orders,
                    previous_cash,
                    previous_account_metrics,
                )?;

                let args = Python::attach(|py| {
                    let bar = b.clone();
                    (bar, py_ctx.clone_ref(py))
                });

                let ret = strategy.call_method1("_on_bar_event_and_flush", args)?;
                let pending_plans: Option<PendingEnginePlans> = if ret.is_none() {
                    None
                } else {
                    let (oco, bracket) = ret.extract()?;
                    Some(PendingEnginePlans {
                        oco_groups: oco,
                        bracket_plans: bracket,
                    })
                };

                // Extract orders and timers
                let mut new_orders = Vec::new();
                let mut new_timers = Vec::new();
                let mut canceled_ids = Vec::new();
                Python::attach(|py| {
                    let ctx_ref = py_ctx.borrow(py);
                    // Read from RwLock
                    if let Ok(orders) = ctx_ref.orders_arc.read() {
                        new_orders.extend(orders.clone());
                    }
                    if let Ok(timers) = ctx_ref.timers_arc.read() {
                        new_timers.extend(timers.clone());
                    }
                    if let Ok(canceled) = ctx_ref.canceled_order_ids_arc.read() {
                        canceled_ids.extend(canceled.clone());
                    }
                });
                Ok((new_orders, new_timers, canceled_ids, pending_plans))
            }
            Event::Tick(t) => {
                let previous_cash = self.state.portfolio.cash;
                let previous_account_metrics = self.current_account_metrics();
                self.last_prices.insert(t.symbol.clone(), t.price);
                let py_ctx = self.get_or_create_strategy_context(
                    slot_index,
                    active_orders,
                    step_trades,
                    step_rejected_orders,
                    previous_cash,
                    previous_account_metrics,
                )?;

                let args = Python::attach(|py| {
                    let tick = t.clone();
                    (tick, py_ctx.clone_ref(py))
                });

                let ret = strategy.call_method1("_on_tick_event_and_flush", args)?;
                let pending_plans: Option<PendingEnginePlans> = if ret.is_none() {
                    None
                } else {
                    let (oco, bracket) = ret.extract()?;
                    Some(PendingEnginePlans {
                        oco_groups: oco,
                        bracket_plans: bracket,
                    })
                };

                // Extract orders and timers
                let mut new_orders = Vec::new();
                let mut new_timers = Vec::new();
                let mut canceled_ids = Vec::new();
                Python::attach(|py| {
                    let ctx_ref = py_ctx.borrow(py);
                    if let Ok(orders) = ctx_ref.orders_arc.read() {
                        new_orders.extend(orders.clone());
                    }
                    if let Ok(timers) = ctx_ref.timers_arc.read() {
                        new_timers.extend(timers.clone());
                    }
                    if let Ok(canceled) = ctx_ref.canceled_order_ids_arc.read() {
                        canceled_ids.extend(canceled.clone());
                    }
                });
                Ok((new_orders, new_timers, canceled_ids, pending_plans))
            }
            Event::Timer(timer) => {
                let previous_cash = self.state.portfolio.cash;
                let previous_account_metrics = self.current_account_metrics();
                let py_ctx = self.get_or_create_strategy_context(
                    slot_index,
                    active_orders,
                    step_trades,
                    step_rejected_orders,
                    previous_cash,
                    previous_account_metrics,
                )?;

                let args = Python::attach(|py| {
                    let payload = timer.payload.as_str();
                    (payload, py_ctx.clone_ref(py))
                });

                let ret = strategy.call_method1("_on_timer_event_and_flush", args)?;
                let pending_plans: Option<PendingEnginePlans> = if ret.is_none() {
                    None
                } else {
                    let (oco, bracket) = ret.extract()?;
                    Some(PendingEnginePlans {
                        oco_groups: oco,
                        bracket_plans: bracket,
                    })
                };

                // Extract orders and timers
                let mut new_orders = Vec::new();
                let mut new_timers = Vec::new();
                let mut canceled_ids = Vec::new();
                Python::attach(|py| {
                    let ctx_ref = py_ctx.borrow(py);
                    if let Ok(orders) = ctx_ref.orders_arc.read() {
                        new_orders.extend(orders.clone());
                    }
                    if let Ok(timers) = ctx_ref.timers_arc.read() {
                        new_timers.extend(timers.clone());
                    }
                    if let Ok(canceled) = ctx_ref.canceled_order_ids_arc.read() {
                        canceled_ids.extend(canceled.clone());
                    }
                });
                Ok((new_orders, new_timers, canceled_ids, pending_plans))
            }
            Event::OrderRequest(_) | Event::OrderValidated(_) | Event::ExecutionReport(_, _) => {
                Ok((Vec::new(), Vec::new(), Vec::new(), None))
            }
        }
    }

    pub(crate) fn flush_terminal_pending_order_events(
        &mut self,
        py: Python<'_>,
        strategy: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if self.state.order_manager.current_step_trades.is_empty()
            && self
                .state
                .order_manager
                .current_step_rejected_orders
                .is_empty()
        {
            return Ok(());
        }

        self.ensure_strategy_slot_exists();
        self.ensure_strategy_context_capacity();
        let slot_count = self.strategy_slots.len();
        let active_orders = Arc::new(self.state.order_manager.active_orders.clone());
        let step_trades = self.state.order_manager.current_step_trades.clone();
        let step_rejected_orders = self
            .state
            .order_manager
            .current_step_rejected_orders
            .clone();
        let previous_cash = self.state.portfolio.cash;
        let previous_account_metrics = self.current_account_metrics();

        for slot_index in 0..slot_count {
            self.active_strategy_slot = slot_index;
            let py_ctx = self.get_or_create_strategy_context(
                slot_index,
                active_orders.clone(),
                step_trades.clone(),
                step_rejected_orders.clone(),
                previous_cash,
                previous_account_metrics,
            )?;
            let slot_strategy = self
                .strategy_slot_strategies
                .get(slot_index)
                .and_then(|slot| slot.as_ref())
                .map(|slot| slot.clone_ref(py));
            if let Some(ref slot_py) = slot_strategy {
                slot_py
                    .bind(py)
                    .call_method1("_flush_pending_order_events", (py_ctx.clone_ref(py),))?;
            } else {
                strategy.call_method1("_flush_pending_order_events", (py_ctx,))?;
            }
        }

        self.state.order_manager.current_step_trades.clear();
        self.state
            .order_manager
            .current_step_rejected_orders
            .clear();
        Ok(())
    }

    pub(crate) fn build_pipeline(&self) -> PipelineRunner {
        let mut pipeline = PipelineRunner::new();
        // 1. Process events from previous iteration (or init)
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 2. Fetch new Data Event
        pipeline.add_processor(Box::new(DataProcessor::new()));

        // 3. Pre-Strategy Execution (Match Pending Orders)
        // For NextOpen/NextClose/NextAverage: Matches orders generated in previous bar against current bar.
        pipeline.add_processor(Box::new(ExecutionProcessor::new(
            ExecutionPhase::PreStrategy,
        )));

        // 4. Process Fills from Pre-Execution immediately (Update Portfolio before Strategy)
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 5. Run Strategy
        pipeline.add_processor(Box::new(StrategyProcessor));

        // 6. Process Order Requests from Strategy immediately (Validate -> Pending)
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 7. Post-Strategy Execution
        // For CurrentClose: Matches orders generated in current bar against current bar.
        pipeline.add_processor(Box::new(ExecutionProcessor::new(
            ExecutionPhase::PostStrategy,
        )));

        // 8. Process Fills from Post-Execution
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 9. Statistics & Cleanup
        pipeline.add_processor(Box::new(StatisticsProcessor));
        pipeline.add_processor(Box::new(CleanupProcessor));

        pipeline
    }
}
