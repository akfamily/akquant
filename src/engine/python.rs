use chrono::{NaiveDate, Offset, Utc};
use chrono_tz::Tz;
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};

use crate::account::calculate_account_metrics;
use crate::analysis::{BacktestResult, PositionSnapshot};
use crate::clock::Clock;
use crate::data::DataFeed;
use crate::event_manager::EventManager;
use crate::execution::{RealtimeExecutionClient, SimulatedExecutionClient};
use crate::history::HistoryBuffer;
use crate::market::corporate_action::CorporateActionManager;
use crate::market::manager::MarketManager;
use crate::model::{
    Bar, ExecutionMode, ExecutionPolicyCore, Instrument, Order, PriceBasis, Timer, Trade,
    TradingSession, corporate_action::CorporateAction,
};
use crate::portfolio::Portfolio;
use crate::risk::{RiskConfig, RiskManager};
use crate::settlement::SettlementManager;
use crate::statistics::StatisticsManager;

use super::core::{Engine, StrategySlot};
use super::state::SharedState;

#[gen_stub_pymethods]
#[pymethods]
impl Engine {
    /// 获取订单列表
    #[getter]
    fn get_orders(&self) -> Vec<Order> {
        self.state.order_manager.get_all_orders()
    }

    /// 获取成交列表
    #[getter]
    fn get_trades(&self) -> Vec<Trade> {
        self.state.order_manager.trades.clone()
    }

    /// 获取投资组合
    #[getter]
    fn get_portfolio(&self) -> Portfolio {
        self.state.portfolio.clone()
    }

    fn get_account_metrics(&self) -> (f64, f64, f64, f64, f64, f64) {
        let prices = self.last_prices.read().expect("last_prices 读锁被污染");
        let metrics = calculate_account_metrics(
            &self.state.portfolio,
            &prices,
            &self.instruments,
            &self.state.order_manager.trade_tracker,
            &self.risk_manager.config,
        );
        (
            metrics.equity.to_f64().unwrap_or_default(),
            metrics.market_value.to_f64().unwrap_or_default(),
            metrics.notional_value.to_f64().unwrap_or_default(),
            metrics.used_margin.to_f64().unwrap_or_default(),
            metrics.unrealized_pnl.to_f64().unwrap_or_default(),
            metrics.maintenance_ratio.to_f64().unwrap_or_default(),
        )
    }

    /// 获取数据源
    #[getter]
    fn get_feed(&self) -> DataFeed {
        self.state.feed.clone()
    }

    /// 获取持仓快照历史
    #[getter]
    fn get_snapshots(&self) -> Vec<(i64, Vec<PositionSnapshot>)> {
        self.statistics_manager.snapshots.clone()
    }

    /// 设置风控配置
    ///
    /// 由于 PyO3 对嵌套结构体的属性访问可能返回副本，
    /// 提供此方法以显式更新风控配置。
    ///
    /// :param config: 新的风控配置
    fn set_risk_config(&mut self, config: RiskConfig) {
        self.risk_manager.config = config;
    }

    fn set_stream_callback(&mut self, callback: Py<PyAny>) {
        self.set_stream_callback_internal(Some(callback));
    }

    fn clear_stream_callback(&mut self) {
        self.set_stream_callback_internal(None);
    }

    fn emit_stream_event_py(
        &mut self,
        py: Python<'_>,
        event_type: &str,
        symbol: Option<String>,
        level: &str,
        payload: HashMap<String, String>,
    ) {
        super::core::Engine::emit_stream_event_owned(self, py, event_type, symbol, level, payload);
    }

    fn add_timer(&mut self, timestamp: i64, payload: String) {
        self.timers.push(Timer { timestamp, payload });
    }

    fn set_stream_options(
        &mut self,
        progress_interval: usize,
        equity_interval: usize,
        batch_size: usize,
        max_buffer: usize,
        error_mode: &str,
        stream_mode: &str,
    ) {
        self.set_stream_options_internal(
            progress_interval,
            equity_interval,
            batch_size,
            max_buffer,
            error_mode,
            stream_mode,
        );
    }

    fn set_default_strategy_id(&mut self, strategy_id: Option<String>) {
        let effective_id = strategy_id.unwrap_or_else(|| "_default".to_string());
        self.default_strategy_id = Some(effective_id.clone());
        self.ensure_strategy_slot_exists();
        self.active_strategy_slot = 0;
        if let Some(slot) = self.strategy_slots.get_mut(0) {
            slot.strategy_id = effective_id;
        }
        self.ensure_strategy_context_capacity();
        if let Some(slot_ctx) = self.strategy_contexts.get_mut(0) {
            *slot_ctx = None;
        }
    }

    fn get_default_strategy_id(&self) -> Option<String> {
        self.default_strategy_id.clone()
    }

    fn register_oco_group(
        &mut self,
        group_id: String,
        first_order_id: String,
        second_order_id: String,
    ) {
        self.state
            .order_manager
            .register_oco_group(group_id, first_order_id, second_order_id);
    }

    #[allow(clippy::too_many_arguments)]
    fn register_bracket_plan(
        &mut self,
        entry_order_id: String,
        stop_trigger_price: Option<f64>,
        take_profit_price: Option<f64>,
        time_in_force: Option<crate::model::TimeInForce>,
        stop_tag: Option<String>,
        take_profit_tag: Option<String>,
    ) {
        let stop_trigger_decimal = stop_trigger_price.and_then(rust_decimal::Decimal::from_f64);
        let take_profit_decimal = take_profit_price.and_then(rust_decimal::Decimal::from_f64);
        self.state.order_manager.register_bracket_plan(
            entry_order_id,
            stop_trigger_decimal,
            take_profit_decimal,
            time_in_force.unwrap_or(crate::model::TimeInForce::GTC),
            stop_tag,
            take_profit_tag,
        );
    }

    fn get_strategy_slot_ids(&self) -> Vec<String> {
        self.strategy_slots
            .iter()
            .map(|slot| slot.strategy_id.clone())
            .collect()
    }

    fn get_strategy_slots(&self) -> Vec<String> {
        self.get_strategy_slot_ids()
    }

    fn get_active_strategy_slot(&self) -> usize {
        self.active_strategy_slot
    }

    fn set_strategy_slots(&mut self, slot_ids: Vec<String>) -> PyResult<()> {
        if slot_ids.is_empty() {
            return Err(PyValueError::new_err("slot_ids cannot be empty"));
        }
        let mut normalized = Vec::with_capacity(slot_ids.len());
        for id in slot_ids {
            let trimmed = id.trim();
            if trimmed.is_empty() {
                return Err(PyValueError::new_err("slot id cannot be empty"));
            }
            if normalized.iter().any(|v: &String| v == trimmed) {
                return Err(PyValueError::new_err(format!(
                    "duplicated slot id: {}",
                    trimmed
                )));
            }
            normalized.push(trimmed.to_string());
        }
        self.strategy_slots = normalized
            .into_iter()
            .map(|strategy_id| StrategySlot { strategy_id })
            .collect();
        self.active_strategy_slot = 0;
        self.default_strategy_id = self
            .strategy_slots
            .first()
            .map(|slot| slot.strategy_id.clone());
        self.strategy_contexts = (0..self.strategy_slots.len()).map(|_| None).collect();
        self.strategy_slot_strategies = (0..self.strategy_slots.len()).map(|_| None).collect();
        Ok(())
    }

    fn set_strategy_for_slot(
        &mut self,
        py: Python<'_>,
        slot_index: usize,
        strategy: Py<PyAny>,
    ) -> PyResult<()> {
        self.ensure_strategy_context_capacity();
        if slot_index >= self.strategy_slots.len() {
            return Err(PyValueError::new_err(format!(
                "slot_index out of range: {}",
                slot_index
            )));
        }
        self.strategy_slot_strategies[slot_index] = Some(strategy.clone_ref(py));
        self.strategy_contexts[slot_index] = None;
        Ok(())
    }

    fn clear_strategy_for_slot(&mut self, slot_index: usize) -> PyResult<()> {
        self.ensure_strategy_context_capacity();
        if slot_index >= self.strategy_slots.len() {
            return Err(PyValueError::new_err(format!(
                "slot_index out of range: {}",
                slot_index
            )));
        }
        self.strategy_slot_strategies[slot_index] = None;
        self.strategy_contexts[slot_index] = None;
        Ok(())
    }

    fn set_strategy_max_order_value_limits(
        &mut self,
        limits: HashMap<String, f64>,
    ) -> PyResult<()> {
        let mut normalized = HashMap::new();
        for (strategy_id, value) in limits {
            let trimmed = strategy_id.trim();
            if trimmed.is_empty() {
                return Err(PyValueError::new_err("strategy id cannot be empty"));
            }
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "invalid max order value for strategy {}: {}",
                    trimmed, value
                )));
            }
            let Some(decimal_value) = Decimal::from_f64(value) else {
                return Err(PyValueError::new_err(format!(
                    "invalid max order value for strategy {}: {}",
                    trimmed, value
                )));
            };
            normalized.insert(trimmed.to_string(), decimal_value);
        }
        self.strategy_max_order_value_limits = normalized;
        Ok(())
    }

    fn set_strategy_max_order_size_limits(&mut self, limits: HashMap<String, f64>) -> PyResult<()> {
        let mut normalized = HashMap::new();
        for (strategy_id, value) in limits {
            let trimmed = strategy_id.trim();
            if trimmed.is_empty() {
                return Err(PyValueError::new_err("strategy id cannot be empty"));
            }
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "invalid max order size for strategy {}: {}",
                    trimmed, value
                )));
            }
            let Some(decimal_value) = Decimal::from_f64(value) else {
                return Err(PyValueError::new_err(format!(
                    "invalid max order size for strategy {}: {}",
                    trimmed, value
                )));
            };
            normalized.insert(trimmed.to_string(), decimal_value);
        }
        self.strategy_max_order_size_limits = normalized;
        Ok(())
    }

    fn set_strategy_max_position_size_limits(
        &mut self,
        limits: HashMap<String, f64>,
    ) -> PyResult<()> {
        let mut normalized = HashMap::new();
        for (strategy_id, value) in limits {
            let trimmed = strategy_id.trim();
            if trimmed.is_empty() {
                return Err(PyValueError::new_err("strategy id cannot be empty"));
            }
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "invalid max position size for strategy {}: {}",
                    trimmed, value
                )));
            }
            let Some(decimal_value) = Decimal::from_f64(value) else {
                return Err(PyValueError::new_err(format!(
                    "invalid max position size for strategy {}: {}",
                    trimmed, value
                )));
            };
            normalized.insert(trimmed.to_string(), decimal_value);
        }
        self.strategy_max_position_size_limits = normalized;
        Ok(())
    }

    fn set_strategy_max_daily_loss_limits(&mut self, limits: HashMap<String, f64>) -> PyResult<()> {
        let mut normalized = HashMap::new();
        for (strategy_id, value) in limits {
            let trimmed = strategy_id.trim();
            if trimmed.is_empty() {
                return Err(PyValueError::new_err("strategy id cannot be empty"));
            }
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "invalid max daily loss for strategy {}: {}",
                    trimmed, value
                )));
            }
            let Some(decimal_value) = Decimal::from_f64(value) else {
                return Err(PyValueError::new_err(format!(
                    "invalid max daily loss for strategy {}: {}",
                    trimmed, value
                )));
            };
            normalized.insert(trimmed.to_string(), decimal_value);
        }
        self.strategy_max_daily_loss_limits = normalized;
        Ok(())
    }

    fn set_strategy_max_drawdown_limits(&mut self, limits: HashMap<String, f64>) -> PyResult<()> {
        let mut normalized = HashMap::new();
        for (strategy_id, value) in limits {
            let trimmed = strategy_id.trim();
            if trimmed.is_empty() {
                return Err(PyValueError::new_err("strategy id cannot be empty"));
            }
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "invalid max drawdown for strategy {}: {}",
                    trimmed, value
                )));
            }
            let Some(decimal_value) = Decimal::from_f64(value) else {
                return Err(PyValueError::new_err(format!(
                    "invalid max drawdown for strategy {}: {}",
                    trimmed, value
                )));
            };
            normalized.insert(trimmed.to_string(), decimal_value);
        }
        self.strategy_max_drawdown_limits = normalized;
        Ok(())
    }

    fn set_strategy_reduce_only_after_risk(
        &mut self,
        flags: HashMap<String, bool>,
    ) -> PyResult<()> {
        let mut enabled = std::collections::HashSet::new();
        for (strategy_id, flag) in flags {
            let trimmed = strategy_id.trim();
            if trimmed.is_empty() {
                return Err(PyValueError::new_err("strategy id cannot be empty"));
            }
            if flag {
                enabled.insert(trimmed.to_string());
            }
        }
        self.strategy_reduce_only_after_risk = enabled;
        Ok(())
    }

    fn set_strategy_risk_cooldown_bars(&mut self, bars: HashMap<String, usize>) -> PyResult<()> {
        let mut normalized = HashMap::new();
        for (strategy_id, cooldown_bars) in bars {
            let trimmed = strategy_id.trim();
            if trimmed.is_empty() {
                return Err(PyValueError::new_err("strategy id cannot be empty"));
            }
            normalized.insert(trimmed.to_string(), cooldown_bars);
        }
        self.strategy_risk_cooldown_bars = normalized;
        Ok(())
    }

    fn set_strategy_priorities(&mut self, priorities: HashMap<String, i32>) -> PyResult<()> {
        let mut normalized = HashMap::new();
        for (strategy_id, priority) in priorities {
            let trimmed = strategy_id.trim();
            if trimmed.is_empty() {
                return Err(PyValueError::new_err("strategy id cannot be empty"));
            }
            normalized.insert(trimmed.to_string(), priority);
        }
        self.strategy_priorities = normalized;
        Ok(())
    }

    fn set_strategy_risk_budget_limits(&mut self, limits: HashMap<String, f64>) -> PyResult<()> {
        let mut normalized = HashMap::new();
        for (strategy_id, value) in limits {
            let trimmed = strategy_id.trim();
            if trimmed.is_empty() {
                return Err(PyValueError::new_err("strategy id cannot be empty"));
            }
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "invalid risk budget for strategy {}: {}",
                    trimmed, value
                )));
            }
            let Some(decimal_value) = Decimal::from_f64(value) else {
                return Err(PyValueError::new_err(format!(
                    "invalid risk budget for strategy {}: {}",
                    trimmed, value
                )));
            };
            normalized.insert(trimmed.to_string(), decimal_value);
        }
        self.strategy_risk_budget_limits = normalized;
        Ok(())
    }

    fn set_portfolio_risk_budget_limit(&mut self, limit: Option<f64>) -> PyResult<()> {
        if let Some(value) = limit {
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "invalid portfolio risk budget: {}",
                    value
                )));
            }
            let Some(decimal_value) = Decimal::from_f64(value) else {
                return Err(PyValueError::new_err(format!(
                    "invalid portfolio risk budget: {}",
                    value
                )));
            };
            self.portfolio_risk_budget_limit = Some(decimal_value);
            return Ok(());
        }
        self.portfolio_risk_budget_limit = None;
        Ok(())
    }

    fn set_risk_budget_mode(&mut self, mode: &str) -> PyResult<()> {
        let normalized = mode.trim().to_ascii_lowercase();
        if normalized != "order_notional" && normalized != "trade_notional" {
            return Err(PyValueError::new_err(
                "risk budget mode must be order_notional or trade_notional",
            ));
        }
        self.risk_budget_mode = normalized;
        Ok(())
    }

    fn set_risk_budget_reset_daily(&mut self, enabled: bool) {
        self.risk_budget_reset_daily = enabled;
    }

    /// 设置实时会话的墙钟截止时刻(纳秒 UTC epoch); None 表示不限
    ///
    /// 到点后主循环**自行结束**, 不依赖行情事件到达 —— 这是真正的墙钟兜底。
    /// 只影响 live 会话; 回测由数据长度决定终点, 不受此影响。
    fn set_session_deadline_ns(&mut self, deadline_ns: Option<i64>) {
        self.session_deadline_ns = deadline_ns.filter(|value| *value > 0);
    }

    /// 取一个外部信号注入端口(paper 模式)
    ///
    /// **必须在 `run()` 之前调用**: 取到的是 channel sender 的克隆, 之后即与引擎
    /// 对象解耦, 可安全交给任意线程。会话开始后再调会撞
    /// `RuntimeError: Already borrowed`(`run(&mut self)` 独占借用引擎)。
    ///
    /// 仅适用于 `trading_mode='paper'`: broker_live 下 `RealtimeExecutionClient`
    /// 不报柜台, 经此注入的订单会过风控进 `active_orders` 但永不成交。
    ///
    /// :param strategy_id: 归属策略 id, 风控限额按它路由(缺省 ``_default``)
    /// :return: SignalPort 实例
    #[pyo3(signature = (strategy_id=None))]
    fn signal_port(&self, strategy_id: Option<&str>) -> crate::signal_port::SignalPort {
        let owner = strategy_id
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("_default")
            .to_string();
        crate::signal_port::SignalPort::new(self.event_manager.sender(), owner)
    }

    /// 初始化回测引擎.
    ///
    /// :return: Engine 实例
    #[new]
    pub fn new() -> Self {
        let initial_cash = Decimal::from(100_000);
        Engine {
            state: SharedState::new(initial_cash),
            last_prices: Arc::new(RwLock::new(HashMap::new())),
            instruments: HashMap::new(),
            current_date: None,
            market_manager: MarketManager::new(),
            corporate_action_manager: CorporateActionManager::new(),
            execution_model: Box::new(SimulatedExecutionClient::new()),
            execution_policy_core_state: ExecutionPolicyCore::default(),
            clock: Clock::new(),
            timers: BinaryHeap::new(),
            force_session_continuous: false,
            risk_manager: RiskManager::new(),
            timezone_offset: 28800, // Default UTC+8
            timezone_name: Some("Asia/Shanghai".to_string()),
            days_per_year: 252.0,
            risk_free_rate: 0.0,
            history_buffer: Arc::new(RwLock::new(HistoryBuffer::new(10000))), // Default large capacity for MAE/MFE
            initial_cash,
            active_start_time_ns: None,
            event_manager: EventManager::new(),
            statistics_manager: StatisticsManager::new(),
            settlement_manager: SettlementManager::new(),
            current_event: None,
            bar_count: 0,
            symbol_whitelist: None,
            progress_total_steps: 0,
            progress_bar: None,
            strategy_contexts: vec![None],
            strategy_slot_strategies: vec![None],
            strategy_slots: vec![StrategySlot {
                strategy_id: "_default".to_string(),
            }],
            active_strategy_slot: 0,
            default_strategy_id: Some("_default".to_string()),
            strategy_priorities: HashMap::new(),
            strategy_risk_budget_limits: HashMap::new(),
            portfolio_risk_budget_limit: None,
            strategy_risk_budget_used: HashMap::new(),
            portfolio_risk_budget_used: Decimal::ZERO,
            risk_budget_mode: "order_notional".to_string(),
            risk_budget_reset_daily: false,
            session_deadline_ns: None,
            risk_budget_usage_day: None,
            strategy_max_order_value_limits: HashMap::new(),
            strategy_max_order_size_limits: HashMap::new(),
            strategy_max_position_size_limits: HashMap::new(),
            strategy_max_daily_loss_limits: HashMap::new(),
            strategy_max_drawdown_limits: HashMap::new(),
            strategy_risk_cooldown_bars: HashMap::new(),
            strategy_risk_cooldown_until_bar: HashMap::new(),
            strategy_reduce_only_after_risk: std::collections::HashSet::new(),
            strategy_positions: HashMap::new(),
            strategy_cashflows: HashMap::new(),
            strategy_daily_loss_day: HashMap::new(),
            strategy_daily_loss_baseline_pnl: HashMap::new(),
            strategy_last_pnl: HashMap::new(),
            strategy_peak_pnl: HashMap::new(),
            strategy_reduce_only_active: std::collections::HashSet::new(),
            margin_accrued_interest: Decimal::ZERO,
            margin_daily_interest: Decimal::ZERO,
            recent_expiry_events: Vec::new(),
            snapshot_time: 0,
            stream_callback: None,
            stream_run_id: None,
            stream_seq: 0,
            stream_progress_interval: 1,
            stream_equity_interval: 1,
            stream_batch_size: 1,
            stream_max_buffer: 1024,
            stream_buffer: Vec::new(),
            stream_fail_fast_on_callback_error: false,
            stream_callback_error_count: 0,
            stream_callback_last_error: None,
            stream_fatal_error: None,
            stream_dropped_event_count: 0,
            stream_dropped_event_count_by_type: HashMap::new(),
            stream_mode: "observability".to_string(),
            stream_sampling_enabled: true,
            stream_drop_non_critical: true,
        }
    }

    /// 导出当前状态为二进制数据
    fn get_state_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let history_state = Some(crate::history::HistoryBufferSnapshot::from(
            &*self.history_buffer.read().unwrap(),
        ));
        let snapshot = crate::engine::state::EngineSnapshot {
            current_time: self.clock.timestamp().unwrap_or(0),
            portfolio: self.state.portfolio.clone(),
            order_manager: self.state.order_manager.clone(),
            last_prices: self
                .last_prices
                .read()
                .expect("last_prices 读锁被污染")
                .clone(),
            instruments: self.instruments.clone(),
            initial_cash: Some(self.initial_cash),
            margin_accrued_interest: self.margin_accrued_interest,
            margin_daily_interest: self.margin_daily_interest,
            history_state,
            strategy_risk_state: crate::engine::state::StrategyRiskStateSnapshot {
                default_strategy_id: self.default_strategy_id.clone(),
                strategy_slots: self
                    .strategy_slots
                    .iter()
                    .map(|slot| slot.strategy_id.clone())
                    .collect(),
                active_strategy_slot: self.active_strategy_slot,
                strategy_priorities: self.strategy_priorities.clone(),
                strategy_risk_budget_limits: self.strategy_risk_budget_limits.clone(),
                portfolio_risk_budget_limit: self.portfolio_risk_budget_limit,
                strategy_risk_budget_used: self.strategy_risk_budget_used.clone(),
                portfolio_risk_budget_used: self.portfolio_risk_budget_used,
                risk_budget_mode: self.risk_budget_mode.clone(),
                risk_budget_reset_daily: self.risk_budget_reset_daily,
                risk_budget_usage_day: self.risk_budget_usage_day,
                strategy_max_order_value_limits: self.strategy_max_order_value_limits.clone(),
                strategy_max_order_size_limits: self.strategy_max_order_size_limits.clone(),
                strategy_max_position_size_limits: self.strategy_max_position_size_limits.clone(),
                strategy_max_daily_loss_limits: self.strategy_max_daily_loss_limits.clone(),
                strategy_max_drawdown_limits: self.strategy_max_drawdown_limits.clone(),
                strategy_risk_cooldown_bars: self.strategy_risk_cooldown_bars.clone(),
                strategy_risk_cooldown_until_bar: self.strategy_risk_cooldown_until_bar.clone(),
                strategy_reduce_only_after_risk: self.strategy_reduce_only_after_risk.clone(),
                strategy_positions: self.strategy_positions.clone(),
                strategy_cashflows: self.strategy_cashflows.clone(),
                strategy_daily_loss_day: self.strategy_daily_loss_day.clone(),
                strategy_daily_loss_baseline_pnl: self.strategy_daily_loss_baseline_pnl.clone(),
                strategy_last_pnl: self.strategy_last_pnl.clone(),
                strategy_peak_pnl: self.strategy_peak_pnl.clone(),
                strategy_reduce_only_active: self.strategy_reduce_only_active.clone(),
            },
        };
        let bytes =
            rmp_serde::to_vec(&snapshot).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// 从二进制数据加载状态
    fn load_state_bytes(&mut self, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        let bytes = data.as_bytes();
        let snapshot: crate::engine::state::EngineSnapshot =
            rmp_serde::from_slice(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;

        self.state.portfolio = snapshot.portfolio;
        self.state.order_manager = snapshot.order_manager;
        *self
            .last_prices
            .write()
            .expect("last_prices 写锁被污染") = snapshot.last_prices;
        self.instruments = snapshot.instruments;
        self.initial_cash = snapshot.initial_cash.unwrap_or(self.state.portfolio.cash);
        self.margin_accrued_interest = snapshot.margin_accrued_interest;
        self.margin_daily_interest = snapshot.margin_daily_interest;
        let history_buffer = snapshot
            .history_state
            .map(crate::history::HistoryBuffer::from);
        if let Some(buffer) = history_buffer {
            self.history_buffer = Arc::new(RwLock::new(buffer));
        }
        self.snapshot_time = snapshot.current_time;
        self.default_strategy_id = snapshot.strategy_risk_state.default_strategy_id;
        self.strategy_slots = snapshot
            .strategy_risk_state
            .strategy_slots
            .into_iter()
            .map(|strategy_id| StrategySlot { strategy_id })
            .collect();
        self.active_strategy_slot = snapshot.strategy_risk_state.active_strategy_slot;
        self.strategy_priorities = snapshot.strategy_risk_state.strategy_priorities;
        self.strategy_risk_budget_limits = snapshot.strategy_risk_state.strategy_risk_budget_limits;
        self.portfolio_risk_budget_limit = snapshot.strategy_risk_state.portfolio_risk_budget_limit;
        self.strategy_risk_budget_used = snapshot.strategy_risk_state.strategy_risk_budget_used;
        self.portfolio_risk_budget_used = snapshot.strategy_risk_state.portfolio_risk_budget_used;
        self.risk_budget_mode = snapshot.strategy_risk_state.risk_budget_mode;
        self.risk_budget_reset_daily = snapshot.strategy_risk_state.risk_budget_reset_daily;
        self.risk_budget_usage_day = snapshot.strategy_risk_state.risk_budget_usage_day;
        self.strategy_max_order_value_limits =
            snapshot.strategy_risk_state.strategy_max_order_value_limits;
        self.strategy_max_order_size_limits =
            snapshot.strategy_risk_state.strategy_max_order_size_limits;
        self.strategy_max_position_size_limits = snapshot
            .strategy_risk_state
            .strategy_max_position_size_limits;
        self.strategy_max_daily_loss_limits =
            snapshot.strategy_risk_state.strategy_max_daily_loss_limits;
        self.strategy_max_drawdown_limits =
            snapshot.strategy_risk_state.strategy_max_drawdown_limits;
        self.strategy_risk_cooldown_bars = snapshot.strategy_risk_state.strategy_risk_cooldown_bars;
        self.strategy_risk_cooldown_until_bar = snapshot
            .strategy_risk_state
            .strategy_risk_cooldown_until_bar;
        self.strategy_reduce_only_after_risk =
            snapshot.strategy_risk_state.strategy_reduce_only_after_risk;
        self.strategy_positions = snapshot.strategy_risk_state.strategy_positions;
        self.strategy_cashflows = snapshot.strategy_risk_state.strategy_cashflows;
        self.strategy_daily_loss_day = snapshot.strategy_risk_state.strategy_daily_loss_day;
        self.strategy_daily_loss_baseline_pnl = snapshot
            .strategy_risk_state
            .strategy_daily_loss_baseline_pnl;
        self.strategy_last_pnl = snapshot.strategy_risk_state.strategy_last_pnl;
        self.strategy_peak_pnl = snapshot.strategy_risk_state.strategy_peak_pnl;
        self.strategy_reduce_only_active = snapshot.strategy_risk_state.strategy_reduce_only_active;
        self.ensure_strategy_slot_exists();

        Ok(())
    }

    /// 设置历史数据长度
    ///
    /// :param depth: 历史数据长度
    fn set_history_depth(&mut self, depth: usize) {
        self.history_buffer.write().unwrap().set_capacity(depth);
    }

    fn configure_history_depth(&mut self, depth: usize) {
        self.history_buffer
            .write()
            .unwrap()
            .set_capacity_preserve_existing(depth);
    }

    /// 设置标的白名单: 只有集合内的标的会被分发给策略。
    ///
    /// 空列表折叠为「未设置」(放行全部) —— Python 侧已在参数解析阶段拒绝空
    /// `symbols`, 此处是防御, 不能让空集合变成「什么都不放行」的静默空回测。
    ///
    /// :param symbols: 白名单标的列表
    fn set_symbol_whitelist(&mut self, symbols: Vec<String>) {
        self.symbol_whitelist = if symbols.is_empty() {
            None
        } else {
            Some(symbols.into_iter().collect())
        };
    }

    /// 设置时区偏移 (秒)
    ///
    /// :param offset: 偏移秒数 (例如 UTC+8 为 28800)
    pub fn set_timezone(&mut self, offset: i32) {
        self.timezone_offset = offset;
        self.timezone_name = None;
    }

    /// 设置时区名称.
    ///
    /// :param timezone: IANA 时区名称 (例如 "Asia/Shanghai", "UTC", "US/Eastern")
    pub fn set_timezone_name(&mut self, timezone: &str) -> PyResult<()> {
        let tz_name = timezone.trim();
        let tz: Tz = tz_name
            .parse()
            .map_err(|_| PyValueError::new_err(format!("Invalid timezone: {}", timezone)))?;
        self.timezone_offset = Utc::now()
            .with_timezone(&tz)
            .offset()
            .fix()
            .local_minus_utc();
        self.timezone_name = Some(tz_name.to_string());
        Ok(())
    }

    /// 设置年化天数因子.
    ///
    /// :param days: 年化使用的天数。A 股等传统市场用 252，数字货币 24/7 市场用 365。
    pub fn set_days_per_year(&mut self, days: f64) {
        self.days_per_year = days;
    }

    /// 设置年化无风险利率.
    ///
    /// :param rate: 年化无风险利率 (例如 0.02 表示 2%)。默认 0.0。
    pub fn set_risk_free_rate(&mut self, rate: f64) {
        self.risk_free_rate = rate;
    }

    /// 启用模拟执行 (回测模式)
    ///
    /// 默认模式。在内存中撮合订单。
    fn use_simulated_execution(&mut self) {
        self.execution_model = Box::new(SimulatedExecutionClient::new());
    }

    /// 启用实盘执行 (CTP/Broker 模式)
    ///
    /// 模拟对接 CTP 或其他 Broker API。
    /// 在此模式下，订单会被标记为 Submitted 并等待回调 (目前仅模拟发送)。
    fn use_realtime_execution(&mut self) {
        self.execution_model = Box::new(RealtimeExecutionClient::new());
    }

    /// 注册自定义撮合器 (Python)
    ///
    /// :param asset_type: 资产类型
    /// :param matcher: Python 撮合器对象 (需实现 match 方法)
    fn register_custom_matcher(&mut self, asset_type: crate::model::AssetType, matcher: Py<PyAny>) {
        use crate::execution::PyExecutionMatcher;
        let py_matcher = Box::new(PyExecutionMatcher::new(matcher));
        self.execution_model
            .register_matcher(asset_type, py_matcher);
    }

    fn set_fill_mode(&mut self, mode: ExecutionMode, timer_timing: &str) -> PyResult<()> {
        let timing = timer_timing.trim().to_ascii_lowercase();
        if timing != "same_cycle" && timing != "next_event" {
            return Err(PyValueError::new_err(
                "timer_timing must be one of: same_cycle, next_event",
            ));
        }
        self.set_execution_policy_core(ExecutionPolicyCore::from_legacy(mode, &timing));
        Ok(())
    }

    fn get_fill_policy(&self) -> (String, u8, String) {
        let policy = self.execution_policy_core();
        let basis = match policy.price_basis {
            PriceBasis::Open => "open",
            PriceBasis::Close => "close",
            PriceBasis::Ohlc4 => "ohlc4",
            PriceBasis::Hl2 => "hl2",
        }
        .to_string();
        (
            basis,
            policy.bar_offset,
            policy.temporal_as_str().to_string(),
        )
    }

    /// 启用 SimpleMarket (7x24小时, T+0, 无税, 简单佣金)
    ///
    /// :param commission_rate: 佣金率
    fn use_simple_market(&mut self, commission_rate: f64) {
        self.market_manager.use_simple_market(commission_rate);
    }

    fn use_simple_market_policy(&mut self, commission_type: String, commission_value: f64) {
        self.market_manager
            .use_simple_market_policy(commission_type, commission_value);
    }

    /// 启用 ChinaMarket (支持 T+1/T+0, 印花税, 过户费, 交易时段等)
    fn use_china_market(&mut self) {
        self.market_manager.use_china_market();
    }

    /// 启用/禁用 T+1 交易规则 (仅针对 ChinaMarket)
    ///
    /// :param enabled: 是否启用 T+1
    /// :type enabled: bool
    fn set_t_plus_one(&mut self, enabled: bool) {
        self.market_manager.set_t_plus_one(enabled);
    }

    /// 强制连续交易时段
    ///
    /// :param enabled: 是否强制连续交易 (忽略午休等)
    fn set_force_session_continuous(&mut self, enabled: bool) {
        self.force_session_continuous = enabled;
    }

    /// 添加公司行为 (除权除息)
    ///
    /// :param action: 公司行为对象
    fn add_corporate_action(&mut self, action: CorporateAction) {
        self.corporate_action_manager.add(action);
    }

    /// 启用中国期货市场默认配置
    /// - 切换到 ChinaMarket
    /// - 设置 T+0
    /// - 保持当前交易时段配置 (需手动设置 set_market_sessions 以匹配特定品种)
    fn use_china_futures_market(&mut self) {
        self.market_manager.use_china_futures_market();
    }

    fn process_option_expiry(&mut self, _local_date: NaiveDate) {
        // Deprecated: logic moved to SettlementManager
    }

    /// 设置股票费率规则 (按交易金额比例)
    ///
    /// :param commission_rate: 佣金率 (如 0.0003)
    /// :param stamp_tax: 印花税率 (如 0.001)
    /// :param transfer_fee: 过户费率 (如 0.00002)
    /// :param min_commission: 最低佣金 (如 5.0)
    pub fn set_stock_fee_rules(
        &mut self,
        commission_rate: f64,
        stamp_tax: f64,
        transfer_fee: f64,
        min_commission: f64,
    ) {
        self.market_manager.set_stock_fee_rules(
            commission_rate,
            stamp_tax,
            transfer_fee,
            min_commission,
        );
    }

    pub fn set_stock_fee_policy(
        &mut self,
        commission_type: String,
        commission_value: f64,
        stamp_tax: f64,
        transfer_fee: f64,
        min_commission: f64,
    ) {
        self.market_manager.set_stock_fee_policy(
            commission_type,
            commission_value,
            stamp_tax,
            transfer_fee,
            min_commission,
        );
    }

    /// 设置期货费率规则
    ///
    /// :param commission_rate: 佣金率 (如 0.0001)
    pub fn set_futures_fee_rules(&mut self, commission_rate: f64) {
        self.market_manager.set_futures_fee_rules(commission_rate);
    }

    pub fn set_futures_fee_rules_by_prefix(&mut self, symbol_prefix: String, commission_rate: f64) {
        self.market_manager
            .set_futures_fee_rules_by_prefix(symbol_prefix, commission_rate);
    }

    fn set_futures_validation_options(&mut self, enforce_tick_size: bool, enforce_lot_size: bool) {
        self.execution_model
            .set_futures_validation_options(enforce_tick_size, enforce_lot_size);
    }

    fn set_futures_validation_options_by_prefix(
        &mut self,
        symbol_prefix: String,
        enforce_tick_size: Option<bool>,
        enforce_lot_size: Option<bool>,
    ) {
        self.execution_model
            .set_futures_validation_options_by_prefix(
                symbol_prefix,
                enforce_tick_size,
                enforce_lot_size,
            );
    }

    fn set_stock_validation_options(&mut self, enforce_tick_size: bool) {
        self.execution_model
            .set_stock_validation_options(enforce_tick_size);
    }

    /// 设置基金费率规则
    ///
    /// :param commission_rate: 佣金率
    /// :param transfer_fee: 过户费率
    /// :param min_commission: 最低佣金
    fn set_fund_fee_rules(&mut self, commission_rate: f64, transfer_fee: f64, min_commission: f64) {
        self.market_manager
            .set_fund_fee_rules(commission_rate, transfer_fee, min_commission);
    }

    /// 设置期权费率规则
    ///
    /// :param commission_per_contract: 每张合约佣金 (如 5.0)
    fn set_option_fee_rules(&mut self, commission_per_contract: f64) {
        self.market_manager
            .set_option_fee_rules(commission_per_contract);
    }

    fn set_options_fee_rules_by_prefix(
        &mut self,
        symbol_prefix: String,
        commission_per_contract: f64,
    ) {
        self.market_manager
            .set_options_fee_rules_by_prefix(symbol_prefix, commission_per_contract);
    }

    /// 设置加密货币费率规则 (按金额比例)
    ///
    /// :param commission_rate: 佣金率 (如 0.001)
    fn set_crypto_fee_rules(&mut self, commission_rate: f64) {
        // Crypto fees are usually percentage based, similar to Stock/Future
        // For now, reuse stock fee rules logic or future logic?
        // Let's assume simple percentage fee for now.
        // Actually, we should probably add specific methods in MarketManager.
        // For now, let's map it to stock rules but with zero tax/transfer fee.
        self.market_manager
            .set_stock_fee_rules(commission_rate, 0.0, 0.0, 0.0);
    }

    /// 设置外汇费率规则 (按金额比例)
    ///
    /// :param commission_rate: 佣金率 (如 0.00005)
    fn set_forex_fee_rules(&mut self, commission_rate: f64) {
        self.market_manager
            .set_stock_fee_rules(commission_rate, 0.0, 0.0, 0.0);
    }

    /// 设置滑点模型
    ///
    /// :param type: 滑点类型 ("fixed" 或 "percent")
    /// :param value: 滑点值 (固定金额 或 百分比如 0.001)
    fn set_slippage(&mut self, type_: String, value: f64) -> PyResult<()> {
        let val = Decimal::from_f64(value).unwrap_or(Decimal::ZERO);
        match type_.as_str() {
            "fixed" => {
                self.execution_model
                    .set_slippage_model(Box::new(crate::execution::FixedSlippage { delta: val }));
            }
            "percent" => {
                self.execution_model
                    .set_slippage_model(Box::new(crate::execution::PercentSlippage { rate: val }));
            }
            "zero" | "none" => {
                self.execution_model
                    .set_slippage_model(Box::new(crate::execution::ZeroSlippage));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Invalid slippage type. Use 'fixed', 'percent', or 'zero'",
                ));
            }
        }
        Ok(())
    }

    /// 设置成交量限制
    ///
    /// :param limit: 限制比例 (0.0-1.0), 0.0 为不限制
    fn set_volume_limit(&mut self, limit: f64) {
        self.execution_model.set_volume_limit(limit);
    }

    /// 设置市场交易时段
    ///
    /// :param sessions: 交易时段列表，每个元素为 (开始时间, 结束时间, 时段类型)
    /// :type sessions: List[Tuple[str, str, TradingSession]]
    ///
    /// 示例::
    ///
    /// ```python
    /// engine.set_market_sessions([
    ///     ("09:30:00", "11:30:00", TradingSession.Normal),
    ///     ("13:00:00", "15:00:00", TradingSession.Normal)
    /// ])
    /// ```
    fn set_market_sessions(
        &mut self,
        sessions: Vec<(String, String, TradingSession)>,
    ) -> PyResult<()> {
        let mut ranges = Vec::with_capacity(sessions.len());
        for (start, end, session) in sessions {
            let start_time = Self::parse_time_string(&start)?;
            let end_time = Self::parse_time_string(&end)?;
            ranges.push((start_time, end_time, session));
        }
        self.market_manager.set_market_sessions(ranges);
        Ok(())
    }

    /// 添加交易标的
    ///
    /// :param instrument: 交易标的对象
    /// :type instrument: Instrument
    pub fn add_instrument(&mut self, instrument: Instrument) {
        self.instruments
            .insert(instrument.symbol().to_string(), instrument);
    }

    /// 设置初始资金
    ///
    /// :param cash: 初始资金数额
    /// :type cash: float
    pub fn set_cash(&mut self, cash: f64) {
        let val = Decimal::from_f64(cash).unwrap_or(Decimal::ZERO);
        self.state.portfolio.cash = val;
        self.initial_cash = val;
    }

    /// 设置绩效统计使用的初始基线资金，不修改当前组合现金
    ///
    /// :param cash: 作为绩效基线的初始权益
    /// :type cash: float
    pub fn set_initial_cash_reference(&mut self, cash: f64) {
        self.initial_cash = Decimal::from_f64(cash).unwrap_or(Decimal::ZERO);
    }

    /// 添加数据源
    ///
    /// :param feed: 数据源对象
    /// :type feed: DataFeed
    fn add_data(&mut self, feed: DataFeed) {
        self.state.feed = feed;
    }

    /// 批量添加 K 线数据
    ///
    /// :param bars: K 线列表
    fn add_bars(&mut self, bars: Vec<Bar>) -> PyResult<()> {
        self.state.feed.add_bars(bars)
    }

    /// 运行回测
    ///
    /// :param strategy: 策略对象
    /// :param show_progress: 是否显示进度条
    /// :type strategy: object
    /// :type show_progress: bool
    /// :return: 回测结果摘要
    /// :rtype: str
    fn run(
        &mut self,
        py: Python<'_>,
        strategy: &Bound<'_, PyAny>,
        show_progress: bool,
    ) -> PyResult<String> {
        // Configure history buffer if strategy has _history_depth set
        if let Ok(depth_attr) = strategy.getattr("_history_depth")
            && let Ok(depth) = depth_attr.extract::<usize>()
            && depth > 0
        {
            self.configure_history_depth(depth);
        }

        if strategy.hasattr("_on_start_internal")? {
            strategy.call_method0("_on_start_internal")?;
        } else {
            strategy.call_method0("on_start")?;
        }

        // Progress Bar Initialization
        let total_progress_steps = self.state.feed.progress_len_hint().unwrap_or(0);
        let pb = if show_progress {
            let pb = if total_progress_steps > 0 {
                let pb = ProgressBar::new(total_progress_steps as u64);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template(
                            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                        )
                        .unwrap()
                        .progress_chars("#>-"),
                );
                pb
            } else {
                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} [{elapsed_precise}] {pos} events processed")
                        .unwrap(),
                );
                pb
            };
            Some(pb)
        } else {
            None
        };
        self.progress_bar = pb;
        self.progress_total_steps = total_progress_steps;
        self.start_stream_run(py, Some(total_progress_steps));

        // Record initial equity
        if self.state.feed.peek_timestamp().is_some() {
            let prices = self.last_prices.read().expect("last_prices 读锁被污染");
            let _equity = self.state.portfolio.calculate_equity(&prices, &self.instruments);
        }

        // Initialize Pipeline
        let mut pipeline = self.build_pipeline();

        // Run Pipeline
        if let Err(e) = pipeline.run(self, py, strategy) {
            let mut payload: HashMap<String, String> = HashMap::new();
            payload.insert("message".to_string(), e.to_string());
            super::core::Engine::emit_stream_event_owned(self, py, "error", None, "error", payload);
            self.finish_stream_run(py, "failed", Some(&e.to_string()));
            // Clean up pb if error
            self.progress_bar = None;
            return Err(e);
        }

        self.flush_terminal_pending_order_events(py, strategy)?;

        // Final cleanup
        self.state.order_manager.cleanup_finished_orders();

        // Record final snapshot if we have data
        if self.current_date.is_some()
            && let Some(timestamp) = self.terminal_result_timestamp()
        {
            let prices = self.last_prices.read().expect("last_prices 读锁被污染");
            self.statistics_manager.record_snapshot(
                timestamp,
                &self.state.portfolio,
                &self.instruments,
                &prices,
                &self.state.order_manager.trade_tracker,
                &self.risk_manager.config,
            );
        }

        if let Some(pb) = &self.progress_bar {
            pb.finish_with_message("Backtest completed");
        }

        let count = self.bar_count;
        self.finish_stream_run(py, "completed", None);
        self.progress_bar = None;

        Ok(format!(
            "Backtest finished. Processed {} events. Total Trades: {}",
            count,
            self.state.order_manager.trades.len()
        ))
    }

    /// 获取回测结果
    ///
    /// :return: BacktestResult
    fn get_results(&self) -> BacktestResult {
        let prices = self.last_prices.read().expect("last_prices 读锁被污染");
        self.statistics_manager.generate_backtest_result(
            &self.state.portfolio,
            &self.instruments,
            &prices,
            &self.state.order_manager,
            &self.risk_manager.config,
            self.initial_cash,
            self.terminal_result_timestamp(),
            self.timezone_name.clone(),
            self.timezone_offset,
            self.days_per_year,
            self.risk_free_rate,
        )
    }
}
