use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::data::DataFeed;
use crate::history::HistoryBufferSnapshot;
use crate::model::Instrument;
use crate::order_manager::OrderManager;
use crate::portfolio::Portfolio;

/// Shared state container for Engine
pub struct SharedState {
    pub portfolio: Portfolio,
    pub order_manager: OrderManager,
    pub feed: DataFeed,
}

impl SharedState {
    pub fn new(initial_cash: Decimal) -> Self {
        Self {
            portfolio: Portfolio {
                cash: initial_cash,
                positions: Arc::new(HashMap::new()),
                available_positions: Arc::new(HashMap::new()),
            },
            order_manager: OrderManager::new(),
            feed: DataFeed::new(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct EngineSnapshot {
    pub current_time: i64,
    pub portfolio: Portfolio,
    pub order_manager: OrderManager,
    #[serde(default)]
    pub last_prices: HashMap<String, Decimal>,
    #[serde(default)]
    pub instruments: HashMap<String, Instrument>,
    #[serde(default)]
    pub initial_cash: Option<Decimal>,
    #[serde(default)]
    pub margin_accrued_interest: Decimal,
    #[serde(default)]
    pub margin_daily_interest: Decimal,
    #[serde(default)]
    pub history_state: Option<HistoryBufferSnapshot>,
    #[serde(default)]
    pub window_aggregator_state: Option<crate::data::WindowAggregatorSnapshot>,
    #[serde(default)]
    pub strategy_risk_state: StrategyRiskStateSnapshot,
}

#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(default)]
pub struct StrategyRiskStateSnapshot {
    pub default_strategy_id: Option<String>,
    pub strategy_slots: Vec<String>,
    pub active_strategy_slot: usize,
    pub strategy_priorities: HashMap<String, i32>,
    pub strategy_risk_budget_limits: HashMap<String, Decimal>,
    pub portfolio_risk_budget_limit: Option<Decimal>,
    pub strategy_risk_budget_used: HashMap<String, Decimal>,
    pub portfolio_risk_budget_used: Decimal,
    pub risk_budget_mode: String,
    pub risk_budget_reset_daily: bool,
    pub risk_budget_usage_day: Option<NaiveDate>,
    pub strategy_max_order_value_limits: HashMap<String, Decimal>,
    pub strategy_max_order_size_limits: HashMap<String, Decimal>,
    pub strategy_max_position_size_limits: HashMap<String, Decimal>,
    pub strategy_max_daily_loss_limits: HashMap<String, Decimal>,
    pub strategy_max_drawdown_limits: HashMap<String, Decimal>,
    pub strategy_risk_cooldown_bars: HashMap<String, usize>,
    pub strategy_risk_cooldown_until_bar: HashMap<String, usize>,
    pub strategy_reduce_only_after_risk: HashSet<String>,
    pub strategy_positions: HashMap<String, HashMap<String, Decimal>>,
    pub strategy_cashflows: HashMap<String, Decimal>,
    pub strategy_daily_loss_day: HashMap<String, NaiveDate>,
    pub strategy_daily_loss_baseline_pnl: HashMap<String, Decimal>,
    pub strategy_last_pnl: HashMap<String, Decimal>,
    pub strategy_peak_pnl: HashMap<String, Decimal>,
    pub strategy_reduce_only_active: HashSet<String>,
}
