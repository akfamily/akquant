use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::account::{
    calculate_account_metrics, estimate_futures_realized_pnl, is_futures_margin_account,
};
use crate::analysis::TradeTracker;
use crate::history::HistoryBuffer;
use crate::market::MarketModel;
use crate::model::{
    Instrument, Order, OrderRole, OrderSide, OrderStatus, OrderType, TimeInForce, Trade,
};
use crate::portfolio::Portfolio;
use crate::risk::RiskConfig;

/// 订单管理器
/// 负责管理订单列表、成交记录及状态流转
#[derive(Clone, Serialize, Deserialize)]
pub struct OrderManager {
    /// 历史订单 (已完成)
    pub orders: Vec<Order>,
    /// 当前活跃订单 (未完成)
    pub active_orders: Vec<Order>,
    /// 历史成交记录
    pub trades: Vec<Trade>,
    /// 当前步生成的成交 (用于通知策略)
    pub current_step_trades: Vec<Trade>,
    /// 当前步收到的拒单回报 (用于通知策略 on_reject)
    #[serde(default)]
    pub current_step_rejected_orders: Vec<Order>,
    /// 交易追踪器 (用于计算 PnL 和统计)
    pub trade_tracker: TradeTracker,
    /// OCO 订单组映射: group_id -> {order_id}
    #[serde(default)]
    pub oco_groups: HashMap<String, HashSet<String>>,
    /// OCO 订单反向索引: order_id -> group_id
    #[serde(default)]
    pub oco_order_to_group: HashMap<String, String>,
    /// Bracket 激活计划: entry_order_id -> plan
    #[serde(default)]
    pub pending_bracket_plans: HashMap<String, PendingBracketPlan>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PendingBracketPlan {
    pub stop_trigger_price: Option<Decimal>,
    pub take_profit_price: Option<Decimal>,
    pub time_in_force: TimeInForce,
    pub stop_tag: String,
    pub take_profit_tag: String,
}

impl OrderManager {
    pub fn new() -> Self {
        OrderManager {
            orders: Vec::new(),
            active_orders: Vec::new(),
            trades: Vec::new(),
            current_step_trades: Vec::new(),
            current_step_rejected_orders: Vec::new(),
            trade_tracker: TradeTracker::new(),
            oco_groups: HashMap::new(),
            oco_order_to_group: HashMap::new(),
            pending_bracket_plans: HashMap::new(),
        }
    }

    pub fn register_oco_group(
        &mut self,
        group_id: String,
        first_order_id: String,
        second_order_id: String,
    ) {
        let first = first_order_id.trim().to_string();
        let second = second_order_id.trim().to_string();
        let group_key = group_id.trim().to_string();
        if first.is_empty() || second.is_empty() || first == second || group_key.is_empty() {
            return;
        }

        self.detach_oco_order(&first);
        self.detach_oco_order(&second);

        let mut members = HashSet::new();
        members.insert(first.clone());
        members.insert(second.clone());
        self.oco_groups.insert(group_key.clone(), members);
        self.oco_order_to_group.insert(first, group_key.clone());
        self.oco_order_to_group.insert(second, group_key);
    }

    pub fn consume_oco_peer_cancels_on_fill(&mut self, order_id: &str) -> Vec<String> {
        let Some(group_id) = self.oco_order_to_group.get(order_id).cloned() else {
            return Vec::new();
        };
        let Some(group_members) = self.oco_groups.remove(&group_id) else {
            self.oco_order_to_group.remove(order_id);
            return Vec::new();
        };

        let mut peers = Vec::new();
        for member in group_members {
            self.oco_order_to_group.remove(&member);
            if member != order_id {
                peers.push(member);
            }
        }
        peers
    }

    pub fn cancel_active_order(&mut self, order_id: &str, updated_at: i64) -> Option<Order> {
        let order = self.active_orders.iter_mut().find(|o| o.id == order_id)?;
        if order.status == OrderStatus::New
            || order.status == OrderStatus::Submitted
            || order.status == OrderStatus::PartiallyFilled
        {
            order.status = OrderStatus::Cancelled;
            order.updated_at = updated_at;
            let snapshot = order.clone();
            self.handle_oco_terminal_without_fill(order_id);
            return Some(snapshot);
        }
        None
    }

    pub fn register_bracket_plan(
        &mut self,
        entry_order_id: String,
        stop_trigger_price: Option<Decimal>,
        take_profit_price: Option<Decimal>,
        time_in_force: TimeInForce,
        stop_tag: Option<String>,
        take_profit_tag: Option<String>,
    ) {
        let entry_id = entry_order_id.trim().to_string();
        if entry_id.is_empty() {
            return;
        }
        if stop_trigger_price.is_none() && take_profit_price.is_none() {
            return;
        }
        self.pending_bracket_plans.insert(
            entry_id,
            PendingBracketPlan {
                stop_trigger_price,
                take_profit_price,
                time_in_force,
                stop_tag: stop_tag.unwrap_or_default(),
                take_profit_tag: take_profit_tag.unwrap_or_default(),
            },
        );
    }

    pub fn consume_bracket_activation_on_fill(&mut self, filled_order: &Order) -> Vec<Order> {
        let Some(plan) = self.pending_bracket_plans.remove(&filled_order.id) else {
            return Vec::new();
        };

        let quantity = if filled_order.filled_quantity > Decimal::ZERO {
            filled_order.filled_quantity
        } else {
            filled_order.quantity
        };
        if quantity <= Decimal::ZERO {
            return Vec::new();
        }

        let created_at = filled_order.updated_at;
        let graph_id = format!("bracket-{}", filled_order.id);
        let mut created_orders: Vec<Order> = Vec::new();

        if let Some(stop_trigger_price) = plan.stop_trigger_price {
            created_orders.push(Order {
                id: Uuid::new_v4().to_string(),
                symbol: filled_order.symbol.clone(),
                side: OrderSide::Sell,
                order_type: OrderType::StopMarket,
                quantity,
                price: None,
                time_in_force: plan.time_in_force,
                trigger_price: Some(stop_trigger_price),
                trail_offset: None,
                trail_reference_price: None,
                fill_policy_override: filled_order.fill_policy_override,
                slippage_type_override: filled_order.slippage_type_override.clone(),
                slippage_value_override: filled_order.slippage_value_override,
                commission_type_override: filled_order.commission_type_override.clone(),
                commission_value_override: filled_order.commission_value_override,
                graph_id: Some(graph_id.clone()),
                parent_order_id: Some(filled_order.id.clone()),
                order_role: OrderRole::StopLoss,
                position_effect: filled_order.position_effect,
                status: OrderStatus::New,
                filled_quantity: Decimal::ZERO,
                average_filled_price: None,
                created_at,
                updated_at: created_at,
                commission: Decimal::ZERO,
                tag: plan.stop_tag.clone(),
                reject_reason: String::new(),
                owner_strategy_id: filled_order.owner_strategy_id.clone(),
                allow_quantity_auto_resize: false,
                reduce_only: filled_order.reduce_only,
            });
        }

        if let Some(take_profit_price) = plan.take_profit_price {
            created_orders.push(Order {
                id: Uuid::new_v4().to_string(),
                symbol: filled_order.symbol.clone(),
                side: OrderSide::Sell,
                order_type: OrderType::Limit,
                quantity,
                price: Some(take_profit_price),
                time_in_force: plan.time_in_force,
                trigger_price: None,
                trail_offset: None,
                trail_reference_price: None,
                fill_policy_override: filled_order.fill_policy_override,
                slippage_type_override: filled_order.slippage_type_override.clone(),
                slippage_value_override: filled_order.slippage_value_override,
                commission_type_override: filled_order.commission_type_override.clone(),
                commission_value_override: filled_order.commission_value_override,
                graph_id: Some(graph_id.clone()),
                parent_order_id: Some(filled_order.id.clone()),
                order_role: OrderRole::TakeProfit,
                position_effect: filled_order.position_effect,
                status: OrderStatus::New,
                filled_quantity: Decimal::ZERO,
                average_filled_price: None,
                created_at,
                updated_at: created_at,
                commission: Decimal::ZERO,
                tag: plan.take_profit_tag.clone(),
                reject_reason: String::new(),
                owner_strategy_id: filled_order.owner_strategy_id.clone(),
                allow_quantity_auto_resize: false,
                reduce_only: filled_order.reduce_only,
            });
        }

        if created_orders.len() == 2 {
            let first = created_orders[0].id.clone();
            let second = created_orders[1].id.clone();
            self.register_oco_group(format!("{}-exit-oco", graph_id), first, second);
        }
        created_orders
    }

    /// 添加新订单 (例如从 OrderValidated 事件)
    pub fn add_active_order(&mut self, order: Order) {
        self.active_orders.push(order);
    }

    /// 处理执行报告 (ExecutionReport)
    /// 更新活跃订单状态
    pub fn on_execution_report(&mut self, report: Order) {
        let mut terminal_without_fill_id: Option<String> = None;
        // Find existing order
        if let Some(existing) = self.active_orders.iter_mut().find(|o| o.id == report.id) {
            existing.status = report.status;
            existing.filled_quantity = report.filled_quantity;
            existing.average_filled_price = report.average_filled_price;
            existing.updated_at = report.updated_at;
            existing.reject_reason = report.reject_reason;
            if existing.status != OrderStatus::Filled
                && (existing.status == OrderStatus::Cancelled
                    || existing.status == OrderStatus::Expired
                    || existing.status == OrderStatus::Rejected)
            {
                terminal_without_fill_id = Some(existing.id.clone());
            }
        } else {
            // If the first report we see is already terminal/filled, keep a snapshot so
            // downstream callbacks/history can still observe the order lifecycle.
            if report.status == OrderStatus::Rejected
                || report.status == OrderStatus::New
                || report.status == OrderStatus::PartiallyFilled
                || report.status == OrderStatus::Submitted
                || report.status == OrderStatus::Filled
            {
                if report.status == OrderStatus::Rejected {
                    terminal_without_fill_id = Some(report.id.clone());
                }
                self.active_orders.push(report);
            }
        }
        if let Some(order_id) = terminal_without_fill_id {
            self.handle_oco_terminal_without_fill(&order_id);
            self.pending_bracket_plans.remove(&order_id);
        }
    }

    /// 清理已完成的订单 (Filled, Cancelled, Expired, Rejected)
    /// 将其移入历史列表
    pub fn cleanup_finished_orders(&mut self) {
        let (finished, active): (Vec<Order>, Vec<Order>) =
            self.active_orders.drain(..).partition(|o| {
                o.status == OrderStatus::Filled
                    || o.status == OrderStatus::Cancelled
                    || o.status == OrderStatus::Expired
                    || o.status == OrderStatus::Rejected
            });

        for order in &finished {
            if order.status != OrderStatus::Filled {
                self.handle_oco_terminal_without_fill(&order.id);
                self.pending_bracket_plans.remove(&order.id);
            }
        }
        self.orders.extend(finished);
        self.active_orders = active;
    }

    /// 获取所有订单 (历史 + 活跃)
    pub fn get_all_orders(&self) -> Vec<Order> {
        let mut all = self.orders.clone();
        all.extend(self.active_orders.clone());
        all
    }

    /// 处理成交列表
    /// 包括资金调整、持仓更新、PnL 计算等
    pub fn process_trades(
        &mut self,
        mut trades: Vec<Trade>,
        portfolio: &mut Portfolio,
        instruments: &HashMap<String, Instrument>,
        risk_config: &RiskConfig,
        market_model: &dyn MarketModel,
        history_buffer: &Arc<RwLock<HistoryBuffer>>,
        last_prices: &HashMap<String, Decimal>,
    ) {
        // Filter out zero quantity trades (just in case)
        trades.retain(|t| t.quantity > Decimal::ZERO);

        for mut trade in trades {
            // 2. Calculate Final Commission
            let instr_opt = instruments.get(&trade.symbol);
            if let Some(instr) = instr_opt {
                let order_override = self
                    .active_orders
                    .iter()
                    .find(|o| o.id == trade.order_id)
                    .and_then(|order| {
                        Some((
                            order
                                .commission_type_override
                                .as_ref()?
                                .trim()
                                .to_ascii_lowercase(),
                            order.commission_value_override?,
                        ))
                    });
                if let Some((override_type, override_value)) = order_override {
                    trade.commission = match override_type.as_str() {
                        "fixed" => override_value,
                        "per_unit" => trade.quantity * override_value,
                        "percent" => {
                            let turnover = trade.price * trade.quantity * instr.multiplier();
                            turnover * override_value
                        }
                        _ => market_model.calculate_commission(
                            instr,
                            trade.side,
                            trade.price,
                            trade.quantity,
                        ),
                    };
                } else {
                    trade.commission = market_model.calculate_commission(
                        instr,
                        trade.side,
                        trade.price,
                        trade.quantity,
                    );
                }
            }

            // 3. Update Portfolio
            portfolio.adjust_cash(-trade.commission);
            let multiplier = instr_opt.map_or(Decimal::ONE, |instr| instr.multiplier());

            if let Some(instr) = instr_opt {
                if is_futures_margin_account(instr, risk_config) {
                    let realized = estimate_futures_realized_pnl(
                        &self.trade_tracker,
                        &trade.symbol,
                        trade.side,
                        trade.quantity,
                        trade.price,
                        multiplier,
                    );
                    portfolio.adjust_cash(realized);
                    if trade.side == crate::model::OrderSide::Buy {
                        portfolio.adjust_position(&trade.symbol, trade.quantity);
                    } else {
                        portfolio.adjust_position(&trade.symbol, -trade.quantity);
                    }
                } else {
                    let cost = trade.price * trade.quantity * multiplier;
                    if trade.side == crate::model::OrderSide::Buy {
                        portfolio.adjust_cash(-cost);
                        portfolio.adjust_position(&trade.symbol, trade.quantity);
                    } else {
                        portfolio.adjust_cash(cost);
                        portfolio.adjust_position(&trade.symbol, -trade.quantity);
                    }
                }
            }

            // Update available positions (T+1/T+0 rules)
            if let Some(instr) = instr_opt {
                market_model.update_available_position(
                    Arc::make_mut(&mut portfolio.available_positions),
                    instr,
                    trade.quantity,
                    trade.side,
                );
            }

            // 4. Update Order Commission
            // Note: filled_quantity and average_filled_price are updated via ExecutionReport
            // in on_execution_report, so we don't need to accumulate them here to avoid double counting.
            if let Some(order) = self
                .active_orders
                .iter_mut()
                .find(|o| o.id == trade.order_id)
            {
                order.commission += trade.commission;

                // Check if fully filled
                // Note: We don't change status to Filled here immediately because
                // execution report might come later or we want to wait for it?
                // Actually Engine logic relied on ExecutionReport to set status to Filled.
                // But here we are processing trade first?
                // In Engine::run, ExecutionReport updates status, THEN process_trades is called.
                // So status might already be Filled if this is the last trade.
                // But if we generated trade internally (simulated), we might need to update status.
                // However, SimulatedExecutionClient sends ExecutionReport with status Filled.
                // So we should rely on that.
                // But we update filled_qty here just in case?
                // Actually, if we rely on ExecutionReport for status, we should be fine.
                // The order update logic in on_execution_report handles it.
                // We just update commission here maybe?
            }

            // 5. Track Trade (PnL)
            let order_tag = self
                .active_orders
                .iter()
                .find(|o| o.id == trade.order_id)
                .map(|o| o.tag.as_str());

            // Get history for MAE/MFE
            // Need to lock history buffer
            //
            // 优先取 bar 容器; 该 symbol 只有 tick 序列时(纯 tick 回测)才落到
            // tick 容器。双流下(两条序列都有)固定取 bar —— 这与
            // context.rs::resolve_use_tick_history 给策略侧 get_history 的
            // freq=None 歧义报错无关: 那是策略主动查询, 这里是引擎内部统计,
            // 不该因为多了一条 tick 序列就改变纯 bar 场景下的既有 MAE/MFE 数值
            // (golden 基线锁定的正是这条路径)。修复前只读 bar 容器, 纯 tick
            // 回测下恒为 None, MAE/MFE/max_drawdown_pct 静默变 0。
            let history_guard = history_buffer.read().unwrap();
            let symbol_history = history_guard
                .get_history(&trade.symbol)
                .or_else(|| history_guard.get_tick_history(&trade.symbol));

            // Calculate Portfolio Value for % metrics
            let portfolio_value = calculate_account_metrics(
                portfolio,
                last_prices,
                instruments,
                &self.trade_tracker,
                risk_config,
            )
            .equity;

            self.trade_tracker.process_trade(
                &trade,
                multiplier,
                order_tag,
                symbol_history,
                portfolio_value,
            );

            // 6. Record Trade
            self.trades.push(trade.clone());
            self.current_step_trades.push(trade);
        }
    }

    fn detach_oco_order(&mut self, order_id: &str) {
        let Some(old_group) = self.oco_order_to_group.remove(order_id) else {
            return;
        };
        if let Some(group_orders) = self.oco_groups.get_mut(&old_group) {
            group_orders.remove(order_id);
            if group_orders.len() <= 1 {
                let leftovers: Vec<String> = group_orders.iter().cloned().collect();
                self.oco_groups.remove(&old_group);
                for oid in leftovers {
                    self.oco_order_to_group.remove(&oid);
                }
            }
        }
    }

    fn handle_oco_terminal_without_fill(&mut self, order_id: &str) {
        self.detach_oco_order(order_id);
    }
}

impl Default for OrderManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{OrderRole, OrderSide, OrderType, TimeInForce};
    use rust_decimal_macros::dec;

    fn make_order(id: &str, status: OrderStatus) -> Order {
        let mut order = Order::test_new(id, "TEST", OrderSide::Buy, OrderType::Limit, dec!(10));
        order.price = Some(dec!(100));
        order.time_in_force = TimeInForce::Day;
        order.order_role = OrderRole::Standalone;
        order.status = status;
        order
    }

    #[test]
    fn test_register_oco_group_and_consume_on_fill() {
        let mut manager = OrderManager::new();
        manager.register_oco_group(
            "oco-1".to_string(),
            "order-a".to_string(),
            "order-b".to_string(),
        );

        let peer_ids = manager.consume_oco_peer_cancels_on_fill("order-a");
        assert_eq!(peer_ids, vec!["order-b".to_string()]);
        assert!(manager.oco_groups.is_empty());
        assert!(manager.oco_order_to_group.is_empty());
    }

    #[test]
    fn test_cancel_active_order_updates_status_and_cleans_oco_mapping() {
        let mut manager = OrderManager::new();
        manager
            .active_orders
            .push(make_order("order-a", OrderStatus::Submitted));
        manager
            .active_orders
            .push(make_order("order-b", OrderStatus::Submitted));
        manager.register_oco_group(
            "oco-1".to_string(),
            "order-a".to_string(),
            "order-b".to_string(),
        );

        let cancelled = manager
            .cancel_active_order("order-b", 123)
            .expect("cancelled");
        assert_eq!(cancelled.status, OrderStatus::Cancelled);
        assert_eq!(cancelled.updated_at, 123);
        assert!(!manager.oco_order_to_group.contains_key("order-b"));
    }
}
