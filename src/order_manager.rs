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
    /// 当前步日终结算置为 Expired 的 Day 单 (用于通知策略 on_order)。
    /// 不入存档: 存档按位置编码(msgpack 数组), 插在中间会让旧存档错位解码失败;
    /// 且它只在结算到下一个策略事件之间非空, 存档时恒为空。
    #[serde(skip)]
    pub current_step_expired_orders: Vec<Order>,
    /// 本步移出在途账本、未成交终结的订单(撤单/过期/拒单)的终态快照。
    /// 多槽位下订单可能被别的槽位撤掉, 所属槽位的 ctx 里没有撤单 id, 靠它回调
    /// on_order(Cancelled)。不入存档(同上, 存档时恒为空)。
    #[serde(skip)]
    pub current_step_finalized_orders: Vec<Order>,
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
    /// 被拆成多笔成交的委托已累计的 `(未套最低佣金的佣金, 成交数量)`, 供最低佣金按
    /// 委托收(见 [`OrderManager::brokerage_floor_adjustment`])。只登记拆单的委托,
    /// 成交完或离开在途账本后删除。入存档且排在最后: 旧存档缺这一项时取空表。
    #[serde(default)]
    pub partial_fill_brokerage: HashMap<String, (Decimal, Decimal)>,
    /// 历史订单的 id 索引, 供 [`OrderManager::find_order`] 用。不入存档, 用到时按
    /// `orders` 现状补建。
    #[serde(skip)]
    finished_index: FinishedOrderIndex,
}

/// `orders`(历史订单)里每个 id 第一次出现的位置。索引记下已覆盖的长度, 查找前把新
/// 追加的部分补进来; 历史表变短则整个重建。
///
/// 前提: 历史表里已有订单的 id 不会被原地改写(现有代码只往表尾追加)。
#[derive(Clone, Default)]
struct FinishedOrderIndex {
    first_position: HashMap<String, usize>,
    covered: usize,
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
            current_step_expired_orders: Vec::new(),
            current_step_finalized_orders: Vec::new(),
            trade_tracker: TradeTracker::new(),
            oco_groups: HashMap::new(),
            oco_order_to_group: HashMap::new(),
            pending_bracket_plans: HashMap::new(),
            partial_fill_brokerage: HashMap::new(),
            finished_index: FinishedOrderIndex::default(),
        }
    }

    /// 按 id 取订单, 优先级同 [`OrderManager::get_all_orders`]: 先历史订单(取入表最早
    /// 的那个), 再在途订单。
    ///
    /// 每个成交回报都要查一次; 历史订单随回测进行越积越多, 逐个比较会让单笔成交的
    /// 开销随累计订单数增长, 所以历史部分走索引。
    pub fn find_order(&mut self, order_id: &str) -> Option<&Order> {
        let index = &mut self.finished_index;
        if self.orders.len() < index.covered {
            // 历史表被截短(不是追加): 索引作废重建。
            *index = FinishedOrderIndex::default();
        }
        for (position, order) in self.orders.iter().enumerate().skip(index.covered) {
            index
                .first_position
                .entry(order.id.clone())
                .or_insert(position);
        }
        index.covered = self.orders.len();
        match index.first_position.get(order_id) {
            Some(&position) if self.orders[position].id == order_id => self.orders.get(position),
            Some(_) => {
                // 索引指向的位置已不是这个 id(历史表被截短后又追加到原长度): 作废
                // 索引, 这次逐个比较。
                *index = FinishedOrderIndex::default();
                self.orders
                    .iter()
                    .chain(self.active_orders.iter())
                    .find(|o| o.id == order_id)
            }
            None => self.active_orders.iter().find(|o| o.id == order_id),
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
        for order in self
            .active_orders
            .iter_mut()
            .filter(|order| order.id == first || order.id == second)
        {
            Self::stamp_oco_graph(order, &group_key);
        }
        self.oco_order_to_group.insert(first, group_key.clone());
        self.oco_order_to_group.insert(second, group_key);
    }

    /// 给 `place_oco` 登记的组员写上组号(`oco:<组号>`), 供只拿得到订单列表的
    /// 在途量统计识别同一组(见 `Order::oco_group_key`)。
    /// 只写 `graph_id` 为空或本身就是这样写入的订单(重新登记到别的组时改写):
    /// 成员已有别的 `graph_id`(如 bracket 子单)时不覆盖, 这样的组在统计里合并
    /// 不了, 各算一次。
    fn stamp_oco_graph(order: &mut Order, group: &str) {
        let ours = order
            .graph_id
            .as_deref()
            .is_none_or(|graph| graph.starts_with(crate::model::OCO_GRAPH_PREFIX));
        if ours {
            order.graph_id = Some(format!("{}{group}", crate::model::OCO_GRAPH_PREFIX));
        }
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

    /// `symbol` 的持仓为 `position` 时, 撤掉该品种上已无持仓可平的 bracket 出场子单
    /// (卖出子单在持仓 <= 0 时、买入子单在持仓 >= 0 时), 原因记为
    /// [`crate::model::BRACKET_EXIT_POSITION_CLOSED`]; 返回被撤订单的快照。
    ///
    /// 持仓只是减少(仍与子单要平的方向一致)时一张都不撤, 成交量的上限由撮合时
    /// 按持仓缩量兜底。只动未完成的子单: 已被 OCO 同伴撤掉或已成交的不再处理。
    pub fn cancel_bracket_exits_without_position(
        &mut self,
        symbol: &str,
        position: Decimal,
        updated_at: i64,
    ) -> Vec<Order> {
        let ids: Vec<String> = self
            .active_orders
            .iter()
            .filter(|order| {
                order.symbol == symbol
                    && order.is_bracket_exit()
                    && matches!(
                        order.status,
                        OrderStatus::New | OrderStatus::Submitted | OrderStatus::PartiallyFilled
                    )
                    && order.closable_quantity(position) <= Decimal::ZERO
            })
            .map(|order| order.id.clone())
            .collect();
        let mut cancelled = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(order) = self.active_orders.iter_mut().find(|o| o.id == id) {
                order.reject_reason = crate::model::BRACKET_EXIT_POSITION_CLOSED.to_string();
            }
            if let Some(snapshot) = self.cancel_active_order(&id, updated_at) {
                cancelled.push(snapshot);
            }
        }
        cancelled
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
    pub fn add_active_order(&mut self, mut order: Order) {
        // 在回调里下单并登记 OCO 时, 登记先于订单过风控进入在途列表。
        if let Some(group) = self.oco_order_to_group.get(&order.id) {
            Self::stamp_oco_graph(&mut order, group);
        }
        self.active_orders.push(order);
    }

    /// 处理执行报告 (ExecutionReport)
    /// 更新活跃订单状态
    pub fn on_execution_report(&mut self, report: Order) {
        let mut terminal_without_fill_id: Option<String> = None;
        // Find existing order
        if let Some(existing) = self.active_orders.iter_mut().find(|o| o.id == report.id) {
            // 这份回报带来了新成交(IOC 成交一部分后撤掉剩余): 不算“无成交终结”, OCO
            // 对手单的撤销与括号单的激活由调用方按成交处理。
            let carries_fill = report.filled_quantity > existing.filled_quantity;
            existing.status = report.status;
            // 成交时缩量会改写订单数量(见 execution::simulated)
            existing.quantity = report.quantity;
            existing.filled_quantity = report.filled_quantity;
            existing.average_filled_price = report.average_filled_price;
            existing.updated_at = report.updated_at;
            existing.reject_reason = report.reject_reason;
            if existing.status != OrderStatus::Filled
                && !carries_fill
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
                // 部分成交后撤单/过期: 不会再有成交, 丢掉按委托累计的佣金记录。
                // 成交完的委托在最后一笔成交记账时删除(见 brokerage_floor_adjustment)。
                self.partial_fill_brokerage.remove(&order.id);
                self.handle_oco_terminal_without_fill(&order.id);
                self.pending_bracket_plans.remove(&order.id);
                self.current_step_finalized_orders.push(order.clone());
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

    /// 最低佣金按委托收: 返回本笔成交佣金相对逐笔套最低佣金的调整额(不大于 0)。
    ///
    /// 整笔委托的佣金为 `max(各笔成交佣金之和, 最低佣金)`: 首笔成交先按最低佣金收
    /// 齐, 之后的成交只补累计佣金超出已收部分的差额。一笔成交就成交完的委托不登记、
    /// 调整额为 0, 与逐笔口径相同。
    fn brokerage_floor_adjustment(
        &mut self,
        order_id: &str,
        order_quantity: Option<Decimal>,
        fill_quantity: Decimal,
        raw: Decimal,
        min_commission: Decimal,
    ) -> Decimal {
        if min_commission <= Decimal::ZERO {
            return Decimal::ZERO;
        }
        let prior = self.partial_fill_brokerage.get(order_id).copied();
        let splits = order_quantity.is_some_and(|total| fill_quantity < total);
        if prior.is_none() && !splits {
            return Decimal::ZERO;
        }
        let (prior_raw, prior_quantity) = prior.unwrap_or((Decimal::ZERO, Decimal::ZERO));
        let charged_before = if prior.is_some() {
            prior_raw.max(min_commission)
        } else {
            Decimal::ZERO
        };
        let cumulative_raw = prior_raw + raw;
        let cumulative_quantity = prior_quantity + fill_quantity;
        let this_fill = cumulative_raw.max(min_commission) - charged_before;
        if order_quantity.is_none_or(|total| cumulative_quantity >= total) {
            self.partial_fill_brokerage.remove(order_id);
        } else {
            self.partial_fill_brokerage
                .insert(order_id.to_string(), (cumulative_raw, cumulative_quantity));
        }
        this_fill - raw.max(min_commission)
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
                // 组合在第 3 步才更新, 此处取到的正是成交前持仓
                let position_before = portfolio
                    .positions
                    .get(&trade.symbol)
                    .copied()
                    .unwrap_or(Decimal::ZERO);
                let active_order = self.active_orders.iter().find(|o| o.id == trade.order_id);
                let order_quantity = active_order.map(|order| order.quantity);
                let order_override = active_order.and_then(|order| {
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
                            position_before,
                        ),
                    };
                } else {
                    trade.commission = market_model.calculate_commission(
                        instr,
                        trade.side,
                        trade.price,
                        trade.quantity,
                        position_before,
                    );
                    if let Some((raw, min_commission)) =
                        market_model.brokerage_floor(instr, trade.price, trade.quantity)
                    {
                        trade.commission += self.brokerage_floor_adjustment(
                            &trade.order_id,
                            order_quantity,
                            trade.quantity,
                            raw,
                            min_commission,
                        );
                    }
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
            // 实际谓词: bar 容器对该 symbol 存在(Some)就取 bar, 否则落到 tick 容器。
            // 纯 tick 回测下 bar 容器恒为 None, 因此恒取 tick 容器。双流下(两条
            // 序列都可能存在)不是"固定取 bar": 在该 symbol 的第一根聚合 bar 落库
            // 之前平的仓, bar 容器此时仍是 None, 这次仍会落到 tick 容器; 落库之后
            // 再平仓才会取到 bar。也就是说同一次回测里 MAE/MFE 的计算基准可能从
            // tick 序列静默切换到 bar 序列, 这是刻意接受的行为(不是回归), 只是
            // 基准不保证同质。这与 context.rs::resolve_history_source 给策略侧
            // get_history 的 freq=None 歧义报错无关: 那是策略主动查询, 这里是引擎
            // 内部统计, 不该因为多了一条 tick 序列就改变纯 bar 场景下的既有
            // MAE/MFE 数值(golden 基线锁定的正是这条路径)。修复前只读 bar 容器,
            // 纯 tick 回测下恒为 None, MAE/MFE/max_drawdown_pct 静默变 0。
            let history_guard = history_buffer.read().unwrap();
            let histories = crate::analysis::ExcursionHistories {
                bar: history_guard.get_history(&trade.symbol),
                tick: history_guard.get_tick_history(&trade.symbol),
            };

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
                Some(histories),
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
                    // 组已解散: 留下的那一张不再是 OCO 组员, 清掉写入的组号,
                    // 免得之后与同名的新组(如别的策略槽位的 `oco-1`)合并统计。
                    if let Some(order) = self.active_orders.iter_mut().find(|o| o.id == oid)
                        && order
                            .graph_id
                            .as_deref()
                            .is_some_and(|graph| graph.starts_with(crate::model::OCO_GRAPH_PREFIX))
                    {
                        order.graph_id = None;
                    }
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

    /// find_order 的参照: 改造前的逐个比较(历史订单在前, 在途订单在后)。
    fn scan(manager: &OrderManager, id: &str) -> Option<(Decimal, OrderStatus)> {
        manager
            .orders
            .iter()
            .chain(manager.active_orders.iter())
            .find(|o| o.id == id)
            .map(|o| (o.quantity, o.status))
    }

    #[test]
    fn find_order_matches_linear_scan() {
        // 线性同余伪随机, 固定种子可复现
        let mut seed: u64 = 20261113;
        let mut next = move |n: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % n
        };
        let mut manager = OrderManager::new();
        let mut serial = 0u32;
        for step in 0..3000 {
            // id 取自小集合: 历史表内、历史与在途之间都会重复
            let id = format!("o{}", next(40));
            serial += 1;
            let mut order = make_order(&id, OrderStatus::New);
            order.quantity = Decimal::from(serial); // 用数量区分同 id 的不同订单
            match next(9) {
                0..=3 => manager.active_orders.push(order),
                4..=6 => {
                    order.status = OrderStatus::Filled;
                    manager.orders.push(order);
                }
                7 => {
                    // 在途订单终结, 经 cleanup 移入历史表
                    if let Some(o) = manager.active_orders.first_mut() {
                        o.status = OrderStatus::Cancelled;
                    }
                    manager.cleanup_finished_orders();
                }
                8 if step % 97 == 0 => {
                    // 历史表被截短
                    let keep = manager.orders.len() / 2;
                    manager.orders.truncate(keep);
                }
                _ => {}
            }
            for probe in 0..41 {
                let probe_id = format!("o{probe}");
                let want = scan(&manager, &probe_id);
                let got = manager
                    .find_order(&probe_id)
                    .map(|o| (o.quantity, o.status));
                assert_eq!(got, want, "step={step} id={probe_id}");
            }
        }
    }

    #[test]
    fn old_layout_checkpoint_decodes_and_new_layout_round_trips() {
        let mut manager = OrderManager::new();
        manager
            .active_orders
            .push(make_order("active-1", OrderStatus::New));
        manager
            .orders
            .push(make_order("done-1", OrderStatus::Expired));
        manager.register_oco_group(
            "oco-1".to_string(),
            "active-1".to_string(),
            "done-1".to_string(),
        );

        // 加入 current_step_expired_orders 之前的字段布局(msgpack 按位置编码)
        let old_encoded: Vec<u8> = rmp_serde::to_vec(&(
            &manager.orders,
            &manager.active_orders,
            &manager.trades,
            &manager.current_step_trades,
            &manager.current_step_rejected_orders,
            &manager.trade_tracker,
            &manager.oco_groups,
            &manager.oco_order_to_group,
            &manager.pending_bracket_plans,
        ))
        .unwrap();
        let old_decoded: OrderManager =
            rmp_serde::from_slice(&old_encoded).expect("旧布局存档应能解码");
        assert_eq!(old_decoded.active_orders.len(), 1);
        assert_eq!(old_decoded.active_orders[0].id, "active-1");
        assert_eq!(old_decoded.orders[0].id, "done-1");
        assert_eq!(
            old_decoded
                .oco_order_to_group
                .get("active-1")
                .map(String::as_str),
            Some("oco-1")
        );
        assert!(old_decoded.current_step_expired_orders.is_empty());

        assert!(old_decoded.partial_fill_brokerage.is_empty());

        // 新布局往返: 过期单缓冲不入存档, 读回为空; 拆单委托的累计佣金排在最后,
        // 其余字段不变
        manager
            .current_step_expired_orders
            .push(make_order("done-1", OrderStatus::Expired));
        manager
            .partial_fill_brokerage
            .insert("active-1".to_string(), (dec!(0.3), dec!(100)));
        let encoded: Vec<u8> = rmp_serde::to_vec(&manager).unwrap();
        let expected: Vec<u8> = rmp_serde::to_vec(&(
            &manager.orders,
            &manager.active_orders,
            &manager.trades,
            &manager.current_step_trades,
            &manager.current_step_rejected_orders,
            &manager.trade_tracker,
            &manager.oco_groups,
            &manager.oco_order_to_group,
            &manager.pending_bracket_plans,
            &manager.partial_fill_brokerage,
        ))
        .unwrap();
        assert_eq!(encoded, expected);
        let decoded: OrderManager = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(
            decoded.partial_fill_brokerage.get("active-1"),
            Some(&(dec!(0.3), dec!(100)))
        );
        assert_eq!(decoded.active_orders[0].id, "active-1");
        assert_eq!(decoded.orders[0].status, OrderStatus::Expired);
        assert_eq!(decoded.oco_groups.len(), 1);
        assert!(decoded.current_step_expired_orders.is_empty());
    }

    #[test]
    fn brokerage_floor_is_charged_once_per_order() {
        let mut manager = OrderManager::new();
        let total = Some(dec!(300));
        // 300 股拆 3 笔, 每笔佣金 0.3, 最低 5: 首笔收 5, 其余不再收
        let first = manager.brokerage_floor_adjustment("o1", total, dec!(100), dec!(0.3), dec!(5));
        assert_eq!(dec!(5) + first, dec!(5));
        let second = manager.brokerage_floor_adjustment("o1", total, dec!(100), dec!(0.3), dec!(5));
        assert_eq!(dec!(5) + second, dec!(0));
        let third = manager.brokerage_floor_adjustment("o1", total, dec!(100), dec!(0.3), dec!(5));
        assert_eq!(dec!(5) + third, dec!(0));
        // 成交完后记录删除
        assert!(manager.partial_fill_brokerage.is_empty());
    }

    #[test]
    fn brokerage_floor_tops_up_once_cumulative_exceeds_minimum() {
        let mut manager = OrderManager::new();
        let total = Some(dec!(300));
        // 每笔佣金 3, 最低 5: 5 → 累计 6 补 1 → 再收 3, 合计 9 = 整笔委托的佣金
        let charged: Vec<Decimal> = (0..3)
            .map(|_| {
                dec!(5)
                    + manager.brokerage_floor_adjustment("o1", total, dec!(100), dec!(3), dec!(5))
            })
            .collect();
        assert_eq!(charged, vec![dec!(5), dec!(1), dec!(3)]);
    }

    #[test]
    fn brokerage_floor_leaves_single_fill_orders_untouched() {
        let mut manager = OrderManager::new();
        // 一笔成交完: 不调整、不登记
        let adj = manager.brokerage_floor_adjustment(
            "o1",
            Some(dec!(100)),
            dec!(100),
            dec!(0.3),
            dec!(5),
        );
        assert_eq!(adj, Decimal::ZERO);
        // 没有最低佣金: 不调整、不登记
        let adj = manager.brokerage_floor_adjustment(
            "o2",
            Some(dec!(300)),
            dec!(100),
            dec!(0.3),
            dec!(0),
        );
        assert_eq!(adj, Decimal::ZERO);
        assert!(manager.partial_fill_brokerage.is_empty());
    }

    #[test]
    fn brokerage_floor_record_dropped_when_partial_order_cancelled() {
        let mut manager = OrderManager::new();
        manager
            .active_orders
            .push(make_order("o1", OrderStatus::Cancelled));
        manager.brokerage_floor_adjustment("o1", Some(dec!(300)), dec!(100), dec!(0.3), dec!(5));
        assert_eq!(manager.partial_fill_brokerage.len(), 1);
        manager.cleanup_finished_orders();
        assert!(manager.partial_fill_brokerage.is_empty());
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

    /// place_oco 的组员带上 oco: 组号(先在途、后在途两种顺序都覆盖); 已有
    /// graph_id 的成员不覆盖; 组解散后留下的那一张清掉组号。
    #[test]
    fn register_oco_group_stamps_members_for_pending_statistics() {
        let mut manager = OrderManager::new();
        manager.add_active_order(make_order("a", OrderStatus::New));
        manager.register_oco_group("g".to_string(), "a".to_string(), "b".to_string());
        manager.add_active_order(make_order("b", OrderStatus::New));
        let mut foreign = make_order("c", OrderStatus::New);
        foreign.graph_id = Some("bracket-e".to_string());
        manager.add_active_order(foreign);
        manager.add_active_order(make_order("d", OrderStatus::New));
        manager.register_oco_group("h".to_string(), "c".to_string(), "d".to_string());

        let graph = |manager: &OrderManager, id: &str| {
            manager
                .active_orders
                .iter()
                .find(|order| order.id == id)
                .and_then(|order| order.graph_id.clone())
        };
        assert_eq!(graph(&manager, "a").as_deref(), Some("oco:g"));
        assert_eq!(graph(&manager, "b").as_deref(), Some("oco:g"));
        assert_eq!(graph(&manager, "c").as_deref(), Some("bracket-e"));
        assert_eq!(graph(&manager, "d").as_deref(), Some("oco:h"));

        manager.handle_oco_terminal_without_fill("a");
        assert_eq!(graph(&manager, "a").as_deref(), Some("oco:g"));
        assert_eq!(graph(&manager, "b"), None);
    }

    fn bracket_exit(id: &str, side: OrderSide, role: OrderRole) -> Order {
        let mut order = Order::test_new(id, "TEST", side, OrderType::StopMarket, dec!(10));
        order.parent_order_id = Some("entry".to_string());
        order.graph_id = Some("bracket-entry".to_string());
        order.order_role = role;
        order
    }

    fn statuses(manager: &OrderManager) -> Vec<(String, OrderStatus)> {
        manager
            .active_orders
            .iter()
            .map(|order| (order.id.clone(), order.status))
            .collect()
    }

    /// 卖出子单(平多)在持仓 <= 0 时撤销, 持仓仍为正时不动; 普通卖单不受影响。
    #[test]
    fn sell_bracket_exits_are_cancelled_once_long_position_is_gone() {
        let mut manager = OrderManager::new();
        manager.add_active_order(bracket_exit("s", OrderSide::Sell, OrderRole::StopLoss));
        manager.add_active_order(bracket_exit("t", OrderSide::Sell, OrderRole::TakeProfit));
        let mut plain = Order::test_new("p", "TEST", OrderSide::Sell, OrderType::Limit, dec!(10));
        plain.price = Some(dec!(12));
        manager.add_active_order(plain);
        let mut other = bracket_exit("o", OrderSide::Sell, OrderRole::StopLoss);
        other.symbol = "OTHER".to_string();
        manager.add_active_order(other);

        // 持仓只是减少: 一张都不撤
        assert!(
            manager
                .cancel_bracket_exits_without_position("TEST", dec!(5), 1)
                .is_empty()
        );

        let cancelled = manager.cancel_bracket_exits_without_position("TEST", dec!(0), 7);
        assert_eq!(
            cancelled.iter().map(|o| o.id.as_str()).collect::<Vec<_>>(),
            vec!["s", "t"]
        );
        for order in &cancelled {
            assert_eq!(order.status, OrderStatus::Cancelled);
            assert_eq!(order.updated_at, 7);
            assert_eq!(order.reject_reason, crate::model::BRACKET_EXIT_POSITION_CLOSED);
        }
        assert_eq!(
            statuses(&manager),
            vec![
                ("s".to_string(), OrderStatus::Cancelled),
                ("t".to_string(), OrderStatus::Cancelled),
                ("p".to_string(), OrderStatus::New),
                ("o".to_string(), OrderStatus::New),
            ]
        );
        // 已撤的不再撤第二次
        assert!(
            manager
                .cancel_bracket_exits_without_position("TEST", dec!(-3), 8)
                .is_empty()
        );
    }

    /// 买入子单(平空)在持仓 >= 0 时撤销, 空头仍在时不动。
    #[test]
    fn buy_bracket_exits_are_cancelled_once_short_position_is_gone() {
        let mut manager = OrderManager::new();
        manager.add_active_order(bracket_exit("s", OrderSide::Buy, OrderRole::StopLoss));
        assert!(
            manager
                .cancel_bracket_exits_without_position("TEST", dec!(-2), 1)
                .is_empty()
        );
        let cancelled = manager.cancel_bracket_exits_without_position("TEST", dec!(4), 2);
        assert_eq!(cancelled.len(), 1);
        assert_eq!(statuses(&manager), vec![("s".to_string(), OrderStatus::Cancelled)]);
    }
}
