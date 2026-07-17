use crate::account::{
    calculate_account_metrics, estimate_futures_realized_pnl, is_futures_margin_account,
    stock_margin_ratio_override,
};
use crate::event::Event;
use crate::execution::matcher::{ExecutionMatcher, MatchContext};
use crate::execution::slippage::{SlippageModel, ZeroSlippage};
use crate::execution::{ExecutionClient, crypto, forex, futures, option, stock};
use crate::log_context::{AkqLogContext, execution_order_context, render_log_message};
use crate::model::{
    AssetType, ExecutionPolicyCore, Order, OrderStatus, PriceBasis, TemporalPolicy, TimeInForce,
    TradingSession,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::{HashMap, HashSet};

/// 模拟交易所执行器 (Simulated Execution Client)
/// 负责在内存中撮合订单 (回测模式)
pub struct SimulatedExecutionClient {
    slippage_model: Box<dyn SlippageModel>,
    volume_limit_pct: Decimal, // 成交量限制比例 (0.0 = 不限制)
    // Map order_id -> Order (O(1) access)
    orders: HashMap<String, Order>,
    // List of order_ids to maintain submission order (for matching fairness)
    order_queue: Vec<String>,
    // Matchers
    matchers: HashMap<AssetType, Box<dyn ExecutionMatcher>>,
    futures_enforce_tick_size: bool,
    futures_enforce_lot_size: bool,
    futures_validation_by_prefix: Vec<(String, Option<bool>, Option<bool>)>,
    slice_tracking_timestamp: Option<i64>,
    attempted_same_cycle_order_ids: HashSet<String>,
    deferred_same_cycle_order_ids: HashSet<String>,
}

impl SimulatedExecutionClient {
    fn order_log_context(order: &Order, event_time: i64) -> AkqLogContext {
        execution_order_context(order, event_time)
    }

    fn insufficient_margin_warning(
        order: &Order,
        current_time: i64,
        total_required: Decimal,
        current_free_margin: Decimal,
    ) -> String {
        render_log_message(
            format!(
                "Rejected order due to insufficient margin during execution. Required: {total_required}, Available: {current_free_margin}"
            ),
            Self::order_log_context(order, current_time),
        )
    }

    fn expired_day_order_warning(order: &Order, event_time: i64) -> String {
        render_log_message(
            "Expired day order at session close",
            Self::order_log_context(order, event_time),
        )
    }

    fn ignored_unknown_cancel_warning(order_id: &str) -> String {
        render_log_message(
            "Ignored cancel request for unknown order",
            AkqLogContext::new()
                .phase("execution")
                .order_id(order_id.to_string()),
        )
    }

    fn ignored_non_cancellable_cancel_warning(order: &Order) -> String {
        let event_time = if order.updated_at != 0 {
            order.updated_at
        } else {
            order.created_at
        };
        render_log_message(
            format!(
                "Ignored cancel request because order is not cancellable in status {:?}",
                order.status
            ),
            Self::order_log_context(order, event_time),
        )
    }

    fn deferred_same_cycle_order_warning(order: &Order, event_time: i64) -> String {
        render_log_message(
            "Deferred same-cycle order until cross-symbol reduce-first orders finish in current slice",
            Self::order_log_context(order, event_time),
        )
    }

    fn rebuild_futures_matcher(&mut self) {
        self.matchers.insert(
            AssetType::Futures,
            Box::new(futures::FuturesMatcher::with_prefix_rules(
                self.futures_enforce_tick_size,
                self.futures_enforce_lot_size,
                self.futures_validation_by_prefix.clone(),
            )),
        );
    }

    pub fn new() -> Self {
        let mut matchers: HashMap<AssetType, Box<dyn ExecutionMatcher>> = HashMap::new();
        matchers.insert(AssetType::Stock, Box::new(stock::StockMatcher));
        matchers.insert(AssetType::Fund, Box::new(stock::StockMatcher)); // Fund uses StockMatcher
        matchers.insert(
            AssetType::Futures,
            Box::new(futures::FuturesMatcher::with_prefix_rules(
                true,
                true,
                Vec::new(),
            )),
        );
        matchers.insert(AssetType::Option, Box::new(option::OptionMatcher));
        matchers.insert(AssetType::Crypto, Box::new(crypto::CryptoMatcher));
        matchers.insert(AssetType::Forex, Box::new(forex::ForexMatcher));

        SimulatedExecutionClient {
            slippage_model: Box::new(ZeroSlippage),
            volume_limit_pct: Decimal::ZERO,
            orders: HashMap::new(),
            order_queue: Vec::new(),
            matchers,
            futures_enforce_tick_size: true,
            futures_enforce_lot_size: true,
            futures_validation_by_prefix: Vec::new(),
            slice_tracking_timestamp: None,
            attempted_same_cycle_order_ids: HashSet::new(),
            deferred_same_cycle_order_ids: HashSet::new(),
        }
    }

    fn prepare_slice_tracking(&mut self, timestamp: i64) {
        if self.slice_tracking_timestamp == Some(timestamp) {
            return;
        }
        self.slice_tracking_timestamp = Some(timestamp);
        self.attempted_same_cycle_order_ids.clear();
        self.deferred_same_cycle_order_ids.clear();
    }

    fn clear_slice_tracking(&mut self) {
        self.attempted_same_cycle_order_ids.clear();
        self.deferred_same_cycle_order_ids.clear();
        self.slice_tracking_timestamp = None;
    }

    fn event_symbol(event: &Event) -> Option<&str> {
        match event {
            Event::Bar(bar) => Some(bar.symbol.as_str()),
            Event::Tick(tick) => Some(tick.symbol.as_str()),
            _ => None,
        }
    }

    fn event_timestamp(event: &Event) -> Option<i64> {
        match event {
            Event::Bar(bar) => Some(bar.timestamp),
            Event::Tick(tick) => Some(tick.timestamp),
            _ => None,
        }
    }

    fn is_order_active(order: &Order) -> bool {
        !matches!(
            order.status,
            OrderStatus::Cancelled
                | OrderStatus::Filled
                | OrderStatus::Rejected
                | OrderStatus::Expired
        )
    }

    fn effective_policy(order: &Order, ctx: &crate::context::EngineContext) -> ExecutionPolicyCore {
        order
            .fill_policy_override
            .unwrap_or(ctx.execution_policy_core)
    }

    fn is_same_cycle_close_policy(policy: ExecutionPolicyCore) -> bool {
        policy.price_basis == PriceBasis::Close
            && policy.bar_offset == 0
            && matches!(policy.temporal, TemporalPolicy::SameCycle)
    }

    fn is_same_cycle_close_order(
        &self,
        order: &Order,
        ctx: &crate::context::EngineContext,
    ) -> bool {
        Self::is_same_cycle_close_policy(Self::effective_policy(order, ctx))
    }

    fn has_cross_symbol_reduce_pending(
        &self,
        event: &Event,
        ctx: &crate::context::EngineContext,
    ) -> bool {
        let Some(event_symbol) = Self::event_symbol(event) else {
            return false;
        };
        self.orders.values().any(|candidate| {
            Self::is_order_active(candidate)
                && candidate.created_at == ctx.current_time
                && candidate.symbol != event_symbol
                && self.is_same_cycle_close_order(candidate, ctx)
                && crate::model::is_reduce_first_order(candidate.side, candidate.position_effect)
        })
    }

    fn should_defer_order_on_current_event(
        &self,
        order: &Order,
        event: &Event,
        ctx: &crate::context::EngineContext,
    ) -> bool {
        if order.created_at != ctx.current_time || !self.is_same_cycle_close_order(order, ctx) {
            return false;
        }
        if crate::model::is_reduce_first_order(order.side, order.position_effect) {
            return false;
        }
        Self::event_symbol(event)
            .map(|symbol| order.symbol == symbol)
            .unwrap_or(false)
            && self.has_cross_symbol_reduce_pending(event, ctx)
    }

    fn should_finalize_order(&self, order: &Order, ctx: &crate::context::EngineContext) -> bool {
        order.created_at == ctx.current_time
            && Self::is_order_active(order)
            && self.is_same_cycle_close_order(order, ctx)
            && (!self.attempted_same_cycle_order_ids.contains(&order.id)
                || self.deferred_same_cycle_order_ids.contains(&order.id))
    }

    fn sorted_queue(&self) -> Vec<String> {
        let mut queue: Vec<String> = self.order_queue.clone();
        queue.sort_by_key(|order_id| {
            self.orders
                .get(order_id)
                .map(|order| match order.side {
                    crate::model::OrderSide::Sell => 0_u8,
                    crate::model::OrderSide::Buy => 1_u8,
                })
                .unwrap_or(2_u8)
        });
        queue
    }

    fn sync_order_from_report(&mut self, report: &Event) {
        if let Event::ExecutionReport(updated_order, _) = report
            && let Some(existing) = self.orders.get_mut(&updated_order.id)
        {
            existing.status = updated_order.status;
            existing.filled_quantity = updated_order.filled_quantity;
            existing.average_filled_price = updated_order.average_filled_price;
            existing.commission = updated_order.commission;
            existing.updated_at = updated_order.updated_at;
        }
    }

    fn cleanup_finished_orders(&mut self) {
        let mut finished_ids = Vec::new();
        for id in &self.order_queue {
            if let Some(order) = self.orders.get(id) {
                if !Self::is_order_active(order) {
                    finished_ids.push(id.clone());
                }
            } else {
                finished_ids.push(id.clone());
            }
        }
        for id in finished_ids {
            self.orders.remove(&id);
        }
        self.order_queue.retain(|id| self.orders.contains_key(id));
    }

    fn match_with_events<F>(
        &mut self,
        events: &[Event],
        ctx: &crate::context::EngineContext,
        mut order_filter: F,
        defer_cross_symbol_increase: bool,
    ) -> Vec<Event>
    where
        F: FnMut(&Order, &Event) -> bool,
    {
        let mut reports = Vec::new();
        let mut projected_portfolio = ctx.portfolio.clone();
        let stock_margin_ratio_override = stock_margin_ratio_override(ctx.risk_config);
        let mut current_free_margin = calculate_free_margin(
            &projected_portfolio,
            ctx.last_prices,
            ctx.instruments,
            ctx.trade_tracker,
            ctx.risk_config,
        );

        for event in events {
            let queue = self.sorted_queue();
            for order_id in &queue {
                let mut synced_report: Option<Event> = None;
                let Some(order_snapshot) = self.orders.get(order_id).cloned() else {
                    continue;
                };
                if !Self::is_order_active(&order_snapshot) || !order_filter(&order_snapshot, event)
                {
                    continue;
                }

                let defer_current = defer_cross_symbol_increase
                    && self.should_defer_order_on_current_event(&order_snapshot, event, ctx);
                let mark_attempted = order_snapshot.created_at == ctx.current_time
                    && self.is_same_cycle_close_order(&order_snapshot, ctx)
                    && Self::event_symbol(event)
                        .map(|symbol| order_snapshot.symbol == symbol)
                        .unwrap_or(false);

                if defer_current {
                    let event_time = Self::event_timestamp(event).unwrap_or(ctx.current_time);
                    let first_defer = self
                        .deferred_same_cycle_order_ids
                        .insert(order_snapshot.id.clone());
                    self.attempted_same_cycle_order_ids
                        .insert(order_snapshot.id.clone());
                    if first_defer {
                        log::warn!(
                            "{}",
                            Self::deferred_same_cycle_order_warning(&order_snapshot, event_time)
                        );
                    }
                    continue;
                }

                if mark_attempted {
                    self.attempted_same_cycle_order_ids
                        .insert(order_snapshot.id.clone());
                }

                if let Some(order) = self.orders.get_mut(order_id) {
                    if let Some(instrument) = ctx.instruments.get(&order.symbol)
                        && let Some(matcher) = self.matchers.get(&instrument.asset_type)
                    {
                        let match_ctx = MatchContext {
                            event,
                            instrument,
                            execution_policy_core: Self::effective_policy(order, ctx),
                            slippage: self.slippage_model.as_ref(),
                            volume_limit_pct: self.volume_limit_pct,
                            bar_index: ctx.bar_index,
                            last_price: ctx.last_prices.get(&order.symbol).copied(),
                        };
                        let report_opt = matcher.match_order(order, &match_ctx);
                        if let Some(mut report) = report_opt {
                            let mut replacement_report: Option<Event> = None;
                            if let Event::ExecutionReport(
                                ref mut report_order,
                                Some(ref mut trade),
                            ) = report
                            {
                                let mut prices_for_margin = ctx.last_prices.clone();
                                prices_for_margin.insert(trade.symbol.clone(), trade.price);

                                let commission = ctx.market_model.calculate_commission(
                                    instrument,
                                    trade.side,
                                    trade.price,
                                    trade.quantity,
                                );

                                let base_used_margin = projected_portfolio
                                    .calculate_used_margin_with_stock_ratio(
                                        &prices_for_margin,
                                        ctx.instruments,
                                        stock_margin_ratio_override,
                                    );
                                let mut margin_projection = projected_portfolio.clone();
                                let current_pos = margin_projection
                                    .positions
                                    .get(&trade.symbol)
                                    .copied()
                                    .unwrap_or(Decimal::ZERO);
                                let next_pos = crate::model::project_position_after(
                                    trade.side,
                                    trade.position_effect,
                                    current_pos,
                                    trade.quantity,
                                );
                                margin_projection
                                    .adjust_position(&trade.symbol, next_pos - current_pos);
                                let next_used_margin = margin_projection
                                    .calculate_used_margin_with_stock_ratio(
                                        &prices_for_margin,
                                        ctx.instruments,
                                        stock_margin_ratio_override,
                                    );
                                let margin_required =
                                    (next_used_margin - base_used_margin).max(Decimal::ZERO);
                                let total_required = margin_required + commission;

                                // Execution-time affordability intentionally does
                                // NOT reuse risk::common::check_affordability: it
                                // gates on `current_free_margin` derived from
                                // calculate_account_metrics (equity incl. futures
                                // unrealized PnL, futures-accurate) at the real
                                // fill price with no safety haircut. The submission
                                // gate uses portfolio free margin at the last/limit
                                // price with a safety buffer — a deliberate
                                // forecast-vs-actual split, not duplicated logic.
                                if total_required > current_free_margin {
                                    if report_order.allow_quantity_auto_resize {
                                        let lot_size = instrument.lot_size();
                                        let safety_factor =
                                            Decimal::from_f64(0.9999).unwrap_or(Decimal::ONE);
                                        let mut new_qty = if total_required > Decimal::ZERO
                                            && current_free_margin > Decimal::ZERO
                                        {
                                            (trade.quantity * current_free_margin * safety_factor
                                                / total_required)
                                                .floor()
                                        } else {
                                            Decimal::ZERO
                                        };
                                        if lot_size > Decimal::ZERO {
                                            new_qty = new_qty - (new_qty % lot_size);
                                        }
                                        if new_qty >= trade.quantity && lot_size > Decimal::ZERO {
                                            new_qty -= lot_size;
                                        }

                                        while new_qty > Decimal::ZERO {
                                            let new_comm = ctx.market_model.calculate_commission(
                                                instrument,
                                                trade.side,
                                                trade.price,
                                                new_qty,
                                            );
                                            let mut resized_projection =
                                                projected_portfolio.clone();
                                            let current_pos = resized_projection
                                                .positions
                                                .get(&trade.symbol)
                                                .copied()
                                                .unwrap_or(Decimal::ZERO);
                                            let next_pos = crate::model::project_position_after(
                                                trade.side,
                                                trade.position_effect,
                                                current_pos,
                                                new_qty,
                                            );
                                            resized_projection.adjust_position(
                                                &trade.symbol,
                                                next_pos - current_pos,
                                            );
                                            let resized_used_margin = resized_projection
                                                .calculate_used_margin_with_stock_ratio(
                                                    &prices_for_margin,
                                                    ctx.instruments,
                                                    stock_margin_ratio_override,
                                                );
                                            let resized_required = (resized_used_margin
                                                - base_used_margin)
                                                .max(Decimal::ZERO);
                                            if resized_required + new_comm <= current_free_margin {
                                                break;
                                            }
                                            if new_qty >= lot_size && lot_size > Decimal::ZERO {
                                                new_qty -= lot_size;
                                            } else {
                                                new_qty = Decimal::ZERO;
                                            }
                                        }

                                        trade.quantity = new_qty;
                                    } else {
                                        let mut rejected_order = report_order.clone();
                                        rejected_order.status = crate::model::OrderStatus::Rejected;
                                        rejected_order.filled_quantity = Decimal::ZERO;
                                        rejected_order.average_filled_price = None;
                                        rejected_order.updated_at = ctx.current_time;
                                        rejected_order.reject_reason = format!(
                                            "Risk: Insufficient margin at execution. Required: {total_required}, Available: {current_free_margin}"
                                        );
                                        log::warn!(
                                            "{}",
                                            Self::insufficient_margin_warning(
                                                &rejected_order,
                                                ctx.current_time,
                                                total_required,
                                                current_free_margin,
                                            )
                                        );
                                        replacement_report =
                                            Some(Event::ExecutionReport(rejected_order, None));
                                    }
                                }
                            }

                            if let Some(new_report) = replacement_report {
                                report = new_report;
                            }

                            if let Event::ExecutionReport(_, Some(ref trade)) = report
                                && trade.quantity > Decimal::ZERO
                            {
                                let mut prices_for_margin = ctx.last_prices.clone();
                                prices_for_margin.insert(trade.symbol.clone(), trade.price);
                                let commission = ctx.market_model.calculate_commission(
                                    instrument,
                                    trade.side,
                                    trade.price,
                                    trade.quantity,
                                );
                                projected_portfolio.adjust_cash(-commission);
                                if is_futures_margin_account(instrument, ctx.risk_config) {
                                    let realized = estimate_futures_realized_pnl(
                                        ctx.trade_tracker,
                                        &trade.symbol,
                                        trade.side,
                                        trade.quantity,
                                        trade.price,
                                        instrument.multiplier(),
                                    );
                                    projected_portfolio.adjust_cash(realized);
                                } else {
                                    let cost =
                                        trade.price * trade.quantity * instrument.multiplier();
                                    if trade.side == crate::model::OrderSide::Buy {
                                        projected_portfolio.adjust_cash(-cost);
                                    } else {
                                        projected_portfolio.adjust_cash(cost);
                                    }
                                }
                                let current_pos = projected_portfolio
                                    .positions
                                    .get(&trade.symbol)
                                    .copied()
                                    .unwrap_or(Decimal::ZERO);
                                let next_pos = crate::model::project_position_after(
                                    trade.side,
                                    trade.position_effect,
                                    current_pos,
                                    trade.quantity,
                                );
                                projected_portfolio
                                    .adjust_position(&trade.symbol, next_pos - current_pos);
                                current_free_margin = calculate_free_margin(
                                    &projected_portfolio,
                                    &prices_for_margin,
                                    ctx.instruments,
                                    ctx.trade_tracker,
                                    ctx.risk_config,
                                );
                            }

                            let keep_report = !matches!(
                                &report,
                                Event::ExecutionReport(_, Some(trade))
                                    if trade.quantity <= Decimal::ZERO
                            );
                            if keep_report {
                                synced_report = Some(report);
                            }
                        }
                    }
                }
                if let Some(report) = synced_report {
                    self.sync_order_from_report(&report);
                    reports.push(report);
                }
            }
        }

        self.cleanup_finished_orders();
        reports
    }
}

impl Default for SimulatedExecutionClient {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionClient for SimulatedExecutionClient {
    fn set_slippage_model(&mut self, model: Box<dyn SlippageModel>) {
        self.slippage_model = model;
    }

    fn set_volume_limit(&mut self, limit: f64) {
        self.volume_limit_pct = Decimal::from_f64(limit).unwrap_or_else(|| {
            log::warn!(
                "{}",
                render_log_message(
                    format!("Invalid volume limit {limit}, defaulting to 0.0"),
                    AkqLogContext::new().phase("execution"),
                )
            );
            Decimal::ZERO
        });
    }

    fn register_matcher(&mut self, asset_type: AssetType, matcher: Box<dyn ExecutionMatcher>) {
        self.matchers.insert(asset_type, matcher);
    }

    fn set_futures_validation_options(&mut self, enforce_tick_size: bool, enforce_lot_size: bool) {
        self.futures_enforce_tick_size = enforce_tick_size;
        self.futures_enforce_lot_size = enforce_lot_size;
        self.rebuild_futures_matcher();
    }

    fn set_futures_validation_options_by_prefix(
        &mut self,
        symbol_prefix: String,
        enforce_tick_size: Option<bool>,
        enforce_lot_size: Option<bool>,
    ) {
        let normalized = symbol_prefix.trim().to_uppercase();
        if normalized.is_empty() {
            return;
        }
        let mut updated = false;
        for (prefix, tick_opt, lot_opt) in &mut self.futures_validation_by_prefix {
            if prefix == &normalized {
                *tick_opt = enforce_tick_size;
                *lot_opt = enforce_lot_size;
                updated = true;
                break;
            }
        }
        if !updated {
            self.futures_validation_by_prefix.push((
                normalized,
                enforce_tick_size,
                enforce_lot_size,
            ));
        }
        self.rebuild_futures_matcher();
    }

    fn on_order(&mut self, order: Order) {
        // 模拟交易所接收订单
        let mut order = order;
        if order.status == OrderStatus::New {
            order.status = OrderStatus::Submitted;
        }
        let id = order.id.clone();
        self.orders.insert(id.clone(), order);
        self.order_queue.push(id);
    }

    fn on_cancel(&mut self, order_id: &str) {
        match self.orders.get_mut(order_id) {
            Some(order)
                if order.status == OrderStatus::Submitted
                    || order.status == OrderStatus::PartiallyFilled
                    || order.status == OrderStatus::New =>
            {
                order.status = OrderStatus::Cancelled;
            }
            Some(order) => {
                log::warn!("{}", Self::ignored_non_cancellable_cancel_warning(order));
            }
            None => {
                log::warn!("{}", Self::ignored_unknown_cancel_warning(order_id));
            }
        }
    }

    fn on_event(&mut self, event: &Event, ctx: &crate::context::EngineContext) -> Vec<Event> {
        self.prepare_slice_tracking(ctx.current_time);
        let mut reports = Vec::new();

        if ctx.session == TradingSession::Break
            || ctx.session == TradingSession::Closed
            || ctx.session == TradingSession::PreOpen
            || ctx.session == TradingSession::PostClose
        {
            if ctx.session == TradingSession::Closed {
                let timestamp = match event {
                    Event::Bar(b) => b.timestamp,
                    Event::Tick(t) => t.timestamp,
                    _ => 0,
                };

                for order_id in &self.order_queue {
                    if let Some(order) = self.orders.get_mut(order_id)
                        && Self::is_order_active(order)
                        && order.time_in_force == TimeInForce::Day
                    {
                        log::warn!("{}", Self::expired_day_order_warning(order, timestamp));
                        order.status = OrderStatus::Expired;
                        order.updated_at = timestamp;
                        reports.push(Event::ExecutionReport(order.clone(), None));
                    }
                }
                self.cleanup_finished_orders();
            }
            return reports;
        }

        self.match_with_events(
            std::slice::from_ref(event),
            ctx,
            |_order, _event| true,
            true,
        )
    }

    fn finalize_timestamp(
        &mut self,
        events: &[Event],
        ctx: &crate::context::EngineContext,
    ) -> Vec<Event> {
        if events.is_empty() {
            self.clear_slice_tracking();
            return Vec::new();
        }
        self.prepare_slice_tracking(ctx.current_time);
        let mut ordered_events: Vec<Event> = events.to_vec();
        let reduce_symbols: HashSet<String> = self
            .orders
            .values()
            .filter(|order| {
                Self::is_order_active(order)
                    && order.created_at == ctx.current_time
                    && self.is_same_cycle_close_order(order, ctx)
                    && crate::model::is_reduce_first_order(order.side, order.position_effect)
            })
            .map(|order| order.symbol.clone())
            .collect();
        ordered_events.sort_by_key(|event| {
            let symbol = Self::event_symbol(event).unwrap_or_default().to_string();
            (
                if reduce_symbols.contains(&symbol) {
                    0_u8
                } else {
                    1_u8
                },
                symbol,
            )
        });
        let finalize_ids: HashSet<String> = self
            .orders
            .values()
            .filter(|order| self.should_finalize_order(order, ctx))
            .map(|order| order.id.clone())
            .collect();
        let reports = self.match_with_events(
            &ordered_events,
            ctx,
            |order, _event| finalize_ids.contains(&order.id),
            false,
        );
        self.clear_slice_tracking();
        reports
    }
}

fn calculate_free_margin(
    portfolio: &crate::portfolio::Portfolio,
    prices: &HashMap<String, Decimal>,
    instruments: &HashMap<String, crate::model::Instrument>,
    trade_tracker: &crate::analysis::TradeTracker,
    risk_config: &crate::risk::RiskConfig,
) -> Decimal {
    let metrics =
        calculate_account_metrics(portfolio, prices, instruments, trade_tracker, risk_config);
    metrics.equity - metrics.used_margin
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        AssetType, Bar, ExecutionPolicyCore, Instrument, PriceBasis, TemporalPolicy, TimeInForce,
    };
    use std::collections::HashMap;

    fn create_test_instruments() -> HashMap<String, Instrument> {
        use crate::model::instrument::{InstrumentEnum, StockInstrument};

        let mut map = HashMap::new();
        let aapl = Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: "AAPL".to_string(),
                lot_size: Decimal::from(100),
                tick_size: Decimal::new(1, 2),
                expiry_date: None,
                sellable_after_days: 1,
            }),
        };
        map.insert("AAPL".to_string(), aapl);
        map
    }

    fn create_test_order(
        symbol: &str,
        side: crate::model::OrderSide,
        order_type: crate::model::OrderType,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Order {
        use uuid::Uuid;
        let mut order = Order::test_new(
            &Uuid::new_v4().to_string(),
            symbol,
            side,
            order_type,
            quantity,
        );
        order.price = price;
        order.time_in_force = TimeInForce::Day;
        order.status = OrderStatus::New;
        order
    }

    fn create_test_bar(
        symbol: &str,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
    ) -> Bar {
        Bar {
            symbol: symbol.to_string(),
            timestamp: 1000,
            open,
            high,
            low,
            close,
            volume: Decimal::from(1000),
            extra: HashMap::new(),
        }
    }

    #[test]
    fn test_execution_market_order() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();
        let order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Market,
            Decimal::from(100),
            None,
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let portfolio = crate::portfolio::Portfolio {
            cash: Decimal::from(100000),
            positions: HashMap::new().into(),
            available_positions: HashMap::new().into(),
        };
        let last_prices = HashMap::new();
        let risk_manager = crate::risk::RiskManager::new();
        let trade_tracker = crate::analysis::TradeTracker::new();

        let china_config = crate::market::ChinaMarketConfig {
            stock: Some(crate::market::stock::StockConfig::default()),
            option: Some(crate::market::option::OptionConfig::default()),
            ..Default::default()
        };
        let market_config = crate::market::MarketConfig::China(china_config);

        let market_model = market_config.create_model();

        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio: &portfolio,
            last_prices: &last_prices,
            trade_tracker: &trade_tracker,
            market_model: market_model.as_ref(),
            execution_policy_core: ExecutionPolicyCore::default(),
            bar_index: 0,
            current_time: 0,
            session: TradingSession::Continuous,
            active_orders: &[],
            risk_config: &risk_manager.config,
            timezone_name: None,
            timezone_offset: 0,
        };

        let events = sim.on_event(&event, &ctx);

        let trades: Vec<crate::model::Trade> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(trade)) = e {
                    Some(trade.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, Decimal::from(100)); // Open price

        let _last_order = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(o, _) = e {
                    Some(o)
                } else {
                    None
                }
            })
            .next_back()
            .unwrap();
    }

    #[test]
    fn test_execution_market_order_next_close() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();
        let order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Market,
            Decimal::from(100),
            None,
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let portfolio = crate::portfolio::Portfolio {
            cash: Decimal::from(100000),
            positions: HashMap::new().into(),
            available_positions: HashMap::new().into(),
        };
        let last_prices = HashMap::new();
        let risk_manager = crate::risk::RiskManager::new();
        let trade_tracker = crate::analysis::TradeTracker::new();

        let china_config = crate::market::ChinaMarketConfig {
            stock: Some(crate::market::stock::StockConfig::default()),
            option: Some(crate::market::option::OptionConfig::default()),
            ..Default::default()
        };
        let market_config = crate::market::MarketConfig::China(china_config);
        let market_model = market_config.create_model();

        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio: &portfolio,
            last_prices: &last_prices,
            trade_tracker: &trade_tracker,
            market_model: market_model.as_ref(),
            execution_policy_core: ExecutionPolicyCore {
                price_basis: PriceBasis::Close,
                bar_offset: 1,
                temporal: TemporalPolicy::SameCycle,
            },
            bar_index: 0,
            current_time: 0,
            session: TradingSession::Continuous,
            active_orders: &[],
            risk_config: &risk_manager.config,
            timezone_name: None,
            timezone_offset: 0,
        };

        let events = sim.on_event(&event, &ctx);
        let trades: Vec<crate::model::Trade> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(trade)) = e {
                    Some(trade.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, Decimal::from(102));
    }

    #[test]
    fn test_execution_limit_buy() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();
        // Buy Limit @ 99.0
        let order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Limit,
            Decimal::from(100),
            Some(Decimal::from(99)),
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let portfolio = crate::portfolio::Portfolio {
            cash: Decimal::from(100000),
            positions: HashMap::new().into(),
            available_positions: HashMap::new().into(),
        };
        let last_prices = HashMap::new();
        let risk_manager = crate::risk::RiskManager::new();
        let trade_tracker = crate::analysis::TradeTracker::new();

        let china_config = crate::market::ChinaMarketConfig {
            stock: Some(crate::market::stock::StockConfig::default()),
            option: Some(crate::market::option::OptionConfig::default()),
            ..Default::default()
        };
        let market_config = crate::market::MarketConfig::China(china_config);

        let market_model = market_config.create_model();

        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio: &portfolio,
            last_prices: &last_prices,
            trade_tracker: &trade_tracker,
            market_model: market_model.as_ref(),
            execution_policy_core: ExecutionPolicyCore::default(),
            bar_index: 0,
            current_time: 0,
            session: TradingSession::Continuous,
            active_orders: &[],
            risk_config: &risk_manager.config,
            timezone_name: None,
            timezone_offset: 0,
        };

        let events = sim.on_event(&event, &ctx);

        // Should be filled at Limit Price (99)
        let trades: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(t)) = e {
                    Some(t)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, Decimal::from(99));
    }

    #[test]
    fn test_execution_limit_buy_no_fill() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();
        // Buy Limit @ 90.0 (Below Low 95)
        let order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Limit,
            Decimal::from(100),
            Some(Decimal::from(90)),
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let portfolio = crate::portfolio::Portfolio {
            cash: Decimal::from(100000),
            positions: HashMap::new().into(),
            available_positions: HashMap::new().into(),
        };
        let last_prices = HashMap::new();
        let risk_manager = crate::risk::RiskManager::new();
        let trade_tracker = crate::analysis::TradeTracker::new();

        let china_config = crate::market::ChinaMarketConfig {
            stock: Some(crate::market::stock::StockConfig::default()),
            ..Default::default()
        };
        let market_config = crate::market::MarketConfig::China(china_config);

        let market_model = market_config.create_model();

        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio: &portfolio,
            last_prices: &last_prices,
            trade_tracker: &trade_tracker,
            market_model: market_model.as_ref(),
            execution_policy_core: ExecutionPolicyCore::default(),
            bar_index: 0,
            current_time: 0,
            session: TradingSession::Continuous,
            active_orders: &[],
            risk_config: &risk_manager.config,
            timezone_name: None,
            timezone_offset: 0,
        };

        let events = sim.on_event(&event, &ctx);

        let trades: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(t)) = e {
                    Some(t)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(trades.len(), 0);
    }

    #[test]
    fn test_dynamic_position_sizing() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();

        // Buy 1000 shares @ 100 = 100,000 value.
        // But cash is only 50,000.
        let order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Market,
            Decimal::from(1000),
            None,
        );
        let mut order = order;
        order.allow_quantity_auto_resize = true;
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let portfolio = crate::portfolio::Portfolio {
            cash: Decimal::from(50000), // Only 50k
            positions: HashMap::new().into(),
            available_positions: HashMap::new().into(),
        };
        let last_prices = HashMap::new();
        let risk_manager = crate::risk::RiskManager::new();
        let trade_tracker = crate::analysis::TradeTracker::new();

        let china_config = crate::market::ChinaMarketConfig {
            stock: Some(crate::market::stock::StockConfig::default()),
            ..Default::default()
        };
        let market_config = crate::market::MarketConfig::China(china_config);

        let market_model = market_config.create_model();

        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio: &portfolio,
            last_prices: &last_prices,
            trade_tracker: &trade_tracker,
            market_model: market_model.as_ref(),
            execution_policy_core: ExecutionPolicyCore::default(),
            bar_index: 0,
            current_time: 0,
            session: TradingSession::Continuous,
            active_orders: &[],
            risk_config: &risk_manager.config,
            timezone_name: None,
            timezone_offset: 0,
        };

        let events = sim.on_event(&event, &ctx);

        let trades: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(t)) = e {
                    Some(t)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(trades.len(), 1);

        // Should be reduced to approx 400 shares (due to lot size 100 and safety margin)
        // 50000 / 100 = 500. 500 * 0.9999 = 499.95 -> floor to 400 (lot size 100)
        assert_eq!(trades[0].quantity, Decimal::from(400));
    }

    #[test]
    fn test_dynamic_position_sizing_for_short_option_margin() {
        use crate::model::instrument::{InstrumentEnum, OptionInstrument, StockInstrument};
        use crate::model::{OptionMarginModel, OptionType};
        use rust_decimal_macros::dec;

        let mut sim = SimulatedExecutionClient::new();
        let mut instruments = HashMap::new();
        instruments.insert(
            "OPT_P".to_string(),
            Instrument {
                asset_type: AssetType::Option,
                inner: InstrumentEnum::Option(OptionInstrument {
                    symbol: "OPT_P".to_string(),
                    multiplier: dec!(100),
                    margin_ratio: dec!(0.2),
                    tick_size: dec!(0.01),
                    option_margin_model: OptionMarginModel::USBrokerSingleLegVolAdjusted,
                    option_type: OptionType::Put,
                    strike_price: dec!(100),
                    expiry_date: 20260101,
                    underlying_symbol: "UL".to_string(),
                    settlement_type: None,
                    implied_volatility: Some(dec!(0.3)),
                    reference_volatility: Some(dec!(0.2)),
                }),
            },
        );
        instruments.insert(
            "UL".to_string(),
            Instrument {
                asset_type: AssetType::Stock,
                inner: InstrumentEnum::Stock(StockInstrument {
                    symbol: "UL".to_string(),
                    lot_size: Decimal::from(100),
                    tick_size: Decimal::new(1, 2),
                    expiry_date: None,
                    sellable_after_days: 1,
                }),
            },
        );

        let mut order = create_test_order(
            "OPT_P",
            crate::model::OrderSide::Sell,
            crate::model::OrderType::Market,
            Decimal::from(2),
            None,
        );
        order.allow_quantity_auto_resize = true;
        sim.on_order(order);

        let bar = create_test_bar(
            "OPT_P",
            Decimal::from(4),
            Decimal::from(4),
            Decimal::from(4),
            Decimal::from(4),
        );
        let event = Event::Bar(bar);

        let portfolio = crate::portfolio::Portfolio {
            cash: Decimal::from(6000),
            positions: HashMap::new().into(),
            available_positions: HashMap::new().into(),
        };
        let mut last_prices = HashMap::new();
        last_prices.insert("UL".to_string(), Decimal::from(95));
        let risk_manager = crate::risk::RiskManager::new();
        let trade_tracker = crate::analysis::TradeTracker::new();

        let china_config = crate::market::ChinaMarketConfig {
            stock: Some(crate::market::stock::StockConfig::default()),
            option: Some(crate::market::option::OptionConfig::default()),
            ..Default::default()
        };
        let market_config = crate::market::MarketConfig::China(china_config);
        let market_model = market_config.create_model();

        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio: &portfolio,
            last_prices: &last_prices,
            trade_tracker: &trade_tracker,
            market_model: market_model.as_ref(),
            execution_policy_core: ExecutionPolicyCore::default(),
            bar_index: 0,
            current_time: 0,
            session: TradingSession::Continuous,
            active_orders: &[],
            risk_config: &risk_manager.config,
            timezone_name: None,
            timezone_offset: 0,
        };

        let events = sim.on_event(&event, &ctx);

        let trades: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(t)) = e {
                    Some(t)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, Decimal::ONE);
        assert_eq!(trades[0].price, Decimal::from(4));
    }

    #[test]
    fn test_short_option_rejected_when_margin_insufficient_without_resize() {
        use crate::model::instrument::{InstrumentEnum, OptionInstrument, StockInstrument};
        use crate::model::{OptionMarginModel, OptionType};
        use rust_decimal_macros::dec;

        let mut sim = SimulatedExecutionClient::new();
        let mut instruments = HashMap::new();
        instruments.insert(
            "OPT_P".to_string(),
            Instrument {
                asset_type: AssetType::Option,
                inner: InstrumentEnum::Option(OptionInstrument {
                    symbol: "OPT_P".to_string(),
                    multiplier: dec!(100),
                    margin_ratio: dec!(0.2),
                    tick_size: dec!(0.01),
                    option_margin_model: OptionMarginModel::USBrokerSingleLegVolAdjusted,
                    option_type: OptionType::Put,
                    strike_price: dec!(100),
                    expiry_date: 20260101,
                    underlying_symbol: "UL".to_string(),
                    settlement_type: None,
                    implied_volatility: Some(dec!(0.3)),
                    reference_volatility: Some(dec!(0.2)),
                }),
            },
        );
        instruments.insert(
            "UL".to_string(),
            Instrument {
                asset_type: AssetType::Stock,
                inner: InstrumentEnum::Stock(StockInstrument {
                    symbol: "UL".to_string(),
                    lot_size: Decimal::from(100),
                    tick_size: Decimal::new(1, 2),
                    expiry_date: None,
                    sellable_after_days: 1,
                }),
            },
        );

        let order = create_test_order(
            "OPT_P",
            crate::model::OrderSide::Sell,
            crate::model::OrderType::Market,
            Decimal::from(2),
            None,
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "OPT_P",
            Decimal::from(4),
            Decimal::from(4),
            Decimal::from(4),
            Decimal::from(4),
        );
        let event = Event::Bar(bar);

        let portfolio = crate::portfolio::Portfolio {
            cash: Decimal::from(6000),
            positions: HashMap::new().into(),
            available_positions: HashMap::new().into(),
        };
        let mut last_prices = HashMap::new();
        last_prices.insert("UL".to_string(), Decimal::from(95));
        let risk_manager = crate::risk::RiskManager::new();
        let trade_tracker = crate::analysis::TradeTracker::new();

        let china_config = crate::market::ChinaMarketConfig {
            stock: Some(crate::market::stock::StockConfig::default()),
            option: Some(crate::market::option::OptionConfig::default()),
            ..Default::default()
        };
        let market_config = crate::market::MarketConfig::China(china_config);
        let market_model = market_config.create_model();

        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio: &portfolio,
            last_prices: &last_prices,
            trade_tracker: &trade_tracker,
            market_model: market_model.as_ref(),
            execution_policy_core: ExecutionPolicyCore::default(),
            bar_index: 0,
            current_time: 0,
            session: TradingSession::Continuous,
            active_orders: &[],
            risk_config: &risk_manager.config,
            timezone_name: None,
            timezone_offset: 0,
        };

        let events = sim.on_event(&event, &ctx);

        let reports: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(order, trade) = e {
                    Some((order, trade))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].1.is_none());
        assert_eq!(reports[0].0.status, OrderStatus::Rejected);
        assert!(
            reports[0]
                .0
                .reject_reason
                .contains("Insufficient margin at execution")
        );
    }

    #[test]
    fn test_insufficient_margin_warning_includes_order_context() {
        let mut order = create_test_order(
            "OPT_P",
            crate::model::OrderSide::Sell,
            crate::model::OrderType::Market,
            Decimal::from(2),
            None,
        );
        order.owner_strategy_id = Some("alpha".to_string());

        let rendered = SimulatedExecutionClient::insufficient_margin_warning(
            &order,
            1_000_000_000,
            Decimal::from(7000),
            Decimal::from(6000),
        );

        assert!(rendered.contains("Rejected order due to insufficient margin during execution"));
        assert!(rendered.contains("\"phase\":\"execution\""));
        assert!(rendered.contains("\"symbol\":\"OPT_P\""));
        assert!(rendered.contains(&format!("\"order_id\":\"{}\"", order.id)));
        assert!(rendered.contains("\"strategy_id\":\"alpha\""));
        assert!(rendered.contains("\"slot\":\"alpha\""));
        assert!(rendered.contains("\"event_time_iso\":\"1970-01-01T00:00:01Z\""));
    }

    #[test]
    fn test_expired_day_order_warning_includes_order_context() {
        let mut order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Limit,
            Decimal::from(100),
            Some(Decimal::from(10)),
        );
        order.owner_strategy_id = Some("beta".to_string());

        let rendered = SimulatedExecutionClient::expired_day_order_warning(&order, 2_000_000_000);

        assert!(rendered.contains("Expired day order at session close"));
        assert!(rendered.contains("\"phase\":\"execution\""));
        assert!(rendered.contains("\"symbol\":\"AAPL\""));
        assert!(rendered.contains(&format!("\"order_id\":\"{}\"", order.id)));
        assert!(rendered.contains("\"strategy_id\":\"beta\""));
        assert!(rendered.contains("\"slot\":\"beta\""));
        assert!(rendered.contains("\"event_time_iso\":\"1970-01-01T00:00:02Z\""));
    }

    #[test]
    fn test_ignored_unknown_cancel_warning_includes_order_id() {
        let rendered = SimulatedExecutionClient::ignored_unknown_cancel_warning("ghost-order");

        assert!(rendered.contains("Ignored cancel request for unknown order"));
        assert!(rendered.contains("\"phase\":\"execution\""));
        assert!(rendered.contains("\"order_id\":\"ghost-order\""));
    }

    #[test]
    fn test_ignored_non_cancellable_cancel_warning_includes_order_context() {
        let mut order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Limit,
            Decimal::from(100),
            Some(Decimal::from(10)),
        );
        order.owner_strategy_id = Some("gamma".to_string());
        order.status = OrderStatus::Filled;
        order.updated_at = 3_000_000_000;

        let rendered = SimulatedExecutionClient::ignored_non_cancellable_cancel_warning(&order);

        assert!(
            rendered.contains(
                "Ignored cancel request because order is not cancellable in status Filled"
            )
        );
        assert!(rendered.contains("\"phase\":\"execution\""));
        assert!(rendered.contains("\"symbol\":\"AAPL\""));
        assert!(rendered.contains(&format!("\"order_id\":\"{}\"", order.id)));
        assert!(rendered.contains("\"strategy_id\":\"gamma\""));
        assert!(rendered.contains("\"slot\":\"gamma\""));
        assert!(rendered.contains("\"event_time_iso\":\"1970-01-01T00:00:03Z\""));
    }

    #[test]
    fn test_deferred_same_cycle_order_warning_includes_order_context() {
        let mut order = create_test_order(
            "MSFT",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Limit,
            Decimal::from(50),
            Some(Decimal::from(20)),
        );
        order.owner_strategy_id = Some("delta".to_string());

        let rendered =
            SimulatedExecutionClient::deferred_same_cycle_order_warning(&order, 4_000_000_000);

        assert!(rendered.contains(
            "Deferred same-cycle order until cross-symbol reduce-first orders finish in current slice"
        ));
        assert!(rendered.contains("\"phase\":\"execution\""));
        assert!(rendered.contains("\"symbol\":\"MSFT\""));
        assert!(rendered.contains(&format!("\"order_id\":\"{}\"", order.id)));
        assert!(rendered.contains("\"strategy_id\":\"delta\""));
        assert!(rendered.contains("\"slot\":\"delta\""));
        assert!(rendered.contains("\"event_time_iso\":\"1970-01-01T00:00:04Z\""));
    }
}
