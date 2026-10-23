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
    AssetType, ExecutionPolicyCore, Order, OrderStatus, PriceBasis, TemporalPolicy, TradingSession,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::{HashMap, HashSet};

/// 竞价期内记录的一笔行情。
#[derive(Debug, Clone, Copy)]
struct AuctionQuote {
    price: Decimal,
    /// 最近一笔非零成交量(零量快照只更新价格, 不覆盖已记录的量)。
    volume: Decimal,
    /// 设定该价格的事件时间戳: 开盘统一撮合只纳入在它之前下的单。
    ts: i64,
    /// 价格来自 tick: tick 撮合不看成交量, 合成事件的量不约束成交。
    from_tick: bool,
}

impl AuctionQuote {
    fn updated(
        prev: Option<Self>,
        price: Decimal,
        volume: Decimal,
        ts: i64,
        from_tick: bool,
    ) -> Self {
        let volume = if volume > Decimal::ZERO {
            volume
        } else {
            prev.map_or(Decimal::ZERO, |q| q.volume)
        };
        Self {
            price,
            volume,
            ts,
            from_tick,
        }
    }

    /// 合成事件的成交量: 来自 tick 时不约束(成交量限制与停牌判断都不生效)。
    fn synthetic_volume(&self) -> Decimal {
        if self.from_tick {
            Decimal::MAX
        } else {
            self.volume
        }
    }
}

/// 单个品种当日的集合竞价状态。交易日变化时重置。
#[derive(Debug, Clone, Default)]
struct AuctionState {
    trading_day: Option<chrono::NaiveDate>,
    /// 开盘竞价期内设定的竞价价(竞价价来源 ①)。
    opening: Option<AuctionQuote>,
    opening_done: bool,
    /// 收盘竞价期内最后一笔行情。
    closing: Option<AuctionQuote>,
    closing_done: bool,
}

/// 一次集合竞价统一撮合: 价格全为竞价价的合成事件, 只纳入 `symbol` 在 `cutoff`
/// 之前下的在途订单, 撮合时该品种视为连续竞价。
struct Uncross {
    event: Event,
    symbol: String,
    cutoff: i64,
}

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
    stock_enforce_tick_size: bool,
    futures_enforce_tick_size: bool,
    futures_enforce_lot_size: bool,
    futures_validation_by_prefix: Vec<(String, Option<bool>, Option<bool>)>,
    slice_tracking_timestamp: Option<i64>,
    attempted_same_cycle_order_ids: HashSet<String>,
    deferred_same_cycle_order_ids: HashSet<String>,
    /// 品种 -> 当日集合竞价状态。
    auctions: HashMap<String, AuctionState>,
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

    fn rebuild_stock_matcher(&mut self) {
        self.matchers.insert(
            AssetType::Stock,
            Box::new(stock::StockMatcher::new(self.stock_enforce_tick_size)),
        );
        self.matchers.insert(
            AssetType::Fund,
            Box::new(stock::StockMatcher::new(self.stock_enforce_tick_size)),
        );
    }

    pub fn new() -> Self {
        let mut matchers: HashMap<AssetType, Box<dyn ExecutionMatcher>> = HashMap::new();
        matchers.insert(AssetType::Stock, Box::new(stock::StockMatcher::default()));
        matchers.insert(AssetType::Fund, Box::new(stock::StockMatcher::default())); // Fund uses StockMatcher
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
            stock_enforce_tick_size: true,
            futures_enforce_tick_size: true,
            futures_enforce_lot_size: true,
            futures_validation_by_prefix: Vec::new(),
            slice_tracking_timestamp: None,
            attempted_same_cycle_order_ids: HashSet::new(),
            deferred_same_cycle_order_ids: HashSet::new(),
            auctions: HashMap::new(),
        }
    }

    /// 价格全为竞价价的合成 bar: 复用 match_order 的手续费、滑点、保证金、成交量限制。
    fn auction_event(symbol: &str, ts: i64, quote: &AuctionQuote) -> Event {
        Event::Bar(crate::model::Bar {
            timestamp: ts,
            open: quote.price,
            high: quote.price,
            low: quote.price,
            close: quote.price,
            volume: quote.synthetic_volume(),
            symbol: symbol.to_string(),
            extra: HashMap::new(),
            freq: None,
            interval_start: None,
        })
    }

    /// 在 `ts` 按 `quote` 对 `symbol` 在 `cutoff` 之前下的订单统一撮合; 没有这样的
    /// 在途订单时返回 None(不必构造合成事件、不跑撮合)。
    fn build_uncross(
        &self,
        symbol: &str,
        ts: i64,
        cutoff: i64,
        quote: &AuctionQuote,
    ) -> Option<Uncross> {
        let has_candidates = self.orders.values().any(|order| {
            Self::is_order_active(order) && order.symbol == symbol && order.created_at < cutoff
        });
        has_candidates.then(|| Uncross {
            event: Self::auction_event(symbol, ts, quote),
            symbol: symbol.to_string(),
            cutoff,
        })
    }

    /// 记录本事件的竞价行情, 返回本事件要做的统一撮合。
    fn auction_action(
        &mut self,
        symbol: &str,
        ts: i64,
        price: Decimal,
        volume: Decimal,
        from_tick: bool,
        ctx: &crate::context::EngineContext,
    ) -> Option<Uncross> {
        let session = ctx.sessions.session_for(symbol, ts);
        if !matches!(
            session,
            TradingSession::OpeningAuction
                | TradingSession::AuctionHold
                | TradingSession::ClosingAuction
                | TradingSession::Continuous
        ) {
            return None;
        }
        // 收盘竞价是否在本事件结束(15:00 bar / 15:00 tick): 终点戳事件取 ts - 1
        // 所在段, 本事件到达段终点即结束。
        let closing_ends_here = session == TradingSession::ClosingAuction
            && ctx
                .sessions
                .resolver
                .and_then(|r| {
                    let asset_type = ctx
                        .sessions
                        .instruments
                        .and_then(|m| m.get(symbol))
                        .map(|i| i.asset_type);
                    r.segment_span_at(symbol, asset_type, ts.saturating_sub(1))
                        .ok()
                })
                .is_some_and(|(_, end, _)| ts >= end);
        let state = self.auctions.entry(symbol.to_string()).or_default();
        if state.trading_day != ctx.trading_day {
            *state = AuctionState {
                trading_day: ctx.trading_day,
                ..Default::default()
            };
        }
        // (报价, 只纳入此前下的单的截止时刻)
        let pending: Option<(AuctionQuote, i64)> = match session {
            TradingSession::OpeningAuction => {
                state.opening = Some(AuctionQuote::updated(
                    state.opening,
                    price,
                    volume,
                    ts,
                    from_tick,
                ));
                None
            }
            TradingSession::AuctionHold => {
                // 09:25～09:30 不形成新价: 只在尚无竞价价时设定(Ruling S3-T3-3)
                if state.opening.is_none() {
                    state.opening = Some(AuctionQuote::updated(None, price, volume, ts, from_tick));
                }
                None
            }
            TradingSession::ClosingAuction => {
                let quote = AuctionQuote::updated(state.closing, price, volume, ts, from_tick);
                state.closing = Some(quote);
                if closing_ends_here && !state.closing_done {
                    state.closing_done = true;
                    // 结束收盘竞价的事件上新下的单不参加(Ruling S3-5)
                    Some((quote, ts))
                } else {
                    None
                }
            }
            _ => {
                // 连续竞价: 首个连续竞价事件结束开盘竞价。只在观察到竞价期行情时统一
                // 撮合(Ruling S3-1); 没有竞价期行情时, 本事件上的常规撮合即来源 ②。
                // 竞价价已知后下的单进入连续竞价(Ruling S3-T3-3)。
                if state.opening_done {
                    None
                } else {
                    state.opening_done = true;
                    state.opening.map(|q| (q, q.ts))
                }
            }
        };
        pending.and_then(|(quote, cutoff)| self.build_uncross(symbol, ts, cutoff, &quote))
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

    /// Whether the order is scheduled to attempt its fill during the current
    /// timestamp. Same-cycle-close orders fill at the close of their creation
    /// bar (created_at == now); next-open orders fill at the open of the first
    /// bar after submission (created_at < now while still active).
    fn fills_current_cycle(&self, order: &Order, ctx: &crate::context::EngineContext) -> bool {
        let policy = Self::effective_policy(order, ctx);
        if Self::is_same_cycle_close_policy(policy) {
            order.created_at == ctx.current_time
        } else {
            order.created_at < ctx.current_time
        }
    }

    /// A co-submitted (same created_at) cross-symbol reduce/close order that
    /// also fills this cycle — its proceeds should fund `order` before it fills.
    fn has_cosubmitted_cross_symbol_reduce_pending(
        &self,
        order: &Order,
        ctx: &crate::context::EngineContext,
    ) -> bool {
        self.orders.values().any(|candidate| {
            Self::is_order_active(candidate)
                && candidate.symbol != order.symbol
                && candidate.created_at == order.created_at
                && self.fills_current_cycle(candidate, ctx)
                && crate::model::is_reduce_first_order(candidate.side, candidate.position_effect)
        })
    }

    fn should_defer_order_on_current_event(
        &self,
        order: &Order,
        event: &Event,
        ctx: &crate::context::EngineContext,
    ) -> bool {
        if !self.fills_current_cycle(order, ctx) {
            return false;
        }
        if crate::model::is_reduce_first_order(order.side, order.position_effect) {
            return false;
        }
        Self::event_symbol(event)
            .map(|symbol| order.symbol == symbol)
            .unwrap_or(false)
            && self.has_cosubmitted_cross_symbol_reduce_pending(order, ctx)
    }

    fn should_finalize_order(&self, order: &Order, ctx: &crate::context::EngineContext) -> bool {
        Self::is_order_active(order)
            && (self.deferred_same_cycle_order_ids.contains(&order.id)
                || (order.created_at == ctx.current_time
                    && self.is_same_cycle_close_order(order, ctx)
                    && !self.attempted_same_cycle_order_ids.contains(&order.id)))
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
        uncrosses: &[Uncross],
    ) -> Vec<Event>
    where
        F: FnMut(&Order, &Event) -> bool,
    {
        let mut reports = Vec::new();
        let mut projected_portfolio = ctx.portfolio.clone();
        // Fill-time mirror of `portfolio.available_positions`, advanced by each
        // fill in this slice so cumulative sells cannot oversell a position
        // whose submission-time check was deferred (issue #391).
        let mut projected_available_positions = ctx.portfolio.available_positions.as_ref().clone();
        let stock_margin_ratio_override = stock_margin_ratio_override(ctx.risk_config);
        let mut current_free_margin = calculate_free_margin(
            &projected_portfolio,
            ctx.last_prices,
            ctx.instruments,
            ctx.trade_tracker,
            ctx.risk_config,
        );

        // 集合竞价统一撮合的合成事件排在真实事件之前, 与之共享同一份资金推演
        // (projected_portfolio / current_free_margin), 不会超额成交。
        let passes = uncrosses
            .iter()
            .map(|u| (&u.event, Some(u)))
            .chain(events.iter().map(|e| (e, None)));
        for (event, uncross) in passes {
            let queue = self.sorted_queue();
            for order_id in &queue {
                let mut synced_report: Option<Event> = None;
                let Some(order_snapshot) = self.orders.get(order_id).cloned() else {
                    continue;
                };
                let selected = match uncross {
                    Some(u) => {
                        order_snapshot.symbol == u.symbol && order_snapshot.created_at < u.cutoff
                    }
                    None => order_filter(&order_snapshot, event),
                };
                if !Self::is_order_active(&order_snapshot) || !selected {
                    continue;
                }

                // A next-open (bar_offset>=1) order must not fill on any event at or
                // before its submission timestamp — including a same-timestamp bar of a
                // different symbol processed after the order was queued (e.g. a strategy
                // that submits a cross-symbol order inside another symbol's bar callback).
                // Without this, such an order fills on its own creation-timestamp bar
                // instead of the next one, skewing cross-symbol switch timing (#307).
                let order_policy = Self::effective_policy(&order_snapshot, ctx);
                if order_policy.bar_offset >= 1
                    && Self::event_timestamp(event)
                        .map(|ts| ts <= order_snapshot.created_at)
                        .unwrap_or(false)
                {
                    continue;
                }

                // 非连续竞价时段(竞价、午休、闭市、盘后): 订单排队, 等下一可成交时段
                // (竞价时段的统一撮合见集合竞价撮合)。只对本品种的行情与定时器判断;
                // 其他品种的行情撮合器本就不会让它成交, 不必查时段。
                // 统一撮合的合成事件上该品种视为连续竞价。
                let own_event = uncross.is_none()
                    && Self::event_symbol(event)
                        .is_none_or(|symbol| symbol == order_snapshot.symbol);
                if own_event {
                    let event_ts = Self::event_timestamp(event).unwrap_or(ctx.current_time);
                    if ctx.sessions.session_for(&order_snapshot.symbol, event_ts)
                        != TradingSession::Continuous
                    {
                        continue;
                    }
                }

                let defer_current = uncross.is_none()
                    && defer_cross_symbol_increase
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
                                    projected_portfolio
                                        .positions
                                        .get(&trade.symbol)
                                        .copied()
                                        .unwrap_or(Decimal::ZERO),
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
                                // Signed: negative when this fill releases margin.
                                // Clamping it here instead of the sum would drop
                                // the released funds and leave a reduce-only sell
                                // demanding its commission in cash up front —
                                // rejecting sells in a fully-invested account
                                // (#400). Mirrors risk::common::check_affordability.
                                let margin_required = next_used_margin - base_used_margin;
                                let total_required =
                                    (margin_required + commission).max(Decimal::ZERO);

                                // Execution-time affordability intentionally does
                                // NOT reuse risk::common::check_affordability: it
                                // gates on `current_free_margin` derived from
                                // calculate_account_metrics (equity incl. futures
                                // unrealized PnL, futures-accurate) at the real
                                // fill price with no safety haircut. The submission
                                // gate uses portfolio free margin at the last/limit
                                // price with a safety buffer — a deliberate
                                // forecast-vs-actual split, not duplicated logic.
                                //
                                // When check_cash is disabled the account owner has
                                // opted into overdrafting (negative cash/margin) so
                                // the backtest can run to completion — the submission
                                // gate already skips its check (risk::common /
                                // risk::futures), so this execution gate must skip
                                // too. Both auto-resize and rejection are bypassed:
                                // there is no affordability budget to size against,
                                // so the order fills in full (#280).
                                //
                                // `total_required > 0` is load-bearing, not
                                // decoration: free margin can legitimately be
                                // negative (an underwater margin account), and
                                // without this guard EVERY zero-cost order —
                                // including the reduce-only sells that are the
                                // account's only way out — is rejected with the
                                // nonsensical `Required: 0, Available: -20007`
                                // and the account stays frozen forever (#400).
                                if ctx.risk_config.check_cash
                                    && total_required > Decimal::ZERO
                                    && total_required > current_free_margin
                                {
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
                                                projected_portfolio
                                                    .positions
                                                    .get(&trade.symbol)
                                                    .copied()
                                                    .unwrap_or(Decimal::ZERO),
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

                            // Fill-time counterpart of StockAvailablePositionRule,
                            // which the channel stage defers for delayed orders
                            // (issue #391). Availability is read from the
                            // slice-local projection, so settlement has already
                            // unlocked T+1 positions here.
                            if replacement_report.is_none()
                                && let Event::ExecutionReport(_, Some(trade)) = &report
                                && trade.quantity > Decimal::ZERO
                                && trade.side == crate::model::OrderSide::Sell
                                && matches!(
                                    instrument.asset_type,
                                    AssetType::Stock | AssetType::Fund
                                )
                                && ctx.risk_config.active
                                && !(ctx.risk_config.is_margin_account()
                                    && ctx.risk_config.enable_short_sell)
                            {
                                let available = projected_available_positions
                                    .get(&trade.symbol)
                                    .copied()
                                    .unwrap_or(Decimal::ZERO);
                                if available < trade.quantity {
                                    let mut rejected_order = order_snapshot.clone();
                                    rejected_order.status = crate::model::OrderStatus::Rejected;
                                    rejected_order.filled_quantity = Decimal::ZERO;
                                    rejected_order.average_filled_price = None;
                                    rejected_order.updated_at = ctx.current_time;
                                    rejected_order.reject_reason = format!(
                                        "Risk: Insufficient available position for {} at execution. Available: {}, Required: {}",
                                        trade.symbol, available, trade.quantity
                                    );
                                    log::warn!(
                                        "{}",
                                        render_log_message(
                                            format!(
                                                "Rejected order: {}",
                                                rejected_order.reject_reason
                                            ),
                                            Self::order_log_context(
                                                &rejected_order,
                                                ctx.current_time,
                                            ),
                                        )
                                    );
                                    replacement_report =
                                        Some(Event::ExecutionReport(rejected_order, None));
                                }
                            }

                            if let Some(new_report) = replacement_report {
                                report = new_report;
                            }

                            if let Event::ExecutionReport(_, Some(ref trade)) = report
                                && trade.quantity > Decimal::ZERO
                            {
                                if matches!(
                                    instrument.asset_type,
                                    AssetType::Stock | AssetType::Fund
                                ) {
                                    ctx.market_model.update_available_position(
                                        &mut projected_available_positions,
                                        instrument,
                                        trade.quantity,
                                        trade.side,
                                    );
                                }
                                let mut prices_for_margin = ctx.last_prices.clone();
                                prices_for_margin.insert(trade.symbol.clone(), trade.price);
                                let commission = ctx.market_model.calculate_commission(
                                    instrument,
                                    trade.side,
                                    trade.price,
                                    trade.quantity,
                                    projected_portfolio
                                        .positions
                                        .get(&trade.symbol)
                                        .copied()
                                        .unwrap_or(Decimal::ZERO),
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

    fn set_stock_validation_options(&mut self, enforce_tick_size: bool) {
        self.stock_enforce_tick_size = enforce_tick_size;
        self.rebuild_stock_matcher();
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
        // 集合竞价: 竞价时段只记价; 开盘竞价在首个连续竞价事件上、收盘竞价在结束它的
        // 事件上按竞价价统一撮合(与本事件的常规撮合同一次调用, 排在它之前)。
        let quote = match event {
            Event::Bar(b) => Some((b.symbol.as_str(), b.timestamp, b.close, b.volume, false)),
            Event::Tick(t) => Some((t.symbol.as_str(), t.timestamp, t.price, t.volume, true)),
            _ => None,
        };
        let uncross = quote.and_then(|(symbol, ts, price, volume, from_tick)| {
            self.auction_action(symbol, ts, price, volume, from_tick, ctx)
        });
        // 时段按订单品种在 match_with_events 里判断(非连续竞价时段排队), 统一撮合
        // 之后对本事件的常规撮合不会让竞价时段的订单重复成交;
        // Day 单只在日终结算过期, 不因闭市时段的行情作废。
        self.match_with_events(
            std::slice::from_ref(event),
            ctx,
            |_order, _event| true,
            true,
            uncross.as_slice(),
        )
    }

    fn finalize_closing_auctions(&mut self, ctx: &crate::context::EngineContext) -> Vec<Event> {
        // 当日收盘竞价有行情但没有结束它的事件(缺收盘时刻的 bar/tick):
        // 按最后一个收盘竞价事件的价格在收盘时刻(ctx.current_time)统一撮合。
        let mut pending: Vec<(String, AuctionQuote)> = self
            .auctions
            .iter_mut()
            .filter(|(_, s)| s.trading_day == ctx.trading_day && !s.closing_done)
            .filter_map(|(sym, s)| {
                s.closing.map(|q| {
                    s.closing_done = true;
                    (sym.clone(), q)
                })
            })
            .collect();
        // HashMap 迭代无序: 按品种排序, 撮合顺序可复现
        pending.sort_by(|a, b| a.0.cmp(&b.0));
        let close_ts = ctx.current_time;
        let uncrosses: Vec<Uncross> = pending
            .iter()
            .filter_map(|(sym, q)| self.build_uncross(sym, close_ts, close_ts, q))
            .collect();
        if uncrosses.is_empty() {
            return Vec::new();
        }
        // 所有品种的合成事件一次撮合, 共享资金推演
        self.match_with_events(&[], ctx, |_order, _event| false, false, &uncrosses)
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
            &[],
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
            freq: None,
            interval_start: None,
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
            trading_day: None,
            sessions: crate::context::SessionView::continuous(),
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
            trading_day: None,
            sessions: crate::context::SessionView::continuous(),
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
            trading_day: None,
            sessions: crate::context::SessionView::continuous(),
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
            trading_day: None,
            sessions: crate::context::SessionView::continuous(),
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
            trading_day: None,
            sessions: crate::context::SessionView::continuous(),
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
                    settlement_price: None,
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
            trading_day: None,
            sessions: crate::context::SessionView::continuous(),
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
                    settlement_price: None,
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
            trading_day: None,
            sessions: crate::context::SessionView::continuous(),
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
    fn test_short_option_fills_when_check_cash_disabled_despite_insufficient_margin() {
        // #280: with check_cash=false the account owner opts into overdrafting,
        // so the execution-time margin gate must not reject — the order fills in
        // full and cash/margin is allowed to go negative. Mirrors the rejection
        // test above but flips check_cash off.
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
                    settlement_price: None,
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
        let mut risk_manager = crate::risk::RiskManager::new();
        risk_manager.config.check_cash = false;
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
            trading_day: None,
            sessions: crate::context::SessionView::continuous(),
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
        // Order fills in full (quantity 2), not rejected, not resized.
        let (_, trade) = &reports[0];
        let trade = trade.as_ref().expect("order should fill, not reject");
        assert_eq!(trade.quantity, Decimal::from(2));
        assert_eq!(trade.price, Decimal::from(4));
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

    /// 以给定的本时间片品种时段跑一次 `on_event`(AAPL 开盘 100 的 bar)。
    fn on_event_in_session(
        sim: &mut SimulatedExecutionClient,
        session: Option<TradingSession>,
    ) -> Vec<Event> {
        let instruments = create_test_instruments();
        let event = Event::Bar(create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        ));
        let portfolio = crate::portfolio::Portfolio {
            cash: Decimal::from(100000),
            positions: HashMap::new().into(),
            available_positions: HashMap::new().into(),
        };
        let last_prices = HashMap::new();
        let risk_manager = crate::risk::RiskManager::new();
        let trade_tracker = crate::analysis::TradeTracker::new();
        let market_model = crate::market::MarketConfig::China(crate::market::ChinaMarketConfig {
            stock: Some(crate::market::stock::StockConfig::default()),
            ..Default::default()
        })
        .create_model();
        let mut slice = HashMap::new();
        if let Some(s) = session {
            slice.insert("AAPL".to_string(), s);
        }
        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio: &portfolio,
            last_prices: &last_prices,
            trade_tracker: &trade_tracker,
            market_model: market_model.as_ref(),
            execution_policy_core: ExecutionPolicyCore::default(),
            bar_index: 0,
            current_time: 1000,
            session: TradingSession::Continuous,
            active_orders: &[],
            risk_config: &risk_manager.config,
            timezone_name: None,
            timezone_offset: 0,
            trading_day: None,
            sessions: crate::context::SessionView {
                force_continuous: false,
                slice: Some(&slice),
                resolver: None,
                instruments: Some(&instruments),
            },
        };
        sim.on_event(&event, &ctx)
    }

    fn fills(events: &[Event]) -> Vec<Decimal> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::ExecutionReport(_, Some(trade)) => Some(trade.price),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn non_continuous_session_queues_order_until_continuous() {
        let mut sim = SimulatedExecutionClient::new();
        sim.on_order(create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Market,
            Decimal::from(100),
            None,
        ));
        for session in [
            TradingSession::Break,
            TradingSession::ClosingAuction,
            TradingSession::OpeningAuction,
            TradingSession::PostClose,
        ] {
            let events = on_event_in_session(&mut sim, Some(session));
            assert!(events.is_empty(), "{session:?} 时段不应成交: {events:?}");
        }
        // 途经开盘竞价时段的 bar(收盘 102): 首个连续竞价事件上按竞价价统一成交
        let events = on_event_in_session(&mut sim, Some(TradingSession::Continuous));
        assert_eq!(fills(&events), vec![Decimal::from(102)]);
    }

    #[test]
    fn closed_session_event_does_not_expire_day_order() {
        let mut sim = SimulatedExecutionClient::new();
        let order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Market,
            Decimal::from(100),
            None,
        );
        assert_eq!(order.time_in_force, TimeInForce::Day);
        sim.on_order(order);
        let events = on_event_in_session(&mut sim, Some(TradingSession::Closed));
        assert!(events.is_empty(), "闭市时段不作废 Day 单: {events:?}");
        // 仍在队列中, 下一连续竞价事件成交
        let events = on_event_in_session(&mut sim, None);
        assert_eq!(fills(&events), vec![Decimal::from(100)]);
    }

    fn day(d: u32) -> Option<chrono::NaiveDate> {
        chrono::NaiveDate::from_ymd_opt(2024, 1, d)
    }

    fn auction_bar(ts: i64, open: i64, close: i64) -> Event {
        let mut bar = create_test_bar(
            "AAPL",
            Decimal::from(open),
            Decimal::from(open.max(close)),
            Decimal::from(open.min(close)),
            Decimal::from(close),
        );
        bar.timestamp = ts;
        Event::Bar(bar)
    }

    fn market_buy_at(created_at: i64) -> Order {
        let mut order = create_test_order(
            "AAPL",
            crate::model::OrderSide::Buy,
            crate::model::OrderType::Market,
            Decimal::from(100),
            None,
        );
        order.created_at = created_at;
        order
    }

    /// 以给定时段与交易日跑 `on_event`(`finalize` 为真时改调 `finalize_closing_auctions`)。
    fn run_auction_step(
        sim: &mut SimulatedExecutionClient,
        event: &Event,
        session: TradingSession,
        trading_day: Option<chrono::NaiveDate>,
        finalize: bool,
    ) -> Vec<Event> {
        let instruments = create_test_instruments();
        let portfolio = crate::portfolio::Portfolio {
            cash: Decimal::from(100000),
            positions: HashMap::new().into(),
            available_positions: HashMap::new().into(),
        };
        let last_prices = HashMap::new();
        let risk_manager = crate::risk::RiskManager::new();
        let trade_tracker = crate::analysis::TradeTracker::new();
        let market_model = crate::market::MarketConfig::China(crate::market::ChinaMarketConfig {
            stock: Some(crate::market::stock::StockConfig::default()),
            ..Default::default()
        })
        .create_model();
        let mut slice = HashMap::new();
        slice.insert("AAPL".to_string(), session);
        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio: &portfolio,
            last_prices: &last_prices,
            trade_tracker: &trade_tracker,
            market_model: market_model.as_ref(),
            execution_policy_core: ExecutionPolicyCore::default(),
            bar_index: 0,
            current_time: SimulatedExecutionClient::event_timestamp(event).unwrap_or(0),
            session,
            active_orders: &[],
            risk_config: &risk_manager.config,
            timezone_name: None,
            timezone_offset: 0,
            trading_day,
            sessions: crate::context::SessionView {
                force_continuous: false,
                slice: Some(&slice),
                resolver: None,
                instruments: Some(&instruments),
            },
        };
        if finalize {
            sim.finalize_closing_auctions(&ctx)
        } else {
            sim.on_event(event, &ctx)
        }
    }

    fn fill_points(events: &[Event]) -> Vec<(i64, Decimal)> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::ExecutionReport(_, Some(trade)) => Some((trade.timestamp, trade.price)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn opening_auction_records_price_and_uncrosses_on_first_continuous_event() {
        let mut sim = SimulatedExecutionClient::new();
        sim.on_order(market_buy_at(500));
        // 竞价时段事件只记价(收盘价 50), 不成交
        let events = run_auction_step(
            &mut sim,
            &auction_bar(1000, 48, 50),
            TradingSession::OpeningAuction,
            day(2),
            false,
        );
        assert!(events.is_empty(), "竞价时段不成交: {events:?}");
        // 首个连续竞价事件(开盘 100)上按记录的竞价价 50 统一成交, 时间戳为该事件
        let events = run_auction_step(
            &mut sim,
            &auction_bar(2000, 100, 101),
            TradingSession::Continuous,
            day(2),
            false,
        );
        assert_eq!(fill_points(&events), vec![(2000, Decimal::from(50))]);
    }

    #[test]
    fn opening_auction_uncrosses_only_once_per_trading_day() {
        let mut sim = SimulatedExecutionClient::new();
        sim.on_order(market_buy_at(500));
        run_auction_step(
            &mut sim,
            &auction_bar(1000, 48, 50),
            TradingSession::OpeningAuction,
            day(2),
            false,
        );
        let events = run_auction_step(
            &mut sim,
            &auction_bar(2000, 100, 101),
            TradingSession::Continuous,
            day(2),
            false,
        );
        assert_eq!(fills(&events), vec![Decimal::from(50)]);
        // 同一交易日再来一段竞价时段行情(脏数据)与连续竞价事件: 不再统一撮合,
        // 新单按连续竞价事件的开盘价成交
        sim.on_order(market_buy_at(2500));
        run_auction_step(
            &mut sim,
            &auction_bar(3000, 58, 60),
            TradingSession::OpeningAuction,
            day(2),
            false,
        );
        let events = run_auction_step(
            &mut sim,
            &auction_bar(4000, 100, 101),
            TradingSession::Continuous,
            day(2),
            false,
        );
        assert_eq!(fills(&events), vec![Decimal::from(100)]);
    }

    #[test]
    fn auction_state_resets_when_trading_day_changes() {
        let mut sim = SimulatedExecutionClient::new();
        // 01-02 只有竞价时段行情(开盘竞价未结束)
        run_auction_step(
            &mut sim,
            &auction_bar(1000, 48, 50),
            TradingSession::OpeningAuction,
            day(2),
            false,
        );
        sim.on_order(market_buy_at(1500));
        // 01-03 的首个连续竞价事件: 昨天的竞价价不用于今天, 按本事件开盘价成交
        let events = run_auction_step(
            &mut sim,
            &auction_bar(2000, 100, 101),
            TradingSession::Continuous,
            day(3),
            false,
        );
        assert_eq!(fills(&events), vec![Decimal::from(100)]);
    }

    fn bar_with_volume(ts: i64, close: i64, volume: i64) -> Event {
        let mut event = auction_bar(ts, close, close);
        if let Event::Bar(ref mut b) = event {
            b.volume = Decimal::from(volume);
        }
        event
    }

    #[test]
    fn zero_volume_auction_event_keeps_recorded_volume() {
        let mut sim = SimulatedExecutionClient::new();
        sim.on_order(market_buy_at(500));
        let session = TradingSession::OpeningAuction;
        run_auction_step(
            &mut sim,
            &bar_with_volume(1000, 50, 1000),
            session,
            day(2),
            false,
        );
        // 末笔零量快照: 价格更新为 51, 量沿用 1000(否则合成事件零量按停牌不成交)
        run_auction_step(
            &mut sim,
            &bar_with_volume(1100, 51, 0),
            session,
            day(2),
            false,
        );
        let quote = sim.auctions["AAPL"].opening.expect("已记录竞价价");
        assert_eq!(
            (quote.price, quote.volume, quote.ts),
            (Decimal::from(51), Decimal::from(1000), 1100)
        );
        let events = run_auction_step(
            &mut sim,
            &auction_bar(2000, 100, 101),
            TradingSession::Continuous,
            day(2),
            false,
        );
        assert_eq!(fills(&events), vec![Decimal::from(51)]);
    }

    #[test]
    fn auction_hold_sets_price_only_when_none_recorded() {
        let mut sim = SimulatedExecutionClient::new();
        run_auction_step(
            &mut sim,
            &auction_bar(1000, 50, 50),
            TradingSession::OpeningAuction,
            day(2),
            false,
        );
        run_auction_step(
            &mut sim,
            &auction_bar(1100, 55, 55),
            TradingSession::AuctionHold,
            day(2),
            false,
        );
        let quote = sim.auctions["AAPL"].opening.expect("已记录竞价价");
        assert_eq!((quote.price, quote.ts), (Decimal::from(50), 1000));
        // 只见到 AuctionHold 行情时由它设定
        let mut sim = SimulatedExecutionClient::new();
        run_auction_step(
            &mut sim,
            &auction_bar(1100, 55, 55),
            TradingSession::AuctionHold,
            day(2),
            false,
        );
        let quote = sim.auctions["AAPL"].opening.expect("已记录竞价价");
        assert_eq!((quote.price, quote.ts), (Decimal::from(55), 1100));
    }

    #[test]
    fn opening_uncross_excludes_orders_placed_after_auction_price_known() {
        let mut sim = SimulatedExecutionClient::new();
        run_auction_step(
            &mut sim,
            &auction_bar(1000, 50, 50),
            TradingSession::OpeningAuction,
            day(2),
            false,
        );
        // 竞价价(1000 时刻)已知后下的单: 进入连续竞价, 按首个连续事件开盘价 100 成交
        sim.on_order(market_buy_at(1000));
        let events = run_auction_step(
            &mut sim,
            &auction_bar(2000, 100, 101),
            TradingSession::Continuous,
            day(2),
            false,
        );
        assert_eq!(fills(&events), vec![Decimal::from(100)]);
    }

    #[test]
    fn tick_fill_after_partial_fill_uses_weighted_average_price() {
        let mut sim = SimulatedExecutionClient::new();
        let mut order = market_buy_at(500);
        order.status = OrderStatus::PartiallyFilled;
        order.filled_quantity = Decimal::from(30);
        order.average_filled_price = Some(Decimal::from(10));
        let id = order.id.clone();
        sim.on_order(order);
        let tick = Event::Tick(crate::model::Tick {
            timestamp: 1000,
            price: Decimal::new(105, 1),
            volume: Decimal::from(1000),
            symbol: "AAPL".to_string(),
        });
        let events = run_auction_step(&mut sim, &tick, TradingSession::Continuous, day(2), false);
        let avg = events.iter().find_map(|e| match e {
            Event::ExecutionReport(o, Some(_)) if o.id == id => o.average_filled_price,
            _ => None,
        });
        // (30 x 10.0 + 70 x 10.5) / 100
        assert_eq!(avg, Some(Decimal::new(1035, 2)));
    }

    #[test]
    fn finalize_closing_auctions_handles_only_todays_unfinished_auction() {
        let mut sim = SimulatedExecutionClient::new();
        sim.on_order(market_buy_at(500));
        // 收盘竞价时段行情(无解析器 → 判不出结束于本事件), 只记价 70
        let events = run_auction_step(
            &mut sim,
            &auction_bar(1000, 69, 70),
            TradingSession::ClosingAuction,
            day(2),
            false,
        );
        assert!(events.is_empty(), "收盘竞价期间不成交: {events:?}");
        // 别的交易日收尾: 不处理
        let close_next_day = auction_bar(9000, 0, 0);
        let events = run_auction_step(
            &mut sim,
            &close_next_day,
            TradingSession::Closed,
            day(3),
            true,
        );
        assert!(events.is_empty(), "非当日收盘竞价不收尾: {events:?}");
        // 当日收尾: 收盘时刻按最后一个收盘竞价价成交
        let close = auction_bar(1500, 0, 0);
        let events = run_auction_step(&mut sim, &close, TradingSession::Closed, day(2), true);
        assert_eq!(fill_points(&events), vec![(1500, Decimal::from(70))]);
        // 已完成: 再次收尾不重复撮合
        sim.on_order(market_buy_at(1200));
        let events = run_auction_step(&mut sim, &close, TradingSession::Closed, day(2), true);
        assert!(events.is_empty(), "收盘竞价只收尾一次: {events:?}");
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
