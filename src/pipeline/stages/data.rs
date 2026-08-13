use super::shared::{
    apply_execution_report, effective_execution_policy_for_order, flush_accumulated_trades,
};
use crate::context::EngineContext;
use crate::data::FeedAction;
use crate::engine::Engine;
use crate::event::Event;
use crate::model::{Bar, Order, OrderStatus, TradingSession};
use crate::pipeline::processor::{Processor, ProcessorResult};
use pyo3::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct DataProcessor {
    last_timestamp: i64,
    finalized_timestamp: i64,
    seen_symbols: HashSet<String>,
    // Symbols with a *tradable* bar this slice (volume > 0). `seen_symbols`
    // tracks "did a bar appear at all" (drives synthetic-bar backfill); this
    // tracks "was it matchable". A suspended symbol emits a volume=0 placeholder
    // bar — present in `seen_symbols` but absent here — so it must not look
    // tradable to the terminal-state path (issue #334).
    tradable_symbols: HashSet<String>,
    current_symbol_events: HashMap<String, Event>,
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl DataProcessor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            last_timestamp: 0,
            finalized_timestamp: 0,
            seen_symbols: HashSet::new(),
            tradable_symbols: HashSet::new(),
            current_symbol_events: HashMap::new(),
        }
    }

    fn fill_missing_bars(&self, engine: &Engine) {
        if self.last_timestamp == 0 {
            return;
        }
        if let Ok(mut buffer) = engine.history_buffer.write() {
            let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
            for symbol in engine.instruments.keys() {
                // 跑 tick 流的 symbol 不补合成 bar: 双流下合成 bar 打区间结束戳,
                // 与 tick 时间戳天然错开, 于是每个 bar 事件都会让其余 symbol 显得
                // "本批次缺失"。此时用 last_prices(成交价)合成退化 bar 写进 bar
                // 序列, 等于用 tick 价冒充 bar, 且完全静默。该 symbol 的 bar 应当
                // 只来自真实聚合, 故此处直接跳过。
                if buffer.has_tick_history(symbol) {
                    continue;
                }
                if !self.seen_symbols.contains(symbol)
                    && let Some(&last_price) = prices.get(symbol)
                {
                    // Create synthetic bar
                    let bar = Bar {
                        timestamp: self.last_timestamp,
                        symbol: symbol.clone(),
                        open: last_price,
                        high: last_price,
                        low: last_price,
                        close: last_price,
                        volume: Decimal::ZERO,
                        extra: HashMap::default(),
                    };
                    buffer.update(&bar);
                }
            }
        }
    }

    fn reject_missing_symbol_orders(&self, engine: &mut Engine) {
        if self.last_timestamp == 0 {
            return;
        }

        let rejected_orders: Vec<Order> = engine
            .state
            .order_manager
            .active_orders
            .iter()
            .filter(|order| {
                matches!(order.status, OrderStatus::New | OrderStatus::Submitted)
                    // `<=` (not `==`): a timer-created order carries the timer's
                    // timestamp (e.g. 09:00) while the day's bar timestamp is
                    // 00:00, so `== last_timestamp` never matched timer orders —
                    // a suspended symbol's order then lived forever as New,
                    // piling into free-margin projection until trading froze
                    // (issue #329). Rejecting once its matchable slice has passed
                    // without a bar gives it a terminal state instead.
                    && order.created_at <= self.last_timestamp
                    // Was there a *tradable* (volume>0) bar this slice, not merely
                    // any bar? A suspended symbol emits a volume=0 placeholder bar
                    // that the matcher skips (execution/common.rs volume<=0 => None)
                    // but that still landed in `seen_symbols`, so the old check let
                    // the order live forever until the symbol resumed (issue #334).
                    // Keying off tradability aligns this terminal-state path with
                    // the matcher and generalizes the #329 "no bar at all" case.
                    && !self.tradable_symbols.contains(&order.symbol)
                    && effective_execution_policy_for_order(engine, order).bar_offset == 1
            })
            .cloned()
            .map(|mut order| {
                order.status = OrderStatus::Rejected;
                order.updated_at = self.last_timestamp;
                // Distinguish the two untradable causes so audit/log analysis is
                // not misled: a symbol present in `seen_symbols` did emit a bar
                // this slice — it was just not tradable (volume=0 suspension,
                // issue #334); one absent from it had no market data at all
                // (issue #329).
                order.reject_reason = if self.seen_symbols.contains(&order.symbol) {
                    format!(
                        "Symbol {} not tradable (zero volume, suspension) at execution timestamp {}",
                        order.symbol, self.last_timestamp
                    )
                } else {
                    format!(
                        "Missing market data for symbol {} at execution timestamp {}",
                        order.symbol, self.last_timestamp
                    )
                };
                order
            })
            .collect();

        for order in rejected_orders {
            engine.execution_model.on_cancel(&order.id);
            let _ = engine
                .event_manager
                .send(Event::ExecutionReport(order, None));
        }
    }

    fn finalize_current_timestamp(&mut self, engine: &mut Engine, py: Python<'_>) {
        if self.last_timestamp == 0 || self.finalized_timestamp == self.last_timestamp {
            return;
        }
        let current_events: Vec<Event> = self.current_symbol_events.values().cloned().collect();
        if !current_events.is_empty() {
            // 注意:此读锁会跨越下面的 finalize_timestamp 调用,自定义撮合器
            // (register_custom_matcher)会经此路径调回用户 Python 代码。
            // 安全性同 core.rs Engine.last_prices 字段注释:写锁只有 3 处
            // 且都要求 &mut Engine,run() 期间该借用被占用,再入会
            // PyBorrowMutError 而非阻塞。
            let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
            let ctx = EngineContext {
                instruments: &engine.instruments,
                portfolio: &engine.state.portfolio,
                last_prices: &prices,
                trade_tracker: &engine.state.order_manager.trade_tracker,
                market_model: engine.market_manager.model.as_ref(),
                execution_policy_core: engine.execution_policy_core(),
                bar_index: engine.bar_count,
                current_time: self.last_timestamp,
                session: engine.clock.session,
                active_orders: &engine.state.order_manager.active_orders,
                risk_config: &engine.risk_manager.config,
                timezone_name: engine.timezone_name.as_deref(),
                timezone_offset: engine.timezone_offset,
            };
            let reports = engine
                .execution_model
                .finalize_timestamp(&current_events, &ctx);
            drop(ctx);
            drop(prices);
            let mut trades_to_process = Vec::new();
            let mut oco_suppressed_fill_order_ids: HashSet<String> = HashSet::new();
            for report in reports {
                if let Event::ExecutionReport(order, trade) = report {
                    apply_execution_report(
                        engine,
                        py,
                        order,
                        trade,
                        &mut oco_suppressed_fill_order_ids,
                        &mut trades_to_process,
                    );
                }
            }
            flush_accumulated_trades(engine, &mut trades_to_process);
        }
        self.fill_missing_bars(engine);
        self.reject_missing_symbol_orders(engine);
        self.finalized_timestamp = self.last_timestamp;
        self.current_symbol_events.clear();
    }
}

impl Processor for DataProcessor {
    fn process(
        &mut self,
        engine: &mut Engine,
        py: Python<'_>,
        _strategy: &Bound<'_, PyAny>,
    ) -> PyResult<ProcessorResult> {
        // 墙钟兜底: 到点即结束, 不依赖行情事件到达。放在 feed 等待**之前**,
        // 因为等待本身可能长达 1 秒, 而行情停摆时 `Wait` 会无限循环下去。
        if let Some(deadline) = engine.session_deadline_ns
            && chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0) >= deadline
        {
            self.finalize_current_timestamp(engine, py);
            return Ok(ProcessorResult::Break);
        }

        let next_timer_time = engine.timers.peek().map(|t| t.timestamp);
        // 把内部事件通道的接收端交给 feed 一起等: 外部线程注入的 OrderRequest
        // 可立刻打断行情等待(否则最坏要等满 1 秒 timeout)。ChannelProcessor 是
        // pipeline 首位, Wait→Loop 后即被它排空。
        let wakeup = engine.event_manager.receiver();
        let action = engine.state.feed.next_action(next_timer_time, py, wakeup);

        match action {
            FeedAction::Wait => Ok(ProcessorResult::Loop),
            FeedAction::End => {
                self.finalize_current_timestamp(engine, py);
                Ok(ProcessorResult::Break)
            }
            FeedAction::Timer(_timestamp) => {
                if let Some(timer) = engine.timers.pop() {
                    // Invariant: finalize the current timestamp before any timer
                    // advances the clock past it. Previously only framework
                    // cross-section timers did this; user timers
                    // (add_daily_timer / schedule) skipped it, so same-cycle
                    // deferred orders were stranded — the timer's on_event wiped
                    // the slice-tracking registry before the slice was ever
                    // finalized, leaving phantom New buys that zeroed out free
                    // margin and froze trading permanently (issue #329). Framework
                    // cross-section timers are scheduled at source_ts + 1, so they
                    // always satisfy this condition and keep their prior behavior.
                    if timer.timestamp > self.last_timestamp {
                        self.finalize_current_timestamp(engine, py);
                    }
                    let local_time = engine.local_time_from_ns(timer.timestamp);
                    let session = engine.market_manager.get_session_status(local_time);
                    engine.clock.update(timer.timestamp, session);
                    if engine.force_session_continuous {
                        engine.clock.session = TradingSession::Continuous;
                    }
                    engine.current_event = Some(Event::Timer(timer));
                }
                Ok(ProcessorResult::Next)
            }
            FeedAction::Event(event) => {
                let event = *event;
                let timestamp = match &event {
                    Event::Bar(b) => b.timestamp,
                    Event::Tick(t) => t.timestamp,
                    _ => 0,
                };

                if timestamp <= engine.snapshot_time {
                    return Ok(ProcessorResult::Loop);
                }

                if self.last_timestamp != 0 && timestamp > self.last_timestamp {
                    self.finalize_current_timestamp(engine, py);
                    self.seen_symbols.clear();
                    self.tradable_symbols.clear();
                    self.current_symbol_events.clear();

                    if engine.is_active_timestamp(timestamp) {
                        engine.bar_count += 1;
                        if let Some(pb) = &engine.progress_bar {
                            pb.inc(1);
                        }
                        let mut progress_payload = HashMap::new();
                        progress_payload.insert("processed", engine.bar_count.to_string());
                        progress_payload.insert("total", engine.progress_total_steps.to_string());
                        engine.emit_stream_event(py, "progress", None, "info", progress_payload);
                    }
                }
                self.last_timestamp = timestamp;

                // Update Market Manager (Session)
                let session = engine
                    .market_manager
                    .get_session_status(engine.local_time_from_ns(timestamp));

                engine.clock.update(timestamp, session);
                if engine.force_session_continuous {
                    engine.clock.session = TradingSession::Continuous;
                }

                // Daily Snapshot & Settlement
                let local_date = engine.local_date_from_ns(timestamp);
                if engine.is_active_timestamp(timestamp) && engine.current_date != Some(local_date)
                {
                    // Captured before `current_date` is advanced below: the date
                    // whose close this settlement follows. A next-open Day order
                    // created on this date fills at *today's* open (matched after
                    // settlement), so it must be spared from today's expiry sweep.
                    let previous_trading_date = engine.current_date;

                    if engine.current_date.is_some() {
                        let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
                        engine.statistics_manager.record_snapshot(
                            timestamp,
                            &engine.state.portfolio,
                            &engine.instruments,
                            &prices,
                            &engine.state.order_manager.trade_tracker,
                            &engine.risk_manager.config,
                        );
                    }
                    engine.current_date = Some(local_date);

                    // Process Corporate Actions (Split/Dividend)
                    engine.corporate_action_manager.process_date(
                        local_date,
                        &mut engine.state.portfolio,
                        &mut engine.state.order_manager.trade_tracker,
                    );

                    // Settlement Manager (T+1, Option Expiry, Day Order Expiry)
                    let mut expired_orders = Vec::new();
                    // Day orders whose only matchable slice is *this* day's bar
                    // (next-open, created after the previous close) must not be
                    // expired by this settlement — they fill after it runs. Same-
                    // cycle (bar_offset == 0) orders had their slice on their
                    // creation day, so they still expire here.
                    let day_orders_awaiting_fill_slice: HashSet<String> = previous_trading_date
                        .map(|prev| {
                            engine
                                .state
                                .order_manager
                                .active_orders
                                .iter()
                                .filter(|order| {
                                    order.time_in_force == crate::model::TimeInForce::Day
                                        && effective_execution_policy_for_order(engine, order)
                                            .bar_offset
                                            == 1
                                        && engine.local_date_from_ns(order.created_at) == prev
                                })
                                .map(|order| order.id.clone())
                                .collect()
                        })
                        .unwrap_or_default();
                    let settlement_prices =
                        engine.last_prices.read().expect("last_prices 读锁被污染");
                    let settlement_ctx = crate::settlement::manager::SettlementContext {
                        date: local_date,
                        instruments: &engine.instruments,
                        last_prices: &settlement_prices,
                        market_manager: &engine.market_manager,
                        trade_tracker: &engine.state.order_manager.trade_tracker,
                        risk_config: &engine.risk_manager.config,
                        timestamp,
                        bar_index: engine.bar_count,
                        default_strategy_id: engine.default_strategy_id.clone(),
                        day_orders_awaiting_fill_slice: &day_orders_awaiting_fill_slice,
                    };
                    let settlement_outcome = engine.settlement_manager.process_daily_settlement(
                        &mut engine.state.portfolio,
                        &mut engine.state.order_manager.active_orders,
                        &mut expired_orders,
                        &settlement_ctx,
                    );
                    drop(settlement_ctx);
                    drop(settlement_prices);
                    engine.recent_expiry_events = settlement_outcome
                        .expiry_events
                        .iter()
                        .map(|event| crate::context::ExpiryEvent {
                            symbol: event.symbol.clone(),
                            asset_type: event.asset_type,
                            trading_date: event.trading_date.to_string(),
                            expiry_date: event.expiry_date,
                            quantity_before: event.quantity_before.to_f64().unwrap_or_default(),
                            quantity_closed: event.quantity_closed.to_f64().unwrap_or_default(),
                            cash_flow: event.cash_flow.to_f64().unwrap_or_default(),
                            settlement_type: event.settlement_type.clone(),
                            settlement_price: event
                                .settlement_price
                                .and_then(|value| value.to_f64()),
                            reason: event.reason.clone(),
                            description: event.description.clone(),
                        })
                        .collect();
                    engine.margin_daily_interest = settlement_outcome.daily_interest;
                    engine.margin_accrued_interest += settlement_outcome.daily_interest;
                    if settlement_outcome.daily_interest > Decimal::ZERO {
                        let mut settlement_payload = HashMap::new();
                        settlement_payload.insert("date", local_date.to_string());
                        settlement_payload.insert(
                            "daily_interest",
                            settlement_outcome.daily_interest.to_string(),
                        );
                        settlement_payload.insert(
                            "accrued_interest",
                            engine.margin_accrued_interest.to_string(),
                        );
                        engine.emit_stream_event(
                            py,
                            "settlement",
                            None,
                            "info",
                            settlement_payload,
                        );
                    }
                    if settlement_outcome.forced_liquidation {
                        let liquidated_symbols = settlement_outcome.liquidated_symbols.clone();
                        let priority = engine.risk_manager.config.liquidation_priority.clone();
                        engine.statistics_manager.record_liquidation_audit(
                            crate::analysis::LiquidationAudit {
                                timestamp,
                                date: local_date.to_string(),
                                daily_interest: settlement_outcome
                                    .daily_interest
                                    .to_f64()
                                    .unwrap_or_default(),
                                liquidated_count: liquidated_symbols.len(),
                                liquidated_symbols: liquidated_symbols.clone(),
                                priority: priority.clone(),
                            },
                        );
                        let mut risk_payload = HashMap::new();
                        risk_payload.insert("date", local_date.to_string());
                        risk_payload
                            .insert("liquidated_count", liquidated_symbols.len().to_string());
                        risk_payload.insert("liquidated_symbols", liquidated_symbols.join(","));
                        risk_payload.insert("priority", priority);
                        if let Some(owner_strategy_id) = engine
                            .default_strategy_id
                            .clone()
                            .filter(|text| !text.trim().is_empty())
                        {
                            risk_payload.insert("owner_strategy_id", owner_strategy_id);
                        }
                        engine.emit_stream_event(py, "risk", None, "warn", risk_payload);
                    }
                    for event in settlement_outcome.forced_liquidation_events {
                        let _ = engine.event_manager.send(event);
                    }
                    let recent_expiry_events = engine.recent_expiry_events.clone();
                    for expiry_event in recent_expiry_events {
                        let expiry_symbol = expiry_event.symbol.clone();
                        let mut expiry_payload = HashMap::new();
                        expiry_payload.insert("symbol", expiry_symbol.clone());
                        expiry_payload.insert(
                            "asset_type",
                            format!("{:?}", expiry_event.asset_type).to_ascii_uppercase(),
                        );
                        expiry_payload.insert("trading_date", expiry_event.trading_date.clone());
                        if let Some(expiry_date) = expiry_event.expiry_date {
                            expiry_payload.insert("expiry_date", expiry_date.to_string());
                        }
                        expiry_payload
                            .insert("quantity_before", expiry_event.quantity_before.to_string());
                        expiry_payload
                            .insert("quantity_closed", expiry_event.quantity_closed.to_string());
                        expiry_payload.insert("cash_flow", expiry_event.cash_flow.to_string());
                        if let Some(ref settlement_type) = expiry_event.settlement_type {
                            expiry_payload.insert("settlement_type", settlement_type.clone());
                        }
                        if let Some(settlement_price) = expiry_event.settlement_price {
                            expiry_payload.insert("settlement_price", settlement_price.to_string());
                        }
                        expiry_payload.insert("reason", expiry_event.reason.clone());
                        expiry_payload.insert("description", expiry_event.description.clone());
                        if let Some(owner_strategy_id) = engine
                            .default_strategy_id
                            .clone()
                            .filter(|text| !text.trim().is_empty())
                        {
                            expiry_payload.insert("owner_strategy_id", owner_strategy_id);
                        }
                        engine.emit_stream_event(
                            py,
                            "expiry",
                            Some(expiry_symbol.as_str()),
                            "info",
                            expiry_payload,
                        );
                    }

                    for o in expired_orders {
                        // Tell the execution client to drop the order too. Daily
                        // settlement only flips the order-manager copy to Expired;
                        // without this the simulated client keeps it New in its own
                        // book and a later bar crossing the limit fills the zombie
                        // long after it should have died (issue #329 hygiene item).
                        // Mirrors the reject path above (reject_missing_symbol_orders).
                        engine.execution_model.on_cancel(&o.id);
                        engine.state.order_manager.orders.push(o);
                    }
                }

                if let Event::Bar(ref b) = event {
                    self.seen_symbols.insert(b.symbol.clone());
                    if b.volume > Decimal::ZERO {
                        self.tradable_symbols.insert(b.symbol.clone());
                    }
                    self.current_symbol_events
                        .insert(b.symbol.clone(), Event::Bar(b.clone()));
                    // Update History Buffer
                    if let Ok(mut buffer) = engine.history_buffer.write() {
                        buffer.update(b);
                    }
                    // println!("DataProcessor: Bar Symbol={}, TS={}", b.symbol, b.timestamp);
                } else if let Event::Tick(ref t) = event {
                    self.current_symbol_events
                        .insert(t.symbol.clone(), Event::Tick(t.clone()));
                    // tick 是本时间戳该 symbol 的真实行情, 必须登记为"已见"——否则
                    // fill_missing_bars 会把它当成缺失并合成一根退化 bar, 导致每个
                    // tick 被双写历史(一次 update_tick, 一次合成 bar)。
                    self.seen_symbols.insert(t.symbol.clone());
                    // tick 也写入历史缓冲区: 否则纯 tick 回测下 get_history 恒为空,
                    // 且是静默的(不报错也不工作)。tick 以退化 bar 写入。
                    if let Ok(mut buffer) = engine.history_buffer.write() {
                        buffer.update_tick(t);
                    }
                }

                engine.current_event = Some(event);
                if engine.is_active_timestamp(timestamp)
                    && let Some(current) = engine.current_event.clone()
                {
                    match current {
                        Event::Bar(b) => {
                            let mut payload = HashMap::new();
                            payload.insert("timestamp", b.timestamp.to_string());
                            payload.insert("close", b.close.to_string());
                            payload.insert("volume", b.volume.to_string());
                            engine.emit_stream_event(
                                py,
                                "bar",
                                Some(b.symbol.as_str()),
                                "info",
                                payload,
                            );
                        }
                        Event::Tick(t) => {
                            let mut payload = HashMap::new();
                            payload.insert("timestamp", t.timestamp.to_string());
                            payload.insert("price", t.price.to_string());
                            payload.insert("volume", t.volume.to_string());
                            engine.emit_stream_event(
                                py,
                                "tick",
                                Some(t.symbol.as_str()),
                                "info",
                                payload,
                            );
                        }
                        _ => {}
                    }
                }
                Ok(ProcessorResult::Next)
            }
        }
    }
}
