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
            current_symbol_events: HashMap::new(),
        }
    }

    fn fill_missing_bars(&self, engine: &Engine) {
        if self.last_timestamp == 0 {
            return;
        }
        if let Ok(mut buffer) = engine.history_buffer.write() {
            for symbol in engine.instruments.keys() {
                if !self.seen_symbols.contains(symbol)
                    && let Some(&last_price) = engine.last_prices.get(symbol)
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
                    && order.created_at == self.last_timestamp
                    && !self.seen_symbols.contains(&order.symbol)
                    && effective_execution_policy_for_order(engine, order).bar_offset == 1
            })
            .cloned()
            .map(|mut order| {
                order.status = OrderStatus::Rejected;
                order.updated_at = self.last_timestamp;
                order.reject_reason = format!(
                    "Missing market data for symbol {} at execution timestamp {}",
                    order.symbol, self.last_timestamp
                );
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
            let ctx = EngineContext {
                instruments: &engine.instruments,
                portfolio: &engine.state.portfolio,
                last_prices: &engine.last_prices,
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
        let next_timer_time = engine.timers.peek().map(|t| t.timestamp);
        let action = engine.state.feed.next_action(next_timer_time, py);

        match action {
            FeedAction::Wait => Ok(ProcessorResult::Loop),
            FeedAction::End => {
                self.finalize_current_timestamp(engine, py);
                Ok(ProcessorResult::Break)
            }
            FeedAction::Timer(_timestamp) => {
                if let Some(timer) = engine.timers.pop() {
                    if timer
                        .payload
                        .starts_with("__framework_after_bar_rebalance__|")
                    {
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
                    if engine.current_date.is_some() {
                        engine.statistics_manager.record_snapshot(
                            timestamp,
                            &engine.state.portfolio,
                            &engine.instruments,
                            &engine.last_prices,
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
                    let settlement_ctx = crate::settlement::manager::SettlementContext {
                        date: local_date,
                        instruments: &engine.instruments,
                        last_prices: &engine.last_prices,
                        market_manager: &engine.market_manager,
                        trade_tracker: &engine.state.order_manager.trade_tracker,
                        risk_config: &engine.risk_manager.config,
                        timestamp,
                        bar_index: engine.bar_count,
                        default_strategy_id: engine.default_strategy_id.clone(),
                    };
                    let settlement_outcome = engine.settlement_manager.process_daily_settlement(
                        &mut engine.state.portfolio,
                        &mut engine.state.order_manager.active_orders,
                        &mut expired_orders,
                        &settlement_ctx,
                    );
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
                        engine.state.order_manager.orders.push(o);
                    }
                }

                if let Event::Bar(ref b) = event {
                    self.seen_symbols.insert(b.symbol.clone());
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
