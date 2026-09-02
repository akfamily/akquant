use super::shared::{
    ExecutionPhase, apply_execution_report, flush_accumulated_trades, has_orders_matching_phase,
};
use crate::context::EngineContext;
use crate::engine::Engine;
use crate::event::Event;
use crate::model::{Order, OrderStatus};
use crate::pipeline::processor::{Processor, ProcessorResult};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct ChannelProcessor;
fn process_order_request(engine: &mut Engine, py: Python<'_>, mut order: Order) {
    let current_time = engine.context_timestamp();
    engine.maybe_reset_risk_budget_usage(current_time);
    let mut strategy_limit_err = engine.check_strategy_risk_cooldown_mode(&order);
    let mut triggers_risk_fallback = false;
    if strategy_limit_err.is_none() {
        strategy_limit_err = engine.check_strategy_reduce_only_mode(&order);
    }
    if strategy_limit_err.is_none() {
        strategy_limit_err = engine
            .check_strategy_order_size_limit(&order)
            .or_else(|| engine.check_strategy_position_size_limit(&order))
            .or_else(|| engine.check_strategy_order_value_limit(&order));
        triggers_risk_fallback = strategy_limit_err.is_some();
    }
    if strategy_limit_err.is_none() {
        strategy_limit_err = engine.check_strategy_daily_loss_limit(&order, current_time);
        triggers_risk_fallback = strategy_limit_err.is_some();
    }
    if strategy_limit_err.is_none() {
        strategy_limit_err = engine.check_strategy_drawdown_limit(&order);
        triggers_risk_fallback = strategy_limit_err.is_some();
    }
    if strategy_limit_err.is_none() {
        strategy_limit_err = engine.check_strategy_risk_budget_limit(&order);
    }
    if strategy_limit_err.is_none() {
        strategy_limit_err = engine.check_portfolio_risk_budget_limit(&order);
    }
    if triggers_risk_fallback {
        engine.activate_strategy_reduce_only_if_configured(&order);
        engine.activate_strategy_risk_cooldown_if_configured(&order);
    }
    let strategy_limit_err = strategy_limit_err.map(crate::error::AkQuantError::OrderError);
    let check_result = if let Some(err) = strategy_limit_err {
        Err(err)
    } else {
        let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
        let ctx = EngineContext {
            instruments: &engine.instruments,
            portfolio: &engine.state.portfolio,
            last_prices: &prices,
            trade_tracker: &engine.state.order_manager.trade_tracker,
            market_model: engine.market_manager.model.as_ref(),
            execution_policy_core: engine.execution_policy_core(),
            bar_index: engine.bar_count,
            current_time,
            session: engine.clock.session,
            active_orders: &engine.state.order_manager.active_orders,
            risk_config: &engine.risk_manager.config,
            timezone_name: engine.timezone_name.as_deref(),
            timezone_offset: engine.timezone_offset,
        };
        // 回测撮合器会在成交时点重跑可用持仓校验(execution/simulated.rs),
        // 因此这里允许把该规则延后;实盘由柜台在报单时校验,不能延后。
        engine
            .risk_manager
            .check_and_adjust_with_delayed_position_check(
                &mut order,
                &ctx,
                !engine.execution_model.is_live(),
            )
    };
    if let Err(err) = check_result {
        order.status = OrderStatus::Rejected;
        order.reject_reason = err.to_string();
        order.updated_at = current_time;

        let mut risk_payload = HashMap::new();
        risk_payload.insert("order_id", order.id.clone());
        risk_payload.insert("symbol", order.symbol.clone());
        risk_payload.insert("reason", order.reject_reason.clone());
        risk_payload.insert(
            "owner_strategy_id",
            order.owner_strategy_id.clone().unwrap_or_default(),
        );
        engine.emit_stream_event(
            py,
            "risk",
            Some(order.symbol.as_str()),
            "warn",
            risk_payload,
        );
        let _ = engine
            .event_manager
            .send(Event::ExecutionReport(order, None));
    } else {
        let mut order_payload = HashMap::new();
        order_payload.insert("order_id", order.id.clone());
        order_payload.insert("status", format!("{:?}", OrderStatus::New));
        order_payload.insert("symbol", order.symbol.clone());
        order_payload.insert(
            "owner_strategy_id",
            order.owner_strategy_id.clone().unwrap_or_default(),
        );
        engine.emit_stream_event(
            py,
            "order",
            Some(order.symbol.as_str()),
            "info",
            order_payload,
        );
        if !engine.risk_budget_use_trade_mode() {
            engine.apply_risk_budget_usage(&order);
        }
        let _ = engine.event_manager.send(Event::OrderValidated(order));
    }
}
fn should_run_post_strategy_match_now(engine: &Engine, orders: &[Order]) -> bool {
    has_orders_matching_phase(engine, &ExecutionPhase::PostStrategy, orders)
}

fn is_reduce_first_order(order: &Order) -> bool {
    crate::model::is_reduce_first_order(order.side, order.position_effect)
}

fn emit_execution_reports_for_current_event(engine: &mut Engine) {
    let Some(event) = engine.current_event.clone() else {
        return;
    };

    if !matches!(event, Event::Bar(_) | Event::Tick(_) | Event::Timer(_)) {
        return;
    }

    // 注意:此读锁会跨越下面的 on_event 调用,而 on_event 在装了自定义撮合器
    // (register_custom_matcher)时会一路调进用户 Python 代码。这是安全的——
    // 详见 core.rs Engine.last_prices 字段上的锁纪律说明:写锁只有 3 处且都
    // 要求 &mut Engine,run() 期间该借用被占用,任何再入都会 PyBorrowMutError
    // 而非阻塞,不会与本处的读锁死锁。
    let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
    let ctx = EngineContext {
        instruments: &engine.instruments,
        portfolio: &engine.state.portfolio,
        last_prices: &prices,
        trade_tracker: &engine.state.order_manager.trade_tracker,
        market_model: engine.market_manager.model.as_ref(),
        execution_policy_core: engine.execution_policy_core(),
        bar_index: engine.bar_count,
        current_time: engine.context_timestamp(),
        session: engine.clock.session,
        active_orders: &engine.state.order_manager.active_orders,
        risk_config: &engine.risk_manager.config,
        timezone_name: engine.timezone_name.as_deref(),
        timezone_offset: engine.timezone_offset,
    };

    let reports = engine.execution_model.on_event(&event, &ctx);
    drop(ctx);
    drop(prices);
    for report in reports {
        let _ = engine.event_manager.send(report);
    }
}
impl Processor for ChannelProcessor {
    fn process(
        &mut self,
        engine: &mut Engine,
        py: Python<'_>,
        _strategy: &Bound<'_, PyAny>,
    ) -> PyResult<ProcessorResult> {
        let mut trades_to_process = Vec::new();
        let mut pending_order_requests = Vec::new();
        let mut oco_suppressed_fill_order_ids: HashSet<String> = HashSet::new();
        let mut settle_reductions_before_increases = false;
        let mut run_intermediate_reduce_match = false;
        loop {
            let mut drained_event = false;
            while let Some(event) = engine.event_manager.try_recv() {
                drained_event = true;
                match event {
                    Event::OrderRequest(order) => pending_order_requests.push(order),
                    Event::OrderValidated(order) => {
                        engine.execution_model.on_order(order.clone());
                        engine.state.order_manager.add_active_order(order);
                    }
                    Event::ExecutionReport(order, trade) => {
                        apply_execution_report(
                            engine,
                            py,
                            order,
                            trade,
                            &mut oco_suppressed_fill_order_ids,
                            &mut trades_to_process,
                        );
                    }
                    _ => {}
                }
            }

            if !pending_order_requests.is_empty() {
                if settle_reductions_before_increases {
                    flush_accumulated_trades(engine, &mut trades_to_process);
                    settle_reductions_before_increases = false;
                }
                if run_intermediate_reduce_match {
                    emit_execution_reports_for_current_event(engine);
                    run_intermediate_reduce_match = false;
                    settle_reductions_before_increases = true;
                    continue;
                }

                pending_order_requests.sort_by(|left, right| {
                    let left_phase_rank =
                        crate::model::reduction_priority_rank(left.side, left.position_effect);
                    let right_phase_rank =
                        crate::model::reduction_priority_rank(right.side, right.position_effect);
                    let left_priority = engine.strategy_priority_for_order(left);
                    let right_priority = engine.strategy_priority_for_order(right);
                    left_phase_rank
                        .cmp(&right_phase_rank)
                        .then_with(|| right_priority.cmp(&left_priority))
                        .then_with(|| {
                            let left_id =
                                Engine::normalized_order_strategy_id(left).unwrap_or_default();
                            let right_id =
                                Engine::normalized_order_strategy_id(right).unwrap_or_default();
                            left_id.cmp(&right_id)
                        })
                });

                let has_reduce = pending_order_requests.iter().any(is_reduce_first_order);
                let has_increase = pending_order_requests
                    .iter()
                    .any(|order| !is_reduce_first_order(order));
                let can_do_two_phase = has_reduce
                    && has_increase
                    && should_run_post_strategy_match_now(engine, &pending_order_requests);

                if can_do_two_phase {
                    let mut reduce_orders = Vec::new();
                    let mut increase_orders = Vec::new();
                    for order in pending_order_requests.drain(..) {
                        if is_reduce_first_order(&order) {
                            reduce_orders.push(order);
                        } else {
                            increase_orders.push(order);
                        }
                    }

                    for order in reduce_orders {
                        process_order_request(engine, py, order);
                    }

                    if !increase_orders.is_empty() {
                        pending_order_requests.extend(increase_orders);
                        run_intermediate_reduce_match = true;
                    }
                } else if has_reduce && has_increase {
                    // Next-open (bar_offset=1) path: two-phase intra-bar matching is
                    // unavailable, so without this branch the increase's affordability
                    // check runs in the same drain pass — before the reduce's
                    // OrderValidated event lands in active_orders — and a buy funded by a
                    // same-bar sell is wrongly rejected (issue #307). Validate the reduces
                    // now, re-queue the increases, and `continue`: the next loop turn
                    // drains the reduce OrderValidated events into active_orders so the
                    // increase projection (risk/common.rs project_active_orders_into) can
                    // credit the freed cash. Reduces stay `New` (they match next bar), so
                    // the projection sees them as pending sells.
                    let mut reduce_orders = Vec::new();
                    let mut increase_orders = Vec::new();
                    for order in pending_order_requests.drain(..) {
                        if is_reduce_first_order(&order) {
                            reduce_orders.push(order);
                        } else {
                            increase_orders.push(order);
                        }
                    }
                    for order in reduce_orders {
                        process_order_request(engine, py, order);
                    }
                    pending_order_requests.extend(increase_orders);
                    continue;
                } else {
                    for order in pending_order_requests.drain(..) {
                        process_order_request(engine, py, order);
                    }
                }
                continue;
            }

            if !drained_event {
                break;
            }
        }

        flush_accumulated_trades(engine, &mut trades_to_process);

        Ok(ProcessorResult::Next)
    }
}
