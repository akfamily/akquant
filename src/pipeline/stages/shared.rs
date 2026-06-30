use crate::engine::Engine;
use crate::event::Event;
use crate::model::{ExecutionPolicyCore, Order, OrderStatus, PriceBasis, TemporalPolicy, Trade};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

pub(crate) fn effective_execution_policy_for_order(engine: &Engine, order: &Order) -> ExecutionPolicyCore {
    order
        .fill_policy_override
        .unwrap_or(engine.execution_policy_core())
}

pub(crate) fn should_run_phase_for_engine_default(
    phase: &ExecutionPhase,
    engine: &Engine,
    event: Option<&Event>,
) -> bool {
    should_run_phase_for_current_event(phase, engine.execution_policy_core(), event)
}

pub(crate) fn should_run_phase_for_current_event(
    phase: &ExecutionPhase,
    policy: ExecutionPolicyCore,
    event: Option<&Event>,
) -> bool {
    match phase {
        ExecutionPhase::PreStrategy => policy.bar_offset == 1,
        ExecutionPhase::PostStrategy => {
            if !(policy.price_basis == PriceBasis::Close && policy.bar_offset == 0) {
                return false;
            }
            match event {
                Some(Event::Bar(_) | Event::Tick(_)) => true,
                Some(Event::Timer(timer)) => {
                    matches!(policy.temporal, TemporalPolicy::SameCycle)
                        && (!timer.payload.starts_with("__framework_")
                            || timer
                                .payload
                                .starts_with("__framework_after_bar_rebalance__|"))
                }
                _ => false,
            }
        }
    }
}

pub(crate) fn has_orders_matching_phase(engine: &Engine, phase: &ExecutionPhase, orders: &[Order]) -> bool {
    let event = engine.current_event.as_ref();
    orders.iter().any(|order| {
        should_run_phase_for_current_event(
            phase,
            effective_execution_policy_for_order(engine, order),
            event,
        )
    })
}
pub(crate) fn flush_accumulated_trades(engine: &mut Engine, trades_to_process: &mut Vec<Trade>) {
    if trades_to_process.is_empty() {
        return;
    }
    engine.state.order_manager.process_trades(
        std::mem::take(trades_to_process),
        &mut engine.state.portfolio,
        &engine.instruments,
        &engine.risk_manager.config,
        engine.market_manager.model.as_ref(),
        &engine.history_buffer,
        &engine.last_prices,
    );
}

pub(crate) fn apply_execution_report(
    engine: &mut Engine,
    py: Python<'_>,
    order: Order,
    trade: Option<Trade>,
    oco_suppressed_fill_order_ids: &mut HashSet<String>,
    trades_to_process: &mut Vec<Trade>,
) {
    if order.status == OrderStatus::Filled && oco_suppressed_fill_order_ids.contains(&order.id) {
        return;
    }

    let report_order_id = order.id.clone();
    let report_status = order.status;
    let report_updated_at = order.updated_at;
    engine.state.order_manager.on_execution_report(order);
    let updated_order = engine
        .state
        .order_manager
        .get_all_orders()
        .into_iter()
        .find(|o| o.id == report_order_id);
    if let Some(order_snapshot) = updated_order {
        if order_snapshot.status == OrderStatus::Rejected {
            engine
                .state
                .order_manager
                .current_step_rejected_orders
                .push(order_snapshot.clone());
        }
        let mut order_payload = HashMap::new();
        order_payload.insert("order_id", order_snapshot.id.clone());
        order_payload.insert("status", format!("{:?}", order_snapshot.status));
        order_payload.insert("filled_qty", order_snapshot.filled_quantity.to_string());
        order_payload.insert("symbol", order_snapshot.symbol.clone());
        order_payload.insert(
            "owner_strategy_id",
            order_snapshot.owner_strategy_id.clone().unwrap_or_default(),
        );
        engine.emit_stream_event(
            py,
            "order",
            Some(order_snapshot.symbol.as_str()),
            "info",
            order_payload,
        );
    }

    if report_status == OrderStatus::Filled {
        let peer_ids = engine
            .state
            .order_manager
            .consume_oco_peer_cancels_on_fill(&report_order_id);
        for peer_id in peer_ids {
            oco_suppressed_fill_order_ids.insert(peer_id.clone());
            engine.execution_model.on_cancel(&peer_id);
            let cancelled_order_snapshot = engine
                .state
                .order_manager
                .cancel_active_order(&peer_id, report_updated_at);
            if let Some(cancelled_order_snapshot) = cancelled_order_snapshot {
                let mut cancel_payload = HashMap::new();
                cancel_payload.insert("order_id", cancelled_order_snapshot.id.clone());
                cancel_payload.insert("status", format!("{:?}", cancelled_order_snapshot.status));
                cancel_payload.insert(
                    "filled_qty",
                    cancelled_order_snapshot.filled_quantity.to_string(),
                );
                cancel_payload.insert("symbol", cancelled_order_snapshot.symbol.clone());
                cancel_payload.insert(
                    "owner_strategy_id",
                    cancelled_order_snapshot
                        .owner_strategy_id
                        .clone()
                        .unwrap_or_default(),
                );
                engine.emit_stream_event(
                    py,
                    "order",
                    Some(cancelled_order_snapshot.symbol.as_str()),
                    "info",
                    cancel_payload,
                );
            }
        }

        if let Some(filled_order_snapshot) = engine
            .state
            .order_manager
            .get_all_orders()
            .into_iter()
            .find(|o| o.id == report_order_id)
        {
            let bracket_exit_orders = engine
                .state
                .order_manager
                .consume_bracket_activation_on_fill(&filled_order_snapshot);
            for bracket_order in bracket_exit_orders {
                let _ = engine
                    .event_manager
                    .send(Event::OrderRequest(bracket_order));
            }
        }
    }

    if let Some(t) = trade {
        engine.maybe_reset_risk_budget_usage(t.timestamp);
        if engine.risk_budget_use_trade_mode() {
            engine.apply_risk_budget_usage_from_trade(&t);
        }
        engine.apply_strategy_trade_position(&t);
        let mut trade_payload = HashMap::new();
        trade_payload.insert("trade_id", t.id.clone());
        trade_payload.insert("order_id", t.order_id.clone());
        trade_payload.insert("price", t.price.to_string());
        trade_payload.insert("quantity", t.quantity.to_string());
        trade_payload.insert(
            "owner_strategy_id",
            t.owner_strategy_id.clone().unwrap_or_default(),
        );
        engine.emit_stream_event(py, "trade", Some(t.symbol.as_str()), "info", trade_payload);
        trades_to_process.push(t);
    }
}
#[derive(Debug)]
pub enum ExecutionPhase {
    PreStrategy,
    PostStrategy,
}
