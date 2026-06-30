use crate::engine::Engine;
use crate::event::Event;
use crate::pipeline::processor::{Processor, ProcessorResult};
use pyo3::prelude::*;
use rust_decimal::prelude::*;
use std::sync::Arc;

pub struct StrategyProcessor;

fn flush_pending_engine_oco_groups(
    engine: &mut Engine,
    strategy_obj: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let pending_any = match strategy_obj.getattr("_pending_engine_oco_groups") {
        Ok(v) => v,
        Err(_) => return Ok(()),
    };
    let pending_groups: Vec<(String, String, String)> = pending_any.extract().unwrap_or_default();
    if pending_groups.is_empty() {
        return Ok(());
    }
    for (group_id, first_order_id, second_order_id) in pending_groups {
        engine
            .state
            .order_manager
            .register_oco_group(group_id, first_order_id, second_order_id);
    }
    strategy_obj.setattr(
        "_pending_engine_oco_groups",
        Vec::<(String, String, String)>::new(),
    )?;
    Ok(())
}

fn flush_pending_engine_bracket_plans(
    engine: &mut Engine,
    strategy_obj: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let pending_any = match strategy_obj.getattr("_pending_engine_bracket_plans") {
        Ok(v) => v,
        Err(_) => return Ok(()),
    };
    let pending_plans: Vec<(
        String,
        Option<f64>,
        Option<f64>,
        Option<crate::model::TimeInForce>,
        Option<String>,
        Option<String>,
    )> = pending_any.extract().unwrap_or_default();
    if pending_plans.is_empty() {
        return Ok(());
    }
    for (
        entry_order_id,
        stop_trigger_price,
        take_profit_price,
        time_in_force,
        stop_tag,
        take_profit_tag,
    ) in pending_plans
    {
        let stop_trigger_decimal = stop_trigger_price.and_then(rust_decimal::Decimal::from_f64);
        let take_profit_decimal = take_profit_price.and_then(rust_decimal::Decimal::from_f64);
        engine.state.order_manager.register_bracket_plan(
            entry_order_id,
            stop_trigger_decimal,
            take_profit_decimal,
            time_in_force.unwrap_or(crate::model::TimeInForce::GTC),
            stop_tag,
            take_profit_tag,
        );
    }
    strategy_obj.setattr(
        "_pending_engine_bracket_plans",
        Vec::<(
            String,
            Option<f64>,
            Option<f64>,
            Option<crate::model::TimeInForce>,
            Option<String>,
            Option<String>,
        )>::new(),
    )?;
    Ok(())
}

impl Processor for StrategyProcessor {
    fn process(
        &mut self,
        engine: &mut Engine,
        py: Python<'_>,
        strategy: &Bound<'_, PyAny>,
    ) -> PyResult<ProcessorResult> {
        if let Some(event) = engine.current_event.clone() {
            engine.ensure_strategy_slot_exists();
            engine.ensure_strategy_context_capacity();
            let slot_count = engine.strategy_slots.len();
            let active_orders = Arc::new(engine.state.order_manager.active_orders.clone());
            let step_trades = engine.state.order_manager.current_step_trades.clone();
            let step_rejected_orders = engine
                .state
                .order_manager
                .current_step_rejected_orders
                .clone();

            for slot_index in 0..slot_count {
                let slot_strategy = engine
                    .strategy_slot_strategies
                    .get(slot_index)
                    .and_then(|slot| slot.as_ref())
                    .map(|slot| slot.clone_ref(py));
                let (new_orders, new_timers, canceled_ids) =
                    if let Some(ref slot_py) = slot_strategy {
                        let slot_bound = slot_py.bind(py);
                        let result = engine.call_strategy_for_slot(
                            slot_bound,
                            &event,
                            slot_index,
                            active_orders.clone(),
                            step_trades.clone(),
                            step_rejected_orders.clone(),
                        )?;
                        flush_pending_engine_oco_groups(engine, slot_bound)?;
                        flush_pending_engine_bracket_plans(engine, slot_bound)?;
                        result
                    } else {
                        let result = engine.call_strategy_for_slot(
                            strategy,
                            &event,
                            slot_index,
                            active_orders.clone(),
                            step_trades.clone(),
                            step_rejected_orders.clone(),
                        )?;
                        flush_pending_engine_oco_groups(engine, strategy)?;
                        flush_pending_engine_bracket_plans(engine, strategy)?;
                        result
                    };

                for id in canceled_ids {
                    engine.execution_model.on_cancel(&id);
                    if let Some(cancelled_order) = engine
                        .state
                        .order_manager
                        .cancel_active_order(&id, engine.clock.timestamp().unwrap_or(0))
                    {
                        let _ = engine
                            .event_manager
                            .send(Event::ExecutionReport(cancelled_order, None));
                    }
                }
                for order in new_orders {
                    let _ = engine.event_manager.send(Event::OrderRequest(order));
                }
                for t in new_timers {
                    engine.timers.push(t);
                }
            }
            engine.state.order_manager.current_step_trades.clear();
            engine
                .state
                .order_manager
                .current_step_rejected_orders
                .clear();
        }
        Ok(ProcessorResult::Next)
    }
}
