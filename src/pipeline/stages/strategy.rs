use crate::engine::Engine;
use crate::event::Event;
use crate::pipeline::processor::{Processor, ProcessorResult};
use pyo3::prelude::*;
use rust_decimal::prelude::*;
use std::sync::Arc;

pub struct StrategyProcessor;

fn apply_pending_engine_plans(
    engine: &mut Engine,
    plans: Option<crate::engine::core::PendingEnginePlans>,
) {
    let Some(plans) = plans else {
        return;
    };
    for (group_id, first_order_id, second_order_id) in plans.oco_groups {
        engine
            .state
            .order_manager
            .register_oco_group(group_id, first_order_id, second_order_id);
    }
    for (
        entry_order_id,
        stop_trigger_price,
        take_profit_price,
        time_in_force,
        stop_tag,
        take_profit_tag,
    ) in plans.bracket_plans
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
            // 本周期内已知的挂单:引擎既有 active_orders,加上**前面 slot 刚提交**
            // 的单。后者要在 slot 之间累加,否则多策略下 slot 1 看不到 slot 0 本
            // 周期的平仓单——而 ctx.positions 是账户级全局,两边口径必须一致,
            // 否则 auto 拆腿/目标仓位的投影会漏掉别的策略的减仓意图。
            let mut cycle_orders = engine.state.order_manager.active_orders.clone();
            let mut active_orders = Arc::new(cycle_orders.clone());
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
                        let (orders, timers, canceled, plans) = engine.call_strategy_for_slot(
                            py,
                            slot_bound,
                            &event,
                            slot_index,
                            active_orders.clone(),
                            step_trades.clone(),
                            step_rejected_orders.clone(),
                        )?;
                        apply_pending_engine_plans(engine, plans);
                        (orders, timers, canceled)
                    } else {
                        let (orders, timers, canceled, plans) = engine.call_strategy_for_slot(
                            py,
                            strategy,
                            &event,
                            slot_index,
                            active_orders.clone(),
                            step_trades.clone(),
                            step_rejected_orders.clone(),
                        )?;
                        apply_pending_engine_plans(engine, plans);
                        (orders, timers, canceled)
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
                // 只有多策略才需要在 slot 间累加(单策略下无后续 slot 会读它),
                // 避免为单策略回测多付一次 Vec 克隆。
                //
                // 取本 slot Context 的 `orders` 而非上面的 `new_orders`:回测下
                // Context 带 `event_tx`,下单直接进事件通道,`orders_arc` 恒空,
                // `new_orders` 因此取不到东西;而 `StrategyContext::buy/sell` 无论
                // 走哪条路都会先 push 进 `orders`(每周期由 update_state 清空)。
                if slot_count > 1 {
                    if let Some(Some(slot_ctx)) = engine.strategy_contexts.get(slot_index) {
                        let ctx_ref = slot_ctx.borrow(py);
                        cycle_orders.extend(ctx_ref.orders.iter().cloned());
                    }
                    active_orders = Arc::new(cycle_orders.clone());
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
