use super::shared::{
    ExecutionPhase, has_orders_matching_phase, should_run_phase_for_engine_default,
};
use crate::context::EngineContext;
use crate::engine::Engine;
use crate::event::Event;
use crate::pipeline::processor::{Processor, ProcessorResult};
use pyo3::prelude::*;

pub struct ExecutionProcessor {
    phase: ExecutionPhase,
}

impl ExecutionProcessor {
    pub fn new(phase: ExecutionPhase) -> Self {
        Self { phase }
    }
}

impl Processor for ExecutionProcessor {
    fn process(
        &mut self,
        engine: &mut Engine,
        _py: Python<'_>,
        _strategy: &Bound<'_, PyAny>,
    ) -> PyResult<ProcessorResult> {
        let active_orders = engine.state.order_manager.active_orders.clone();
        let event = engine.current_event.as_ref();
        let should_run = should_run_phase_for_engine_default(&self.phase, engine, event)
            || has_orders_matching_phase(engine, &self.phase, &active_orders);

        if !should_run {
            return Ok(ProcessorResult::Next);
        }

        if let Some(timestamp) = engine.current_event_timestamp()
            && !engine.is_active_timestamp(timestamp)
        {
            return Ok(ProcessorResult::Next);
        }

        if let Some(event) = engine.current_event.clone() {
            match event {
                Event::Bar(_) | Event::Tick(_) | Event::Timer(_) => {
                    // Create Context
                    let ctx = EngineContext {
                        instruments: &engine.instruments,
                        portfolio: &engine.state.portfolio,
                        last_prices: &engine.last_prices,
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
                    for report in reports {
                        let _ = engine.event_manager.send(report);
                    }
                }
                _ => {}
            }
        }
        Ok(ProcessorResult::Next)
    }
}
