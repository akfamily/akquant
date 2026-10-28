use super::shared::{
    ExecutionPhase, has_orders_matching_phase, should_run_phase_for_engine_default,
};
use crate::context::{EngineContext, SessionView};
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
        // 交易日生命周期事件(盘前/开始交易/收盘后)不是行情, 不撮合任何订单。
        if let Some(Event::Timer(timer)) = engine.current_event.as_ref()
            && timer.is_lifecycle()
        {
            return Ok(ProcessorResult::Next);
        }
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
                    // 注意:此读锁会跨越下面的 on_event 调用,自定义撮合器
                    // (register_custom_matcher)会经此路径调回用户 Python
                    // 代码。安全性同 core.rs Engine.last_prices 字段注释:
                    // 写锁只有 3 处且都要求 &mut Engine,run() 期间该借用被
                    // 占用,再入会 PyBorrowMutError 而非阻塞。
                    engine.ensure_session_resolver()?;
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
                        trading_day: engine.current_trading_day,
                        sessions: SessionView {
                            slice: Some(&*engine.slice_sessions),
                            resolver: engine.session_resolver.as_deref(),
                            instruments: Some(&engine.instruments),
                        },
                    };

                    let reports = engine.execution_model.on_event(&event, &ctx);
                    drop(ctx);
                    drop(prices);
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
