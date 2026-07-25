use crate::engine::Engine;
use crate::event::Event;
use crate::pipeline::processor::{Processor, ProcessorResult};
use pyo3::prelude::*;
use std::collections::HashMap;

pub struct StatisticsProcessor;

impl Processor for StatisticsProcessor {
    fn process(
        &mut self,
        engine: &mut Engine,
        py: Python<'_>,
        _strategy: &Bound<'_, PyAny>,
    ) -> PyResult<ProcessorResult> {
        if let Some(Event::Bar(_) | Event::Tick(_)) = engine.current_event.clone()
            && let Some(timestamp) = engine.clock.timestamp()
            && engine.is_active_timestamp(timestamp)
        {
            // 复用引擎每事件指标缓存：同一事件内 strategy 阶段已算过一次，
            // 此处零成本命中；used_margin 直接取指标字段，不再重复 O(P) 计算。
            let metrics = engine.current_account_metrics();
            let equity = metrics.equity;
            let margin = metrics.used_margin;
            engine.statistics_manager.update(
                timestamp,
                equity,
                engine.state.portfolio.cash,
                margin,
            );
            let mut payload = HashMap::new();
            payload.insert("timestamp", timestamp.to_string());
            payload.insert("equity", equity.to_string());
            payload.insert("cash", engine.state.portfolio.cash.to_string());
            payload.insert("margin", margin.to_string());
            engine.emit_stream_event(py, "equity", None, "info", payload);
        }
        Ok(ProcessorResult::Next)
    }
}
