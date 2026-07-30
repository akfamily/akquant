use crate::account::calculate_account_metrics;
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
            let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
            let equity = calculate_account_metrics(
                &engine.state.portfolio,
                &prices,
                &engine.instruments,
                &engine.state.order_manager.trade_tracker,
                &engine.risk_manager.config,
            )
            .equity;
            let margin = engine
                .state
                .portfolio
                .calculate_used_margin(&prices, &engine.instruments);
            drop(prices);
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
