use crate::engine::Engine;
use crate::engine::strategy_handler::StrategyHandle;
use pyo3::prelude::*;

#[derive(Debug, PartialEq)]
pub enum ProcessorResult {
    /// Proceed to the next processor in the pipeline
    Next,
    /// Restart the main loop (skip remaining processors)
    Loop,
    /// Break the main loop (finish backtest)
    Break,
}

pub trait Processor {
    fn process(
        &mut self,
        engine: &mut Engine,
        py: Python<'_>,
        strategy: &StrategyHandle,
    ) -> PyResult<ProcessorResult>;
}
