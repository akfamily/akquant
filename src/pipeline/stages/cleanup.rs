use crate::engine::Engine;
use crate::pipeline::processor::{Processor, ProcessorResult};
use pyo3::prelude::*;

pub struct CleanupProcessor;

impl Processor for CleanupProcessor {
    fn process(
        &mut self,
        engine: &mut Engine,
        _py: Python<'_>,
        _strategy: &Bound<'_, PyAny>,
    ) -> PyResult<ProcessorResult> {
        engine.state.order_manager.cleanup_finished_orders();
        Ok(ProcessorResult::Next)
    }
}
