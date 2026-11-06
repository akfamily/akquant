use crate::engine::Engine;
use crate::engine::strategy_handler::StrategyHandle;
use crate::pipeline::processor::{Processor, ProcessorResult};
use pyo3::prelude::*;

pub struct PipelineRunner {
    processors: Vec<Box<dyn Processor>>,
}

impl Default for PipelineRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineRunner {
    #[must_use]
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    pub fn add_processor(&mut self, processor: Box<dyn Processor>) {
        self.processors.push(processor);
    }

    pub fn step(
        &mut self,
        engine: &mut Engine,
        py: Python<'_>,
        strategy: &StrategyHandle,
    ) -> PyResult<bool> {
        'main_loop: loop {
            for processor in &mut self.processors {
                match processor.process(engine, py, strategy)? {
                    ProcessorResult::Next => {}
                    ProcessorResult::Loop => continue 'main_loop,
                    ProcessorResult::Break => {
                        engine.flush_stream_events(py);
                        engine.raise_stream_fatal_error_if_any()?;
                        return Ok(false);
                    }
                }
                engine.raise_stream_fatal_error_if_any()?;
            }
            engine.flush_stream_events(py);
            engine.raise_stream_fatal_error_if_any()?;
            return Ok(true);
        }
    }

    pub fn run(
        &mut self,
        engine: &mut Engine,
        py: Python<'_>,
        strategy: &StrategyHandle,
    ) -> PyResult<()> {
        while self.step(engine, py, strategy)? {}
        Ok(())
    }
}
