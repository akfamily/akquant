use crate::event::Event;
use crate::execution::matcher::{ExecutionMatcher, MatchContext};
use crate::log_context::{execution_order_context_from_event, render_log_message};
use crate::model::{Order, Trade};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

/// Python 自定义撮合器包装器
///
/// 将 Rust 的 ExecutionMatcher trait 转发给 Python 对象。
/// Python 对象需要实现 `match` 方法:
/// `def match(self, order: Order, event: Union[Bar, Tick], instrument: Instrument) -> Optional[Trade]: ...`
pub struct PyExecutionMatcher {
    inner: Py<PyAny>,
}

impl PyExecutionMatcher {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { inner: obj }
    }

    fn invalid_result_warning(order: &Order, event: &Event) -> String {
        render_log_message(
            "Ignored custom Python matcher result because it is not a Trade",
            execution_order_context_from_event(order, event),
        )
    }

    fn exception_warning(order: &Order, event: &Event, error_message: &str) -> String {
        render_log_message(
            format!("Custom Python matcher raised an exception: {error_message}"),
            execution_order_context_from_event(order, event),
        )
    }

    fn order_writeback_warning(order: &Order, event: &Event, error_message: &str) -> String {
        render_log_message(
            format!(
                "Ignored Python matcher order state updates because updated Order could not be extracted: {error_message}"
            ),
            execution_order_context_from_event(order, event),
        )
    }
}

impl ExecutionMatcher for PyExecutionMatcher {
    fn match_order(&self, order: &mut Order, ctx: &MatchContext) -> Option<Event> {
        let event = ctx.event;
        let instrument = ctx.instrument;
        let bar_index = ctx.bar_index;

        Python::attach(|py| {
            // 1. Convert arguments to Python objects
            // Use clone() to create a copy for Python
            let py_order = order.clone().into_py_any(py).ok()?;

            let py_event = match event {
                Event::Bar(b) => b.clone().into_py_any(py).ok()?,
                Event::Tick(t) => t.clone().into_py_any(py).ok()?,
                _ => return None,
            };

            let py_instrument = instrument.clone().into_py_any(py).ok()?;

            // 2. Call Python `match` method
            // Signature: match(self, order, event, instrument) -> Optional[Trade]
            // Note: The Python method is expected to modify `order` in-place if needed,
            // and return a Trade if a trade occurred.
            let args = (py_order.clone_ref(py), py_event, py_instrument, bar_index);

            match self.inner.call_method1(py, "match", args) {
                Ok(result) => {
                    // 3. Check result
                    if result.is_none(py) {
                        return None;
                    }

                    // 4. If result is a Trade, extract it
                    if let Ok(trade) = result.extract::<Trade>(py) {
                        // 5. Sync back modified Order state from Python object
                        match py_order.extract::<Order>(py) {
                            Ok(updated_order) => {
                                *order = updated_order;
                            }
                            Err(error) => {
                                let error_message = error.to_string();
                                log::warn!(
                                    "{}",
                                    Self::order_writeback_warning(order, event, &error_message)
                                );
                            }
                        }

                        // 6. Return ExecutionReport
                        Some(Event::ExecutionReport(order.clone(), Some(trade)))
                    } else {
                        // 自定义撮合返回非 Trade 结果: 该单未能执行, 属执行错误。
                        log::error!("{}", Self::invalid_result_warning(order, event));
                        None
                    }
                }
                Err(e) => {
                    // 自定义撮合回调抛异常: 该单未能执行, 属执行错误。
                    let error_message = e.to_string();
                    log::error!("{}", Self::exception_warning(order, event, &error_message));
                    None
                }
            }
        })
    }
}
