use crate::event::Event;
use crate::execution::matcher::{ExecutionMatcher, MatchContext};
use crate::log_context::{execution_order_context_from_event, render_log_message};
use crate::model::{Order, OrderStatus, Trade};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use rust_decimal::Decimal;

/// Python 自定义撮合器包装器
///
/// 将 Rust 的 ExecutionMatcher trait 转发给 Python 对象。
/// Python 对象需要实现 `match` 方法:
/// `def match(self, order: Order, event: Union[Bar, Tick], instrument: Instrument, bar_index: int) -> Optional[Trade]: ...`
///
/// 返回 `Trade` 表示成交: 引擎按这笔成交推进订单的已成交量、均价与状态(分批成交时
/// 订单留在队列里); 订单号、品种、方向与订单不符或数量不在 (0, 剩余量] 内的成交被
/// 忽略并记错误日志。成交的手续费入账时由引擎按费率重算, 不采用 `Trade.commission`。
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

    fn invalid_trade_error(order: &Order, event: &Event, reason: &str) -> String {
        render_log_message(
            format!("Ignored invalid trade from custom Python matcher: {reason}"),
            execution_order_context_from_event(order, event),
        )
    }

    /// 撮合器返回的成交是否能记到这张订单上(按调用撮合器之前的订单判断)。
    fn validate_trade(order: &Order, trade: &Trade) -> Result<(), String> {
        if trade.order_id != order.id {
            return Err(format!(
                "trade order_id {} does not match order {}",
                trade.order_id, order.id
            ));
        }
        if trade.symbol != order.symbol {
            return Err(format!(
                "trade symbol {} does not match order symbol {}",
                trade.symbol, order.symbol
            ));
        }
        if trade.side != order.side {
            return Err(format!(
                "trade side {:?} does not match order side {:?}",
                trade.side, order.side
            ));
        }
        if trade.quantity <= Decimal::ZERO {
            return Err(format!("trade quantity {} is not positive", trade.quantity));
        }
        let remaining = order.quantity - order.filled_quantity;
        if trade.quantity > remaining {
            return Err(format!(
                "trade quantity {} exceeds remaining order quantity {}",
                trade.quantity, remaining
            ));
        }
        Ok(())
    }

    /// 按这笔成交推进订单: 已成交量、均价、状态与更新时间都从调用撮合器之前的
    /// `before` 算起, 撮合器自己就地写过的这几个字段一律被覆盖, 不会重复累加。
    fn apply_fill(order: &mut Order, before: &Order, trade: &Trade) {
        let filled = before.filled_quantity + trade.quantity;
        let prev_avg = before.average_filled_price.unwrap_or(Decimal::ZERO);
        order.filled_quantity = filled;
        order.average_filled_price =
            Some((prev_avg * before.filled_quantity + trade.price * trade.quantity) / filled);
        order.status = if filled >= before.quantity {
            OrderStatus::Filled
        } else {
            OrderStatus::PartiallyFilled
        };
        order.updated_at = trade.timestamp;
    }
}

impl ExecutionMatcher for PyExecutionMatcher {
    fn match_order(&self, order: &mut Order, ctx: &MatchContext) -> Option<Event> {
        let event = ctx.event;
        let instrument = ctx.instrument;
        let bar_index = ctx.bar_index;

        Python::attach(|py| {
            // 调用撮合器之前的订单: 校验成交与推进成交量都以它为准。
            let before = order.clone();
            // 1. Convert arguments to Python objects
            // Use clone() to create a copy for Python
            let py_order = before.clone().into_py_any(py).ok()?;

            let py_event = match event {
                Event::Bar(b) => b.clone().into_py_any(py).ok()?,
                Event::Tick(t) => t.clone().into_py_any(py).ok()?,
                _ => return None,
            };

            let py_instrument = instrument.clone().into_py_any(py).ok()?;

            // 2. Call Python `match` method
            // Signature: match(self, order, event, instrument, bar_index) -> Optional[Trade]
            // 返回 Trade 表示成交、None 表示本次不成交; 订单的成交量与状态由
            // 下面按这笔成交推进, 撮合器不必(也不应)自己改。
            let args = (py_order.clone_ref(py), py_event, py_instrument, bar_index);

            match self.inner.call_method1(py, "match", args) {
                Ok(result) => {
                    // 3. Check result
                    if result.is_none(py) {
                        return None;
                    }

                    // 4. If result is a Trade, extract it
                    if let Ok(trade) = result.extract::<Trade>(py) {
                        // 不合法的成交: 不入账、不改订单, 按本次未成交处理。
                        if let Err(reason) = Self::validate_trade(&before, &trade) {
                            log::error!("{}", Self::invalid_trade_error(&before, event, &reason));
                            return None;
                        }
                        // 5. Sync back other Order fields the matcher may have set
                        // (e.g. tag); fill fields are recomputed by apply_fill below.
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
                        Self::apply_fill(order, &before, &trade);

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
