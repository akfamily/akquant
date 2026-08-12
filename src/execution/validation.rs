//! 撮合前的价格/数量校验共享 helper。
//!
//! 期货侧原本把 `is_multiple` / `reject` 作为 `FuturesMatcher` 的私有关联函数，
//! 股票侧要复用只能提取出来——两边必须是同一套判断，否则同一笔非对齐价格在
//! 股票和期货上行为不一致。

use crate::event::Event;
use crate::execution::matcher::MatchContext;
use crate::log_context::{execution_order_context_from_event, render_log_message};
use crate::model::{Order, OrderStatus, OrderType};
use rust_decimal::Decimal;

pub fn is_multiple(value: Decimal, step: Decimal) -> bool {
    if step <= Decimal::ZERO {
        return true;
    }
    // `value`/`step` 常年经 Python `float` -> `extract_decimal`
    // (`Decimal::from_f64_retain`) 传入, 后者保留了 f64 二进制展开的全部有效位,
    // 例如 `2.83_f64` 的十进制精确值并非 `2.83`。这会让本应对齐的价格(如
    // 2.83 相对 tick 0.01)在算 `value % step` 时得到一个极小的非零余数,
    // 被误判为"未对齐"。真正的错位(如 2.8314 相对 0.01)偏差远大于此容差,
    // 因此用"步数"(value / step) 距最近整数的偏差是否小于 1e-6 来判断,
    // 而不是要求余数精确为零。
    let quotient = value / step;
    let nearest = quotient.round();
    let tolerance = Decimal::new(1, 6);
    (quotient - nearest).abs() <= tolerance
}

/// 纯渲染函数：只拼装拒单日志文案，不做任何状态变更。
///
/// `category` 决定文案措辞（如 "futures" -> "Rejected futures order because
/// ..."）——历史上期货侧的文案里带着资产品类，提取成共享函数后必须保留这个
/// 措辞，否则会静默改变已有的可观测日志输出（Python 侧按文案做过断言）。
pub fn render_reject_warning(
    order: &Order,
    ctx: &MatchContext,
    reason: &str,
    category: &str,
) -> String {
    render_log_message(
        format!("Rejected {category} order because {reason}"),
        execution_order_context_from_event(order, ctx.event),
    )
}

/// 拒单：写回订单状态并构造 `ExecutionReport` 事件。
///
/// `category` 同时驱动日志文案与 `log::warn!` 的 target——期货侧历史上的日志
/// target 是调用处所在模块路径（`akquant::execution::futures`），Python 侧
/// 测试按 `record.name` 精确匹配这个 target。把 `log::warn!` 挪进共享模块后，
/// 如果用默认 target（宏定义处的 `module_path!()`），target 会变成
/// `akquant::execution::validation`，破坏既有断言，因此这里显式传入 target
/// 字符串以保持提取前后完全一致。
pub fn reject_order(
    order: &mut Order,
    ctx: &MatchContext,
    reason: String,
    category: &str,
) -> Option<Event> {
    let warning = render_reject_warning(order, ctx, &reason, category);
    order.status = OrderStatus::Rejected;
    order.reject_reason = reason;
    match ctx.event {
        Event::Bar(b) => order.updated_at = b.timestamp,
        Event::Tick(t) => order.updated_at = t.timestamp,
        _ => {}
    }
    let log_target = format!("akquant::execution::{category}");
    log::warn!(target: &log_target, "{}", warning);
    Some(Event::ExecutionReport(order.clone(), None))
}

/// 校验委托里所有价格字段是否为 tick_size 的整数倍。
///
/// 返回 `Some(reason)` 表示不合规；`None` 表示通过。按 order_type 决定校验哪些
/// 字段——市价单没有 price，追踪单还要校验 trail_offset。
pub fn validate_tick_size(order: &Order, tick_size: Decimal) -> Option<String> {
    if tick_size <= Decimal::ZERO {
        return None;
    }
    let mut prices_to_check: Vec<(&str, Decimal)> = Vec::new();
    match order.order_type {
        OrderType::Limit => {
            if let Some(price) = order.price {
                prices_to_check.push(("price", price));
            }
        }
        OrderType::StopMarket => {
            if let Some(trigger_price) = order.trigger_price {
                prices_to_check.push(("trigger_price", trigger_price));
            }
        }
        OrderType::StopLimit => {
            if let Some(price) = order.price {
                prices_to_check.push(("price", price));
            }
            if let Some(trigger_price) = order.trigger_price {
                prices_to_check.push(("trigger_price", trigger_price));
            }
        }
        OrderType::StopTrail => {
            if let Some(trigger_price) = order.trigger_price {
                prices_to_check.push(("trigger_price", trigger_price));
            }
            if let Some(trail_offset) = order.trail_offset {
                prices_to_check.push(("trail_offset", trail_offset));
            }
        }
        OrderType::StopTrailLimit => {
            if let Some(price) = order.price {
                prices_to_check.push(("price", price));
            }
            if let Some(trigger_price) = order.trigger_price {
                prices_to_check.push(("trigger_price", trigger_price));
            }
            if let Some(trail_offset) = order.trail_offset {
                prices_to_check.push(("trail_offset", trail_offset));
            }
        }
        _ => {}
    }
    for (field_name, value) in prices_to_check {
        if !is_multiple(value, tick_size) {
            return Some(format!(
                "{} {} is not aligned with tick size {}",
                field_name, value, tick_size
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_is_multiple_treats_non_positive_step_as_pass() {
        assert!(is_multiple(dec!(2.8314), dec!(0)));
        assert!(is_multiple(dec!(2.8314), dec!(-1)));
    }

    #[test]
    fn test_is_multiple_detects_misalignment() {
        assert!(is_multiple(dec!(2.83), dec!(0.01)));
        assert!(!is_multiple(dec!(2.8314), dec!(0.01)));
        assert!(is_multiple(dec!(2.831), dec!(0.001)));
    }

    /// `extract_decimal` (Python `float` 入参) 走 `Decimal::from_f64_retain`,
    /// 保留 f64 二进制展开的全部有效位而非最短可读表示——`2.83_f64` 的精确
    /// 十进制值带一长串噪声位。回归 `is_multiple` 用容差而非精确取模判断,
    /// 否则这条路径传入的合法对齐价会被误拒。
    #[test]
    fn test_is_multiple_tolerates_f64_retain_noise() {
        let value = Decimal::from_f64_retain(2.83).unwrap();
        let step = Decimal::from_f64_retain(0.01).unwrap();
        assert!(is_multiple(value, step));

        let misaligned = Decimal::from_f64_retain(2.8314).unwrap();
        assert!(!is_multiple(misaligned, step));
    }

    fn make_limit_order(price: Decimal) -> Order {
        let mut order = Order::test_new(
            "1",
            "AAPL",
            crate::model::OrderSide::Buy,
            OrderType::Limit,
            dec!(100),
        );
        order.price = Some(price);
        order
    }

    #[test]
    fn test_validate_tick_size_accepts_aligned_limit_price() {
        let order = make_limit_order(dec!(2.83));
        assert_eq!(validate_tick_size(&order, dec!(0.01)), None);
    }

    #[test]
    fn test_validate_tick_size_rejects_misaligned_limit_price() {
        let order = make_limit_order(dec!(2.8314));
        let reason = validate_tick_size(&order, dec!(0.01));
        assert!(reason.is_some());
        let reason = reason.unwrap();
        assert!(reason.contains("price"));
        assert!(reason.contains("0.01"));
    }

    #[test]
    fn test_validate_tick_size_passes_when_tick_size_non_positive() {
        let order = make_limit_order(dec!(2.8314));
        assert_eq!(validate_tick_size(&order, dec!(0)), None);
        assert_eq!(validate_tick_size(&order, dec!(-1)), None);
    }
}
