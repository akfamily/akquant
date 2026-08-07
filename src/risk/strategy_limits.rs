//! 策略级限额(`strategy_max_*`)的无状态比较逻辑。
//!
//! 这些函数不依赖 `&Engine`,原因是 broker_live 的前置风控必须能在**策略回调内**
//! 调用,而那一刻 `Engine::run(&mut self)` 正独占借用引擎对象 —— 任何经 Python 侧
//! 触达 `Engine` 的调用都会 `RuntimeError: Already borrowed`。
//!
//! `Engine::check_strategy_*_limit` 与 Python 侧的 `check_strategy_limits`
//! 都转发到这里,保证两条路径用的是**同一份**判定逻辑,不产生第二套规则。

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use std::collections::HashMap;

/// 单笔委托名义超限?
pub fn exceeds_order_value(
    strategy_id: &str,
    quantity: Decimal,
    price: Option<Decimal>,
    max_value: Decimal,
) -> Option<String> {
    let price = price?;
    let value = price * quantity;
    if value > max_value {
        return Some(format!(
            "Risk: Strategy {strategy_id} order value {value} exceeds strategy limit {max_value}"
        ));
    }
    None
}

/// 单笔委托数量超限?
pub fn exceeds_order_size(
    strategy_id: &str,
    quantity: Decimal,
    max_size: Decimal,
) -> Option<String> {
    if quantity > max_size {
        return Some(format!(
            "Risk: Strategy {strategy_id} order quantity {quantity} exceeds strategy limit {max_size}"
        ));
    }
    None
}

/// 成交后持仓将超限?
///
/// `current_qty` 为该策略在此标的上的现有持仓(带符号),`signed_delta` 为本笔
/// 委托对持仓的带符号影响。判定用**绝对值**(多空同一上限),但消息里打印**带符号**
/// 的投影持仓——与 `Engine::check_strategy_position_size_limit` 的文案逐字一致。
pub fn exceeds_position_size(
    strategy_id: &str,
    current_qty: Decimal,
    signed_delta: Decimal,
    max_size: Decimal,
) -> Option<String> {
    let projected = current_qty + signed_delta;
    if projected.abs() > max_size {
        return Some(format!(
            "Risk: Strategy {strategy_id} projected position {projected} exceeds strategy position limit {max_size}"
        ));
    }
    None
}

/// 按限额表逐项校验一笔委托,返回首个命中的拒单原因。
///
/// 只覆盖**不需要引擎运行态**的三项限额(order_value / order_size / position_size)。
/// daily_loss / drawdown / risk_budget 依赖引擎累计的盈亏与预算用量,不在此处;
/// 它们在 broker_live 下仍由 `ChannelProcessor` 那条路负责(见 signal-ingestion-rfc 3.3)。
#[allow(clippy::too_many_arguments)]
pub fn check_all(
    strategy_id: &str,
    symbol: &str,
    quantity: Decimal,
    price: Option<Decimal>,
    signed_delta: Decimal,
    current_positions: &HashMap<String, Decimal>,
    max_order_value: Option<Decimal>,
    max_order_size: Option<Decimal>,
    max_position_size: Option<Decimal>,
) -> Option<String> {
    if let Some(limit) = max_order_size
        && let Some(err) = exceeds_order_size(strategy_id, quantity, limit)
    {
        return Some(err);
    }
    if let Some(limit) = max_position_size {
        let current = current_positions
            .get(symbol)
            .copied()
            .unwrap_or(Decimal::ZERO);
        if let Some(err) = exceeds_position_size(strategy_id, current, signed_delta, limit) {
            return Some(err);
        }
    }
    if let Some(limit) = max_order_value
        && let Some(err) = exceeds_order_value(strategy_id, quantity, price, limit)
    {
        return Some(err);
    }
    None
}

fn to_decimal(value: f64) -> Option<Decimal> {
    Decimal::from_f64(value)
}

/// 校验一笔委托是否触碰策略级限额(broker_live 报单前的前置风控)。
///
/// 独立于 `Engine`: broker_live 的报单发生在策略回调内, 那一刻
/// `Engine::run(&mut self)` 正独占借用引擎对象, 经 Python 侧触达 `Engine` 的任何
/// 调用都会 `RuntimeError: Already borrowed`。故本函数只吃纯数据。
///
/// 判定逻辑与 `Engine::check_strategy_*_limit` 共用同一批自由函数, 拒单文案逐字一致。
///
/// :param strategy_id: 归属策略 id(通常 `_default`)
/// :param symbol: 标的代码
/// :param side: `"Buy"` / `"Sell"`(决定持仓投影方向)
/// :param quantity: 委托数量
/// :param price: 委托价; None 时跳过名义校验(无参考价无法折算)
/// :param current_positions: 该策略当前持仓 {symbol: 带符号数量}
/// :param max_order_value: 单笔名义上限(None 表示不限)
/// :param max_order_size: 单笔数量上限(None 表示不限)
/// :param max_position_size: 持仓上限(None 表示不限)
/// :return: 拒单原因; None 表示通过
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    strategy_id,
    symbol,
    side,
    quantity,
    price=None,
    current_positions=None,
    max_order_value=None,
    max_order_size=None,
    max_position_size=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn check_strategy_limits(
    strategy_id: &str,
    symbol: &str,
    side: &str,
    quantity: f64,
    price: Option<f64>,
    current_positions: Option<HashMap<String, f64>>,
    max_order_value: Option<f64>,
    max_order_size: Option<f64>,
    max_position_size: Option<f64>,
) -> Option<String> {
    let quantity_dec = to_decimal(quantity)?;
    let price_dec = price.and_then(to_decimal);
    let signed_delta = if side.trim().eq_ignore_ascii_case("sell") {
        -quantity_dec
    } else {
        quantity_dec
    };
    let positions: HashMap<String, Decimal> = current_positions
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(key, value)| to_decimal(value).map(|dec| (key, dec)))
        .collect();

    check_all(
        strategy_id,
        symbol,
        quantity_dec,
        price_dec,
        signed_delta,
        &positions,
        max_order_value.and_then(to_decimal),
        max_order_size.and_then(to_decimal),
        max_position_size.and_then(to_decimal),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(value: i64) -> Decimal {
        Decimal::from(value)
    }

    #[test]
    fn order_value_rejects_above_limit_and_passes_at_boundary() {
        // 100 * 100 = 10000 > 500 → 拒
        let err = exceeds_order_value("s1", dec(100), Some(dec(100)), dec(500));
        assert!(err.unwrap().contains("order value 10000"));
        // 恰好等于上限不拒(判定是严格大于)
        assert!(exceeds_order_value("s1", dec(5), Some(dec(100)), dec(500)).is_none());
    }

    #[test]
    fn order_value_skips_check_when_price_unknown() {
        // 无参考价无法折算名义, 不能凭空拒单
        assert!(exceeds_order_value("s1", dec(100), None, dec(1)).is_none());
    }

    #[test]
    fn order_size_rejects_above_limit() {
        assert!(exceeds_order_size("s1", dec(101), dec(100)).is_some());
        assert!(exceeds_order_size("s1", dec(100), dec(100)).is_none());
    }

    #[test]
    fn position_size_uses_absolute_projection_for_both_sides() {
        // 多头累加超限
        assert!(exceeds_position_size("s1", dec(80), dec(30), dec(100)).is_some());
        // 空头方向同一上限(绝对值判定)
        assert!(exceeds_position_size("s1", dec(-80), dec(-30), dec(100)).is_some());
        // 减仓使投影回落到限内 → 放行
        assert!(exceeds_position_size("s1", dec(120), dec(-30), dec(100)).is_none());
    }

    #[test]
    fn position_size_message_keeps_sign() {
        // 文案须与 Engine::check_strategy_position_size_limit 逐字一致(带符号)
        let err = exceeds_position_size("s1", dec(-80), dec(-30), dec(100)).unwrap();
        assert!(err.contains("projected position -110"), "{err}");
        assert!(err.contains("exceeds strategy position limit 100"), "{err}");
    }

    #[test]
    fn check_all_reports_size_before_value() {
        // 同时触碰数量与名义上限时, 先报数量(与 Engine 的 or_else 串联顺序一致)
        let err = check_all(
            "s1",
            "X",
            dec(200),
            Some(dec(100)),
            dec(200),
            &HashMap::new(),
            Some(dec(500)),
            Some(dec(100)),
            None,
        )
        .unwrap();
        assert!(err.contains("order quantity"), "{err}");
    }

    #[test]
    fn check_all_passes_when_no_limit_configured() {
        assert!(
            check_all(
                "s1",
                "X",
                dec(10_000),
                Some(dec(10_000)),
                dec(10_000),
                &HashMap::new(),
                None,
                None,
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn pyfunction_maps_sell_to_negative_delta() {
        let mut positions = HashMap::new();
        positions.insert("X".to_string(), 100.0);
        // 卖 30 → 投影 70, 限 100 → 放行
        assert!(
            check_strategy_limits(
                "s1",
                "X",
                "Sell",
                30.0,
                Some(1.0),
                Some(positions.clone()),
                None,
                None,
                Some(100.0),
            )
            .is_none()
        );
        // 买 30 → 投影 130 > 100 → 拒
        assert!(
            check_strategy_limits(
                "s1",
                "X",
                "Buy",
                30.0,
                Some(1.0),
                Some(positions),
                None,
                None,
                Some(100.0),
            )
            .is_some()
        );
    }
}
