use crate::analysis::TradeTracker;
use crate::model::{AssetType, Instrument, Order, OrderSide, OrderStatus};
use crate::portfolio::Portfolio;
use crate::risk::RiskConfig;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;

/// 该订单是否还占用资金: 未终结且有未成交数量。
pub fn order_reserves_cash(order: &Order) -> bool {
    matches!(
        order.status,
        OrderStatus::New | OrderStatus::Submitted | OrderStatus::PartiallyFilled
    ) && order.quantity - order.filled_quantity > Decimal::ZERO
}

/// 在途订单预占的资金: 把每笔在途订单的未成交部分投影进持仓后, 占用保证金比当前
/// 多出来的部分(不为负)。
///
/// `base_used_margin` 是当前持仓的占用保证金(同一价格表、同一比例下算出), 由调用方
/// 传入, 这里只算投影后的那一遍。
pub fn frozen_cash(
    positions: &Arc<HashMap<String, Decimal>>,
    active_orders: &[Order],
    prices: &HashMap<String, Decimal>,
    instruments: &HashMap<String, Instrument>,
    stock_margin_ratio_override: Option<Decimal>,
    base_used_margin: Decimal,
) -> Decimal {
    // 没有占用资金的在途订单时投影后的持仓与当前相同, 差值恒为 0。
    if !active_orders.iter().any(order_reserves_cash) {
        return Decimal::ZERO;
    }
    // 保证金只看持仓, 现金与可用持仓不参与。
    let mut projected = Portfolio {
        cash: Decimal::ZERO,
        positions: Arc::clone(positions),
        available_positions: Arc::default(),
    };
    for order in active_orders {
        if !order_reserves_cash(order) {
            continue;
        }
        let remaining = order.quantity - order.filled_quantity;
        let current_pos = projected
            .positions
            .get(&order.symbol)
            .copied()
            .unwrap_or(Decimal::ZERO);
        let next_pos = crate::model::project_position_after(
            order.side,
            order.position_effect,
            current_pos,
            remaining,
        );
        projected.adjust_position(&order.symbol, next_pos - current_pos);
    }
    let projected_used = projected.calculate_used_margin_with_stock_ratio(
        prices,
        instruments,
        stock_margin_ratio_override,
    );
    (projected_used - base_used_margin).max(Decimal::ZERO)
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AccountMetrics {
    pub equity: Decimal,
    pub market_value: Decimal,
    pub notional_value: Decimal,
    pub used_margin: Decimal,
    pub unrealized_pnl: Decimal,
    pub short_market_value: Decimal,
    pub maintenance_ratio: Decimal,
}

pub fn is_futures_margin_account(instrument: &Instrument, risk_config: &RiskConfig) -> bool {
    risk_config.is_margin_account() && instrument.asset_type == AssetType::Futures
}

pub fn stock_margin_ratio_override(risk_config: &RiskConfig) -> Option<Decimal> {
    if risk_config.is_margin_account() {
        Some(risk_config.stock_initial_margin_ratio())
    } else {
        None
    }
}

pub fn estimate_futures_realized_pnl(
    trade_tracker: &TradeTracker,
    symbol: &str,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    multiplier: Decimal,
) -> Decimal {
    let mut realized = Decimal::ZERO;
    let mut remaining = quantity;
    match side {
        OrderSide::Buy => {
            if let Some(inventory) = trade_tracker.short_inventory.get(symbol) {
                for entry in inventory {
                    if remaining <= Decimal::ZERO {
                        break;
                    }
                    let covered_qty = remaining.min(entry.quantity);
                    realized += (entry.price - price) * covered_qty * multiplier;
                    remaining -= covered_qty;
                }
            }
        }
        OrderSide::Sell => {
            if let Some(inventory) = trade_tracker.long_inventory.get(symbol) {
                for entry in inventory {
                    if remaining <= Decimal::ZERO {
                        break;
                    }
                    let covered_qty = remaining.min(entry.quantity);
                    realized += (price - entry.price) * covered_qty * multiplier;
                    remaining -= covered_qty;
                }
            }
        }
    }
    realized
}

pub fn calculate_account_metrics(
    portfolio: &Portfolio,
    prices: &HashMap<String, Decimal>,
    instruments: &HashMap<String, Instrument>,
    trade_tracker: &TradeTracker,
    risk_config: &RiskConfig,
) -> AccountMetrics {
    let stock_ratio_override = stock_margin_ratio_override(risk_config);
    let used_margin =
        portfolio.calculate_used_margin_with_stock_ratio(prices, instruments, stock_ratio_override);
    let mut metrics = AccountMetrics {
        equity: portfolio.cash,
        used_margin,
        ..AccountMetrics::default()
    };

    let mut maintenance_denominator = Decimal::ZERO;
    // 股票类多头敞口(正值)；market_value 是带符号净值，不能直接当分母。
    let mut long_market_value = Decimal::ZERO;
    let mut futures_margin_component = Decimal::ZERO;

    for (symbol, quantity) in portfolio.positions.iter() {
        if quantity.is_zero() {
            continue;
        }
        let Some(price) = prices.get(symbol).copied() else {
            continue;
        };
        let Some(instrument) = instruments.get(symbol) else {
            let exposure = *quantity * price;
            metrics.market_value += exposure;
            metrics.equity += exposure;
            if *quantity < Decimal::ZERO {
                metrics.short_market_value += quantity.abs() * price;
            } else {
                long_market_value += exposure;
            }
            continue;
        };

        let multiplier = instrument.multiplier();
        let notional = quantity.abs() * price * multiplier;
        if is_futures_margin_account(instrument, risk_config) {
            let unrealized = trade_tracker.get_unrealized_pnl(symbol, price);
            let margin = crate::margin::MarginEngine::position_margin(
                *quantity,
                price,
                instrument,
                prices,
                stock_ratio_override,
            );
            metrics.notional_value += notional;
            metrics.unrealized_pnl += unrealized;
            metrics.equity += unrealized;
            futures_margin_component += margin;
            continue;
        }

        let exposure = *quantity * price * multiplier;
        metrics.market_value += exposure;
        metrics.equity += exposure;
        if *quantity < Decimal::ZERO {
            metrics.short_market_value += notional;
        } else {
            long_market_value += exposure;
        }
    }

    // 分母 = 多头敞口 + 空头敞口(总敞口)；market_value 为净值，空头会与 short_market_value 抵消。
    maintenance_denominator += long_market_value + metrics.short_market_value;
    maintenance_denominator += futures_margin_component;
    if maintenance_denominator > Decimal::ZERO {
        metrics.maintenance_ratio = metrics.equity / maintenance_denominator;
    }

    metrics
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::instrument::{InstrumentEnum, StockInstrument};
    use std::sync::Arc;

    fn stock(symbol: &str) -> Instrument {
        Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.to_string(),
                lot_size: Decimal::ONE,
                tick_size: Decimal::new(1, 2),
                expiry_date: None,
                sellable_after_days: 0,
            }),
        }
    }

    fn metrics_for(cash: i64, positions: &[(&str, i64)]) -> AccountMetrics {
        let mut pos = HashMap::new();
        let mut prices = HashMap::new();
        let mut instruments = HashMap::new();
        for (sym, qty) in positions {
            pos.insert(sym.to_string(), Decimal::from(*qty));
            prices.insert(sym.to_string(), Decimal::from(100));
            instruments.insert(sym.to_string(), stock(sym));
        }
        let portfolio = Portfolio {
            cash: Decimal::from(cash),
            positions: Arc::new(pos),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut config = RiskConfig::new();
        config.account_mode = "margin".to_string();
        calculate_account_metrics(
            &portfolio,
            &prices,
            &instruments,
            &TradeTracker::new(),
            &config,
        )
    }

    #[test]
    fn maintenance_ratio_pure_short_uses_short_exposure() {
        let m = metrics_for(20000, &[("AAA", -100)]);
        assert_eq!(m.equity, Decimal::from(10000));
        assert_eq!(m.maintenance_ratio, Decimal::ONE);
        assert_eq!(m.market_value, Decimal::from(-10000));
    }

    #[test]
    fn maintenance_ratio_mixed_uses_gross_exposure() {
        // 权益 = 5000 + 10000 - 5000 = 10000，总敞口 15000
        let m = metrics_for(5000, &[("AAA", 100), ("BBB", -50)]);
        assert_eq!(m.equity, Decimal::from(10000));
        assert_eq!(m.market_value, Decimal::from(5000));
        assert_eq!(
            m.maintenance_ratio,
            Decimal::from(10000) / Decimal::from(15000)
        );
    }

    #[test]
    fn maintenance_ratio_long_only_unchanged() {
        let m = metrics_for(5000, &[("AAA", 100)]);
        assert_eq!(
            m.maintenance_ratio,
            Decimal::from(15000) / Decimal::from(10000)
        );
    }
}
