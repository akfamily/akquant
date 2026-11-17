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

/// 账户口径的权益: 保证金账户里期货持仓只计浮动盈亏, 不计名义市值。
///
/// [`Portfolio::calculate_equity`] 是“现金 + 持仓名义市值”, 只在持仓都按全额入账
/// 时才等于权益; 保证金账户开期货仓不扣名义额, 用它会把多头名义额加进权益、把空头
/// 名义额减出权益。风控规则、可用资金一律用本函数的口径。
pub fn account_equity(
    portfolio: &Portfolio,
    prices: &HashMap<String, Decimal>,
    instruments: &HashMap<String, Instrument>,
    trade_tracker: &TradeTracker,
    risk_config: &RiskConfig,
) -> Decimal {
    if !risk_config.is_margin_account() {
        return portfolio.calculate_equity(prices, instruments);
    }
    calculate_account_metrics(portfolio, prices, instruments, trade_tracker, risk_config).equity
}

/// 保证金账户里期货持仓的带符号名义市值之和(非保证金账户为 0)。
pub fn futures_margin_notional(
    positions: &HashMap<String, Decimal>,
    prices: &HashMap<String, Decimal>,
    instruments: &HashMap<String, Instrument>,
    risk_config: &RiskConfig,
) -> Decimal {
    if !risk_config.is_margin_account() {
        return Decimal::ZERO;
    }
    let mut notional = Decimal::ZERO;
    for (symbol, quantity) in positions.iter() {
        if quantity.is_zero() {
            continue;
        }
        let (Some(price), Some(instrument)) = (prices.get(symbol), instruments.get(symbol)) else {
            continue;
        };
        if is_futures_margin_account(instrument, risk_config) {
            notional += *quantity * *price * instrument.multiplier();
        }
    }
    notional
}

/// “现金 + 持仓名义市值”口径相对账户口径权益的修正额(加到前者上得到后者)。
///
/// 用于在投影了挂单的组合上算可用资金: 投影按全额名义价调现金与持仓, 两者相抵,
/// 需要修正的只有**已有**期货持仓——它们开仓时没扣名义额。
pub fn futures_margin_equity_correction(
    portfolio: &Portfolio,
    prices: &HashMap<String, Decimal>,
    instruments: &HashMap<String, Instrument>,
    trade_tracker: &TradeTracker,
    risk_config: &RiskConfig,
) -> Decimal {
    if !risk_config.is_margin_account() {
        return Decimal::ZERO;
    }
    account_equity(portfolio, prices, instruments, trade_tracker, risk_config)
        - portfolio.calculate_equity(prices, instruments)
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

    fn futures(symbol: &str) -> Instrument {
        use crate::model::instrument::FuturesInstrument;
        Instrument {
            asset_type: AssetType::Futures,
            inner: InstrumentEnum::Futures(FuturesInstrument {
                symbol: symbol.to_string(),
                multiplier: Decimal::from(10),
                margin_ratio: Decimal::new(1, 1),
                tick_size: Decimal::ONE,
                expiry_date: None,
                settlement_type: None,
                settlement_price: None,
            }),
        }
    }

    /// 保证金账户持有 `quantity` 手期货(价格 4000、乘数 10)、现金 50 万、无浮动盈亏。
    fn futures_account(quantity: i64, mode: &str) -> (Portfolio, RiskConfig) {
        let mut positions = HashMap::new();
        positions.insert("RB".to_string(), Decimal::from(quantity));
        let portfolio = Portfolio {
            cash: Decimal::from(500_000),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut config = RiskConfig::new();
        config.account_mode = mode.to_string();
        (portfolio, config)
    }

    fn futures_market() -> (HashMap<String, Decimal>, HashMap<String, Instrument>) {
        let mut prices = HashMap::new();
        prices.insert("RB".to_string(), Decimal::from(4000));
        let mut instruments = HashMap::new();
        instruments.insert("RB".to_string(), futures("RB"));
        (prices, instruments)
    }

    #[test]
    fn account_equity_excludes_futures_notional_in_margin_account() {
        let (prices, instruments) = futures_market();
        let tracker = TradeTracker::new();
        for quantity in [5, -5] {
            let (portfolio, config) = futures_account(quantity, "margin");
            // 多头不把名义额 20 万加进权益, 空头也不减出去: 无浮动盈亏时权益就是现金
            assert_eq!(
                account_equity(&portfolio, &prices, &instruments, &tracker, &config),
                Decimal::from(500_000)
            );
            let notional = Decimal::from(quantity * 40_000);
            assert_eq!(
                futures_margin_notional(&portfolio.positions, &prices, &instruments, &config),
                notional
            );
            assert_eq!(
                futures_margin_equity_correction(
                    &portfolio,
                    &prices,
                    &instruments,
                    &tracker,
                    &config
                ),
                -notional
            );
        }
    }

    #[test]
    fn account_equity_matches_cash_plus_notional_in_cash_account() {
        // 现金账户里期货按全额名义价入账, 现金 + 名义市值就是权益, 不做修正
        let (prices, instruments) = futures_market();
        let tracker = TradeTracker::new();
        let (portfolio, config) = futures_account(5, "cash");
        assert_eq!(
            account_equity(&portfolio, &prices, &instruments, &tracker, &config),
            Decimal::from(700_000)
        );
        assert_eq!(
            futures_margin_notional(&portfolio.positions, &prices, &instruments, &config),
            Decimal::ZERO
        );
        assert_eq!(
            futures_margin_equity_correction(&portfolio, &prices, &instruments, &tracker, &config),
            Decimal::ZERO
        );
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
