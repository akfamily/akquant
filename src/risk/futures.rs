use crate::account::{calculate_account_metrics, estimate_futures_realized_pnl};
use crate::error::AkQuantError;
use crate::model::{AssetType, Order, OrderSide, OrderStatus};
use crate::portfolio::Portfolio;
use rust_decimal::Decimal;
use std::collections::HashMap;

use super::rule::{RiskCheckContext, RiskRule};

#[derive(Debug, Clone)]
pub struct FuturesMarginRule;

impl RiskRule for FuturesMarginRule {
    fn name(&self) -> &'static str {
        "FuturesMarginRule"
    }

    fn check(&self, order: &Order, ctx: &RiskCheckContext) -> Result<(), AkQuantError> {
        if !ctx.config.check_cash
            || !ctx.config.is_margin_account()
            || ctx.instrument.asset_type != AssetType::Futures
        {
            return Ok(());
        }

        let maintenance_ratio = ctx.config.maintenance_margin_ratio_decimal();
        if maintenance_ratio <= Decimal::ZERO {
            return Ok(());
        }

        let Some(order_price) = resolve_order_price(order, ctx.current_prices) else {
            return Ok(());
        };

        let mut prices_for_order = ctx.current_prices.clone();
        prices_for_order.insert(order.symbol.clone(), order_price);

        let mut current_portfolio = ctx.portfolio.clone();
        let pending: Vec<&Order> = ctx
            .active_orders
            .iter()
            .filter(|o| o.status == OrderStatus::New)
            .collect();
        // 同一 OCO 组(如 bracket 的止损止盈)只会成交一条腿, 按一组投影一次
        for active_order in crate::model::dedup_oco_legs(pending, crate::model::larger_remaining)
        {
            let Some(active_price) = resolve_order_price(active_order, ctx.current_prices) else {
                continue;
            };
            apply_order_to_portfolio(
                &mut current_portfolio,
                active_order,
                active_price,
                ctx.instruments,
                ctx.trade_tracker,
            );
        }

        let current_metrics = calculate_account_metrics(
            &current_portfolio,
            &prices_for_order,
            ctx.instruments,
            ctx.trade_tracker,
            ctx.config,
        );

        let mut next_portfolio = current_portfolio.clone();
        apply_order_to_portfolio(
            &mut next_portfolio,
            order,
            order_price,
            ctx.instruments,
            ctx.trade_tracker,
        );
        let next_metrics = calculate_account_metrics(
            &next_portfolio,
            &prices_for_order,
            ctx.instruments,
            ctx.trade_tracker,
            ctx.config,
        );
        if next_metrics.used_margin <= Decimal::ZERO {
            return Ok(());
        }

        let next_ratio = next_metrics.maintenance_ratio;
        if next_ratio < maintenance_ratio && next_metrics.used_margin >= current_metrics.used_margin
        {
            return Err(AkQuantError::OrderError(format!(
                "Risk: Futures maintenance margin breach. Ratio: {next_ratio}, Required: {maintenance_ratio}",
            )));
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn RiskRule> {
        Box::new(self.clone())
    }
}

fn resolve_order_price(
    order: &Order,
    current_prices: &HashMap<String, Decimal>,
) -> Option<Decimal> {
    order
        .price
        .or_else(|| current_prices.get(&order.symbol).copied())
}

fn apply_order_to_portfolio(
    portfolio: &mut Portfolio,
    order: &Order,
    price: Decimal,
    instruments: &HashMap<String, crate::model::Instrument>,
    trade_tracker: &crate::analysis::TradeTracker,
) {
    let Some(instr) = instruments.get(&order.symbol) else {
        return;
    };
    let multiplier = instr.multiplier();
    let realized = estimate_futures_realized_pnl(
        trade_tracker,
        &order.symbol,
        order.side,
        order.quantity,
        price,
        multiplier,
    );
    portfolio.adjust_cash(realized);
    match order.side {
        OrderSide::Buy => {
            portfolio.adjust_position(&order.symbol, order.quantity);
        }
        OrderSide::Sell => {
            portfolio.adjust_position(&order.symbol, -order.quantity);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::TradeTracker;
    use crate::model::instrument::{FuturesInstrument, InstrumentEnum};
    use crate::model::{Instrument, OrderType, TimeInForce};
    use crate::risk::RiskConfig;
    use rust_decimal::prelude::FromStr;
    use std::sync::Arc;

    fn make_futures_instrument(symbol: &str) -> Instrument {
        Instrument {
            asset_type: AssetType::Futures,
            inner: InstrumentEnum::Futures(FuturesInstrument {
                symbol: symbol.to_string(),
                multiplier: Decimal::from(10),
                margin_ratio: Decimal::from_str("0.1").expect("decimal"),
                tick_size: Decimal::from_str("0.2").expect("decimal"),
                expiry_date: None,
                settlement_type: None,
                settlement_price: None,
            }),
        }
    }

    fn make_order(symbol: &str, side: OrderSide, quantity: Decimal) -> Order {
        let mut order = Order::test_new("test-order", symbol, side, OrderType::Limit, quantity);
        order.price = Some(Decimal::from(100));
        order.time_in_force = TimeInForce::Day;
        order.status = OrderStatus::New;
        order
    }

    fn test_market_model() -> &'static dyn crate::market::MarketModel {
        use crate::market::{SimpleMarket, SimpleMarketConfig};
        Box::leak(Box::new(SimpleMarket::from_config(
            SimpleMarketConfig::default(),
        )))
    }

    fn make_context<'a>(
        portfolio: &'a Portfolio,
        instrument: &'a Instrument,
        instruments: &'a HashMap<String, Instrument>,
        current_prices: &'a HashMap<String, Decimal>,
        trade_tracker: &'a TradeTracker,
        config: &'a RiskConfig,
    ) -> RiskCheckContext<'a> {
        RiskCheckContext {
            portfolio,
            instrument,
            instruments,
            active_orders: &[],
            current_prices,
            trade_tracker,
            market_model: test_market_model(),
            current_time: 0,
            config,
            timezone_name: None,
            timezone_offset: 0,
            trading_day: None,
        }
    }

    #[test]
    fn futures_rule_rejects_new_order_below_maintenance_ratio() {
        let symbol = "IF9999".to_string();
        let instrument = make_futures_instrument(&symbol);
        let mut instruments = HashMap::new();
        instruments.insert(symbol.clone(), instrument.clone());
        let mut current_prices = HashMap::new();
        current_prices.insert(symbol.clone(), Decimal::from(100));
        let portfolio = Portfolio {
            cash: Decimal::from(100_000),
            positions: Arc::new(HashMap::new()),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut config = RiskConfig::new();
        config.account_mode = "margin".to_string();
        config.maintenance_margin_ratio = 3.0;
        let tracker = TradeTracker::new();
        let order = make_order(&symbol, OrderSide::Buy, Decimal::from(400));
        let rule = FuturesMarginRule;
        let ctx = make_context(
            &portfolio,
            &instrument,
            &instruments,
            &current_prices,
            &tracker,
            &config,
        );

        let err = rule.check(&order, &ctx).expect_err("should reject order");
        assert!(err.to_string().contains("maintenance margin breach"));
    }

    #[test]
    fn futures_rule_allows_reducing_position_when_below_maintenance_ratio() {
        let symbol = "IF9999".to_string();
        let instrument = make_futures_instrument(&symbol);
        let mut instruments = HashMap::new();
        instruments.insert(symbol.clone(), instrument.clone());
        let mut current_prices = HashMap::new();
        current_prices.insert(symbol.clone(), Decimal::from(100));
        let mut positions = HashMap::new();
        positions.insert(symbol.clone(), Decimal::from(500));
        let portfolio = Portfolio {
            cash: Decimal::from(-400_000),
            positions: Arc::new(positions.clone()),
            available_positions: Arc::new(positions),
        };
        let mut config = RiskConfig::new();
        config.account_mode = "margin".to_string();
        config.maintenance_margin_ratio = 0.3;
        let tracker = TradeTracker::new();
        let order = make_order(&symbol, OrderSide::Sell, Decimal::from(100));
        let rule = FuturesMarginRule;
        let ctx = make_context(
            &portfolio,
            &instrument,
            &instruments,
            &current_prices,
            &tracker,
            &config,
        );

        assert!(rule.check(&order, &ctx).is_ok());
    }

    /// 在途的两张互斥买单(同一 OCO 组)只投影一条腿: 各 400 手, 再买 200 手时
    /// 投影占用 60000(维持率 1.67)而不是 100000(1.0)。不是 OCO 组员的同图订单
    /// 照旧各算一次, 维持率跌破 1.2 被拒。
    #[test]
    fn futures_rule_projects_an_oco_group_once() {
        let symbol = "IF9999".to_string();
        let instrument = make_futures_instrument(&symbol);
        let instruments = HashMap::from([(symbol.clone(), instrument.clone())]);
        let current_prices = HashMap::from([(symbol.clone(), Decimal::from(100))]);
        let portfolio = Portfolio {
            cash: Decimal::from(100_000),
            positions: Arc::new(HashMap::new()),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut config = RiskConfig::new();
        config.account_mode = "margin".to_string();
        config.maintenance_margin_ratio = 1.2;
        let tracker = TradeTracker::new();
        let leg = |id: &str, graph: &str| {
            let mut order = make_order(&symbol, OrderSide::Buy, Decimal::from(400));
            order.id = id.to_string();
            order.graph_id = Some(graph.to_string());
            order
        };
        let order = make_order(&symbol, OrderSide::Buy, Decimal::from(200));
        let check = |active: &[Order]| {
            let mut ctx = make_context(
                &portfolio,
                &instrument,
                &instruments,
                &current_prices,
                &tracker,
                &config,
            );
            ctx.active_orders = active;
            FuturesMarginRule.check(&order, &ctx)
        };

        check(&[leg("a", "oco:g"), leg("b", "oco:g")]).expect("OCO legs project once");
        let err = check(&[leg("a", "plan-1"), leg("b", "plan-1")])
            .expect_err("non-OCO orders are each projected");
        assert!(err.to_string().contains("Ratio: 1.0, Required: 1.2"), "{err}");
    }
}
