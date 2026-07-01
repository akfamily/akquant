use crate::error::AkQuantError;
use crate::model::{Order, OrderSide, OrderStatus};
use rust_decimal::Decimal;

use super::rule::{RiskCheckContext, RiskRule};

/// Check available position for selling (T+1 rule usually implied by `available_positions`)
#[derive(Debug, Clone)]
pub struct StockAvailablePositionRule;

impl RiskRule for StockAvailablePositionRule {
    fn name(&self) -> &'static str {
        "StockAvailablePositionRule"
    }

    fn check(&self, order: &Order, ctx: &RiskCheckContext) -> Result<(), AkQuantError> {
        if order.side == OrderSide::Sell {
            if ctx.config.is_margin_account() && ctx.config.enable_short_sell {
                return Ok(());
            }
            let available = ctx
                .portfolio
                .available_positions
                .get(&order.symbol)
                .copied()
                .unwrap_or(Decimal::ZERO);

            let pending_sell: Decimal = ctx
                .active_orders
                .iter()
                .filter(|o| o.symbol == order.symbol && o.side == OrderSide::Sell)
                .filter(|o| {
                    o.status == OrderStatus::New
                        || o.status == OrderStatus::Submitted
                        || o.status == OrderStatus::PartiallyFilled
                })
                .map(|o| o.quantity - o.filled_quantity)
                .sum();

            if available - pending_sell < order.quantity {
                return Err(AkQuantError::OrderError(format!(
                    "Risk: Insufficient available position for {}. Available: {}, Pending Sell: {}, Required: {}",
                    order.symbol, available, pending_sell, order.quantity
                )));
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn RiskRule> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::instrument::{InstrumentEnum, StockInstrument};
    use crate::model::{AssetType, Instrument, OrderRole, OrderType, TimeInForce};
    use crate::portfolio::Portfolio;
    use crate::risk::RiskConfig;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn make_order(
        id: &str,
        symbol: &str,
        side: OrderSide,
        qty: Decimal,
        status: OrderStatus,
    ) -> Order {
        let mut order = Order::test_new(id, symbol, side, OrderType::Limit, qty);
        order.price = Some(Decimal::from(10));
        order.time_in_force = TimeInForce::IOC;
        order.order_role = OrderRole::Standalone;
        order.status = status;
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
        active_orders: &'a [Order],
        current_prices: &'a HashMap<String, Decimal>,
        trade_tracker: &'a crate::analysis::TradeTracker,
        config: &'a RiskConfig,
    ) -> super::super::rule::RiskCheckContext<'a> {
        super::super::rule::RiskCheckContext {
            portfolio,
            instrument,
            instruments,
            active_orders,
            current_prices,
            trade_tracker,
            market_model: test_market_model(),
            current_time: 0,
            config,
            timezone_name: None,
            timezone_offset: 0,
        }
    }

    #[test]
    fn test_cancelled_ioc_sell_not_counted_as_pending() {
        let symbol = "sz300274".to_string();
        let mut available_positions = HashMap::new();
        available_positions.insert(symbol.clone(), Decimal::from(68900));

        let portfolio = Portfolio {
            cash: Decimal::from(1_000_000),
            positions: Arc::new(HashMap::new()),
            available_positions: Arc::new(available_positions),
        };

        let instrument = Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.clone(),
                lot_size: Decimal::from(100),
                tick_size: Decimal::new(1, 2),
                expiry_date: None,
            }),
        };

        let mut instruments = HashMap::new();
        instruments.insert(symbol.clone(), instrument.clone());
        let current_prices = HashMap::new();
        let config = RiskConfig::new();
        let tracker = crate::analysis::TradeTracker::new();

        let cancelled_ioc = make_order(
            "o1",
            &symbol,
            OrderSide::Sell,
            Decimal::from(68900),
            OrderStatus::Cancelled,
        );
        let active_orders = vec![cancelled_ioc];
        let ctx = make_context(
            &portfolio,
            &instrument,
            &instruments,
            &active_orders,
            &current_prices,
            &tracker,
            &config,
        );

        let new_sell = make_order(
            "o2",
            &symbol,
            OrderSide::Sell,
            Decimal::from(68900),
            OrderStatus::New,
        );
        let rule = StockAvailablePositionRule;
        let result = rule.check(&new_sell, &ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_submitted_sell_still_counted_as_pending() {
        let symbol = "sz300274".to_string();
        let mut available_positions = HashMap::new();
        available_positions.insert(symbol.clone(), Decimal::from(68900));

        let portfolio = Portfolio {
            cash: Decimal::from(1_000_000),
            positions: Arc::new(HashMap::new()),
            available_positions: Arc::new(available_positions),
        };

        let instrument = Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.clone(),
                lot_size: Decimal::from(100),
                tick_size: Decimal::new(1, 2),
                expiry_date: None,
            }),
        };

        let mut instruments = HashMap::new();
        instruments.insert(symbol.clone(), instrument.clone());
        let current_prices = HashMap::new();
        let config = RiskConfig::new();
        let tracker = crate::analysis::TradeTracker::new();

        let submitted_sell = make_order(
            "o1",
            &symbol,
            OrderSide::Sell,
            Decimal::from(68900),
            OrderStatus::Submitted,
        );
        let active_orders = vec![submitted_sell];
        let ctx = make_context(
            &portfolio,
            &instrument,
            &instruments,
            &active_orders,
            &current_prices,
            &tracker,
            &config,
        );

        let new_sell = make_order(
            "o2",
            &symbol,
            OrderSide::Sell,
            Decimal::from(100),
            OrderStatus::New,
        );
        let rule = StockAvailablePositionRule;
        let result = rule.check(&new_sell, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_margin_account_can_short_sell_without_available_position() {
        let symbol = "sz300274".to_string();
        let portfolio = Portfolio {
            cash: Decimal::from(1_000_000),
            positions: Arc::new(HashMap::new()),
            available_positions: Arc::new(HashMap::new()),
        };

        let instrument = Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.clone(),
                lot_size: Decimal::from(100),
                tick_size: Decimal::new(1, 2),
                expiry_date: None,
            }),
        };

        let mut instruments = HashMap::new();
        instruments.insert(symbol.clone(), instrument.clone());
        let current_prices = HashMap::new();
        let mut config = RiskConfig::new();
        config.account_mode = "margin".to_string();
        config.enable_short_sell = true;
        let tracker = crate::analysis::TradeTracker::new();

        let active_orders: Vec<Order> = vec![];
        let ctx = make_context(
            &portfolio,
            &instrument,
            &instruments,
            &active_orders,
            &current_prices,
            &tracker,
            &config,
        );

        let new_sell = make_order(
            "o2",
            &symbol,
            OrderSide::Sell,
            Decimal::from(100),
            OrderStatus::New,
        );
        let rule = StockAvailablePositionRule;
        let result = rule.check(&new_sell, &ctx);
        assert!(result.is_ok());
    }
}
