use crate::event::Event;
use crate::execution::common::CommonMatcher;
use crate::execution::matcher::{ExecutionMatcher, MatchContext};
use crate::execution::validation::{is_multiple, reject_order, validate_tick_size};
use crate::model::Order;
use rust_decimal::Decimal;

pub struct FuturesMatcher {
    default_enforce_tick_size: bool,
    default_enforce_lot_size: bool,
    validation_by_prefix: Vec<(String, Option<bool>, Option<bool>)>,
}

impl FuturesMatcher {
    pub fn new(enforce_tick_size: bool, enforce_lot_size: bool) -> Self {
        Self {
            default_enforce_tick_size: enforce_tick_size,
            default_enforce_lot_size: enforce_lot_size,
            validation_by_prefix: Vec::new(),
        }
    }

    pub fn with_prefix_rules(
        enforce_tick_size: bool,
        enforce_lot_size: bool,
        validation_by_prefix: Vec<(String, Option<bool>, Option<bool>)>,
    ) -> Self {
        Self {
            default_enforce_tick_size: enforce_tick_size,
            default_enforce_lot_size: enforce_lot_size,
            validation_by_prefix,
        }
    }

    fn validation_flags_for_symbol(&self, symbol: &str) -> (bool, bool) {
        let mut enforce_tick_size = self.default_enforce_tick_size;
        let mut enforce_lot_size = self.default_enforce_lot_size;
        let mut best_match_len = 0usize;
        let symbol_upper = symbol.to_uppercase();
        for (prefix, tick_opt, lot_opt) in &self.validation_by_prefix {
            let normalized = prefix.trim().to_uppercase();
            if normalized.is_empty() {
                continue;
            }
            if symbol_upper.starts_with(&normalized) && normalized.len() > best_match_len {
                if let Some(tick) = tick_opt {
                    enforce_tick_size = *tick;
                }
                if let Some(lot) = lot_opt {
                    enforce_lot_size = *lot;
                }
                best_match_len = normalized.len();
            }
        }
        (enforce_tick_size, enforce_lot_size)
    }

    fn validate_order(&self, order: &mut Order, ctx: &MatchContext) -> Option<Event> {
        let (enforce_tick_size, enforce_lot_size) =
            self.validation_flags_for_symbol(ctx.instrument.symbol());
        let lot_size = ctx.instrument.lot_size();
        if enforce_lot_size && lot_size > Decimal::ZERO && !is_multiple(order.quantity, lot_size) {
            return reject_order(
                order,
                ctx,
                format!(
                    "Quantity {} is not a multiple of lot size {}",
                    order.quantity, lot_size
                ),
                "futures",
            );
        }

        let tick_size = ctx.instrument.tick_size();
        if !enforce_tick_size {
            return None;
        }
        if let Some(reason) = validate_tick_size(order, tick_size) {
            return reject_order(order, ctx, reason, "futures");
        }
        None
    }
}

impl ExecutionMatcher for FuturesMatcher {
    fn match_order(&self, order: &mut Order, ctx: &MatchContext) -> Option<Event> {
        if let Some(report) = self.validate_order(order, ctx) {
            return Some(report);
        }
        CommonMatcher::match_order(order, ctx, false)
    }
}

impl Default for FuturesMatcher {
    fn default() -> Self {
        Self::new(true, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::instrument::{FuturesInstrument, InstrumentEnum};
    use crate::model::{
        AssetType, ExecutionPolicyCore, Instrument, OrderRole, OrderSide, OrderStatus, OrderType,
        TimeInForce,
    };
    use rust_decimal::prelude::FromStr;
    use rust_decimal_macros::dec;

    fn create_futures_instrument() -> Instrument {
        Instrument {
            asset_type: AssetType::Futures,
            inner: InstrumentEnum::Futures(FuturesInstrument {
                symbol: "RB2310".to_string(),
                multiplier: dec!(10),
                margin_ratio: dec!(0.1),
                tick_size: dec!(0.2),
                expiry_date: None,
                settlement_type: None,
                settlement_price: None,
            }),
        }
    }

    fn create_order(side: OrderSide) -> Order {
        let mut order = Order::test_new("1", "RB2310", side, OrderType::Limit, dec!(1));
        order.price = Some(dec!(3500.0));
        order.time_in_force = TimeInForce::Day;
        order.status = OrderStatus::New;
        order.order_role = OrderRole::Standalone;
        order
    }

    fn create_context<'a>(
        event: &'a Event,
        instrument: &'a Instrument,
    ) -> crate::execution::matcher::MatchContext<'a> {
        crate::execution::matcher::MatchContext {
            event,
            instrument,
            execution_policy_core: ExecutionPolicyCore::default(),
            slippage: &crate::execution::slippage::ZeroSlippage,
            volume_limit_pct: Decimal::ZERO,
            bar_index: 0,
            last_price: None,
        }
    }

    #[test]
    fn test_futures_reject_non_multiple_lot_size_for_sell() {
        let matcher = FuturesMatcher::default();
        let mut order = create_order(OrderSide::Sell);
        order.quantity = Decimal::from_str("1.5").unwrap();
        let instrument = create_futures_instrument();
        let bar = crate::model::Bar {
            timestamp: 100,
            symbol: "RB2310".to_string(),
            open: dec!(3500.0),
            high: dec!(3510.0),
            low: dec!(3490.0),
            close: dec!(3505.0),
            volume: dec!(1000),
            extra: Default::default(),
        };
        let event = Event::Bar(bar);
        let ctx = create_context(&event, &instrument);
        let res = matcher.match_order(&mut order, &ctx);
        assert!(res.is_some());
        assert_eq!(order.status, OrderStatus::Rejected);
        assert!(order.reject_reason.contains("lot size"));
    }

    #[test]
    fn test_futures_reject_non_tick_aligned_limit_price() {
        let matcher = FuturesMatcher::default();
        let mut order = create_order(OrderSide::Buy);
        order.price = Some(dec!(3500.1));
        let instrument = create_futures_instrument();
        let tick = crate::model::Tick {
            timestamp: 100,
            price: dec!(3500.0),
            volume: dec!(10),
            symbol: "RB2310".to_string(),
        };
        let event = Event::Tick(tick);
        let ctx = create_context(&event, &instrument);
        let res = matcher.match_order(&mut order, &ctx);
        assert!(res.is_some());
        assert_eq!(order.status, OrderStatus::Rejected);
        assert!(order.reject_reason.contains("tick size"));
    }

    #[test]
    fn test_validation_reject_warning_includes_order_context_for_lot_size() {
        let mut order = create_order(OrderSide::Sell);
        order.quantity = Decimal::from_str("1.5").unwrap();
        order.owner_strategy_id = Some("fut-alpha".to_string());
        let instrument = create_futures_instrument();
        let bar = crate::model::Bar {
            timestamp: 2_000_000_000,
            symbol: "RB2310".to_string(),
            open: dec!(3500.0),
            high: dec!(3510.0),
            low: dec!(3490.0),
            close: dec!(3505.0),
            volume: dec!(1000),
            extra: Default::default(),
        };
        let event = Event::Bar(bar);
        let ctx = create_context(&event, &instrument);

        let rendered = crate::execution::validation::render_reject_warning(
            &order,
            &ctx,
            "Quantity 1.5 is not a multiple of lot size 1",
            "futures",
        );

        assert!(rendered.contains("Rejected futures order because Quantity 1.5"));
        assert!(rendered.contains("\"phase\":\"execution\""));
        assert!(rendered.contains("\"symbol\":\"RB2310\""));
        assert!(rendered.contains(&format!("\"order_id\":\"{}\"", order.id)));
        assert!(rendered.contains("\"strategy_id\":\"fut-alpha\""));
        assert!(rendered.contains("\"slot\":\"fut-alpha\""));
        assert!(rendered.contains("\"event_time_iso\":\"1970-01-01T00:00:02Z\""));
    }

    #[test]
    fn test_validation_reject_warning_includes_tick_context() {
        let mut order = create_order(OrderSide::Buy);
        order.price = Some(dec!(3500.1));
        let instrument = create_futures_instrument();
        let tick = crate::model::Tick {
            timestamp: 3_000_000_000,
            price: dec!(3500.0),
            volume: dec!(10),
            symbol: "RB2310".to_string(),
        };
        let event = Event::Tick(tick);
        let ctx = create_context(&event, &instrument);

        let rendered = crate::execution::validation::render_reject_warning(
            &order,
            &ctx,
            "price 3500.1 is not aligned with tick size 0.2",
            "futures",
        );

        assert!(rendered.contains("Rejected futures order because price 3500.1"));
        assert!(rendered.contains("\"phase\":\"execution\""));
        assert!(rendered.contains("\"symbol\":\"RB2310\""));
        assert!(rendered.contains(&format!("\"order_id\":\"{}\"", order.id)));
        assert!(rendered.contains("\"event_time_iso\":\"1970-01-01T00:00:03Z\""));
    }

    #[test]
    fn test_futures_accept_tick_aligned_prices() {
        let matcher = FuturesMatcher::default();
        let mut order = create_order(OrderSide::Buy);
        order.order_type = OrderType::StopLimit;
        order.price = Some(dec!(3500.2));
        order.trigger_price = Some(dec!(3501.0));
        let instrument = create_futures_instrument();
        let bar = crate::model::Bar {
            timestamp: 100,
            symbol: "RB2310".to_string(),
            open: dec!(3501.0),
            high: dec!(3510.0),
            low: dec!(3490.0),
            close: dec!(3505.0),
            volume: dec!(1000),
            extra: Default::default(),
        };
        let event = Event::Bar(bar);
        let ctx = create_context(&event, &instrument);
        let _ = matcher.match_order(&mut order, &ctx);
        assert_ne!(order.status, OrderStatus::Rejected);
    }

    #[test]
    fn test_futures_can_disable_tick_validation() {
        let matcher = FuturesMatcher::new(false, true);
        let mut order = create_order(OrderSide::Buy);
        order.price = Some(dec!(3500.1));
        let instrument = create_futures_instrument();
        let tick = crate::model::Tick {
            timestamp: 100,
            price: dec!(3500.0),
            volume: dec!(10),
            symbol: "RB2310".to_string(),
        };
        let event = Event::Tick(tick);
        let ctx = create_context(&event, &instrument);
        let _ = matcher.match_order(&mut order, &ctx);
        assert_ne!(order.status, OrderStatus::Rejected);
    }

    #[test]
    fn test_futures_can_disable_lot_validation() {
        let matcher = FuturesMatcher::new(true, false);
        let mut order = create_order(OrderSide::Sell);
        order.quantity = Decimal::from_str("1.5").unwrap();
        let instrument = create_futures_instrument();
        let bar = crate::model::Bar {
            timestamp: 100,
            symbol: "RB2310".to_string(),
            open: dec!(3500.0),
            high: dec!(3510.0),
            low: dec!(3490.0),
            close: dec!(3505.0),
            volume: dec!(1000),
            extra: Default::default(),
        };
        let event = Event::Bar(bar);
        let ctx = create_context(&event, &instrument);
        let _ = matcher.match_order(&mut order, &ctx);
        assert_ne!(order.status, OrderStatus::Rejected);
    }

    #[test]
    fn test_futures_prefix_rules_override_default_validation() {
        let matcher = FuturesMatcher::with_prefix_rules(
            true,
            true,
            vec![("RB".to_string(), Some(false), Some(false))],
        );
        let mut order = create_order(OrderSide::Sell);
        order.quantity = Decimal::from_str("1.5").unwrap();
        order.price = Some(dec!(3500.1));
        let instrument = create_futures_instrument();
        let bar = crate::model::Bar {
            timestamp: 100,
            symbol: "RB2310".to_string(),
            open: dec!(3500.0),
            high: dec!(3510.0),
            low: dec!(3490.0),
            close: dec!(3505.0),
            volume: dec!(1000),
            extra: Default::default(),
        };
        let event = Event::Bar(bar);
        let ctx = create_context(&event, &instrument);
        let _ = matcher.match_order(&mut order, &ctx);
        assert_ne!(order.status, OrderStatus::Rejected);
    }
}
