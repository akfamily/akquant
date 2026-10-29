use crate::event::Event;
use crate::execution::common::CommonMatcher;
use crate::execution::matcher::{ExecutionMatcher, MatchContext};
use crate::execution::validation::{reject_order, validate_tick_size};
use crate::model::Order;

/// A 股/基金撮合器。
///
/// `enforce_tick_size` 默认开启：A 股股票 0.01、基金/债券 0.001 的规则是确定
/// 的，且 `Instrument` 的缺省 tick 已按 asset_type 分流（见 model/instrument.rs），
/// 不会出现 Lean 那种"元数据缺失就默认 0.01 导致误拒"的情况。确需关闭时用
/// `set_stock_validation_options(false)`。
pub struct StockMatcher {
    enforce_tick_size: bool,
}

impl StockMatcher {
    pub fn new(enforce_tick_size: bool) -> Self {
        Self { enforce_tick_size }
    }
}

impl Default for StockMatcher {
    fn default() -> Self {
        Self::new(true)
    }
}

impl ExecutionMatcher for StockMatcher {
    fn match_order(&self, order: &mut Order, ctx: &MatchContext) -> Option<Event> {
        if self.enforce_tick_size {
            let tick_size = ctx.instrument.tick_size();
            if let Some(reason) = validate_tick_size(order, tick_size) {
                return reject_order(order, ctx, reason, "stock");
            }
        }
        // Stock: Check Lot Size for Buy orders (e.g. A-Share 100 shares)
        CommonMatcher::match_order(
            order, ctx, true, // check_lot_size = true for Stock
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        ExecutionPolicyCore, Instrument, InstrumentEnum, OrderSide, OrderStatus, OrderType,
        PriceBasis, StockInstrument, TemporalPolicy, TimeInForce,
    };
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    fn create_order(
        side: OrderSide,
        type_: OrderType,
        price: Option<Decimal>,
        stop: Option<Decimal>,
    ) -> Order {
        let mut order = Order::test_new("1", "AAPL", side, type_, dec!(100));
        order.price = price;
        order.trigger_price = stop;
        order.time_in_force = TimeInForce::Day;
        order.status = OrderStatus::New;
        order
    }

    fn create_instrument() -> Instrument {
        Instrument {
            asset_type: crate::model::AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: "AAPL".to_string(),
                lot_size: dec!(100),
                tick_size: dec!(0.01),
                expiry_date: None,
                sellable_after_days: 1,
            }),
        }
    }

    fn create_tick(price: Decimal, timestamp: i64) -> crate::model::Tick {
        crate::model::Tick {
            timestamp,
            price,
            volume: dec!(100),
            symbol: "AAPL".to_string(),
        }
    }

    fn default_policy_core() -> ExecutionPolicyCore {
        ExecutionPolicyCore::default()
    }

    fn tick_ctx<'a>(event: &'a Event, instr: &'a crate::model::Instrument) -> MatchContext<'a> {
        MatchContext {
            event,
            instrument: instr,
            execution_policy_core: default_policy_core(),
            slippage: &crate::execution::slippage::ZeroSlippage,
            volume_limit_pct: Decimal::ZERO,
            ignore_volume: false,
            bar_index: 0,
            last_price: None,
        }
    }

    #[test]
    fn test_stock_rejects_price_not_aligned_with_tick() {
        let matcher = StockMatcher::default();
        let mut order = create_order(OrderSide::Buy, OrderType::Limit, Some(dec!(2.8314)), None);
        let instr = create_instrument(); // tick_size = 0.01
        let event = Event::Tick(create_tick(dec!(2.83), 100));
        let ctx = tick_ctx(&event, &instr);
        let result = matcher.match_order(&mut order, &ctx);
        assert!(result.is_some());
        assert_eq!(order.status, OrderStatus::Rejected);
        assert!(order.reject_reason.contains("tick size"));
    }

    #[test]
    fn test_stock_accepts_aligned_price() {
        let matcher = StockMatcher::default();
        let mut order = create_order(OrderSide::Buy, OrderType::Limit, Some(dec!(2.83)), None);
        let instr = create_instrument();
        let event = Event::Tick(create_tick(dec!(2.83), 100));
        let ctx = tick_ctx(&event, &instr);
        matcher.match_order(&mut order, &ctx);
        assert_ne!(order.status, OrderStatus::Rejected);
    }

    #[test]
    fn test_stock_skips_tick_check_when_disabled() {
        let matcher = StockMatcher::new(false);
        let mut order = create_order(OrderSide::Buy, OrderType::Limit, Some(dec!(2.8314)), None);
        let instr = create_instrument();
        let event = Event::Tick(create_tick(dec!(2.83), 100));
        let ctx = tick_ctx(&event, &instr);
        matcher.match_order(&mut order, &ctx);
        assert_ne!(order.status, OrderStatus::Rejected);
    }

    #[test]
    fn test_buy_stop_in_bar() {
        let matcher = StockMatcher::default();
        let mut order = create_order(OrderSide::Buy, OrderType::StopMarket, None, Some(dec!(105)));
        let instr = create_instrument();

        let bar = crate::model::Bar {
            timestamp: 100,
            symbol: "AAPL".to_string(),
            open: dec!(100),
            high: dec!(110),
            low: dec!(90),
            close: dec!(108),
            volume: dec!(1000),
            extra: Default::default(),
            freq: None,
            interval_start: None,
        };

        let event = Event::Bar(bar);
        let ctx = MatchContext {
            event: &event,
            instrument: &instr,
            execution_policy_core: default_policy_core(),
            slippage: &crate::execution::slippage::ZeroSlippage,
            volume_limit_pct: Decimal::ZERO,
            ignore_volume: false,
            bar_index: 0,
            last_price: None,
        };
        let res = matcher.match_order(&mut order, &ctx);

        assert!(res.is_some());
        if let Some(Event::ExecutionReport(o, Some(t))) = res {
            assert_eq!(o.status, OrderStatus::Filled);
            assert_eq!(t.price, dec!(105)); // Executed at Trigger
        } else {
            panic!("Unexpected result");
        }
    }

    #[test]
    fn test_buy_limit_check_low() {
        let matcher = StockMatcher::default();
        let mut order = create_order(OrderSide::Buy, OrderType::Limit, Some(dec!(90)), None);
        let instr = create_instrument();

        // Bar Low is 95. Limit is 90. Should NOT fill.
        let bar = crate::model::Bar {
            timestamp: 100,
            symbol: "AAPL".to_string(),
            open: dec!(100),
            high: dec!(110),
            low: dec!(95),
            close: dec!(98),
            volume: dec!(1000),
            extra: Default::default(),
            freq: None,
            interval_start: None,
        };

        let event = Event::Bar(bar);
        let ctx = MatchContext {
            event: &event,
            instrument: &instr,
            execution_policy_core: default_policy_core(),
            slippage: &crate::execution::slippage::ZeroSlippage,
            volume_limit_pct: Decimal::ZERO,
            ignore_volume: false,
            bar_index: 0,
            last_price: None,
        };
        let res = matcher.match_order(&mut order, &ctx);

        assert!(res.is_none());
    }

    #[test]
    fn test_sell_limit_uses_open_price_improvement_under_open_policy() {
        let matcher = StockMatcher::default();
        let mut order = create_order(OrderSide::Sell, OrderType::Limit, Some(dec!(108)), None);
        let instr = create_instrument();

        let bar = crate::model::Bar {
            timestamp: 100,
            symbol: "AAPL".to_string(),
            open: dec!(110),
            high: dec!(112),
            low: dec!(105),
            close: dec!(107),
            volume: dec!(1000),
            extra: Default::default(),
            freq: None,
            interval_start: None,
        };

        let event = Event::Bar(bar);
        let ctx = MatchContext {
            event: &event,
            instrument: &instr,
            execution_policy_core: default_policy_core(),
            slippage: &crate::execution::slippage::ZeroSlippage,
            volume_limit_pct: Decimal::ZERO,
            ignore_volume: false,
            bar_index: 0,
            last_price: None,
        };
        let res = matcher.match_order(&mut order, &ctx);

        if let Some(Event::ExecutionReport(o, Some(t))) = res {
            assert_eq!(o.status, OrderStatus::Filled);
            assert_eq!(t.price, dec!(110));
        } else {
            panic!("Unexpected result");
        }
    }

    #[test]
    fn test_sell_limit_does_not_use_open_price_improvement_under_close_policy() {
        let matcher = StockMatcher::default();
        let mut order = create_order(OrderSide::Sell, OrderType::Limit, Some(dec!(108)), None);
        let instr = create_instrument();

        let bar = crate::model::Bar {
            timestamp: 100,
            symbol: "AAPL".to_string(),
            open: dec!(110),
            high: dec!(112),
            low: dec!(105),
            close: dec!(107),
            volume: dec!(1000),
            extra: Default::default(),
            freq: None,
            interval_start: None,
        };

        let event = Event::Bar(bar);
        let ctx = MatchContext {
            event: &event,
            instrument: &instr,
            execution_policy_core: ExecutionPolicyCore {
                price_basis: PriceBasis::Close,
                bar_offset: 0,
                temporal: TemporalPolicy::NextEvent,
            },
            slippage: &crate::execution::slippage::ZeroSlippage,
            volume_limit_pct: Decimal::ZERO,
            ignore_volume: false,
            bar_index: 0,
            last_price: None,
        };
        let res = matcher.match_order(&mut order, &ctx);

        if let Some(Event::ExecutionReport(o, Some(t))) = res {
            assert_eq!(o.status, OrderStatus::Filled);
            assert_eq!(t.price, dec!(108));
        } else {
            panic!("Unexpected result");
        }
    }

    #[test]
    fn test_sell_stop_trail_updates_and_triggers_on_bar() {
        let matcher = StockMatcher::default();
        let mut order = create_order(OrderSide::Sell, OrderType::StopTrail, None, Some(dec!(95)));
        order.trail_offset = Some(dec!(5));
        let instr = create_instrument();

        let first_bar = crate::model::Bar {
            timestamp: 100,
            symbol: "AAPL".to_string(),
            open: dec!(108),
            high: dec!(110),
            low: dec!(106),
            close: dec!(109),
            volume: dec!(1000),
            extra: Default::default(),
            freq: None,
            interval_start: None,
        };
        let first_event = Event::Bar(first_bar);
        let first_ctx = MatchContext {
            event: &first_event,
            instrument: &instr,
            execution_policy_core: default_policy_core(),
            slippage: &crate::execution::slippage::ZeroSlippage,
            volume_limit_pct: Decimal::ZERO,
            ignore_volume: false,
            bar_index: 0,
            last_price: None,
        };
        let first_res = matcher.match_order(&mut order, &first_ctx);
        assert!(first_res.is_none());
        assert_eq!(order.trail_reference_price, Some(dec!(110)));
        assert_eq!(order.trigger_price, Some(dec!(105)));

        let second_bar = crate::model::Bar {
            timestamp: 200,
            symbol: "AAPL".to_string(),
            open: dec!(110),
            high: dec!(111),
            low: dec!(106),
            close: dec!(107),
            volume: dec!(1000),
            extra: Default::default(),
            freq: None,
            interval_start: None,
        };
        let second_event = Event::Bar(second_bar);
        let second_ctx = MatchContext {
            event: &second_event,
            instrument: &instr,
            execution_policy_core: default_policy_core(),
            slippage: &crate::execution::slippage::ZeroSlippage,
            volume_limit_pct: Decimal::ZERO,
            ignore_volume: false,
            bar_index: 1,
            last_price: None,
        };
        let second_res = matcher.match_order(&mut order, &second_ctx);
        assert!(second_res.is_some());
        if let Some(Event::ExecutionReport(o, Some(t))) = second_res {
            assert_eq!(o.status, OrderStatus::Filled);
            assert_eq!(t.price, dec!(106));
        } else {
            panic!("Unexpected result");
        }
    }

    #[test]
    fn test_buy_stop_trail_limit_updates_and_triggers_on_tick() {
        let matcher = StockMatcher::default();
        let mut order = create_order(
            OrderSide::Buy,
            OrderType::StopTrailLimit,
            Some(dec!(101)),
            None,
        );
        order.trail_offset = Some(dec!(2));
        let instr = create_instrument();

        let first_tick_event = Event::Tick(create_tick(dec!(100), 100));
        let first_ctx = MatchContext {
            event: &first_tick_event,
            instrument: &instr,
            execution_policy_core: default_policy_core(),
            slippage: &crate::execution::slippage::ZeroSlippage,
            volume_limit_pct: Decimal::ZERO,
            ignore_volume: false,
            bar_index: 0,
            last_price: None,
        };
        let first_res = matcher.match_order(&mut order, &first_ctx);
        assert!(first_res.is_none());
        assert_eq!(order.trail_reference_price, Some(dec!(100)));
        assert_eq!(order.trigger_price, Some(dec!(102)));

        let second_tick_event = Event::Tick(create_tick(dec!(98), 200));
        let second_ctx = MatchContext {
            event: &second_tick_event,
            instrument: &instr,
            execution_policy_core: default_policy_core(),
            slippage: &crate::execution::slippage::ZeroSlippage,
            volume_limit_pct: Decimal::ZERO,
            ignore_volume: false,
            bar_index: 1,
            last_price: None,
        };
        let second_res = matcher.match_order(&mut order, &second_ctx);
        assert!(second_res.is_none());
        assert_eq!(order.trail_reference_price, Some(dec!(98)));
        assert_eq!(order.trigger_price, Some(dec!(100)));

        let third_tick_event = Event::Tick(create_tick(dec!(100), 300));
        let third_ctx = MatchContext {
            event: &third_tick_event,
            instrument: &instr,
            execution_policy_core: default_policy_core(),
            slippage: &crate::execution::slippage::ZeroSlippage,
            volume_limit_pct: Decimal::ZERO,
            ignore_volume: false,
            bar_index: 2,
            last_price: None,
        };
        let third_res = matcher.match_order(&mut order, &third_ctx);
        assert!(third_res.is_some());
        if let Some(Event::ExecutionReport(o, Some(t))) = third_res {
            assert_eq!(o.status, OrderStatus::Filled);
            assert_eq!(o.order_type, OrderType::Limit);
            assert_eq!(t.price, dec!(100));
        } else {
            panic!("Unexpected result");
        }
    }

    fn bar_event(volume: Decimal) -> Event {
        Event::Bar(crate::model::Bar {
            timestamp: 100,
            symbol: "AAPL".to_string(),
            open: dec!(10.0),
            high: dec!(10.2),
            low: dec!(9.8),
            close: dec!(10.0),
            volume,
            extra: Default::default(),
            freq: None,
            interval_start: None,
        })
    }

    #[test]
    fn test_volume_limit_caps_fill_to_whole_lots() {
        // 0.2499 * 10000 = 2499 股, 按 lot_size=100 向下取整为 2400 (#409)
        let matcher = StockMatcher::default();
        let mut order = create_order(OrderSide::Buy, OrderType::Limit, Some(dec!(10.5)), None);
        order.quantity = dec!(3000);
        let instr = create_instrument();
        let event = bar_event(dec!(10000));
        let mut ctx = tick_ctx(&event, &instr);
        ctx.volume_limit_pct = dec!(0.2499);
        let report = matcher.match_order(&mut order, &ctx);
        assert!(report.is_some());
        assert_eq!(order.filled_quantity, dec!(2400));
        assert_eq!(order.status, OrderStatus::PartiallyFilled);
    }

    #[test]
    fn test_volume_limit_below_one_lot_fills_nothing() {
        // 0.25 * 300 = 75 股, 不足一手: 本根不成交, 订单继续挂着
        let matcher = StockMatcher::default();
        let mut order = create_order(OrderSide::Buy, OrderType::Limit, Some(dec!(10.5)), None);
        order.quantity = dec!(1000);
        let instr = create_instrument();
        let event = bar_event(dec!(300));
        let mut ctx = tick_ctx(&event, &instr);
        ctx.volume_limit_pct = dec!(0.25);
        let report = matcher.match_order(&mut order, &ctx);
        assert!(report.is_none());
        assert_eq!(order.filled_quantity, Decimal::ZERO);
        // 订单仍挂着, 状态不变(未成交也未撤销)
        assert_eq!(order.status, OrderStatus::New);
    }

    #[test]
    fn test_volume_limit_odd_lot_remainder_fills_in_one_piece() {
        // 剩余 50 股(零股) <= 截断上限 0.25 * 300 = 75, 整笔一次成交; 不能因为
        // 75 向下取整到 0 手而卡住(#409)
        let matcher = StockMatcher::default();
        let mut order = create_order(OrderSide::Sell, OrderType::Limit, Some(dec!(9.5)), None);
        order.quantity = dec!(50);
        let instr = create_instrument();
        let event = bar_event(dec!(300));
        let mut ctx = tick_ctx(&event, &instr);
        ctx.volume_limit_pct = dec!(0.25);
        let report = matcher.match_order(&mut order, &ctx);
        assert!(report.is_some());
        assert_eq!(order.filled_quantity, dec!(50));
        assert_eq!(order.status, OrderStatus::Filled);
    }

    #[test]
    fn test_volume_limit_below_one_lot_cancels_ioc() {
        // 截断上限取整为 0 手时, IOC 当根撤销, 不能继续挂着
        let matcher = StockMatcher::default();
        let mut order = create_order(OrderSide::Buy, OrderType::Limit, Some(dec!(10.5)), None);
        order.quantity = dec!(1000);
        order.time_in_force = TimeInForce::IOC;
        let instr = create_instrument();
        let event = bar_event(dec!(300));
        let mut ctx = tick_ctx(&event, &instr);
        ctx.volume_limit_pct = dec!(0.25);
        let report = matcher.match_order(&mut order, &ctx);
        assert!(matches!(report, Some(Event::ExecutionReport(_, None))));
        assert_eq!(order.status, OrderStatus::Cancelled);
        assert_eq!(order.filled_quantity, Decimal::ZERO);
    }
}
