use crate::error::AkQuantError;
use crate::model::{AssetType, Order, OrderSide, project_position_after};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

use super::rule::{RiskCheckContext, RiskRule};

const RISK_OVERFLOW_PREFIX: &str = "AKQ-RISK-OVERFLOW";

fn risk_overflow_error(symbol: &str, field: &str) -> AkQuantError {
    AkQuantError::OrderError(format!(
        "[{RISK_OVERFLOW_PREFIX}] overflow while calculating margin for {symbol} at {field}",
    ))
}

fn checked_sub_or_zero(lhs: Decimal, rhs: Decimal) -> Decimal {
    lhs.checked_sub(rhs).unwrap_or(Decimal::ZERO)
}

fn checked_mul_or_err(
    lhs: Decimal,
    rhs: Decimal,
    symbol: &str,
    field: &str,
) -> Result<Decimal, AkQuantError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| risk_overflow_error(symbol, field))
}

fn checked_sub_or_err(
    lhs: Decimal,
    rhs: Decimal,
    symbol: &str,
    field: &str,
) -> Result<Decimal, AkQuantError> {
    lhs.checked_sub(rhs)
        .ok_or_else(|| risk_overflow_error(symbol, field))
}

/// Check restricted list
#[derive(Debug, Clone)]
pub struct RestrictedListRule;

impl RiskRule for RestrictedListRule {
    fn name(&self) -> &'static str {
        "RestrictedListRule"
    }

    fn check(&self, order: &Order, ctx: &RiskCheckContext) -> Result<(), AkQuantError> {
        if ctx.config.restricted_list.contains(&order.symbol) {
            return Err(AkQuantError::OrderError(format!(
                "Risk: Symbol {} is restricted",
                order.symbol
            )));
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn RiskRule> {
        Box::new(self.clone())
    }
}

/// Check max order size
#[derive(Debug, Clone)]
pub struct MaxOrderSizeRule;

impl RiskRule for MaxOrderSizeRule {
    fn name(&self) -> &'static str {
        "MaxOrderSizeRule"
    }

    fn check(&self, order: &Order, ctx: &RiskCheckContext) -> Result<(), AkQuantError> {
        if let Some(max_size) = ctx.config.max_order_size
            && order.quantity > max_size
        {
            return Err(AkQuantError::OrderError(format!(
                "Risk: Order quantity {} exceeds limit {}",
                order.quantity, max_size
            )));
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn RiskRule> {
        Box::new(self.clone())
    }
}

/// Check max order value
#[derive(Debug, Clone)]
pub struct MaxOrderValueRule;

impl RiskRule for MaxOrderValueRule {
    fn name(&self) -> &'static str {
        "MaxOrderValueRule"
    }

    fn check(&self, order: &Order, ctx: &RiskCheckContext) -> Result<(), AkQuantError> {
        if let Some(max_value) = ctx.config.max_order_value {
            let price = if let Some(p) = order.price {
                Some(p)
            } else {
                ctx.current_prices.get(&order.symbol).copied()
            };

            if let Some(p) = price {
                let value = p * order.quantity;
                if value > max_value {
                    return Err(AkQuantError::OrderError(format!(
                        "Risk: Order value {value} exceeds limit {max_value}",
                    )));
                }
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn RiskRule> {
        Box::new(self.clone())
    }
}

/// Check max position size
#[derive(Debug, Clone)]
pub struct MaxPositionSizeRule;

impl RiskRule for MaxPositionSizeRule {
    fn name(&self) -> &'static str {
        "MaxPositionSizeRule"
    }

    fn check(&self, order: &Order, ctx: &RiskCheckContext) -> Result<(), AkQuantError> {
        if let Some(max_pos) = ctx.config.max_position_size {
            let current_pos = ctx
                .portfolio
                .positions
                .get(&order.symbol)
                .copied()
                .unwrap_or(Decimal::ZERO);
            let new_pos = project_position_after(
                order.side,
                order.position_effect,
                current_pos,
                order.quantity,
            );
            if new_pos.abs() > max_pos {
                return Err(AkQuantError::OrderError(format!(
                    "Risk: Resulting position {new_pos} exceeds limit {max_pos}",
                )));
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn RiskRule> {
        Box::new(self.clone())
    }
}

/// Check cash / margin sufficiency
#[derive(Debug, Clone)]
pub struct CashMarginRule;

impl RiskRule for CashMarginRule {
    fn name(&self) -> &'static str {
        "CashMarginRule"
    }

    fn check(&self, order: &Order, ctx: &RiskCheckContext) -> Result<(), AkQuantError> {
        // Futures in a margin account are intentionally delegated to
        // FuturesMarginRule (maintenance-ratio based), so this cash/free-margin
        // gate steps aside. This is an UNCONDITIONAL hand-off: it relies on
        // FuturesMarginRule being registered for AssetType::Futures in
        // RiskManager::init_rules. If that registration is ever removed, futures
        // margin accounts would lose ALL submission-time margin checks silently.
        // `risk_rules_cover_futures_margin_account` guards that invariant.
        if ctx.config.is_margin_account() && ctx.instrument.asset_type == AssetType::Futures {
            return Ok(());
        }
        if ctx.config.check_cash {
            // Submission-time affordability: the fill price is not yet known, so
            // approximate with the limit price, else the last price. If neither
            // is available we cannot check and let the order through.
            let order_price = if let Some(p) = order.price {
                p
            } else if let Some(p) = ctx.current_prices.get(&order.symbol) {
                *p
            } else {
                return Ok(());
            };

            let safety_margin =
                Decimal::from_f64(ctx.config.safety_margin).unwrap_or(Decimal::ZERO);
            let result = check_affordability(
                order,
                ctx.portfolio,
                ctx.current_prices,
                ctx.instruments,
                ctx.config,
                ctx.market_model,
                ctx.active_orders,
                order_price,
                safety_margin,
            )?;

            // Nothing to fund: the margin this order releases already covers its
            // own fees, so they come out of the net proceeds and no cash needs to
            // be held up front. This is the reduce-only sell path (#400) — a sell
            // whose fees exceed its proceeds still lands below with a positive
            // `required` and is gated normally.
            if result.required.is_zero() {
                return Ok(());
            }

            if !result.affordable {
                let (required, available) = (result.required, result.available);
                return Err(AkQuantError::OrderError(format!(
                    "Risk: Insufficient margin. Required: {required}, Available: {available} (Safety: {})",
                    ctx.config.safety_margin
                )));
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn RiskRule> {
        Box::new(self.clone())
    }
}

/// Signed margin delta for a stock/fund order in a margin account.
///
/// **Negative for a reducing order** — closing a position releases margin. The
/// sign must survive: clamping it to zero here would drop the released funds and
/// leave a pure reduce-only sell looking like it needs its commission in cash up
/// front, which rejects sells in a fully-invested account (#400). Callers clamp
/// the final `margin_delta + commission` sum instead.
fn stock_margin_delta(
    order: &Order,
    current_pos: Decimal,
    price: Decimal,
    multiplier: Decimal,
    initial_margin_ratio: Decimal,
) -> Result<Decimal, AkQuantError> {
    let current_abs = current_pos.abs();
    let safe_price = price.abs();
    let safe_multiplier = multiplier.abs();
    let safe_initial_margin_ratio = initial_margin_ratio.abs();
    let next_pos = project_position_after(
        order.side,
        order.position_effect,
        current_pos,
        order.quantity,
    );
    let next_abs = next_pos.abs();
    let current_notional = checked_mul_or_err(
        current_abs,
        safe_price,
        &order.symbol,
        "current_abs * price",
    )?;
    let current_gross = checked_mul_or_err(
        current_notional,
        safe_multiplier,
        &order.symbol,
        "current_notional * multiplier",
    )?;
    let current_margin = checked_mul_or_err(
        current_gross,
        safe_initial_margin_ratio,
        &order.symbol,
        "current_gross * initial_margin_ratio",
    )?;

    let next_notional =
        checked_mul_or_err(next_abs, safe_price, &order.symbol, "next_abs * price")?;
    let next_gross = checked_mul_or_err(
        next_notional,
        safe_multiplier,
        &order.symbol,
        "next_notional * multiplier",
    )?;
    let next_margin = checked_mul_or_err(
        next_gross,
        safe_initial_margin_ratio,
        &order.symbol,
        "next_gross * initial_margin_ratio",
    )?;

    Ok(checked_sub_or_zero(next_margin, current_margin))
}

/// Signed margin delta for any order. See [`stock_margin_delta`] on why the sign
/// is preserved rather than clamped to zero (#400).
fn calc_required_margin_delta(
    order: &Order,
    instruments: &HashMap<String, crate::model::Instrument>,
    config: &crate::risk::RiskConfig,
    prices: &HashMap<String, Decimal>,
    portfolio: &crate::portfolio::Portfolio,
) -> Result<Decimal, AkQuantError> {
    if let Some(instr) = instruments.get(&order.symbol)
        && config.is_margin_account()
        && (instr.asset_type == AssetType::Stock || instr.asset_type == AssetType::Fund)
    {
        let price = prices
            .get(&order.symbol)
            .copied()
            .unwrap_or_else(|| order.price.unwrap_or(Decimal::ZERO));
        if price <= Decimal::ZERO {
            return Ok(Decimal::ZERO);
        }
        let current_pos = portfolio
            .positions
            .get(&order.symbol)
            .copied()
            .unwrap_or(Decimal::ZERO);
        return stock_margin_delta(
            order,
            current_pos,
            price,
            instr.multiplier(),
            config.stock_initial_margin_ratio(),
        );
    }

    let stock_ratio_override = if config.is_margin_account() {
        Some(config.stock_initial_margin_ratio())
    } else {
        None
    };

    let mut projected_portfolio = portfolio.clone();
    let base_used = projected_portfolio.calculate_used_margin_with_stock_ratio(
        prices,
        instruments,
        stock_ratio_override,
    );
    if base_used == Decimal::MAX {
        return Err(risk_overflow_error(&order.symbol, "base_used_margin"));
    }
    {
        let positions = std::sync::Arc::make_mut(&mut projected_portfolio.positions);
        let entry = positions
            .entry(order.symbol.clone())
            .or_insert(Decimal::ZERO);
        *entry = project_position_after(order.side, order.position_effect, *entry, order.quantity);
    }
    let next_used = projected_portfolio.calculate_used_margin_with_stock_ratio(
        prices,
        instruments,
        stock_ratio_override,
    );
    if next_used == Decimal::MAX {
        return Err(risk_overflow_error(&order.symbol, "next_used_margin"));
    }
    checked_sub_or_err(next_used, base_used, &order.symbol, "next_used - base_used")
}

/// Outcome of a unified affordability check. Single source of truth for
/// "can this order be funded" across submission-time risk gating and
/// execution-time margin checks (issue #292).
#[derive(Debug, Clone)]
pub(crate) struct AffordabilityResult {
    /// Cash/margin this order must be able to fund: its signed margin delta plus
    /// commission, floored at zero. Zero for a reducing order whose released
    /// margin already covers its own fees (#400).
    pub required: Decimal,
    /// Free margin available after projecting pending orders, net of the
    /// caller-supplied safety haircut.
    pub available: Decimal,
    /// Whether `required <= available`.
    pub affordable: bool,
    /// Largest lot-rounded quantity that would be affordable (for auto-resize
    /// callers). Equals `order.quantity` when the order already fits.
    pub max_affordable_qty: Decimal,
}

/// Project all pending `New` orders (other than a fill already reflected in the
/// portfolio) into a cloned portfolio: buys spend cash and add position, sells
/// release cash and reduce position. This is what lets a sell free up cash for a
/// same-cycle buy instead of the buy being rejected against pre-sale cash.
pub(crate) fn project_active_orders_into(
    portfolio: &crate::portfolio::Portfolio,
    active_orders: &[Order],
    prices: &HashMap<String, Decimal>,
    instruments: &HashMap<String, crate::model::Instrument>,
) -> crate::portfolio::Portfolio {
    let mut projected = portfolio.clone();
    for o in active_orders {
        if o.status != crate::model::OrderStatus::New {
            continue;
        }
        let active_price = if let Some(p) = o.price {
            p
        } else if let Some(p) = prices.get(&o.symbol) {
            *p
        } else {
            continue;
        };
        if let Some(instr) = instruments.get(&o.symbol) {
            let cost = active_price * o.quantity * instr.multiplier();
            let current_pos = projected
                .positions
                .get(&o.symbol)
                .copied()
                .unwrap_or(Decimal::ZERO);
            let next_pos =
                project_position_after(o.side, o.position_effect, current_pos, o.quantity);
            let delta = next_pos - current_pos;
            match o.side {
                OrderSide::Buy => {
                    projected.adjust_cash(-cost);
                    projected.adjust_position(&o.symbol, delta);
                }
                OrderSide::Sell => {
                    projected.adjust_cash(cost);
                    projected.adjust_position(&o.symbol, delta);
                }
            }
        }
    }
    projected
}

/// Unified affordability check. Every checkpoint (submission gate, submission
/// auto-resize, execution gate, execution auto-resize) funnels through this so
/// price source, pending-order projection, safety haircut, and commission are
/// computed one way.
///
/// - `price`: the authoritative price for THIS order — submission passes the
///   last/limit price (a necessary approximation), execution passes the real
///   fill price.
/// - `market_model`: used to compute the order's commission, which is folded
///   into `required` (both submission and execution include it). Commission is
///   computed after the checked margin delta so absurd inputs surface as a
///   graceful margin-overflow error rather than a panic. The margin delta is
///   SIGNED, so a reducing order's released margin nets the commission off
///   before the sum is clamped — fees on a reduce come out of the proceeds
///   rather than being demanded in cash up front (#400).
/// - `safety_margin`: applied as `(1 - safety_margin)` to available margin —
///   submission passes `config.safety_margin`, execution passes `0` (the fill
///   price and commission are already real, so no buffer is warranted).
/// - `active_orders`: pending `New` orders to project (pass `&[]` when the
///   caller has already folded fills into `portfolio`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_affordability(
    order: &Order,
    portfolio: &crate::portfolio::Portfolio,
    prices: &HashMap<String, Decimal>,
    instruments: &HashMap<String, crate::model::Instrument>,
    config: &crate::risk::RiskConfig,
    market_model: &dyn crate::market::MarketModel,
    active_orders: &[Order],
    price: Decimal,
    safety_margin: Decimal,
) -> Result<AffordabilityResult, AkQuantError> {
    let mut prices_for_order = prices.clone();
    prices_for_order.insert(order.symbol.clone(), price);

    let projected_portfolio =
        project_active_orders_into(portfolio, active_orders, prices, instruments);

    let margin_delta = calc_required_margin_delta(
        order,
        instruments,
        config,
        &prices_for_order,
        &projected_portfolio,
    )?;

    // Commission after the checked margin delta so overflow surfaces via the
    // margin path rather than panicking in the market model's raw arithmetic.
    let commission = instruments
        .get(&order.symbol)
        .map(|instr| market_model.calculate_commission(instr, order.side, price, order.quantity))
        .unwrap_or(Decimal::ZERO);
    let required = (margin_delta + commission).max(Decimal::ZERO);

    let stock_ratio_override = if config.is_margin_account() {
        Some(config.stock_initial_margin_ratio())
    } else {
        None
    };
    let free_margin = projected_portfolio.calculate_free_margin_with_stock_ratio(
        prices,
        instruments,
        stock_ratio_override,
    );
    let safety_factor = (Decimal::ONE - safety_margin).max(Decimal::ZERO);
    let available = free_margin
        .checked_mul(safety_factor)
        .unwrap_or(Decimal::ZERO)
        .max(Decimal::ZERO);

    let affordable = required <= available;

    // Linear estimate of the largest affordable quantity, lot-rounded. Used only
    // by auto-resize callers; `required` scales ~linearly with quantity for the
    // dominant cash/stock path, so this matches the prior resize seed.
    let max_affordable_qty = if affordable {
        order.quantity
    } else if required > Decimal::ZERO
        && available > Decimal::ZERO
        && order.quantity > Decimal::ZERO
    {
        let raw = order.quantity * available / required;
        let lot = instruments
            .get(&order.symbol)
            .map(|instr| instr.lot_size())
            .unwrap_or(Decimal::ONE);
        let mut qty = raw.floor();
        if lot > Decimal::ZERO {
            qty -= qty % lot;
        }
        qty.max(Decimal::ZERO)
    } else {
        Decimal::ZERO
    };

    Ok(AffordabilityResult {
        required,
        available,
        affordable,
        max_affordable_qty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::margin::MarginEngine;
    use crate::model::instrument::{InstrumentEnum, OptionInstrument, StockInstrument};
    use crate::model::{Instrument, OptionMarginModel, OptionType, OrderType, TimeInForce};
    use crate::portfolio::Portfolio;
    use crate::risk::RiskConfig;
    use rust_decimal_macros::dec;
    use std::sync::Arc;

    fn create_stock_instrument(symbol: &str) -> Instrument {
        Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.to_string(),
                lot_size: dec!(100),
                tick_size: dec!(0.01),
                expiry_date: None,
                sellable_after_days: 1,
            }),
        }
    }

    fn create_china_short_put(symbol: &str, underlying_symbol: &str) -> Instrument {
        Instrument {
            asset_type: AssetType::Option,
            inner: InstrumentEnum::Option(OptionInstrument {
                symbol: symbol.to_string(),
                multiplier: dec!(100),
                margin_ratio: dec!(0.2),
                tick_size: dec!(0.01),
                option_margin_model: OptionMarginModel::ChinaSingleLeg,
                option_type: OptionType::Put,
                strike_price: dec!(100),
                expiry_date: 20260101,
                underlying_symbol: underlying_symbol.to_string(),
                settlement_type: None,
                implied_volatility: None,
                reference_volatility: None,
            }),
        }
    }

    fn create_order(symbol: &str, side: OrderSide, quantity: Decimal, price: Decimal) -> Order {
        let mut order = Order::test_new("test", symbol, side, OrderType::Limit, quantity);
        order.price = Some(price);
        order.time_in_force = TimeInForce::Day;
        order.status = crate::model::OrderStatus::New;
        order
    }

    fn test_market_model() -> &'static dyn crate::market::MarketModel {
        use crate::market::{SimpleMarket, SimpleMarketConfig};
        Box::leak(Box::new(SimpleMarket::from_config(
            SimpleMarketConfig::default(),
        )))
    }

    #[test]
    fn required_margin_delta_for_china_short_put_matches_margin_engine() {
        let option = create_china_short_put("OPT_P", "510050.SH");
        let mut instruments = HashMap::new();
        instruments.insert("OPT_P".to_string(), option.clone());

        let mut prices = HashMap::new();
        prices.insert("OPT_P".to_string(), dec!(4));
        prices.insert("510050.SH".to_string(), dec!(110));

        let portfolio = Portfolio {
            cash: dec!(100000),
            positions: Arc::new(HashMap::new()),
            available_positions: Arc::new(HashMap::new()),
        };
        let instrument_ref = instruments.get("OPT_P").unwrap();
        let config = RiskConfig::new();
        let tracker = crate::analysis::TradeTracker::new();
        let ctx = RiskCheckContext {
            portfolio: &portfolio,
            instrument: instrument_ref,
            instruments: &instruments,
            active_orders: &[],
            current_prices: &prices,
            trade_tracker: &tracker,
            market_model: test_market_model(),
            current_time: 0,
            config: &config,
            timezone_name: None,
            timezone_offset: 0,
        };
        let order = create_order("OPT_P", OrderSide::Sell, dec!(1), dec!(4));

        let required =
            calc_required_margin_delta(&order, ctx.instruments, ctx.config, &prices, &portfolio)
                .unwrap();
        let expected =
            MarginEngine::position_margin(dec!(-1), dec!(4), instrument_ref, &prices, None);

        assert_eq!(required, dec!(1100));
        assert_eq!(required, expected);
    }

    #[test]
    fn cash_margin_rule_matches_execution_resize_boundary_for_stock_buys() {
        let stock = create_stock_instrument("AAPL");
        let mut instruments = HashMap::new();
        instruments.insert("AAPL".to_string(), stock.clone());

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), dec!(100));

        let portfolio = Portfolio {
            cash: dec!(50000),
            positions: Arc::new(HashMap::new()),
            available_positions: Arc::new(HashMap::new()),
        };
        let instrument_ref = instruments.get("AAPL").unwrap();
        let config = RiskConfig::new();
        let tracker = crate::analysis::TradeTracker::new();
        let ctx = RiskCheckContext {
            portfolio: &portfolio,
            instrument: instrument_ref,
            instruments: &instruments,
            active_orders: &[],
            current_prices: &prices,
            trade_tracker: &tracker,
            market_model: test_market_model(),
            current_time: 0,
            config: &config,
            timezone_name: None,
            timezone_offset: 0,
        };
        let rule = CashMarginRule;

        let rejected = create_order("AAPL", OrderSide::Buy, dec!(500), dec!(100));
        let accepted = create_order("AAPL", OrderSide::Buy, dec!(400), dec!(100));

        let rejected_result = rule.check(&rejected, &ctx);
        let accepted_result = rule.check(&accepted, &ctx);

        assert!(rejected_result.is_err());
        assert!(
            rejected_result
                .unwrap_err()
                .to_string()
                .contains("Insufficient margin")
        );
        assert!(accepted_result.is_ok());
    }

    #[test]
    fn pending_cross_symbol_sell_frees_cash_for_same_cycle_buy() {
        // Issue #292: a buy must not be rejected against pre-sale cash when a
        // same-cycle sell of another symbol will release funds. CashMarginRule
        // projects the pending sell before gating the buy.
        let mut instruments = HashMap::new();
        instruments.insert("AAA".to_string(), create_stock_instrument("AAA"));
        instruments.insert("BBB".to_string(), create_stock_instrument("BBB"));

        let mut prices = HashMap::new();
        prices.insert("AAA".to_string(), dec!(1));
        prices.insert("BBB".to_string(), dec!(1));

        // Almost fully invested in AAA, little free cash.
        let mut positions = HashMap::new();
        positions.insert("AAA".to_string(), dec!(980000));
        let portfolio = Portfolio {
            cash: dec!(20000),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };
        let config = RiskConfig::new();
        let tracker = crate::analysis::TradeTracker::new();
        let instrument_ref = instruments.get("BBB").unwrap();

        // A pending sell of the whole AAA position releases ~980k cash.
        let pending_sell = create_order("AAA", OrderSide::Sell, dec!(980000), dec!(1));
        let active_orders = vec![pending_sell];

        let ctx = RiskCheckContext {
            portfolio: &portfolio,
            instrument: instrument_ref,
            instruments: &instruments,
            active_orders: &active_orders,
            current_prices: &prices,
            trade_tracker: &tracker,
            market_model: test_market_model(),
            current_time: 0,
            config: &config,
            timezone_name: None,
            timezone_offset: 0,
        };
        let rule = CashMarginRule;

        // Buying 900k of BBB costs ~900k — unaffordable against 20k cash, but
        // affordable once the pending AAA sale is projected in.
        let buy = create_order("BBB", OrderSide::Buy, dec!(900000), dec!(1));
        assert!(rule.check(&buy, &ctx).is_ok());

        // Without the pending sell, the same buy must be rejected.
        let ctx_no_sell = RiskCheckContext {
            active_orders: &[],
            ..ctx
        };
        assert!(rule.check(&buy, &ctx_no_sell).is_err());
    }

    /// A-share sell fees (commission + stamp tax + transfer fee) so a sell's
    /// fees can exceed the cash left after a full-position buy.
    fn fee_market_model() -> &'static dyn crate::market::MarketModel {
        use crate::market::{SimpleMarket, SimpleMarketConfig};
        Box::leak(Box::new(SimpleMarket::from_config(SimpleMarketConfig {
            commission_rate: dec!(0.0003),
            stamp_tax: dec!(0.001),
            transfer_fee: dec!(0.00002),
            min_commission: dec!(5),
            ..SimpleMarketConfig::default()
        })))
    }

    #[test]
    fn reduce_only_sell_passes_when_proceeds_cover_fees_cash_account() {
        // Issue #400: fully invested cash account, 68 yuan left. Selling the
        // whole 100k position costs 132 in fees but releases 100k of cash, so
        // the fees come out of the net proceeds — the sell must not be gated on
        // pre-fill cash.
        let mut instruments = HashMap::new();
        instruments.insert("600000".to_string(), create_stock_instrument("600000"));
        let mut prices = HashMap::new();
        prices.insert("600000".to_string(), dec!(10));

        let mut positions = HashMap::new();
        positions.insert("600000".to_string(), dec!(10000));
        let portfolio = Portfolio {
            cash: dec!(68),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };
        let config = RiskConfig::new();
        let tracker = crate::analysis::TradeTracker::new();
        let instrument_ref = instruments.get("600000").unwrap();
        let ctx = RiskCheckContext {
            portfolio: &portfolio,
            instrument: instrument_ref,
            instruments: &instruments,
            active_orders: &[],
            current_prices: &prices,
            trade_tracker: &tracker,
            market_model: fee_market_model(),
            current_time: 0,
            config: &config,
            timezone_name: None,
            timezone_offset: 0,
        };

        let sell = create_order("600000", OrderSide::Sell, dec!(10000), dec!(10));
        assert!(CashMarginRule.check(&sell, &ctx).is_ok());
    }

    #[test]
    fn reduce_only_sell_passes_in_margin_account_when_free_margin_is_negative() {
        // Same defect on the margin-account code path (`stock_margin_delta`),
        // which a cash account never reaches. The position is underwater so free
        // margin is tiny, yet closing it releases margin rather than consuming
        // any — the sell must be allowed (#400).
        let mut instruments = HashMap::new();
        instruments.insert("600000".to_string(), create_stock_instrument("600000"));
        let mut prices = HashMap::new();
        prices.insert("600000".to_string(), dec!(10));

        let mut positions = HashMap::new();
        positions.insert("600000".to_string(), dec!(20000));
        let portfolio = Portfolio {
            cash: dec!(-99900),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut config = RiskConfig::new();
        config.account_mode = "margin".to_string();
        config.initial_margin_ratio = 0.5;
        let tracker = crate::analysis::TradeTracker::new();
        let instrument_ref = instruments.get("600000").unwrap();
        let ctx = RiskCheckContext {
            portfolio: &portfolio,
            instrument: instrument_ref,
            instruments: &instruments,
            active_orders: &[],
            current_prices: &prices,
            trade_tracker: &tracker,
            market_model: fee_market_model(),
            current_time: 0,
            config: &config,
            timezone_name: None,
            timezone_offset: 0,
        };

        let sell = create_order("600000", OrderSide::Sell, dec!(20000), dec!(10));
        assert!(CashMarginRule.check(&sell, &ctx).is_ok());
    }

    #[test]
    fn reduce_only_sell_still_rejected_when_fees_exceed_proceeds() {
        // The relief is "fees come out of the net proceeds", NOT "a close is
        // always allowed": when the proceeds cannot cover the fees the order
        // still needs funding, and letting it through would overdraft cash —
        // which only `check_cash=False` may do (#280).
        let mut instruments = HashMap::new();
        instruments.insert("600000".to_string(), create_stock_instrument("600000"));
        let mut prices = HashMap::new();
        prices.insert("600000".to_string(), dec!(0.01));

        let mut positions = HashMap::new();
        positions.insert("600000".to_string(), dec!(100));
        let portfolio = Portfolio {
            cash: Decimal::ZERO,
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };
        let config = RiskConfig::new();
        let tracker = crate::analysis::TradeTracker::new();
        let instrument_ref = instruments.get("600000").unwrap();
        let ctx = RiskCheckContext {
            portfolio: &portfolio,
            instrument: instrument_ref,
            instruments: &instruments,
            active_orders: &[],
            current_prices: &prices,
            trade_tracker: &tracker,
            market_model: fee_market_model(),
            current_time: 0,
            config: &config,
            timezone_name: None,
            timezone_offset: 0,
        };

        // Proceeds 1 yuan vs >5 yuan of fees (min_commission dominates).
        let sell = create_order("600000", OrderSide::Sell, dec!(100), dec!(0.01));
        assert!(CashMarginRule.check(&sell, &ctx).is_err());
    }
}
