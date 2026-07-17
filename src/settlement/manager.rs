use crate::account::{calculate_account_metrics, estimate_futures_realized_pnl};
use crate::analysis::TradeTracker;
use crate::event::Event;
use crate::market::manager::MarketManager;
use crate::model::{
    Instrument, Order, OrderRole, OrderSide, OrderStatus, OrderType, PositionEffect, TimeInForce,
    Trade,
};
use crate::portfolio::Portfolio;
use crate::risk::RiskConfig;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use super::expiry::ExpirySettlementHandler;
use super::handler::SettlementHandler;
use super::option::OptionSettlementHandler;

/// Settlement Context
pub struct SettlementContext<'a> {
    pub date: NaiveDate,
    pub instruments: &'a HashMap<String, Instrument>,
    pub last_prices: &'a HashMap<String, Decimal>,
    pub market_manager: &'a MarketManager,
    pub trade_tracker: &'a TradeTracker,
    pub risk_config: &'a RiskConfig,
    pub timestamp: i64,
    pub bar_index: usize,
    pub default_strategy_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ExecutedExpiryEvent {
    pub symbol: String,
    pub asset_type: crate::model::types::AssetType,
    pub trading_date: NaiveDate,
    pub expiry_date: Option<u32>,
    pub quantity_before: Decimal,
    pub quantity_closed: Decimal,
    pub cash_flow: Decimal,
    pub settlement_type: Option<String>,
    pub settlement_price: Option<Decimal>,
    pub reason: String,
    pub description: String,
}

#[derive(Debug, Clone, Default)]
pub struct SettlementOutcome {
    pub daily_interest: Decimal,
    pub forced_liquidation: bool,
    pub liquidated_symbols: Vec<String>,
    pub expiry_events: Vec<ExecutedExpiryEvent>,
    pub forced_liquidation_events: Vec<Event>,
}

/// Settlement Manager
/// Centralizes daily settlement logic including T+1 settlement, option expiry, and order expiration.
pub struct SettlementManager {
    option_handler: OptionSettlementHandler,
    expiry_handler: ExpirySettlementHandler,
}

impl SettlementManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            option_handler: OptionSettlementHandler,
            expiry_handler: ExpirySettlementHandler,
        }
    }

    /// Process daily settlement routine
    /// 1. T+1 Settlement (`MarketManager`)
    /// 2. Option Expiry
    /// 3. Order Expiration (Day orders)
    pub fn process_daily_settlement(
        &self,
        portfolio: &mut Portfolio,
        active_orders: &mut Vec<Order>,
        expired_orders_out: &mut Vec<Order>,
        ctx: &SettlementContext,
    ) -> SettlementOutcome {
        let outcome = self.process_margin_interest_and_liquidation(portfolio, ctx);

        // 1. Market Settlement (T+1 logic)
        // Note: MarketManager::on_day_close handles T+1 by moving positions from 'positions' to 'available_positions'
        // But wait, `portfolio.positions` is T+0/Total, `available_positions` is Sellable.
        // `on_day_close` updates `available_positions`.
        ctx.market_manager.on_day_close(
            &portfolio.positions,
            Arc::make_mut(&mut portfolio.available_positions),
            ctx.instruments,
        );

        // 2. Option Expiry
        let mut tasks = self.option_handler.check_settlement(
            ctx.date,
            portfolio,
            ctx.instruments,
            ctx.last_prices,
        );
        // 3. Futures/Stock Expiry
        tasks.extend(self.expiry_handler.check_settlement(
            ctx.date,
            portfolio,
            ctx.instruments,
            ctx.last_prices,
        ));

        let mut expiry_events = Vec::new();
        for task in tasks {
            let quantity_before = portfolio
                .positions
                .get(&task.symbol)
                .copied()
                .unwrap_or(Decimal::ZERO);
            // Execute settlement task
            // 1. Adjust Cash
            if !task.cash_flow.is_zero() {
                portfolio.cash += task.cash_flow;
            }

            // 2. Close Position
            // Remove from total positions
            {
                let positions = Arc::make_mut(&mut portfolio.positions);
                if let Some(qty) = positions.get_mut(&task.symbol) {
                    *qty -= task.quantity;
                    if qty.is_zero() {
                        positions.remove(&task.symbol);
                    }
                }
            }

            // Remove from available positions
            {
                let avail_pos = Arc::make_mut(&mut portfolio.available_positions);
                if let Some(qty) = avail_pos.get_mut(&task.symbol) {
                    *qty -= task.quantity;
                    if qty.is_zero() {
                        avail_pos.remove(&task.symbol);
                    }
                }
            }

            expiry_events.push(ExecutedExpiryEvent {
                symbol: task.symbol.clone(),
                asset_type: task.asset_type,
                trading_date: ctx.date,
                expiry_date: task.expiry_date,
                quantity_before,
                quantity_closed: task.quantity,
                cash_flow: task.cash_flow,
                settlement_type: task.settlement_type.clone(),
                settlement_price: task.settlement_price,
                reason: task.reason.clone(),
                description: task.description.clone(),
            });
        }

        // 4. Order Expiration (Day Orders)
        // Partition orders into expired and kept
        let (expired, kept): (Vec<Order>, Vec<Order>) = active_orders
            .drain(..)
            .partition(|o| o.time_in_force == TimeInForce::Day);

        *active_orders = kept;

        for mut o in expired {
            o.status = OrderStatus::Expired;
            expired_orders_out.push(o);
        }
        SettlementOutcome {
            expiry_events,
            ..outcome
        }
    }

    fn process_margin_interest_and_liquidation(
        &self,
        portfolio: &mut Portfolio,
        ctx: &SettlementContext,
    ) -> SettlementOutcome {
        let mut outcome = SettlementOutcome::default();
        if !ctx.risk_config.is_margin_account() {
            return outcome;
        }

        let borrowed_cash = (-portfolio.cash).max(Decimal::ZERO);
        let pre_interest_metrics = calculate_account_metrics(
            portfolio,
            ctx.last_prices,
            ctx.instruments,
            ctx.trade_tracker,
            ctx.risk_config,
        );
        let financing_rate =
            Decimal::from_f64(ctx.risk_config.financing_rate_annual).unwrap_or(Decimal::ZERO);
        let borrow_rate =
            Decimal::from_f64(ctx.risk_config.borrow_rate_annual).unwrap_or(Decimal::ZERO);
        let day_divisor = Decimal::from(365);
        let financing_interest = borrowed_cash * financing_rate / day_divisor;
        let borrow_interest = pre_interest_metrics.short_market_value * borrow_rate / day_divisor;
        let total_interest = (financing_interest + borrow_interest).max(Decimal::ZERO);
        if total_interest > Decimal::ZERO {
            portfolio.cash -= total_interest;
            outcome.daily_interest = total_interest;
        }

        if !ctx.risk_config.allow_force_liquidation {
            return outcome;
        }

        let required_maintenance = ctx.risk_config.maintenance_margin_ratio_decimal();
        let mut simulated_portfolio = portfolio.clone();
        let mut simulated_trade_tracker = ctx.trade_tracker.clone();
        let account_metrics = calculate_account_metrics(
            &simulated_portfolio,
            ctx.last_prices,
            ctx.instruments,
            &simulated_trade_tracker,
            ctx.risk_config,
        );
        if account_metrics.used_margin <= Decimal::ZERO {
            return outcome;
        }
        if account_metrics.maintenance_ratio >= required_maintenance {
            return outcome;
        }

        let mut symbols_to_close: Vec<(String, Decimal)> = portfolio
            .positions
            .iter()
            .filter_map(|(symbol, qty)| {
                if qty.is_zero() {
                    return None;
                }
                let price = ctx.last_prices.get(symbol).copied()?;
                if price <= Decimal::ZERO {
                    return None;
                }
                let multiplier = ctx
                    .instruments
                    .get(symbol)
                    .map(|i| i.multiplier())
                    .unwrap_or(Decimal::ONE);
                Some((symbol.clone(), qty.abs() * price * multiplier))
            })
            .collect();
        let short_first = ctx.risk_config.liquidation_short_first();
        symbols_to_close.sort_by(
            |(left_symbol, left_exposure), (right_symbol, right_exposure)| {
                let left_qty = portfolio
                    .positions
                    .get(left_symbol)
                    .copied()
                    .unwrap_or(Decimal::ZERO);
                let right_qty = portfolio
                    .positions
                    .get(right_symbol)
                    .copied()
                    .unwrap_or(Decimal::ZERO);
                let left_rank = if short_first {
                    if left_qty < Decimal::ZERO { 0 } else { 1 }
                } else if left_qty > Decimal::ZERO {
                    0
                } else {
                    1
                };
                let right_rank = if short_first {
                    if right_qty < Decimal::ZERO { 0 } else { 1 }
                } else if right_qty > Decimal::ZERO {
                    0
                } else {
                    1
                };
                left_rank
                    .cmp(&right_rank)
                    .then_with(|| right_exposure.cmp(left_exposure))
            },
        );

        let symbols_to_close: Vec<String> = symbols_to_close
            .into_iter()
            .map(|(symbol, _)| symbol)
            .collect();
        for symbol in symbols_to_close {
            let qty = simulated_portfolio
                .positions
                .get(&symbol)
                .copied()
                .unwrap_or(Decimal::ZERO);
            if qty.is_zero() {
                continue;
            }
            let Some(price) = ctx.last_prices.get(&symbol).copied() else {
                continue;
            };
            let side = if qty > Decimal::ZERO {
                OrderSide::Sell
            } else {
                OrderSide::Buy
            };
            let quantity = qty.abs();
            let order_id = Uuid::new_v4().to_string();
            let trade_id = Uuid::new_v4().to_string();
            let owner_strategy_id = normalized_strategy_id(ctx.default_strategy_id.clone());
            let mut filled_order = forced_liquidation_order(
                order_id.clone(),
                symbol.clone(),
                side,
                quantity,
                price,
                ctx.timestamp,
                owner_strategy_id.clone(),
            );
            filled_order.status = OrderStatus::Filled;
            filled_order.filled_quantity = quantity;
            filled_order.average_filled_price = Some(price);
            filled_order.updated_at = ctx.timestamp;
            let trade = Trade {
                id: trade_id,
                order_id,
                symbol: symbol.clone(),
                side,
                position_effect: PositionEffect::Close,
                quantity,
                price,
                commission: Decimal::ZERO,
                timestamp: ctx.timestamp,
                bar_index: ctx.bar_index,
                owner_strategy_id,
            };
            outcome
                .forced_liquidation_events
                .push(Event::ExecutionReport(filled_order, Some(trade.clone())));

            apply_liquidation_trade_simulation(
                &mut simulated_portfolio,
                &mut simulated_trade_tracker,
                &trade,
                ctx.instruments,
                ctx.last_prices,
                ctx.risk_config,
            );
            outcome.liquidated_symbols.push(symbol.clone());

            let next_metrics = calculate_account_metrics(
                &simulated_portfolio,
                ctx.last_prices,
                ctx.instruments,
                &simulated_trade_tracker,
                ctx.risk_config,
            );
            if next_metrics.used_margin <= Decimal::ZERO {
                break;
            }
            if next_metrics.maintenance_ratio >= required_maintenance {
                break;
            }
        }
        outcome.forced_liquidation = !outcome.forced_liquidation_events.is_empty();
        outcome
    }
}

fn normalized_strategy_id(value: Option<String>) -> Option<String> {
    value
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

fn forced_liquidation_order(
    id: String,
    symbol: String,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    timestamp: i64,
    owner_strategy_id: Option<String>,
) -> Order {
    Order {
        id,
        symbol,
        side,
        order_type: OrderType::Market,
        quantity,
        price: Some(price),
        time_in_force: TimeInForce::Day,
        trigger_price: None,
        trail_offset: None,
        trail_reference_price: None,
        fill_policy_override: None,
        slippage_type_override: None,
        slippage_value_override: None,
        commission_type_override: None,
        commission_value_override: None,
        graph_id: None,
        parent_order_id: None,
        order_role: OrderRole::Standalone,
        position_effect: PositionEffect::Close,
        status: OrderStatus::New,
        filled_quantity: Decimal::ZERO,
        average_filled_price: None,
        created_at: timestamp,
        updated_at: timestamp,
        commission: Decimal::ZERO,
        tag: "__forced_liquidation__".to_string(),
        reject_reason: String::new(),
        owner_strategy_id,
        allow_quantity_auto_resize: false,
        reduce_only: true,
    }
}

fn apply_liquidation_trade_simulation(
    portfolio: &mut Portfolio,
    trade_tracker: &mut TradeTracker,
    trade: &Trade,
    instruments: &HashMap<String, Instrument>,
    prices: &HashMap<String, Decimal>,
    risk_config: &RiskConfig,
) {
    let Some(instr) = instruments.get(&trade.symbol) else {
        return;
    };
    let multiplier = instr.multiplier();

    if crate::account::is_futures_margin_account(instr, risk_config) {
        let realized = estimate_futures_realized_pnl(
            trade_tracker,
            &trade.symbol,
            trade.side,
            trade.quantity,
            trade.price,
            multiplier,
        );
        portfolio.adjust_cash(realized);
        if trade.side == OrderSide::Buy {
            portfolio.adjust_position(&trade.symbol, trade.quantity);
        } else {
            portfolio.adjust_position(&trade.symbol, -trade.quantity);
        }
    } else {
        let notional = trade.price * trade.quantity * multiplier;
        if trade.side == OrderSide::Buy {
            portfolio.adjust_cash(-notional);
            portfolio.adjust_position(&trade.symbol, trade.quantity);
        } else {
            portfolio.adjust_cash(notional);
            portfolio.adjust_position(&trade.symbol, -trade.quantity);
        }
    }

    let portfolio_value =
        calculate_account_metrics(portfolio, prices, instruments, trade_tracker, risk_config)
            .equity;
    trade_tracker.process_trade(
        trade,
        multiplier,
        Some("__forced_liquidation__"),
        None,
        portfolio_value,
    );
}

impl Default for SettlementManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::TradeTracker;
    use crate::model::instrument::{InstrumentEnum, StockInstrument};
    use crate::model::types::AssetType;
    use std::str::FromStr;

    fn stock_instrument(symbol: &str) -> Instrument {
        Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.to_string(),
                lot_size: Decimal::ONE,
                tick_size: Decimal::new(1, 2),
                expiry_date: None,
                sellable_after_days: 1,
            }),
        }
    }

    #[test]
    fn test_margin_interest_is_deducted_daily() {
        let mut portfolio = Portfolio {
            cash: Decimal::from(-5000),
            positions: Arc::new(HashMap::new()),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut instruments = HashMap::new();
        instruments.insert("AAA".to_string(), stock_instrument("AAA"));
        let mut prices = HashMap::new();
        prices.insert("AAA".to_string(), Decimal::from(100));
        let mut risk = RiskConfig::new();
        risk.account_mode = "margin".to_string();
        risk.financing_rate_annual = 36.5;
        risk.borrow_rate_annual = 0.0;
        risk.allow_force_liquidation = false;
        let market_manager = MarketManager::new();
        let tracker = TradeTracker::new();

        let ctx = SettlementContext {
            date: NaiveDate::from_ymd_opt(2024, 1, 2).expect("valid date"),
            instruments: &instruments,
            last_prices: &prices,
            market_manager: &market_manager,
            trade_tracker: &tracker,
            risk_config: &risk,
            timestamp: 0,
            bar_index: 0,
            default_strategy_id: None,
        };
        let outcome = SettlementManager::new().process_daily_settlement(
            &mut portfolio,
            &mut Vec::new(),
            &mut Vec::new(),
            &ctx,
        );
        assert_eq!(portfolio.cash, Decimal::from(-5500));
        assert_eq!(outcome.daily_interest, Decimal::from(500));
        assert!(!outcome.forced_liquidation);
    }

    #[test]
    fn test_margin_liquidation_closes_positions_when_maintenance_breached() {
        let mut pos = HashMap::new();
        pos.insert(
            "AAA".to_string(),
            Decimal::from_str("150").expect("decimal"),
        );
        let mut portfolio = Portfolio {
            cash: Decimal::from(-5000),
            positions: Arc::new(pos),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut instruments = HashMap::new();
        instruments.insert("AAA".to_string(), stock_instrument("AAA"));
        let mut prices = HashMap::new();
        prices.insert("AAA".to_string(), Decimal::from(20));
        let mut risk = RiskConfig::new();
        risk.account_mode = "margin".to_string();
        risk.financing_rate_annual = 0.0;
        risk.borrow_rate_annual = 0.0;
        risk.allow_force_liquidation = true;
        risk.maintenance_margin_ratio = 0.5;
        let market_manager = MarketManager::new();
        let tracker = TradeTracker::new();

        let ctx = SettlementContext {
            date: NaiveDate::from_ymd_opt(2024, 1, 2).expect("valid date"),
            instruments: &instruments,
            last_prices: &prices,
            market_manager: &market_manager,
            trade_tracker: &tracker,
            risk_config: &risk,
            timestamp: 0,
            bar_index: 0,
            default_strategy_id: None,
        };
        let outcome = SettlementManager::new().process_daily_settlement(
            &mut portfolio,
            &mut Vec::new(),
            &mut Vec::new(),
            &ctx,
        );
        assert!(outcome.forced_liquidation);
        assert_eq!(outcome.liquidated_symbols, vec!["AAA".to_string()]);
        assert_eq!(outcome.forced_liquidation_events.len(), 1);
    }

    #[test]
    fn test_liquidation_priority_controls_close_order() {
        let mut positions = HashMap::new();
        positions.insert("LONG".to_string(), Decimal::from(100));
        positions.insert("SHORT".to_string(), Decimal::from(-50));

        let mut instruments = HashMap::new();
        instruments.insert("LONG".to_string(), stock_instrument("LONG"));
        instruments.insert("SHORT".to_string(), stock_instrument("SHORT"));

        let mut prices = HashMap::new();
        prices.insert("LONG".to_string(), Decimal::from(100));
        prices.insert("SHORT".to_string(), Decimal::from(100));

        let mut risk_short_first = RiskConfig::new();
        risk_short_first.account_mode = "margin".to_string();
        risk_short_first.allow_force_liquidation = true;
        risk_short_first.financing_rate_annual = 0.0;
        risk_short_first.borrow_rate_annual = 0.0;
        risk_short_first.maintenance_margin_ratio = 0.5;
        risk_short_first.liquidation_priority = "short_first".to_string();

        let mut risk_long_first = risk_short_first.clone();
        risk_long_first.liquidation_priority = "long_first".to_string();

        let market_manager = MarketManager::new();
        let date = NaiveDate::from_ymd_opt(2024, 1, 2).expect("valid date");
        let tracker_short = TradeTracker::new();
        let tracker_long = TradeTracker::new();

        let mut portfolio_short_first = Portfolio {
            cash: Decimal::from(-1000),
            positions: Arc::new(positions.clone()),
            available_positions: Arc::new(HashMap::new()),
        };
        let ctx_short_first = SettlementContext {
            date,
            instruments: &instruments,
            last_prices: &prices,
            market_manager: &market_manager,
            trade_tracker: &tracker_short,
            risk_config: &risk_short_first,
            timestamp: 0,
            bar_index: 0,
            default_strategy_id: None,
        };
        let outcome_short_first = SettlementManager::new().process_daily_settlement(
            &mut portfolio_short_first,
            &mut Vec::new(),
            &mut Vec::new(),
            &ctx_short_first,
        );
        assert!(outcome_short_first.forced_liquidation);
        assert_eq!(
            outcome_short_first.liquidated_symbols.first(),
            Some(&"SHORT".to_string())
        );

        let mut portfolio_long_first = Portfolio {
            cash: Decimal::from(-1000),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };
        let ctx_long_first = SettlementContext {
            date,
            instruments: &instruments,
            last_prices: &prices,
            market_manager: &market_manager,
            trade_tracker: &tracker_long,
            risk_config: &risk_long_first,
            timestamp: 0,
            bar_index: 0,
            default_strategy_id: None,
        };
        let outcome_long_first = SettlementManager::new().process_daily_settlement(
            &mut portfolio_long_first,
            &mut Vec::new(),
            &mut Vec::new(),
            &ctx_long_first,
        );
        assert!(outcome_long_first.forced_liquidation);
        assert_eq!(
            outcome_long_first.liquidated_symbols.first(),
            Some(&"LONG".to_string())
        );
    }
}
