use crate::error::AkQuantError;
use crate::model::{AssetType, Instrument, Order};
use crate::portfolio::Portfolio;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

use super::common::{
    CashMarginRule, MaxOrderSizeRule, MaxOrderValueRule, MaxPositionSizeRule, RestrictedListRule,
};
use super::config::RiskConfig;
use super::futures::FuturesMarginRule;
use super::option::OptionGreekRiskRule;
use super::portfolio::{
    MaxDailyLossRule, MaxDrawdownRule, MaxLeverageRule, MaxPositionPercentRule,
    SectorConcentrationRule, StopLossRule,
};
use super::rule::RiskRule;
use super::stock::StockAvailablePositionRule;

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct RiskManager {
    #[pyo3(get, set)]
    pub config: RiskConfig,

    // Internal fields, not exposed to Python directly (unless we add getters)
    // No #[pyo3(skip)] needed as fields are private by default in #[pyclass]
    common_rules: Vec<Box<dyn RiskRule>>,
    asset_rules: HashMap<AssetType, Vec<Box<dyn RiskRule>>>,
    dynamic_rules: Vec<Box<dyn RiskRule>>,
}

impl Default for RiskManager {
    fn default() -> Self {
        let mut manager = Self {
            config: RiskConfig::new(),
            common_rules: Vec::new(),
            asset_rules: HashMap::new(),
            dynamic_rules: Vec::new(),
        };
        manager.init_rules();
        manager
    }
}

#[pymethods]
impl RiskManager {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn check(
        &self,
        order: &Order,
        portfolio: &Portfolio,
        instruments: HashMap<String, Instrument>,
        active_orders: Vec<Order>,
        current_prices: Option<HashMap<String, f64>>,
    ) -> Option<String> {
        let prices_dec: HashMap<String, Decimal> = if let Some(cp) = current_prices {
            cp.into_iter()
                .map(|(k, v)| (k, Decimal::from_f64(v).unwrap_or(Decimal::ZERO)))
                .collect()
        } else {
            HashMap::new()
        };

        // Create a dummy market model for context
        use crate::market::{SimpleMarket, SimpleMarketConfig};
        let market_model = SimpleMarket::from_config(SimpleMarketConfig::default());
        let trade_tracker = crate::analysis::TradeTracker::new();

        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio,
            last_prices: &prices_dec,
            trade_tracker: &trade_tracker,
            market_model: &market_model,
            execution_policy_core: crate::model::ExecutionPolicyCore::default(),
            bar_index: 0,
            current_time: 0,
            session: crate::model::TradingSession::Continuous,
            active_orders: &active_orders,
            risk_config: &self.config,
            timezone_name: None,
            timezone_offset: 0,
        };

        match self.check_internal(order, &ctx) {
            Ok(_) => None,
            Err(e) => Some(e.to_string()),
        }
    }

    /// Add max position percentage rule (0.1 = 10%)
    pub fn add_max_position_percent_rule(&mut self, max_pct: f64) {
        self.dynamic_rules.push(Box::new(MaxPositionPercentRule {
            max_pct: Decimal::from_f64(max_pct).unwrap_or(Decimal::ZERO),
        }));
    }

    /// Add max leverage rule (e.g. 1.5 = 150%)
    pub fn add_max_leverage_rule(&mut self, max_leverage: f64) {
        self.dynamic_rules.push(Box::new(MaxLeverageRule {
            max_leverage: Decimal::from_f64(max_leverage).unwrap_or(Decimal::ZERO),
        }));
    }

    /// Add sector concentration rule
    pub fn add_sector_concentration_rule(
        &mut self,
        max_pct: f64,
        sector_map: HashMap<String, String>,
    ) {
        self.dynamic_rules.push(Box::new(SectorConcentrationRule {
            max_pct: Decimal::from_f64(max_pct).unwrap_or(Decimal::ZERO),
            sector_map,
        }));
    }

    /// Add max drawdown rule (0.2 = max 20% drawdown)
    pub fn add_max_drawdown_rule(&mut self, limit: f64) {
        if !(0.0..=1.0).contains(&limit) {
            return;
        }
        self.dynamic_rules.push(Box::new(MaxDrawdownRule::new(
            Decimal::from_f64(limit).unwrap_or(Decimal::ZERO),
        )));
    }

    /// Add max daily loss rule (0.05 = max 5% daily loss)
    pub fn add_max_daily_loss_rule(&mut self, limit: f64) {
        if !(0.0..=1.0).contains(&limit) {
            return;
        }
        self.dynamic_rules.push(Box::new(MaxDailyLossRule::new(
            Decimal::from_f64(limit).unwrap_or(Decimal::ZERO),
        )));
    }

    /// Add stop loss rule (0.8 = stop if equity < initial_equity * 0.8)
    pub fn add_stop_loss_rule(&mut self, threshold: f64) {
        if !(0.0..=1.0).contains(&threshold) {
            return;
        }
        self.dynamic_rules.push(Box::new(StopLossRule::new(
            Decimal::from_f64(threshold).unwrap_or(Decimal::ZERO),
        )));
    }
}

impl RiskManager {
    fn init_rules(&mut self) {
        // Common rules
        self.common_rules.push(Box::new(RestrictedListRule));
        self.common_rules.push(Box::new(MaxOrderSizeRule));
        self.common_rules.push(Box::new(MaxOrderValueRule));
        self.common_rules.push(Box::new(MaxPositionSizeRule));
        self.common_rules.push(Box::new(CashMarginRule));

        // Stock rules
        self.asset_rules
            .entry(AssetType::Stock)
            .or_default()
            .push(Box::new(StockAvailablePositionRule));
        self.asset_rules
            .entry(AssetType::Fund)
            .or_default()
            .push(Box::new(StockAvailablePositionRule));

        // Futures rules. FuturesMarginRule is REQUIRED here: CashMarginRule
        // unconditionally steps aside for futures in a margin account (see
        // risk::common::CashMarginRule::check) and hands the margin check to this
        // rule. Removing it would silently drop all submission-time margin checks
        // for futures margin accounts. Guarded by the test
        // `risk_rules_cover_futures_margin_account`.
        self.asset_rules
            .entry(AssetType::Futures)
            .or_default()
            .push(Box::new(FuturesMarginRule));

        // Option rules
        self.asset_rules
            .entry(AssetType::Option)
            .or_default()
            .push(Box::new(OptionGreekRiskRule));
    }

    fn should_defer_available_position_check(
        order: &Order,
        ctx: &crate::context::EngineContext,
    ) -> bool {
        if order.side != crate::model::OrderSide::Sell {
            return false;
        }

        let Some(instrument) = ctx.instruments.get(&order.symbol) else {
            return false;
        };
        if !matches!(instrument.asset_type, AssetType::Stock | AssetType::Fund) {
            return false;
        }

        // The order will not be matched on its submission timestamp when the
        // policy uses a delayed bar offset. Its available-position check must
        // therefore use the portfolio state at the eventual execution event,
        // after any T+1 settlement has run.
        order
            .fill_policy_override
            .unwrap_or(ctx.execution_policy_core)
            .bar_offset
            >= 1
    }

    pub fn check_and_adjust(
        &self,
        order: &mut Order,
        ctx: &crate::context::EngineContext,
    ) -> Result<(), AkQuantError> {
        self.check_and_adjust_with_delayed_position_check(order, ctx, false)
    }

    pub(crate) fn check_and_adjust_with_delayed_position_check(
        &self,
        order: &mut Order,
        ctx: &crate::context::EngineContext,
        allow_deferred_position_check: bool,
    ) -> Result<(), AkQuantError> {
        let defer_available_position_check = allow_deferred_position_check
            && Self::should_defer_available_position_check(order, ctx);

        // 1. Initial Check
        if let Err(err) = self.check_internal_with_deferred_position_check(
            order,
            ctx,
            defer_available_position_check,
        ) {
            let err_msg = err.to_string();
            // Check for insufficient cash/margin to attempt auto-reduction
            // This logic was moved from OrderManager.
            //
            // NOTE on the `side == Buy` guard: submission-time auto-resize is
            // buy-only. The execution-time gate (execution::simulated) resizes
            // either side, so there is a narrow, currently-UNREACHABLE asymmetry:
            // a short-opening sell that is affordable at the submission price but
            // not at the real fill price would be resized at execution, whereas an
            // unaffordable-at-submission short sell is rejected here rather than
            // resized. It is unreachable because the public API (`ctx.sell`)
            // hardcodes `allow_quantity_auto_resize = false` for sells — only
            // directly-constructed orders (e.g. tests) can trigger the execution
            // path. Aligning the two would also require deciding whether resizing
            // a *reducing/closing* sell is ever correct (it is not — a close
            // should always be allowed even below the commission budget), which is
            // a separate concern from cash/margin gating. Left buy-only pending
            // that design decision; do not "fix" by simply dropping this guard.
            if order.allow_quantity_auto_resize
                && (err_msg.contains("Insufficient cash")
                    || err_msg.contains("Insufficient margin"))
                && order.side == crate::model::OrderSide::Buy
                && let Some(instr) = ctx.instruments.get(&order.symbol)
            {
                // Get price (Limit or Last)
                let price = if let Some(p) = order.price {
                    p
                } else {
                    *ctx.last_prices.get(&order.symbol).unwrap_or(&Decimal::ZERO)
                };

                if price > Decimal::ZERO {
                    // Resize through the unified affordability check so the max
                    // quantity is derived from free margin (not raw cash) with the
                    // same safety/commission convention as the gate that rejected.
                    let _ = instr;
                    let safety_margin =
                        Decimal::from_f64(self.config.safety_margin).unwrap_or(Decimal::ZERO);
                    let result = super::common::check_affordability(
                        order,
                        ctx.portfolio,
                        ctx.last_prices,
                        ctx.instruments,
                        &self.config,
                        ctx.market_model,
                        ctx.active_orders,
                        price,
                        safety_margin,
                    )?;

                    let new_qty = result.max_affordable_qty;
                    if new_qty > Decimal::ZERO && new_qty < order.quantity {
                        order.quantity = new_qty;
                        // Re-check with new quantity
                        return self.check_internal_with_deferred_position_check(
                            order,
                            ctx,
                            defer_available_position_check,
                        );
                    }
                }
            }
            return Err(err);
        }
        Ok(())
    }

    pub fn check_internal(
        &self,
        order: &Order,
        ctx: &crate::context::EngineContext,
    ) -> Result<(), AkQuantError> {
        self.check_internal_with_deferred_position_check(order, ctx, false)
    }

    fn check_internal_with_deferred_position_check(
        &self,
        order: &Order,
        ctx: &crate::context::EngineContext,
        defer_available_position_check: bool,
    ) -> Result<(), AkQuantError> {
        if !self.config.active {
            return Ok(());
        }

        let instrument = ctx.instruments.get(&order.symbol).ok_or_else(|| {
            AkQuantError::OrderError(format!("Instrument not found for {}", order.symbol))
        })?;

        let risk_ctx = crate::risk::rule::RiskCheckContext {
            portfolio: ctx.portfolio,
            instrument,
            instruments: ctx.instruments,
            active_orders: ctx.active_orders,
            current_prices: ctx.last_prices,
            trade_tracker: ctx.trade_tracker,
            market_model: ctx.market_model,
            current_time: ctx.current_time,
            config: &self.config,
            timezone_name: ctx.timezone_name,
            timezone_offset: ctx.timezone_offset,
        };

        // Check common rules
        for rule in &self.common_rules {
            rule.check(order, &risk_ctx)?;
        }

        // Check asset-specific rules
        if let Some(rules) = self.asset_rules.get(&instrument.asset_type) {
            for rule in rules {
                if defer_available_position_check && rule.defer_for_delayed_execution() {
                    continue;
                }
                rule.check(order, &risk_ctx)?;
            }
        }

        // Check dynamic rules
        for rule in &self.dynamic_rules {
            rule.check(order, &risk_ctx)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::{SimpleMarket, SimpleMarketConfig};
    use crate::model::instrument::{InstrumentEnum, StockInstrument};
    use crate::model::{ExecutionPolicyCore, OrderSide, OrderType, TradingSession};
    use std::sync::Arc;

    #[test]
    fn risk_rules_cover_futures_margin_account() {
        // CashMarginRule delegates futures margin accounts to FuturesMarginRule
        // (see risk::common::CashMarginRule::check). This asserts the delegate is
        // actually registered, so the hand-off never leaves futures margin
        // accounts with no submission-time margin check.
        let manager = RiskManager::new();
        let futures_rules = manager
            .asset_rules
            .get(&AssetType::Futures)
            .expect("Futures asset rules must be registered");
        assert!(
            futures_rules
                .iter()
                .any(|rule| rule.name() == "FuturesMarginRule"),
            "FuturesMarginRule must be registered for AssetType::Futures"
        );
    }

    #[test]
    fn public_check_and_adjust_rejects_unavailable_delayed_stock() {
        let symbol = "AAPL".to_string();
        let instrument = Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.clone(),
                lot_size: Decimal::from(100),
                tick_size: Decimal::new(1, 2),
                expiry_date: None,
                sellable_after_days: 1,
            }),
        };
        let instruments = HashMap::from([(symbol.clone(), instrument)]);
        let portfolio = Portfolio {
            cash: Decimal::from(100_000),
            positions: Arc::new(HashMap::from([(symbol.clone(), Decimal::from(100))])),
            available_positions: Arc::new(HashMap::new()),
        };
        let market_model = SimpleMarket::from_config(SimpleMarketConfig::default());
        let trade_tracker = crate::analysis::TradeTracker::new();
        let manager = RiskManager::new();
        let mut order = Order::test_new(
            "sell-unavailable",
            &symbol,
            OrderSide::Sell,
            OrderType::Limit,
            Decimal::from(100),
        );
        order.price = Some(Decimal::from(10));
        order.fill_policy_override = Some(ExecutionPolicyCore::default());
        let prices = HashMap::from([(symbol.clone(), Decimal::from(10))]);
        let ctx = crate::context::EngineContext {
            instruments: &instruments,
            portfolio: &portfolio,
            last_prices: &prices,
            trade_tracker: &trade_tracker,
            market_model: &market_model,
            execution_policy_core: ExecutionPolicyCore::default(),
            bar_index: 0,
            current_time: 0,
            session: TradingSession::Continuous,
            active_orders: &[],
            risk_config: &manager.config,
            timezone_name: None,
            timezone_offset: 0,
        };

        let err = manager
            .check_and_adjust(&mut order, &ctx)
            .expect_err("public check must retain submission-time availability checks");
        assert!(err.to_string().contains("Insufficient available position"));
    }
}
