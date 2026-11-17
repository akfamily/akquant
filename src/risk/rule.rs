use crate::analysis::TradeTracker;
use crate::error::AkQuantError;
use crate::market::MarketModel;
use crate::model::{Instrument, Order};
use crate::portfolio::Portfolio;
use chrono::{NaiveDate, TimeZone, Utc};
use chrono_tz::Tz;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::fmt::Debug;

use super::RiskConfig;

/// Context for risk checks
pub struct RiskCheckContext<'a> {
    pub portfolio: &'a Portfolio,
    pub instrument: &'a Instrument,
    pub instruments: &'a HashMap<String, Instrument>,
    pub active_orders: &'a [Order],
    pub current_prices: &'a HashMap<String, Decimal>,
    pub trade_tracker: &'a TradeTracker,
    pub market_model: &'a dyn MarketModel,
    pub current_time: i64,
    pub config: &'a RiskConfig,
    pub timezone_name: Option<&'a str>,
    pub timezone_offset: i32,
    pub trading_day: Option<chrono::NaiveDate>,
}

impl<'a> RiskCheckContext<'a> {
    fn datetime_from_ns(timestamp: i64) -> chrono::DateTime<Utc> {
        let secs = timestamp.div_euclid(1_000_000_000);
        let nanos = timestamp.rem_euclid(1_000_000_000) as u32;
        Utc.timestamp_opt(secs, nanos)
            .single()
            .expect("Invalid timestamp")
    }

    /// 账户口径的权益(保证金账户里期货只计浮动盈亏), 见 [`crate::account::account_equity`]。
    pub fn account_equity(&self) -> Decimal {
        crate::account::account_equity(
            self.portfolio,
            self.current_prices,
            self.instruments,
            self.trade_tracker,
            self.config,
        )
    }

    /// 这笔委托是否只减仓: 方向与现有持仓相反, 且数量不超过持仓(超过会反手开仓)。
    ///
    /// 账户级规则(回撤、日亏损、止损线、杠杆、集中度)触发后只拒会增加风险的委托,
    /// 减仓单放行, 否则仓位平不掉。只看已成交持仓, 不计同标的其它在途减仓单。
    pub fn is_reducing_order(&self, order: &Order) -> bool {
        let position = self
            .portfolio
            .positions
            .get(&order.symbol)
            .copied()
            .unwrap_or(Decimal::ZERO);
        match order.side {
            crate::model::OrderSide::Buy => position < Decimal::ZERO && order.quantity <= -position,
            crate::model::OrderSide::Sell => position > Decimal::ZERO && order.quantity <= position,
        }
    }

    pub fn local_date(&self) -> NaiveDate {
        if let Some(d) = self.trading_day {
            return d;
        }
        let utc_dt = Self::datetime_from_ns(self.current_time);
        if let Some(tz) = self.timezone_name.and_then(|name| name.parse::<Tz>().ok()) {
            return utc_dt.with_timezone(&tz).date_naive();
        }
        let offset_ns = i64::from(self.timezone_offset) * 1_000_000_000;
        Self::datetime_from_ns(self.current_time + offset_ns).date_naive()
    }
}

/// Trait for risk check rules
pub trait RiskRule: Send + Sync + Debug {
    /// Check if the order passes the risk rule
    fn check(&self, order: &Order, ctx: &RiskCheckContext) -> Result<(), AkQuantError>;

    /// Whether this rule may be deferred from submission time to fill time.
    ///
    /// A delayed order (`NextOpen`, i.e. `bar_offset >= 1`) is submitted on one
    /// bar and matched on a later one, with daily settlement running in
    /// between. Rules whose verdict depends on post-settlement state would
    /// otherwise reject such an order against a stale snapshot. Opting in here
    /// skips the submission-time check; the simulated matcher re-checks the
    /// same condition against fill-time state (issue #391).
    fn defer_for_delayed_execution(&self) -> bool {
        false
    }

    /// Get the name of the rule
    fn name(&self) -> &'static str;

    /// Clone the rule
    fn clone_box(&self) -> Box<dyn RiskRule>;
}

impl Clone for Box<dyn RiskRule> {
    fn clone(&self) -> Box<dyn RiskRule> {
        self.clone_box()
    }
}
