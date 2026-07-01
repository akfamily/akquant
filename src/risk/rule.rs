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
}

impl<'a> RiskCheckContext<'a> {
    fn datetime_from_ns(timestamp: i64) -> chrono::DateTime<Utc> {
        let secs = timestamp.div_euclid(1_000_000_000);
        let nanos = timestamp.rem_euclid(1_000_000_000) as u32;
        Utc.timestamp_opt(secs, nanos)
            .single()
            .expect("Invalid timestamp")
    }

    pub fn local_date(&self) -> NaiveDate {
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
