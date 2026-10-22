use crate::analysis::tracker::TradeTracker;
use crate::log_context::{AkqLogContext, render_log_message};
use crate::model::corporate_action::{CorporateAction, CorporateActionType};
use crate::portfolio::Portfolio;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;

const CORP_ACTION_ALERT_PREFIX: &str = "AKQ-CORP-ACTION-ALERT";
const MIN_SPLIT_RATIO_SCALE: i64 = 1;
const MIN_SPLIT_RATIO_DP: u32 = 4;
const MAX_SPLIT_RATIO_SCALE: i64 = 10000;
const MAX_SPLIT_RATIO_DP: u32 = 0;

fn checked_mul_or_cap(lhs: Decimal, rhs: Decimal) -> Decimal {
    lhs.checked_mul(rhs).unwrap_or(Decimal::MAX)
}

fn checked_add_or_cap(lhs: Decimal, rhs: Decimal) -> Decimal {
    lhs.checked_add(rhs).unwrap_or(Decimal::MAX)
}

fn split_ratio_bounds() -> (Decimal, Decimal) {
    (
        Decimal::new(MIN_SPLIT_RATIO_SCALE, MIN_SPLIT_RATIO_DP),
        Decimal::new(MAX_SPLIT_RATIO_SCALE, MAX_SPLIT_RATIO_DP),
    )
}

pub struct CorporateActionManager {
    actions: HashMap<NaiveDate, Vec<CorporateAction>>,
}

impl Default for CorporateActionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CorporateActionManager {
    pub fn new() -> Self {
        Self {
            actions: HashMap::new(),
        }
    }

    pub fn add(&mut self, action: CorporateAction) {
        self.actions.entry(action.date).or_default().push(action);
    }

    /// 按日期升序处理日期落在 `[from, to]` 内的全部公司行动。
    pub fn process_range(
        &self,
        from: NaiveDate,
        to: NaiveDate,
        portfolio: &mut Portfolio,
        trade_tracker: &mut TradeTracker,
    ) {
        if from > to {
            return;
        }
        let mut dates: Vec<NaiveDate> = self
            .actions
            .keys()
            .copied()
            .filter(|d| *d >= from && *d <= to)
            .collect();
        dates.sort();
        for date in dates {
            self.process_date(date, portfolio, trade_tracker);
        }
    }

    pub fn process_date(
        &self,
        date: NaiveDate,
        portfolio: &mut Portfolio,
        trade_tracker: &mut TradeTracker,
    ) {
        if let Some(actions) = self.actions.get(&date) {
            for action in actions {
                match action.action_type {
                    CorporateActionType::Split => {
                        let ratio = action.value.abs();
                        let (min_ratio, max_ratio) = split_ratio_bounds();
                        if ratio <= Decimal::ZERO || ratio < min_ratio || ratio > max_ratio {
                            log::warn!(
                                "{}",
                                render_log_message(
                                    format!(
                                        "[{CORP_ACTION_ALERT_PREFIX}] skip split action for {} on {} due to abnormal ratio {}",
                                        action.symbol, action.date, action.value
                                    ),
                                    AkqLogContext::new()
                                        .phase("data")
                                        .symbol(action.symbol.clone()),
                                )
                            );
                            continue;
                        }

                        let positions = Arc::make_mut(&mut portfolio.positions);
                        if let Some(qty) = positions.get_mut(&action.symbol) {
                            *qty = checked_mul_or_cap(*qty, ratio);
                        }

                        let available = Arc::make_mut(&mut portfolio.available_positions);
                        if let Some(avail) = available.get_mut(&action.symbol) {
                            *avail = checked_mul_or_cap(*avail, ratio);
                        }

                        trade_tracker.on_split(&action.symbol, ratio);
                    }
                    CorporateActionType::Dividend => {
                        if let Some(qty) = portfolio.positions.get(&action.symbol)
                            && !qty.is_zero()
                        {
                            let dividend_amount = checked_mul_or_cap(*qty, action.value);
                            portfolio.cash = checked_add_or_cap(portfolio.cash, dividend_amount);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::corporate_action::CorporateAction;
    use chrono::NaiveDate;

    #[test]
    fn split_caps_position_on_overflow() {
        let mut manager = CorporateActionManager::new();
        manager.add(CorporateAction {
            symbol: "AAPL".to_string(),
            date: NaiveDate::from_ymd_opt(2025, 1, 1).expect("valid date"),
            action_type: CorporateActionType::Split,
            value: Decimal::MAX,
        });

        let mut portfolio = Portfolio {
            cash: Decimal::ZERO,
            positions: Arc::new(HashMap::from([("AAPL".to_string(), Decimal::MAX)])),
            available_positions: Arc::new(HashMap::from([("AAPL".to_string(), Decimal::MAX)])),
        };
        let mut tracker = TradeTracker::new();

        manager.process_date(
            NaiveDate::from_ymd_opt(2025, 1, 1).expect("valid date"),
            &mut portfolio,
            &mut tracker,
        );

        assert_eq!(portfolio.positions.get("AAPL"), Some(&Decimal::MAX));
        assert_eq!(
            portfolio.available_positions.get("AAPL"),
            Some(&Decimal::MAX)
        );
    }

    #[test]
    fn dividend_caps_cash_on_overflow() {
        let mut manager = CorporateActionManager::new();
        manager.add(CorporateAction {
            symbol: "AAPL".to_string(),
            date: NaiveDate::from_ymd_opt(2025, 1, 2).expect("valid date"),
            action_type: CorporateActionType::Dividend,
            value: Decimal::MAX,
        });

        let mut portfolio = Portfolio {
            cash: Decimal::MAX,
            positions: Arc::new(HashMap::from([("AAPL".to_string(), Decimal::MAX)])),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut tracker = TradeTracker::new();

        manager.process_date(
            NaiveDate::from_ymd_opt(2025, 1, 2).expect("valid date"),
            &mut portfolio,
            &mut tracker,
        );

        assert_eq!(portfolio.cash, Decimal::MAX);
    }

    #[test]
    fn split_out_of_range_ratio_is_skipped() {
        let mut manager = CorporateActionManager::new();
        manager.add(CorporateAction {
            symbol: "AAPL".to_string(),
            date: NaiveDate::from_ymd_opt(2025, 1, 3).expect("valid date"),
            action_type: CorporateActionType::Split,
            value: Decimal::new(1, 5),
        });

        let mut portfolio = Portfolio {
            cash: Decimal::ZERO,
            positions: Arc::new(HashMap::from([("AAPL".to_string(), Decimal::from(100))])),
            available_positions: Arc::new(HashMap::from([("AAPL".to_string(), Decimal::from(80))])),
        };
        let mut tracker = TradeTracker::new();

        manager.process_date(
            NaiveDate::from_ymd_opt(2025, 1, 3).expect("valid date"),
            &mut portfolio,
            &mut tracker,
        );

        assert_eq!(portfolio.positions.get("AAPL"), Some(&Decimal::from(100)));
        assert_eq!(
            portfolio.available_positions.get("AAPL"),
            Some(&Decimal::from(80))
        );
    }
}
