use crate::model::Instrument;
use crate::model::types::{AssetType, OptionType};
use crate::portfolio::Portfolio;
use chrono::{Datelike, NaiveDate};
use rust_decimal::Decimal;
use std::collections::HashMap;

use super::handler::{SettlementHandler, SettlementTask};

/// Handles Option Expiration and Settlement
#[derive(Debug, Clone, Default)]
pub struct OptionSettlementHandler;

impl OptionSettlementHandler {
    /// 找出应到期结算的期权持仓。
    ///
    /// 返回 `(tasks, deferred)`: `deferred` 是本应到期结算、但既没有配置
    /// `settlement_price` 也拿不到标的任何价格而延后的 symbol。
    pub fn check_with_deferred(
        &self,
        date: NaiveDate,
        portfolio: &Portfolio,
        instruments: &HashMap<String, Instrument>,
        last_prices: &HashMap<String, Decimal>,
    ) -> (Vec<SettlementTask>, Vec<String>) {
        let mut tasks = Vec::new();
        let mut deferred = Vec::new();

        // Convert NaiveDate to YYYYMMDD u32 for comparison
        let (_, year_ce) = date.year_ce();
        let current_date_int = year_ce * 10000 + date.month() * 100 + date.day();

        for (symbol, qty) in portfolio.positions.iter() {
            if qty.is_zero() {
                continue;
            }

            let Some(instr) = instruments.get(symbol) else {
                continue;
            };
            if instr.asset_type != AssetType::Option {
                continue;
            }
            let Some(expiry_date_int) = instr.expiry_date() else {
                continue;
            };
            // 结算在"进入新交易日"时触发, 所以必须严格大于: 进入到期日当天就结算会用
            // 前一日收盘价, 且策略在到期日当天再也交易不到这张合约。
            if current_date_int <= expiry_date_int {
                continue;
            }

            let strike = instr.strike_price().unwrap_or(Decimal::ZERO);
            // 标的价格来源: 配置的到期结算价 > 标的最近已知价格(价格表不按天清空)。
            // 两者都没有时不能按 0 结算——那等于把实值期权静默作废; 延后到拿到价格为止。
            let underlying_price = instr.settlement_price().or_else(|| {
                instr
                    .underlying_symbol()
                    .and_then(|us| last_prices.get(us.as_str()).copied())
            });
            let Some(underlying_price) = underlying_price.filter(|p| *p > Decimal::ZERO) else {
                deferred.push(symbol.clone());
                continue;
            };
            let payoff_per_unit = match instr.option_type() {
                Some(OptionType::Call) => (underlying_price - strike).max(Decimal::ZERO),
                Some(OptionType::Put) => (strike - underlying_price).max(Decimal::ZERO),
                None => Decimal::ZERO,
            };

            // Total Cash Flow
            // Long (Qty > 0): Receives Payoff * Multiplier * Qty
            // Short (Qty < 0): Pays Payoff * Multiplier * Abs(Qty) -> Qty * Payoff * Multiplier
            let cash_flow = *qty * payoff_per_unit * instr.multiplier();

            tasks.push(SettlementTask {
                symbol: symbol.clone(),
                asset_type: instr.asset_type,
                expiry_date: Some(expiry_date_int),
                quantity: *qty, // Full position quantity to close
                cash_flow,
                fee: Decimal::ZERO,
                settlement_type: None,
                settlement_price: Some(underlying_price),
                reason: "expiry".to_string(),
                description: format!("Option Expiry for {symbol}"),
            });
        }

        (tasks, deferred)
    }
}

impl SettlementHandler for OptionSettlementHandler {
    fn check_settlement(
        &self,
        date: NaiveDate,
        portfolio: &Portfolio,
        instruments: &HashMap<String, Instrument>,
        last_prices: &HashMap<String, Decimal>,
    ) -> Vec<SettlementTask> {
        self.check_with_deferred(date, portfolio, instruments, last_prices)
            .0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::instrument::{InstrumentEnum, OptionInstrument};
    use crate::model::types::{AssetType, OptionType};
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use std::sync::Arc;

    fn create_test_option(
        symbol: &str,
        expiry_date: u32,
        option_type: OptionType,
        strike: Decimal,
    ) -> Instrument {
        Instrument {
            asset_type: AssetType::Option,
            inner: InstrumentEnum::Option(OptionInstrument {
                symbol: symbol.to_string(),
                multiplier: dec!(100),
                margin_ratio: dec!(0.2),
                tick_size: dec!(0.01),
                option_margin_model: crate::model::OptionMarginModel::ChinaSingleLeg,
                option_type,
                strike_price: strike,
                expiry_date,
                underlying_symbol: "UNDERLYING".to_string(),
                settlement_type: None,
                settlement_price: None,
                implied_volatility: None,
                reference_volatility: None,
            }),
        }
    }

    #[test]
    fn test_option_expiry_call_in_the_money() {
        let handler = OptionSettlementHandler;
        let expiry_date = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();

        let mut positions = HashMap::new();
        positions.insert("OPT_CALL".to_string(), dec!(10)); // 10 Long Calls

        let portfolio = Portfolio {
            cash: dec!(100000),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };

        let mut instruments = HashMap::new();
        instruments.insert(
            "OPT_CALL".to_string(),
            create_test_option("OPT_CALL", 20240101, OptionType::Call, dec!(100)),
        );

        let mut last_prices = HashMap::new();
        last_prices.insert("UNDERLYING".to_string(), dec!(110)); // Underlying > Strike (ITM)

        let tasks = handler.check_settlement(expiry_date, &portfolio, &instruments, &last_prices);

        assert_eq!(tasks.len(), 1);
        let task = &tasks[0];
        assert_eq!(task.symbol, "OPT_CALL");
        assert_eq!(task.quantity, dec!(10));

        // Payoff = (110 - 100) = 10
        // Cash Flow = 10 * 100 (multiplier) * 10 (qty) = 10000
        assert_eq!(task.cash_flow, dec!(10000));
    }

    #[test]
    fn test_option_expiry_out_of_the_money() {
        let handler = OptionSettlementHandler;
        let expiry_date = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();

        let mut positions = HashMap::new();
        positions.insert("OPT_PUT".to_string(), dec!(1));

        let portfolio = Portfolio {
            cash: dec!(10000),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };

        let mut instruments = HashMap::new();
        instruments.insert(
            "OPT_PUT".to_string(),
            create_test_option("OPT_PUT", 20240101, OptionType::Put, dec!(100)),
        );

        let mut last_prices = HashMap::new();
        last_prices.insert("UNDERLYING".to_string(), dec!(110)); // Underlying > Strike (OTM for Put)

        let tasks = handler.check_settlement(expiry_date, &portfolio, &instruments, &last_prices);

        // 虚值到期也生成任务, 现金流为 0, 用于平掉持仓

        assert_eq!(tasks.len(), 1);
        let task = &tasks[0];
        assert_eq!(task.cash_flow, dec!(0));
    }
    #[test]
    fn test_option_not_settled_on_expiry_date_itself() {
        let handler = OptionSettlementHandler;
        let mut positions = HashMap::new();
        positions.insert("OPT_CALL".to_string(), dec!(10));
        let portfolio = Portfolio {
            cash: dec!(100000),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut instruments = HashMap::new();
        instruments.insert(
            "OPT_CALL".to_string(),
            create_test_option("OPT_CALL", 20240101, OptionType::Call, dec!(100)),
        );
        let mut last_prices = HashMap::new();
        last_prices.insert("UNDERLYING".to_string(), dec!(110));
        // 到期日当天还能交易, 结算要等到期日收盘后(进入下一个交易日时)才发生
        let tasks = handler.check_settlement(
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            &portfolio,
            &instruments,
            &last_prices,
        );
        assert!(tasks.is_empty());
    }

    fn one_long_call_portfolio() -> Portfolio {
        let mut positions = HashMap::new();
        positions.insert("OPT_CALL".to_string(), dec!(1));
        Portfolio {
            cash: dec!(0),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        }
    }

    #[test]
    fn test_missing_underlying_price_defers_instead_of_zero_payoff() {
        let handler = OptionSettlementHandler;
        let mut instruments = HashMap::new();
        instruments.insert(
            "OPT_CALL".to_string(),
            create_test_option("OPT_CALL", 20240101, OptionType::Call, dec!(100)),
        );
        let (tasks, deferred) = handler.check_with_deferred(
            NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            &one_long_call_portfolio(),
            &instruments,
            &HashMap::new(),
        );
        assert!(tasks.is_empty());
        assert_eq!(deferred, vec!["OPT_CALL".to_string()]);
    }

    #[test]
    fn test_configured_settlement_price_wins_over_last_price() {
        let handler = OptionSettlementHandler;
        let mut option = create_test_option("OPT_CALL", 20240101, OptionType::Call, dec!(100));
        if let InstrumentEnum::Option(ref mut o) = option.inner {
            o.settlement_price = Some(dec!(130));
        }
        let mut instruments = HashMap::new();
        instruments.insert("OPT_CALL".to_string(), option);
        let mut last_prices = HashMap::new();
        last_prices.insert("UNDERLYING".to_string(), dec!(110));
        let (tasks, deferred) = handler.check_with_deferred(
            NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            &one_long_call_portfolio(),
            &instruments,
            &last_prices,
        );
        assert!(deferred.is_empty());
        // (130 - 100) × 乘数 100 × 1 张
        assert_eq!(tasks[0].cash_flow, dec!(3000));
        assert_eq!(tasks[0].settlement_price, Some(dec!(130)));
    }

    #[test]
    fn test_option_without_expiry_date_never_expires() {
        // expiry_date = 0 表示未配置到期日: 任何日期都不结算, 也不算延后
        let handler = OptionSettlementHandler;
        let mut instruments = HashMap::new();
        instruments.insert(
            "OPT_CALL".to_string(),
            create_test_option("OPT_CALL", 0, OptionType::Call, dec!(100)),
        );
        let mut last_prices = HashMap::new();
        last_prices.insert("UNDERLYING".to_string(), dec!(110));
        for date in [
            NaiveDate::from_ymd_opt(1970, 1, 2).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            NaiveDate::from_ymd_opt(2099, 12, 31).unwrap(),
        ] {
            let (tasks, deferred) = handler.check_with_deferred(
                date,
                &one_long_call_portfolio(),
                &instruments,
                &last_prices,
            );
            assert!(tasks.is_empty(), "{date}: 不应生成到期任务");
            assert!(deferred.is_empty(), "{date}: 不应进入延后列表");
        }
    }
}
