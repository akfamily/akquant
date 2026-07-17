use crate::model::Instrument;
use crate::model::types::{AssetType, SettlementType};
use crate::portfolio::Portfolio;
use chrono::{Datelike, NaiveDate};
use rust_decimal::Decimal;
use std::collections::HashMap;

use super::handler::{SettlementHandler, SettlementTask};

#[derive(Debug, Clone, Default)]
pub struct ExpirySettlementHandler;

impl SettlementHandler for ExpirySettlementHandler {
    fn check_settlement(
        &self,
        date: NaiveDate,
        portfolio: &Portfolio,
        instruments: &HashMap<String, Instrument>,
        last_prices: &HashMap<String, Decimal>,
    ) -> Vec<SettlementTask> {
        let mut tasks = Vec::new();
        let (_, year_ce) = date.year_ce();
        let current_date_int = year_ce * 10000 + date.month() * 100 + date.day();

        for (symbol, qty) in portfolio.positions.iter() {
            if qty.is_zero() {
                continue;
            }
            let Some(instr) = instruments.get(symbol) else {
                continue;
            };
            if instr.asset_type != AssetType::Stock && instr.asset_type != AssetType::Futures {
                continue;
            }
            let Some(expiry_date_int) = instr.expiry_date() else {
                continue;
            };
            if current_date_int < expiry_date_int {
                continue;
            }
            let maybe_price = if instr.asset_type == AssetType::Futures {
                match instr.settlement_type().unwrap_or(SettlementType::Cash) {
                    SettlementType::Cash => instr
                        .settlement_price()
                        .or_else(|| last_prices.get(symbol).copied()),
                    SettlementType::Physical => last_prices.get(symbol).copied(),
                    SettlementType::ForceClose => last_prices.get(symbol).copied(),
                }
            } else {
                last_prices.get(symbol).copied()
            };
            let Some(settle_price) = maybe_price else {
                continue;
            };
            let cash_flow = *qty * settle_price * instr.multiplier();
            tasks.push(SettlementTask {
                symbol: symbol.clone(),
                asset_type: instr.asset_type,
                expiry_date: Some(expiry_date_int),
                quantity: *qty,
                cash_flow,
                settlement_type: match instr.asset_type {
                    AssetType::Futures => Some(
                        match instr.settlement_type().unwrap_or(SettlementType::Cash) {
                            SettlementType::Cash => {
                                if instr.settlement_price().is_some() {
                                    "settlement_price"
                                } else {
                                    "cash"
                                }
                            }
                            SettlementType::Physical => "physical",
                            SettlementType::ForceClose => "force_close",
                        }
                        .to_string(),
                    ),
                    _ => None,
                },
                settlement_price: Some(settle_price),
                reason: "expiry".to_string(),
                description: format!(
                    "Expiry settlement for {symbol} ({:?})",
                    instr.settlement_type().unwrap_or(SettlementType::Cash)
                ),
            });
        }
        tasks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::instrument::{FuturesInstrument, InstrumentEnum, StockInstrument};
    use crate::model::types::AssetType;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use std::sync::Arc;

    #[test]
    fn test_stock_expiry_generates_settlement_task() {
        let handler = ExpirySettlementHandler;
        let mut positions = HashMap::new();
        positions.insert("STK_EXP".to_string(), dec!(100));
        let portfolio = Portfolio {
            cash: dec!(1000),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut instruments = HashMap::new();
        instruments.insert(
            "STK_EXP".to_string(),
            Instrument {
                asset_type: AssetType::Stock,
                inner: InstrumentEnum::Stock(StockInstrument {
                    symbol: "STK_EXP".to_string(),
                    lot_size: dec!(1),
                    tick_size: dec!(0.01),
                    expiry_date: Some(20260131),
                    sellable_after_days: 1,
                }),
            },
        );
        let mut prices = HashMap::new();
        prices.insert("STK_EXP".to_string(), dec!(10));
        let tasks = handler.check_settlement(
            NaiveDate::from_ymd_opt(2026, 1, 31).expect("valid date"),
            &portfolio,
            &instruments,
            &prices,
        );
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].cash_flow, dec!(1000));
    }

    #[test]
    fn test_futures_expiry_generates_settlement_task() {
        let handler = ExpirySettlementHandler;
        let mut positions = HashMap::new();
        positions.insert("FUT_EXP".to_string(), dec!(2));
        let portfolio = Portfolio {
            cash: dec!(1000),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut instruments = HashMap::new();
        instruments.insert(
            "FUT_EXP".to_string(),
            Instrument {
                asset_type: AssetType::Futures,
                inner: InstrumentEnum::Futures(FuturesInstrument {
                    symbol: "FUT_EXP".to_string(),
                    multiplier: dec!(10),
                    margin_ratio: dec!(0.1),
                    tick_size: dec!(0.2),
                    expiry_date: Some(20260131),
                    settlement_type: None,
                    settlement_price: None,
                }),
            },
        );
        let mut prices = HashMap::new();
        prices.insert("FUT_EXP".to_string(), dec!(100));
        let tasks = handler.check_settlement(
            NaiveDate::from_ymd_opt(2026, 1, 31).expect("valid date"),
            &portfolio,
            &instruments,
            &prices,
        );
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].cash_flow, dec!(2000));
    }

    #[test]
    fn test_futures_expiry_settlement_price_mode_uses_configured_price() {
        let handler = ExpirySettlementHandler;
        let mut positions = HashMap::new();
        positions.insert("FUT_SETTLE".to_string(), dec!(1));
        let portfolio = Portfolio {
            cash: dec!(1000),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };
        let mut instruments = HashMap::new();
        instruments.insert(
            "FUT_SETTLE".to_string(),
            Instrument {
                asset_type: AssetType::Futures,
                inner: InstrumentEnum::Futures(FuturesInstrument {
                    symbol: "FUT_SETTLE".to_string(),
                    multiplier: dec!(10),
                    margin_ratio: dec!(0.1),
                    tick_size: dec!(0.2),
                    expiry_date: Some(20260131),
                    settlement_type: Some(SettlementType::Cash),
                    settlement_price: Some(dec!(88)),
                }),
            },
        );
        let mut prices = HashMap::new();
        prices.insert("FUT_SETTLE".to_string(), dec!(100));
        let tasks = handler.check_settlement(
            NaiveDate::from_ymd_opt(2026, 1, 31).expect("valid date"),
            &portfolio,
            &instruments,
            &prices,
        );
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].cash_flow, dec!(880));
    }
}
