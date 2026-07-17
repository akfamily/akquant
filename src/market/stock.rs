use crate::model::{Instrument, OrderSide};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommissionMode {
    Percent,
    Fixed,
    PerUnit,
}

#[derive(Clone, Debug)]
pub struct StockConfig {
    pub commission_mode: CommissionMode,
    pub commission_rate: Decimal,
    pub stamp_tax: Decimal,
    pub transfer_fee: Decimal,
    pub min_commission: Decimal,
    pub t_plus_one: bool,
}

impl Default for StockConfig {
    fn default() -> Self {
        Self {
            commission_mode: CommissionMode::Percent,
            commission_rate: Decimal::from_str("0.0003").unwrap(),
            stamp_tax: Decimal::from_str("0.0005").unwrap(),
            transfer_fee: Decimal::from_str("0.00001").unwrap(),
            min_commission: Decimal::from(5),
            t_plus_one: true,
        }
    }
}

/// 计算股票交易费用 (佣金 + 印花税 + 过户费)
pub fn calculate_commission(
    config: &StockConfig,
    _instrument: &Instrument,
    side: OrderSide,
    price: Decimal,
    quantity: Decimal,
    multiplier: Decimal,
) -> Decimal {
    let transaction_value = price * quantity * multiplier;
    let mut commission = Decimal::ZERO;

    // 1. 佣金
    let mut brokerage = match config.commission_mode {
        CommissionMode::Percent => transaction_value * config.commission_rate,
        CommissionMode::Fixed => config.commission_rate,
        CommissionMode::PerUnit => quantity * config.commission_rate,
    };
    if brokerage < config.min_commission {
        brokerage = config.min_commission;
    }
    commission += brokerage;

    // 2. 印花税 (仅卖出收取)
    if side == OrderSide::Sell {
        commission += transaction_value * config.stamp_tax;
    }

    // 3. 过户费
    commission += transaction_value * config.transfer_fee;

    commission
}

/// 更新股票可用持仓 (按 per-instrument sellable_after_days 处理 T+0/T+1)
pub fn update_available_position(
    sellable_after_days: u32,
    available_positions: &mut HashMap<String, Decimal>,
    symbol: &str,
    quantity: Decimal,
    side: OrderSide,
) {
    match side {
        OrderSide::Buy => {
            available_positions
                .entry(symbol.to_string())
                .or_insert(Decimal::ZERO);

            // T+0(==0)买入立即可卖;T+N(>=1)不立即增加可用持仓,待 on_day_close 解锁
            if sellable_after_days == 0
                && let Some(pos) = available_positions.get_mut(symbol)
            {
                *pos += quantity;
            }
        }
        OrderSide::Sell => {
            // 卖出立即扣减可用持仓
            if let Some(pos) = available_positions.get_mut(symbol) {
                *pos -= quantity;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AssetType, Instrument};

    fn create_stock_instrument(symbol: &str) -> Instrument {
        use crate::model::instrument::{InstrumentEnum, StockInstrument};
        Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.to_string(),
                lot_size: Decimal::from(100),
                tick_size: Decimal::from_str("0.01").unwrap(),
                expiry_date: None,
                sellable_after_days: 1,
            }),
        }
    }

    #[test]
    fn test_stock_commission() {
        let config = StockConfig::default();
        let instr = create_stock_instrument("600000");

        // Case 1: Minimum Commission (5 RMB)
        // Buy 100 @ 10. Value = 1000.
        // Brokerage = 1000 * 0.0003 = 0.3 < 5 -> 5.
        // Transfer = 1000 * 0.00001 = 0.01.
        // Total = 5.01.
        let comm_buy = calculate_commission(
            &config,
            &instr,
            OrderSide::Buy,
            Decimal::from(10),
            Decimal::from(100),
            instr.multiplier(),
        );
        assert_eq!(comm_buy, Decimal::from_str("5.01").unwrap());

        // Case 2: Sell (Add Stamp Tax)
        // Sell 100 @ 10. Value = 1000.
        // Brokerage = 5.
        // Stamp Tax = 1000 * 0.0005 = 0.5.
        // Transfer = 0.01.
        // Total = 5.51.
        let comm_sell = calculate_commission(
            &config,
            &instr,
            OrderSide::Sell,
            Decimal::from(10),
            Decimal::from(100),
            instr.multiplier(),
        );
        assert_eq!(comm_sell, Decimal::from_str("5.51").unwrap());
    }

    #[test]
    fn test_stock_t_plus_one() {
        let instr = create_stock_instrument("600000");
        let mut available = HashMap::new();

        // Buy 100 under T+1 (sellable_after_days=1): available shouldn't increase immediately.
        update_available_position(
            1,
            &mut available,
            instr.symbol(),
            Decimal::from(100),
            OrderSide::Buy,
        );
        assert!(available.get("600000").unwrap_or(&Decimal::ZERO).is_zero());
    }
}
