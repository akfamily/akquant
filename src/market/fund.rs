use crate::model::{Instrument, OrderSide};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct FundConfig {
    pub commission_rate: Decimal,
    pub transfer_fee: Decimal,
    pub min_commission: Decimal,
    pub t_plus_one: bool,
}

impl Default for FundConfig {
    fn default() -> Self {
        Self {
            commission_rate: Decimal::from_str("0.0003").unwrap(),
            transfer_fee: Decimal::from_str("0.00001").unwrap(),
            min_commission: Decimal::from(5),
            t_plus_one: true,
        }
    }
}

/// 计算基金交易费用
pub fn calculate_commission(
    config: &FundConfig,
    _instrument: &Instrument,
    _side: OrderSide,
    price: Decimal,
    quantity: Decimal,
    multiplier: Decimal,
) -> Decimal {
    let transaction_value = price * quantity * multiplier;
    let mut commission = Decimal::ZERO;

    // 1. 佣金
    let mut brokerage = transaction_value * config.commission_rate;
    if brokerage < config.min_commission {
        brokerage = config.min_commission;
    }
    commission += brokerage;

    // 2. 过户费
    commission += transaction_value * config.transfer_fee;

    commission
}

/// 更新基金可用持仓 (按 per-instrument sellable_after_days 处理 T+0/T+1)
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

            if sellable_after_days == 0
                && let Some(pos) = available_positions.get_mut(symbol)
            {
                *pos += quantity;
            }
        }
        OrderSide::Sell => {
            if let Some(pos) = available_positions.get_mut(symbol) {
                *pos -= quantity;
            }
        }
    }
}
