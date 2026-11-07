use crate::market::stock::CommissionMode;
use crate::model::{Instrument, OrderSide};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct FundConfig {
    /// 佣金口径, 与股票相同: 按成交额比例 / 每笔固定 / 按数量
    pub commission_mode: CommissionMode,
    pub commission_rate: Decimal,
    pub transfer_fee: Decimal,
    pub min_commission: Decimal,
    pub t_plus_one: bool,
    /// 印花税率(仅卖出收取), 场内基金/可转债现行为 0, 留作按品种覆盖
    pub stamp_tax: Decimal,
}

impl Default for FundConfig {
    fn default() -> Self {
        Self {
            commission_mode: CommissionMode::Percent,
            commission_rate: Decimal::from_str("0.0003").unwrap(),
            transfer_fee: Decimal::from_str("0.00001").unwrap(),
            min_commission: Decimal::from(5),
            t_plus_one: true,
            stamp_tax: Decimal::ZERO,
        }
    }
}

/// 计算基金交易费用
pub fn calculate_commission(
    config: &FundConfig,
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

    // 2. 过户费
    commission += transaction_value * config.transfer_fee;

    // 3. 印花税 (仅卖出收取)
    if side == OrderSide::Sell {
        commission += transaction_value * config.stamp_tax;
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::AssetType;
    use crate::model::instrument::{FundInstrument, InstrumentEnum};

    fn fund_instrument() -> Instrument {
        Instrument {
            asset_type: AssetType::Fund,
            inner: InstrumentEnum::Fund(FundInstrument {
                symbol: "510300".to_string(),
                lot_size: Decimal::from(100),
                tick_size: Decimal::from_str("0.001").unwrap(),
                sellable_after_days: 1,
            }),
        }
    }

    fn fee(config: &FundConfig, side: OrderSide) -> Decimal {
        let instr = fund_instrument();
        // 1000 份 × 10 元 = 成交额 10000
        calculate_commission(
            config,
            &instr,
            side,
            Decimal::from(10),
            Decimal::from(1000),
            instr.multiplier(),
        )
    }

    fn config(mode: CommissionMode, rate: &str) -> FundConfig {
        FundConfig {
            commission_mode: mode,
            commission_rate: Decimal::from_str(rate).unwrap(),
            transfer_fee: Decimal::ZERO,
            min_commission: Decimal::ZERO,
            ..FundConfig::default()
        }
    }

    #[test]
    fn fund_commission_modes() {
        // 按成交额: 10000 × 0.0001 = 1
        let percent = config(CommissionMode::Percent, "0.0001");
        assert_eq!(fee(&percent, OrderSide::Buy), Decimal::from(1));
        // 每笔固定 3 元
        let fixed = config(CommissionMode::Fixed, "3");
        assert_eq!(fee(&fixed, OrderSide::Buy), Decimal::from(3));
        // 按数量: 1000 × 0.002 = 2
        let per_unit = config(CommissionMode::PerUnit, "0.002");
        assert_eq!(fee(&per_unit, OrderSide::Buy), Decimal::from(2));
    }

    #[test]
    fn fund_default_has_no_stamp_tax() {
        let config = FundConfig::default();
        // 默认: 佣金 max(10000 × 0.0003, 5) = 5, 过户费 0.1, 卖出不收印花税
        let expected = Decimal::from_str("5.1").unwrap();
        assert_eq!(fee(&config, OrderSide::Buy), expected);
        assert_eq!(fee(&config, OrderSide::Sell), expected);
    }
}
