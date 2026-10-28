use crate::model::{Instrument, OrderSide};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;

use super::core::MarketModel;
use super::fee_override::FeeOverride;
use super::stock::CommissionMode;

/// 简单市场配置
#[derive(Clone, Debug)]
pub struct SimpleMarketConfig {
    pub commission_mode: CommissionMode,
    pub commission_rate: Decimal,
    pub stamp_tax: Decimal,
    pub transfer_fee: Decimal,
    pub min_commission: Decimal,
    /// 按品种费用覆盖, 键为归一化后的 symbol
    pub fee_overrides: HashMap<String, FeeOverride>,
}

impl Default for SimpleMarketConfig {
    fn default() -> Self {
        Self {
            commission_mode: CommissionMode::Percent,
            commission_rate: Decimal::from_str("0.0003").unwrap(),
            stamp_tax: Decimal::ZERO,
            transfer_fee: Decimal::ZERO,
            min_commission: Decimal::ZERO,
            fee_overrides: HashMap::new(),
        }
    }
}

/// 简单市场模型 (如加密货币/外汇)
pub struct SimpleMarket {
    pub config: SimpleMarketConfig,
}

impl SimpleMarket {
    pub fn from_config(config: SimpleMarketConfig) -> Self {
        Self { config }
    }
}

impl MarketModel for SimpleMarket {
    fn calculate_commission(
        &self,
        instrument: &Instrument,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
        _position_before: Decimal,
    ) -> Decimal {
        let cfg: Cow<'_, SimpleMarketConfig> =
            match self.config.fee_overrides.get(instrument.symbol()) {
                Some(o) => Cow::Owned(o.apply_simple(&self.config)),
                None => Cow::Borrowed(&self.config),
            };
        let turnover = price * quantity * instrument.multiplier();
        let mut commission = match cfg.commission_mode {
            CommissionMode::Percent => turnover * cfg.commission_rate,
            CommissionMode::Fixed => cfg.commission_rate,
            CommissionMode::PerUnit => quantity * cfg.commission_rate,
        };
        if commission < cfg.min_commission {
            commission = cfg.min_commission;
        }
        let tax = if side == OrderSide::Sell {
            turnover * cfg.stamp_tax
        } else {
            Decimal::ZERO
        };
        let transfer = turnover * cfg.transfer_fee;
        commission + tax + transfer
    }

    fn update_available_position(
        &self,
        available_positions: &mut HashMap<String, Decimal>,
        instrument: &Instrument,
        quantity: Decimal,
        side: OrderSide,
    ) {
        let symbol = instrument.symbol();
        match side {
            OrderSide::Buy => {
                available_positions
                    .entry(symbol.to_string())
                    .or_insert(Decimal::ZERO);
                if let Some(pos) = available_positions.get_mut(symbol) {
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

    /// SimpleMarket 本身按 T+0 维护可用持仓, 这里只解冻已冻结的多头持仓
    /// (可用 < 持仓): 例如 T+1 下存档、以 T+0 续跑时存档前当日买入的部分。
    /// 冻结状态不依赖当前 T+1 开关, 否则这部分持仓永不可卖。
    fn on_day_close(
        &self,
        positions: &HashMap<String, Decimal>,
        available_positions: &mut HashMap<String, Decimal>,
        _instruments: &HashMap<String, Instrument>,
    ) {
        for (symbol, quantity) in positions {
            if *quantity <= Decimal::ZERO {
                continue;
            }
            let available = available_positions
                .get(symbol)
                .copied()
                .unwrap_or(Decimal::ZERO);
            if available < *quantity {
                available_positions.insert(symbol.clone(), *quantity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_close_releases_frozen_long_positions_only() {
        let market = SimpleMarket::from_config(SimpleMarketConfig::default());
        let positions = HashMap::from([
            ("FROZEN".to_string(), Decimal::from(100)),
            ("PARTIAL".to_string(), Decimal::from(300)),
            ("FREE".to_string(), Decimal::from(200)),
            ("SHORT".to_string(), Decimal::from(-50)),
        ]);
        let mut available = HashMap::from([
            ("PARTIAL".to_string(), Decimal::from(100)),
            ("FREE".to_string(), Decimal::from(200)),
        ]);
        market.on_day_close(&positions, &mut available, &HashMap::new());
        assert_eq!(available.get("FROZEN"), Some(&Decimal::from(100)));
        assert_eq!(available.get("PARTIAL"), Some(&Decimal::from(300)));
        assert_eq!(available.get("FREE"), Some(&Decimal::from(200)));
        assert!(!available.contains_key("SHORT"));
    }
}
