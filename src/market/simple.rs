use crate::model::{Instrument, OrderSide, TradingSession};
use chrono::NaiveTime;
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
    fn get_session_status(&self, _time: NaiveTime) -> TradingSession {
        TradingSession::Continuous
    }

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

    fn on_day_close(
        &self,
        _positions: &HashMap<String, Decimal>,
        _available_positions: &mut HashMap<String, Decimal>,
        _instruments: &HashMap<String, Instrument>,
    ) {
    }
}
