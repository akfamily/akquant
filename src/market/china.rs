use crate::model::{AssetType, Instrument, OrderSide, TradingSession};
use chrono::NaiveTime;
use rust_decimal::Decimal;
use std::collections::HashMap;

use super::core::MarketModel;
use super::{fund, futures, option, stock};

#[derive(Clone, Debug)]
pub struct SessionRange {
    pub start: NaiveTime,
    pub end: NaiveTime,
    pub session: TradingSession,
}

#[derive(Clone, Debug)]
pub struct ChinaMarketConfig {
    pub stock: Option<stock::StockConfig>,
    pub futures: Option<futures::FuturesConfig>,
    pub fund: Option<fund::FundConfig>,
    pub option: Option<option::OptionConfig>,
    pub sessions: Vec<SessionRange>,
    pub futures_fee_by_prefix: Vec<(String, futures::FuturesConfig)>,
    pub options_fee_by_prefix: Vec<(String, option::OptionConfig)>,
}

fn default_sessions() -> Vec<SessionRange> {
    let t_9_15 = NaiveTime::from_hms_opt(9, 15, 0).unwrap();
    let t_9_25 = NaiveTime::from_hms_opt(9, 25, 0).unwrap();
    let t_9_30 = NaiveTime::from_hms_opt(9, 30, 0).unwrap();
    let t_11_30 = NaiveTime::from_hms_opt(11, 30, 0).unwrap();
    let t_13_00 = NaiveTime::from_hms_opt(13, 0, 0).unwrap();
    let t_14_57 = NaiveTime::from_hms_opt(14, 57, 0).unwrap();
    let t_15_00 = NaiveTime::from_hms_opt(15, 0, 1).unwrap();
    vec![
        SessionRange {
            start: t_9_15,
            end: t_9_25,
            session: TradingSession::CallAuction,
        },
        SessionRange {
            start: t_9_25,
            end: t_9_30,
            session: TradingSession::PreOpen,
        },
        SessionRange {
            start: t_9_30,
            end: t_11_30,
            session: TradingSession::Continuous,
        },
        SessionRange {
            start: t_11_30,
            end: t_13_00,
            session: TradingSession::Break,
        },
        SessionRange {
            start: t_13_00,
            end: t_14_57,
            session: TradingSession::Continuous,
        },
        SessionRange {
            start: t_14_57,
            end: t_15_00,
            session: TradingSession::CallAuction,
        },
    ]
}

impl Default for ChinaMarketConfig {
    fn default() -> Self {
        Self {
            stock: None,
            futures: None,
            fund: None,
            option: None,
            sessions: default_sessions(),
            futures_fee_by_prefix: Vec::new(),
            options_fee_by_prefix: Vec::new(),
        }
    }
}

pub struct ChinaMarket {
    pub config: ChinaMarketConfig,
}

impl Default for ChinaMarket {
    fn default() -> Self {
        Self::new()
    }
}

impl ChinaMarket {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            config: ChinaMarketConfig::default(),
        }
    }
    pub fn from_config(config: ChinaMarketConfig) -> Self {
        Self { config }
    }

    fn futures_config_for_symbol(&self, symbol: &str) -> Option<&futures::FuturesConfig> {
        let mut best_match: Option<&futures::FuturesConfig> = None;
        let mut best_len = 0usize;
        let symbol_upper = symbol.to_uppercase();
        for (prefix, cfg) in &self.config.futures_fee_by_prefix {
            let normalized = prefix.trim().to_uppercase();
            if normalized.is_empty() {
                continue;
            }
            if symbol_upper.starts_with(&normalized) && normalized.len() > best_len {
                best_match = Some(cfg);
                best_len = normalized.len();
            }
        }
        best_match
    }

    fn option_config_for_symbol(&self, symbol: &str) -> Option<&option::OptionConfig> {
        let mut best_match: Option<&option::OptionConfig> = None;
        let mut best_len = 0usize;
        let symbol_upper = symbol.to_uppercase();
        for (prefix, cfg) in &self.config.options_fee_by_prefix {
            let normalized = prefix.trim().to_uppercase();
            if normalized.is_empty() {
                continue;
            }
            if symbol_upper.starts_with(&normalized) && normalized.len() > best_len {
                best_match = Some(cfg);
                best_len = normalized.len();
            }
        }
        best_match
    }
}

impl MarketModel for ChinaMarket {
    fn get_session_status(&self, time: NaiveTime) -> TradingSession {
        for range in &self.config.sessions {
            if time >= range.start && time < range.end {
                return range.session;
            }
        }
        TradingSession::Closed
    }

    fn calculate_commission(
        &self,
        instrument: &Instrument,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
    ) -> Decimal {
        match instrument.asset_type {
            AssetType::Stock => {
                if let Some(config) = &self.config.stock {
                    stock::calculate_commission(
                        config,
                        instrument,
                        side,
                        price,
                        quantity,
                        instrument.multiplier(),
                    )
                } else {
                    panic!("Stock market configuration not found but received stock order");
                }
            }
            AssetType::Futures => {
                if let Some(config) = self
                    .futures_config_for_symbol(instrument.symbol())
                    .or(self.config.futures.as_ref())
                {
                    futures::calculate_commission(
                        config,
                        instrument,
                        side,
                        price,
                        quantity,
                        instrument.multiplier(),
                    )
                } else {
                    panic!("Futures market configuration not found but received futures order");
                }
            }
            AssetType::Fund => {
                if let Some(config) = &self.config.fund {
                    fund::calculate_commission(
                        config,
                        instrument,
                        side,
                        price,
                        quantity,
                        instrument.multiplier(),
                    )
                } else {
                    panic!("Fund market configuration not found but received fund order");
                }
            }
            AssetType::Option => {
                if let Some(config) = self
                    .option_config_for_symbol(instrument.symbol())
                    .or(self.config.option.as_ref())
                {
                    option::calculate_commission(
                        config,
                        instrument,
                        side,
                        price,
                        quantity,
                        instrument.multiplier(),
                    )
                } else {
                    panic!("Option market configuration not found but received option order");
                }
            }
            AssetType::Crypto | AssetType::Forex => {
                panic!("Crypto/Forex not supported in ChinaMarket");
            }
        }
    }

    fn update_available_position(
        &self,
        available_positions: &mut HashMap<String, Decimal>,
        instrument: &Instrument,
        quantity: Decimal,
        side: OrderSide,
    ) {
        let symbol = &instrument.symbol();
        match instrument.asset_type {
            AssetType::Stock => {
                stock::update_available_position(
                    instrument.sellable_after_days(),
                    available_positions,
                    symbol,
                    quantity,
                    side,
                );
            }
            AssetType::Fund => {
                fund::update_available_position(
                    instrument.sellable_after_days(),
                    available_positions,
                    symbol,
                    quantity,
                    side,
                );
            }
            AssetType::Futures => {
                if let Some(config) = &self.config.futures {
                    futures::update_available_position(
                        config,
                        available_positions,
                        symbol,
                        quantity,
                        side,
                    );
                } else {
                    panic!("Futures market configuration not found for position update");
                }
            }
            AssetType::Option => {
                if let Some(config) = &self.config.option {
                    option::update_available_position(
                        config,
                        available_positions,
                        symbol,
                        quantity,
                        side,
                    );
                } else {
                    panic!("Option market configuration not found for position update");
                }
            }
            AssetType::Crypto | AssetType::Forex => {
                panic!("Crypto/Forex not supported in ChinaMarket");
            }
        }
    }

    fn on_day_close(
        &self,
        positions: &HashMap<String, Decimal>,
        available_positions: &mut HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
    ) {
        for (symbol, quantity) in positions {
            // sellable_after_days>=1 的标的:持有进入新交易日后全量可卖(T+1)。
            // T+0(==0)标的的可卖量由 update_available_position 在成交时即时维护,
            // 此处不覆盖。
            let releases = instruments
                .get(symbol)
                .map(|instr| instr.sellable_after_days() >= 1)
                .unwrap_or(false);
            if releases {
                available_positions.insert(symbol.clone(), *quantity);
            }
        }
    }
}
