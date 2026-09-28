pub mod china;
pub mod core;
pub mod corporate_action;
pub mod fee_override;
pub mod fund;
pub mod futures;
pub mod manager;
pub mod option;
pub mod simple;
pub mod stock;

pub use china::{ChinaMarket, ChinaMarketConfig, SessionRange};
pub use core::{MarketConfig, MarketModel};
pub use simple::{SimpleMarket, SimpleMarketConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AssetType, Instrument, OrderSide, TradingSession};
    use chrono::NaiveTime;
    use rust_decimal::Decimal;
    use rust_decimal::prelude::*;

    #[test]
    fn test_china_market_session() {
        let market = ChinaMarket::new();

        // 9:15 -> CallAuction
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(9, 15, 0).unwrap()),
            TradingSession::CallAuction
        );
        // 9:30 -> Continuous
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(9, 30, 0).unwrap()),
            TradingSession::Continuous
        );
        // 12:00 -> Break
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(12, 0, 0).unwrap()),
            TradingSession::Break
        );
        // 18:00 -> Closed
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(18, 0, 0).unwrap()),
            TradingSession::Closed
        );
    }

    #[test]
    #[should_panic(expected = "Futures market configuration not found")]
    fn test_china_market_missing_config_panic() {
        // Create config with only Stock enabled
        let config = ChinaMarketConfig {
            stock: Some(stock::StockConfig::default()),
            ..Default::default()
        };
        // futures is None by default

        let market = ChinaMarket::from_config(config);

        // Create a Futures instrument
        use crate::model::instrument::{FuturesInstrument, InstrumentEnum};
        let instr = Instrument {
            asset_type: AssetType::Futures,
            inner: InstrumentEnum::Futures(FuturesInstrument {
                symbol: "IF2206".to_string(),
                multiplier: Decimal::from(300),
                tick_size: Decimal::from_str("0.2").unwrap(),
                margin_ratio: Decimal::from_str("0.1").unwrap(),
                expiry_date: None,
                settlement_type: None,
                settlement_price: None,
            }),
        };

        // This should panic because futures config is missing
        market.calculate_commission(
            &instr,
            OrderSide::Buy,
            Decimal::from(4000),
            Decimal::from(1),
            Decimal::ZERO,
        );
    }

    #[test]
    fn test_china_market_futures_fee_by_prefix() {
        let mut config = ChinaMarketConfig {
            futures: Some(futures::FuturesConfig::default()),
            ..Default::default()
        };
        config.futures_fee_by_prefix.push((
            "RB".to_string(),
            futures::FuturesConfig {
                commission_rate: Decimal::from_str("0.0005").unwrap(),
            },
        ));

        let market = ChinaMarket::from_config(config);
        use crate::model::instrument::{FuturesInstrument, InstrumentEnum};
        let instr = Instrument {
            asset_type: AssetType::Futures,
            inner: InstrumentEnum::Futures(FuturesInstrument {
                symbol: "RB2310".to_string(),
                multiplier: Decimal::from(10),
                tick_size: Decimal::from_str("1").unwrap(),
                margin_ratio: Decimal::from_str("0.1").unwrap(),
                expiry_date: None,
                settlement_type: None,
                settlement_price: None,
            }),
        };

        let commission = market.calculate_commission(
            &instr,
            OrderSide::Buy,
            Decimal::from(3500),
            Decimal::from(2),
            Decimal::ZERO,
        );
        assert_eq!(commission, Decimal::from_str("35").unwrap());
    }

    #[test]
    fn test_china_market_options_fee_by_prefix() {
        let mut config = ChinaMarketConfig {
            option: Some(option::OptionConfig::default()),
            ..Default::default()
        };
        config.options_fee_by_prefix.push((
            "OPT".to_string(),
            option::OptionConfig {
                commission_per_contract: Decimal::from(12),
                ..Default::default()
            },
        ));

        let market = ChinaMarket::from_config(config);
        use crate::model::instrument::{InstrumentEnum, OptionInstrument};
        let instr = Instrument {
            asset_type: AssetType::Option,
            inner: InstrumentEnum::Option(OptionInstrument {
                symbol: "OPT_510050_C_202601".to_string(),
                multiplier: Decimal::from(1),
                margin_ratio: Decimal::from_str("0.2").unwrap(),
                tick_size: Decimal::from_str("0.0001").unwrap(),
                option_margin_model: crate::model::OptionMarginModel::ChinaSingleLeg,
                strike_price: Decimal::from(2),
                option_type: crate::model::OptionType::Call,
                expiry_date: 20260131,
                underlying_symbol: "510050.SH".to_string(),
                settlement_type: None,
                settlement_price: None,
                implied_volatility: None,
                reference_volatility: None,
            }),
        };

        let commission = market.calculate_commission(
            &instr,
            OrderSide::Buy,
            Decimal::from_str("0.1234").unwrap(),
            Decimal::from(3),
            Decimal::ZERO,
        );
        assert_eq!(commission, Decimal::from_str("40.8").unwrap());
    }
}
