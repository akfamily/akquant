use crate::model::{Instrument, OptionMarginModel, OptionType};
use rust_decimal::Decimal;

fn checked_mul_or_cap(lhs: Decimal, rhs: Decimal) -> Decimal {
    lhs.checked_mul(rhs).unwrap_or(Decimal::MAX)
}

fn checked_add_or_cap(lhs: Decimal, rhs: Decimal) -> Decimal {
    lhs.checked_add(rhs).unwrap_or(Decimal::MAX)
}

fn checked_sub_or_zero(lhs: Decimal, rhs: Decimal) -> Decimal {
    lhs.checked_sub(rhs)
        .unwrap_or(Decimal::ZERO)
        .max(Decimal::ZERO)
}

fn legacy_option_margin(
    abs_qty: Decimal,
    option_price: Decimal,
    underlying_price: Option<Decimal>,
    instrument: &Instrument,
) -> Decimal {
    let multiplier = instrument.multiplier().abs();
    let margin_ratio = instrument.margin_ratio().abs();
    let underlying_price = underlying_price.unwrap_or(Decimal::ZERO).abs();
    let margin_per_unit = if underlying_price > Decimal::ZERO {
        checked_add_or_cap(
            option_price,
            checked_mul_or_cap(underlying_price, margin_ratio),
        )
    } else {
        checked_mul_or_cap(option_price, checked_add_or_cap(Decimal::ONE, margin_ratio))
    };
    checked_mul_or_cap(checked_mul_or_cap(margin_per_unit, multiplier), abs_qty)
}

fn calculate_single_leg_margin_per_unit(
    option_type: OptionType,
    option_price: Decimal,
    strike_price: Decimal,
    underlying_price: Decimal,
    exposure_ratio: Decimal,
    floor_ratio: Decimal,
) -> Decimal {
    let otm = match option_type {
        OptionType::Call => checked_sub_or_zero(strike_price, underlying_price),
        OptionType::Put => checked_sub_or_zero(underlying_price, strike_price),
    };
    let exposure_component =
        checked_sub_or_zero(checked_mul_or_cap(underlying_price, exposure_ratio), otm);
    let floor_base = match option_type {
        OptionType::Call => underlying_price,
        OptionType::Put => strike_price,
    };
    let floor_component = checked_mul_or_cap(floor_base, floor_ratio);
    checked_add_or_cap(option_price, exposure_component.max(floor_component))
}

/// Trait for calculating margin requirements for a position.
pub trait MarginCalculator: Send + Sync {
    /// Calculate the margin required for a position.
    ///
    /// # Arguments
    /// * `quantity` - The position quantity (positive for long, negative for short).
    /// * `price` - The current price of the instrument.
    /// * `instrument` - The instrument details.
    /// * `underlying_price` - The price of the underlying asset (optional, for derivatives).
    ///
    /// # Returns
    /// The required margin amount (always non-negative).
    fn calculate_margin(
        &self,
        quantity: Decimal,
        price: Decimal,
        instrument: &Instrument,
        underlying_price: Option<Decimal>,
    ) -> Decimal;
}

/// Default calculator for linear instruments (Stocks, Funds).
/// Margin = Quantity * Price * Multiplier * MarginRatio
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearMarginCalculator;

impl MarginCalculator for LinearMarginCalculator {
    fn calculate_margin(
        &self,
        quantity: Decimal,
        price: Decimal,
        instrument: &Instrument,
        _underlying_price: Option<Decimal>,
    ) -> Decimal {
        let multiplier = instrument.multiplier().abs();
        let margin_ratio = instrument.margin_ratio().abs();
        let notional = checked_mul_or_cap(quantity.abs(), price.abs());
        let gross = checked_mul_or_cap(notional, multiplier);
        checked_mul_or_cap(gross, margin_ratio)
    }
}

/// Calculator for Futures.
/// Same as Linear, but conceptually distinct for potential future complexity (e.g. spread margin).
#[derive(Debug, Clone, Copy, Default)]
pub struct FuturesMarginCalculator;

impl MarginCalculator for FuturesMarginCalculator {
    fn calculate_margin(
        &self,
        quantity: Decimal,
        price: Decimal,
        instrument: &Instrument,
        _underlying_price: Option<Decimal>,
    ) -> Decimal {
        let multiplier = instrument.multiplier().abs();
        let margin_ratio = instrument.margin_ratio().abs();
        let notional = checked_mul_or_cap(quantity.abs(), price.abs());
        let gross = checked_mul_or_cap(notional, multiplier);
        checked_mul_or_cap(gross, margin_ratio)
    }
}

/// Calculator for Options.
/// Handles Long (usually 0 margin) and Short (complex margin) positions.
#[derive(Debug, Clone, Copy, Default)]
pub struct OptionMarginCalculator;

impl MarginCalculator for OptionMarginCalculator {
    fn calculate_margin(
        &self,
        quantity: Decimal,
        price: Decimal,
        instrument: &Instrument,
        underlying_price: Option<Decimal>,
    ) -> Decimal {
        if quantity > Decimal::ZERO {
            return Decimal::ZERO;
        }

        let abs_qty = quantity.abs();
        let option_price = price.abs();
        let Some(option_type) = instrument.option_type() else {
            return legacy_option_margin(abs_qty, option_price, underlying_price, instrument);
        };
        let strike_price = instrument.strike_price().unwrap_or(Decimal::ZERO).abs();
        let maybe_underlying = underlying_price.map(|value| value.abs());

        let margin = match (
            instrument.option_margin_model().unwrap_or_default(),
            maybe_underlying,
        ) {
            (OptionMarginModel::Ratio, _) | (_, None) => {
                legacy_option_margin(abs_qty, option_price, maybe_underlying, instrument)
            }
            (OptionMarginModel::ChinaSingleLeg, Some(underlying)) => {
                let margin_per_unit = calculate_single_leg_margin_per_unit(
                    option_type,
                    option_price,
                    strike_price,
                    underlying,
                    Decimal::new(12, 2),
                    Decimal::new(7, 2),
                );
                checked_mul_or_cap(
                    checked_mul_or_cap(margin_per_unit, instrument.multiplier().abs()),
                    abs_qty,
                )
            }
            (OptionMarginModel::USBrokerSingleLeg, Some(underlying)) => {
                let margin_per_unit = calculate_single_leg_margin_per_unit(
                    option_type,
                    option_price,
                    strike_price,
                    underlying,
                    Decimal::new(20, 2),
                    Decimal::new(10, 2),
                );
                checked_mul_or_cap(
                    checked_mul_or_cap(margin_per_unit, instrument.multiplier().abs()),
                    abs_qty,
                )
            }
            (OptionMarginModel::USBrokerSingleLegVolAdjusted, Some(underlying)) => {
                let base_margin = calculate_single_leg_margin_per_unit(
                    option_type,
                    option_price,
                    strike_price,
                    underlying,
                    Decimal::new(20, 2),
                    Decimal::new(10, 2),
                );
                let volatility_factor = match (
                    instrument.implied_volatility(),
                    instrument.reference_volatility(),
                ) {
                    (Some(implied), Some(reference)) if reference > Decimal::ZERO => {
                        checked_add_or_cap(
                            Decimal::ONE,
                            checked_mul_or_cap(implied.abs(), Decimal::ONE / reference.abs()),
                        )
                    }
                    _ => Decimal::ONE,
                };
                let adjusted_margin = checked_mul_or_cap(base_margin, volatility_factor);
                checked_mul_or_cap(
                    checked_mul_or_cap(adjusted_margin, instrument.multiplier().abs()),
                    abs_qty,
                )
            }
        };

        margin
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::instrument::{FuturesInstrument, InstrumentEnum, OptionInstrument};
    use crate::model::types::{AssetType, OptionMarginModel, OptionType};
    use rust_decimal_macros::dec;

    #[test]
    fn linear_margin_caps_on_overflow() {
        let instrument = Instrument {
            asset_type: AssetType::Futures,
            inner: InstrumentEnum::Futures(FuturesInstrument {
                symbol: "FUT".to_string(),
                multiplier: Decimal::MAX,
                margin_ratio: Decimal::MAX,
                tick_size: Decimal::ONE,
                expiry_date: None,
                settlement_type: None,
                settlement_price: None,
            }),
        };

        let calc = LinearMarginCalculator;
        let margin = calc.calculate_margin(Decimal::MAX, Decimal::MAX, &instrument, None);
        assert_eq!(margin, Decimal::MAX);
    }

    #[test]
    fn option_margin_caps_on_overflow() {
        let instrument = Instrument {
            asset_type: AssetType::Option,
            inner: InstrumentEnum::Option(OptionInstrument {
                symbol: "OPT".to_string(),
                multiplier: Decimal::MAX,
                margin_ratio: Decimal::MAX,
                tick_size: Decimal::ONE,
                option_margin_model: OptionMarginModel::Ratio,
                option_type: OptionType::Call,
                strike_price: Decimal::ZERO,
                expiry_date: 0,
                underlying_symbol: "UL".to_string(),
                settlement_type: None,
                settlement_price: None,
                implied_volatility: None,
                reference_volatility: None,
            }),
        };

        let calc = OptionMarginCalculator;
        let margin = calc.calculate_margin(
            Decimal::NEGATIVE_ONE,
            Decimal::MAX,
            &instrument,
            Some(Decimal::MAX),
        );
        assert_eq!(margin, Decimal::MAX);
    }

    #[test]
    fn option_margin_uses_china_single_leg_formula() {
        let instrument = Instrument {
            asset_type: AssetType::Option,
            inner: InstrumentEnum::Option(OptionInstrument {
                symbol: "OPT".to_string(),
                multiplier: dec!(100),
                margin_ratio: dec!(0.2),
                tick_size: dec!(0.01),
                option_margin_model: OptionMarginModel::ChinaSingleLeg,
                option_type: OptionType::Call,
                strike_price: dec!(105),
                expiry_date: 20260101,
                underlying_symbol: "UL".to_string(),
                settlement_type: None,
                settlement_price: None,
                implied_volatility: None,
                reference_volatility: None,
            }),
        };

        let calc = OptionMarginCalculator;
        let margin = calc.calculate_margin(dec!(-2), dec!(3), &instrument, Some(dec!(100)));
        assert_eq!(margin, dec!(2000));
    }

    #[test]
    fn option_margin_uses_china_single_leg_put_floor() {
        let instrument = Instrument {
            asset_type: AssetType::Option,
            inner: InstrumentEnum::Option(OptionInstrument {
                symbol: "OPT".to_string(),
                multiplier: dec!(100),
                margin_ratio: dec!(0.2),
                tick_size: dec!(0.01),
                option_margin_model: OptionMarginModel::ChinaSingleLeg,
                option_type: OptionType::Put,
                strike_price: dec!(100),
                expiry_date: 20260101,
                underlying_symbol: "UL".to_string(),
                settlement_type: None,
                settlement_price: None,
                implied_volatility: None,
                reference_volatility: None,
            }),
        };

        let calc = OptionMarginCalculator;
        let margin = calc.calculate_margin(dec!(-1), dec!(4), &instrument, Some(dec!(110)));
        assert_eq!(margin, dec!(1100));
    }

    #[test]
    fn option_margin_uses_us_broker_vol_adjusted_formula() {
        let instrument = Instrument {
            asset_type: AssetType::Option,
            inner: InstrumentEnum::Option(OptionInstrument {
                symbol: "OPT".to_string(),
                multiplier: dec!(100),
                margin_ratio: dec!(0.2),
                tick_size: dec!(0.01),
                option_margin_model: OptionMarginModel::USBrokerSingleLegVolAdjusted,
                option_type: OptionType::Put,
                strike_price: dec!(100),
                expiry_date: 20260101,
                underlying_symbol: "UL".to_string(),
                settlement_type: None,
                settlement_price: None,
                implied_volatility: Some(dec!(0.3)),
                reference_volatility: Some(dec!(0.2)),
            }),
        };

        let calc = OptionMarginCalculator;
        let margin = calc.calculate_margin(dec!(-1), dec!(4), &instrument, Some(dec!(95)));
        assert_eq!(margin, dec!(5750));
    }

    #[test]
    fn option_margin_vol_adjustment_grows_with_implied_volatility() {
        let base_option = OptionInstrument {
            symbol: "OPT".to_string(),
            multiplier: dec!(100),
            margin_ratio: dec!(0.2),
            tick_size: dec!(0.01),
            option_margin_model: OptionMarginModel::USBrokerSingleLegVolAdjusted,
            option_type: OptionType::Call,
            strike_price: dec!(105),
            expiry_date: 20260101,
            underlying_symbol: "UL".to_string(),
            settlement_type: None,
            settlement_price: None,
            implied_volatility: Some(dec!(0.1)),
            reference_volatility: Some(dec!(0.2)),
        };
        let low_vol_instrument = Instrument {
            asset_type: AssetType::Option,
            inner: InstrumentEnum::Option(base_option.clone()),
        };
        let high_vol_instrument = Instrument {
            asset_type: AssetType::Option,
            inner: InstrumentEnum::Option(OptionInstrument {
                implied_volatility: Some(dec!(0.4)),
                ..base_option
            }),
        };

        let calc = OptionMarginCalculator;
        let low_margin =
            calc.calculate_margin(dec!(-1), dec!(3), &low_vol_instrument, Some(dec!(100)));
        let high_margin =
            calc.calculate_margin(dec!(-1), dec!(3), &high_vol_instrument, Some(dec!(100)));

        assert_eq!(low_margin, dec!(2700));
        assert_eq!(high_margin, dec!(5400));
        assert!(high_margin > low_margin);
    }
}
