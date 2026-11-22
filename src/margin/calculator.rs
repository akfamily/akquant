use crate::model::{AssetType, Instrument, OptionMarginModel, OptionType};
use rust_decimal::Decimal;
use std::collections::HashMap;

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

/// 期货交易所期权卖方保证金的计算输入, 由 [`exchange_option_inputs`] 校验后给出.
#[derive(Debug, Clone, Copy)]
pub struct ExchangeOptionInputs {
    pub model: OptionMarginModel,
    pub option_type: OptionType,
    pub strike_price: Decimal,
    pub underlying_price: Decimal,
    /// 商品期权为标的期货保证金率, 股指期权为合约保证金调整系数
    pub margin_rate: Decimal,
}

/// 期货交易所期权保证金模型(`ChinaCommodity`、`ChinaIndex`)的计算输入.
///
/// 其他模型返回 `None`; 取不到输入时返回确切原因。商品期权的保证金率取已登记的
/// 标的期货合约的 `margin_ratio`, 股指期权的调整系数取期权合约自身的 `margin_ratio`,
/// 标的价格取 `prices`。
pub fn exchange_option_inputs(
    instrument: &Instrument,
    instruments: &HashMap<String, Instrument>,
    prices: &HashMap<String, Decimal>,
) -> Option<Result<ExchangeOptionInputs, String>> {
    let model = instrument.option_margin_model()?;
    if !matches!(
        model,
        OptionMarginModel::ChinaCommodity | OptionMarginModel::ChinaIndex
    ) {
        return None;
    }
    Some(resolve_exchange_option_inputs(
        model,
        instrument,
        instruments,
        prices,
    ))
}

fn resolve_exchange_option_inputs(
    model: OptionMarginModel,
    instrument: &Instrument,
    instruments: &HashMap<String, Instrument>,
    prices: &HashMap<String, Decimal>,
) -> Result<ExchangeOptionInputs, String> {
    let (Some(option_type), Some(strike_price), Some(underlying)) = (
        instrument.option_type(),
        instrument.strike_price(),
        instrument.underlying_symbol(),
    ) else {
        return Err("缺少期权类型、行权价或标的代码".to_string());
    };
    let (label, margin_rate) = if model == OptionMarginModel::ChinaCommodity {
        let Some(underlying_instrument) = instruments.get(underlying.as_str()) else {
            return Err(format!("标的期货 {underlying} 未登记为合约, 取不到标的期货保证金率"));
        };
        if underlying_instrument.asset_type != AssetType::Futures {
            return Err(format!(
                "标的 {underlying} 登记的资产类型是 {:?}, 商品期权保证金模型要求标的为期货",
                underlying_instrument.asset_type
            ));
        }
        let rate = underlying_instrument.margin_ratio();
        if rate <= Decimal::ZERO {
            return Err(format!("标的期货 {underlying} 的保证金率 {rate} 不为正"));
        }
        ("标的期货", rate)
    } else {
        let rate = instrument.margin_ratio();
        if rate <= Decimal::ZERO {
            return Err(format!(
                "期权 {} 的保证金调整系数(margin_ratio) {rate} 不为正",
                instrument.symbol()
            ));
        }
        ("标的指数", rate)
    };
    let Some(&underlying_price) = prices.get(underlying.as_str()) else {
        return Err(format!("{label} {underlying} 没有价格"));
    };
    if underlying_price <= Decimal::ZERO {
        return Err(format!("{label} {underlying} 的价格 {underlying_price} 不为正"));
    }
    Ok(ExchangeOptionInputs {
        model,
        option_type,
        strike_price,
        underlying_price,
        margin_rate,
    })
}

/// 期货交易所期权持仓保证金(多头为 0), 按每单位标的计算后乘合约乘数与手数.
///
/// - 商品期权(大商所、郑商所、上期所、上期能源、广期所): 权利金 +
///   max(标的期货保证金 - 虚值额/2, 标的期货保证金/2), 标的期货保证金 = 标的价 × 保证金率;
/// - 股指期权(中金所): 权利金 + max(标的价 × 调整系数 - 虚值额,
///   0.5 × 调整系数 × 下限基数), 下限基数认购取标的价、认沽取行权价。
pub fn exchange_option_margin(
    quantity: Decimal,
    option_price: Decimal,
    instrument: &Instrument,
    inputs: &ExchangeOptionInputs,
) -> Decimal {
    if quantity >= Decimal::ZERO {
        return Decimal::ZERO;
    }
    let half = Decimal::new(5, 1);
    let exposure = checked_mul_or_cap(inputs.underlying_price, inputs.margin_rate);
    let otm = match inputs.option_type {
        OptionType::Call => checked_sub_or_zero(inputs.strike_price, inputs.underlying_price),
        OptionType::Put => checked_sub_or_zero(inputs.underlying_price, inputs.strike_price),
    };
    let margin = if inputs.model == OptionMarginModel::ChinaIndex {
        let floor_base = match inputs.option_type {
            OptionType::Call => inputs.underlying_price,
            OptionType::Put => inputs.strike_price,
        };
        let floor = checked_mul_or_cap(checked_mul_or_cap(half, inputs.margin_rate), floor_base);
        checked_sub_or_zero(exposure, otm).max(floor)
    } else {
        let reduced = checked_sub_or_zero(exposure, checked_mul_or_cap(otm, half));
        reduced.max(checked_mul_or_cap(exposure, half))
    };
    let per_unit = checked_add_or_cap(option_price.abs(), margin);
    checked_mul_or_cap(
        checked_mul_or_cap(per_unit, instrument.multiplier().abs()),
        quantity.abs(),
    )
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
            // 需要标的合约或调整系数, 只经 MarginEngine::position_margin 计算; 直接调用本计算器时
            // 按无法计算处理, 不退化为其他公式
            (OptionMarginModel::ChinaCommodity | OptionMarginModel::ChinaIndex, _) => {
                Decimal::MAX
            }
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

    fn exchange_option(
        model: OptionMarginModel,
        margin_ratio: Decimal,
        symbol: &str,
        underlying: &str,
        option_type: OptionType,
        strike: Decimal,
        unit: Decimal,
    ) -> Instrument {
        Instrument {
            asset_type: AssetType::Option,
            inner: InstrumentEnum::Option(OptionInstrument {
                symbol: symbol.to_string(),
                multiplier: unit,
                margin_ratio,
                tick_size: dec!(0.2),
                option_margin_model: model,
                option_type,
                strike_price: strike,
                expiry_date: 20261216,
                underlying_symbol: underlying.to_string(),
                settlement_type: None,
                settlement_price: None,
                implied_volatility: None,
                reference_volatility: None,
            }),
        }
    }

    fn futures(symbol: &str, unit: Decimal, margin_ratio: Decimal) -> Instrument {
        Instrument {
            asset_type: AssetType::Futures,
            inner: InstrumentEnum::Futures(FuturesInstrument {
                symbol: symbol.to_string(),
                multiplier: unit,
                margin_ratio,
                tick_size: Decimal::ONE,
                expiry_date: None,
                settlement_type: None,
                settlement_price: None,
            }),
        }
    }

    /// 期望值取自 SimNow 实测(交易日 20261120): CTP ReqQryOptionInstrTradeCost 以期权、
    /// 标的昨结算价计算的 Royalty + FixedMargin, 与实际卖开 1 手后的持仓占用一致。
    #[test]
    fn china_commodity_margin_matches_simnow_samples() {
        use crate::margin::MarginEngine;
        // (期权, 标的, 类型, 行权价, 交易单位, 标的保证金率, 标的价, 期权价, 1 手保证金)
        let cases = [
            ("eb2701-C-8000", "eb2701", OptionType::Call, dec!(8000), dec!(5), dec!(0.19), dec!(8468), dec!(607.5), dec!(11082.10)),
            ("eb2701-C-8500", "eb2701", OptionType::Call, dec!(8500), dec!(5), dec!(0.19), dec!(8468), dec!(313), dec!(9529.60)),
            ("eb2701-P-6000", "eb2701", OptionType::Put, dec!(6000), dec!(5), dec!(0.19), dec!(8468), dec!(2.5), dec!(4034.80)),
            ("FG701C910", "FG701", OptionType::Call, dec!(910), dec!(20), dec!(0.17), dec!(907), dec!(23), dec!(3513.80)),
            ("FG701P730", "FG701", OptionType::Put, dec!(730), dec!(20), dec!(0.17), dec!(907), dec!(0.5), dec!(1551.90)),
            ("ag2702C10300", "ag2702", OptionType::Call, dec!(10300), dec!(15), dec!(0.19), dec!(10806), dec!(1182), dec!(48527.10)),
            ("ag2702C22100", "ag2702", OptionType::Call, dec!(22100), dec!(15), dec!(0.19), dec!(10806), dec!(0.5), dec!(15406.05)),
            ("sc2701P610", "sc2701", OptionType::Put, dec!(610), dec!(1000), dec!(0.17), dec!(576.9), dec!(56.8), dec!(154873.00)),
            ("sc2701C580", "sc2701", OptionType::Call, dec!(580), dec!(1000), dec!(0.17), dec!(576.9), dec!(35.45), dec!(131973.00)),
            ("lc2704-P-89000", "lc2704", OptionType::Put, dec!(89000), dec!(1), dec!(0.09), dec!(89400), dec!(13610), dec!(21456.00)),
            ("lc2704-P-69000", "lc2704", OptionType::Put, dec!(69000), dec!(1), dec!(0.09), dec!(89400), dec!(4630), dec!(8653.00)),
        ];
        for (symbol, underlying, kind, strike, unit, ratio, f, p, expected) in cases {
            let option = exchange_option(
                OptionMarginModel::ChinaCommodity,
                dec!(0),
                symbol,
                underlying,
                kind,
                strike,
                unit,
            );
            let instruments = HashMap::from([
                (symbol.to_string(), option.clone()),
                (underlying.to_string(), futures(underlying, unit, ratio)),
            ]);
            let prices = HashMap::from([(symbol.to_string(), p), (underlying.to_string(), f)]);
            let one = MarginEngine::position_margin(dec!(-1), p, &option, &prices, &instruments, None);
            assert_eq!(one, expected, "{symbol}");
            let three =
                MarginEngine::position_margin(dec!(-3), p, &option, &prices, &instruments, None);
            assert_eq!(three, expected * dec!(3), "{symbol}");
            let long = MarginEngine::position_margin(dec!(2), p, &option, &prices, &instruments, None);
            assert_eq!(long, Decimal::ZERO, "{symbol}");
        }
    }

    #[test]
    fn china_commodity_inputs_report_exact_cause() {
        let option = exchange_option(
            OptionMarginModel::ChinaCommodity,
            dec!(0),
            "m2701-C-3000",
            "m2701",
            OptionType::Call,
            dec!(3000),
            dec!(10),
        );
        let priced = HashMap::from([("m2701".to_string(), dec!(3000))]);
        let cause = |instruments: &HashMap<String, Instrument>, prices: &HashMap<String, Decimal>| {
            exchange_option_inputs(&option, instruments, prices)
                .unwrap()
                .unwrap_err()
        };

        assert_eq!(
            cause(&HashMap::new(), &priced),
            "标的期货 m2701 未登记为合约, 取不到标的期货保证金率"
        );
        let mut stock = futures("m2701", dec!(10), dec!(0.1));
        stock.asset_type = AssetType::Stock;
        assert_eq!(
            cause(&HashMap::from([("m2701".to_string(), stock)]), &priced),
            "标的 m2701 登记的资产类型是 Stock, 商品期权保证金模型要求标的为期货"
        );
        let zero_ratio = HashMap::from([("m2701".to_string(), futures("m2701", dec!(10), dec!(0)))]);
        assert_eq!(cause(&zero_ratio, &priced), "标的期货 m2701 的保证金率 0 不为正");
        let registered = HashMap::from([("m2701".to_string(), futures("m2701", dec!(10), dec!(0.1)))]);
        assert_eq!(cause(&registered, &HashMap::new()), "标的期货 m2701 没有价格");
        let zero_price = HashMap::from([("m2701".to_string(), dec!(0))]);
        assert_eq!(cause(&registered, &zero_price), "标的期货 m2701 的价格 0 不为正");
    }

    /// 期望值取自宏源期货实盘账户的 CTP ReqQryOptionInstrTradeCost(交易日 20261123, 只查询):
    /// 输入期权昨结算价与给定标的指数价, 柜台与交易所口径一致, 合约保证金调整系数 12%。
    #[test]
    fn china_index_margin_matches_cffex_samples() {
        use crate::margin::MarginEngine;
        // (期权, 类型, 行权价, 期权价, 标的指数价, 1 手保证金)
        let cases = [
            ("IO2612-C-3650", OptionType::Call, dec!(3650), dec!(242.2), dec!(3866.4), dec!(70616.8)),
            ("IO2612-C-3850", OptionType::Call, dec!(3850), dec!(95.2), dec!(3673.1), dec!(35907.2)),
            ("IO2612-C-5600", OptionType::Call, dec!(5600), dec!(0.6), dec!(3866.4), dec!(23258.4)),
            ("IO2612-P-3550", OptionType::Put, dec!(3550), dec!(14.0), dec!(3866.4), dec!(22700.0)),
            ("IO2612-P-3550", OptionType::Put, dec!(3550), dec!(14.0), dec!(3673.1), dec!(33167.2)),
            ("MO2612-C-5600", OptionType::Call, dec!(5600), dec!(245.8), dec!(5287.3), dec!(56757.6)),
            ("MO2612-P-5300", OptionType::Put, dec!(5300), dec!(157.6), dec!(5565.6), dec!(55987.2)),
            ("HO2612-C-2700", OptionType::Call, dec!(2700), dec!(53.6), dec!(2716.8), dec!(37961.6)),
            ("HO2612-P-2850", OptionType::Put, dec!(2850), dec!(148.8), dec!(2581.0), dec!(45852.0)),
            ("HO2612-P-2450", OptionType::Put, dec!(2450), dec!(2.4), dec!(2716.8), dec!(14940.0)),
        ];
        for (symbol, kind, strike, p, s, expected) in cases {
            let option = exchange_option(
                OptionMarginModel::ChinaIndex,
                dec!(0.12),
                symbol,
                "IDX",
                kind,
                strike,
                dec!(100),
            );
            let instruments = HashMap::from([(symbol.to_string(), option.clone())]);
            let prices = HashMap::from([(symbol.to_string(), p), ("IDX".to_string(), s)]);
            let short = MarginEngine::position_margin(dec!(-2), p, &option, &prices, &instruments, None);
            assert_eq!(short, expected * dec!(2), "{symbol} @ {s}");
            let long = MarginEngine::position_margin(dec!(1), p, &option, &prices, &instruments, None);
            assert_eq!(long, Decimal::ZERO, "{symbol}");
        }
    }

    #[test]
    fn china_index_inputs_report_exact_cause() {
        let cause = |ratio: Decimal, prices: &HashMap<String, Decimal>| {
            let option = exchange_option(
                OptionMarginModel::ChinaIndex,
                ratio,
                "IO2612-C-3850",
                "000300",
                OptionType::Call,
                dec!(3850),
                dec!(100),
            );
            exchange_option_inputs(&option, &HashMap::new(), prices)
                .unwrap()
                .unwrap_err()
        };
        let priced = HashMap::from([("000300".to_string(), dec!(3866.4))]);
        assert_eq!(
            cause(dec!(0), &priced),
            "期权 IO2612-C-3850 的保证金调整系数(margin_ratio) 0 不为正"
        );
        assert_eq!(cause(dec!(0.12), &HashMap::new()), "标的指数 000300 没有价格");
    }
}
