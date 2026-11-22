//! 期货交易所期权组合保证金(大商所、广期所、郑商所).
//!
//! 只对保证金模型为 `ChinaCommodityCombination` 的期权生效。按交易所规定的组合策略
//! 优先级逐项配对(与交易所结算时组合的做法相同, 不求全局最优), 同一策略内按每组节省
//! 的保证金从大到小配对。总占用 = 各合约单腿保证金之和 - 组合优惠。
//!
//! 策略与公式(每组各腿 1 手, 期权保证金含权利金, 权利金 = 期权价 × 交易单位):
//!
//! | 策略 | 腿 | 组合保证金 |
//! |---|---|---|
//! | 买入垂直价差 | 买低价认购 + 卖高价认购; 买高价认沽 + 卖低价认沽 | X × 卖期权保证金 |
//! | 卖出期权期货组合 | 卖认购 + 买期货; 卖认沽 + 卖期货 | 期货保证金 + 期权权利金 |
//! | 期权跨式 | 卖同执行价认购、认沽 | max(两腿保证金) + 另一腿权利金 |
//! | 期权宽跨式 | 卖低价认沽 + 卖高价认购 | 同跨式 |
//! | 卖出垂直价差 | 卖低价认购 + 买高价认购; 卖高价认沽 + 买低价认沽 | min(执行价差 × 交易单位, 卖期权保证金) |
//! | 买入期权期货组合 | 买认购 + 卖期货; 买认沽 + 买期货 | X × 期货保证金 |
//!
//! 跨式、宽跨式两腿保证金相等时为 max(两腿保证金) + max(两腿权利金)。
//!
//! 优先级: 大商所(2026-05-21 结算时起)买入垂直价差、卖出期权期货组合、跨式、宽跨式、
//! 卖出垂直价差、买入期权期货组合; 广期所卖出期权期货组合、跨式、宽跨式、买入垂直
//! 价差、卖出垂直价差; 郑商所卖出期权期货组合、跨式、宽跨式。期货对锁、跨期、跨品种
//! 与期权对锁不在此计算(持仓按合约净额记账, 同一合约不会同时有多空)。
//!
//! X 取期货公司 CTP `ReqQryCombPromotionParam` 的查询值(2026-11): 大商所、广期所买入
//! 垂直价差 0.2, 大商所买入期权期货组合 0.8。

use std::collections::HashMap;

use rust_decimal::Decimal;

use super::MarginEngine;
use crate::model::{AssetType, Instrument, OptionMarginModel, OptionType};

/// 期权组合保证金规则所属交易所.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComboExchange {
    Dce,
    Gfex,
    Czce,
    /// 上期所、上期能源: 期权没有组合保证金优惠
    Shfe,
}

/// 有商品期权的标的期货品种(期货公司 CTP 合约查询, 2026-11)
const DCE_PRODUCTS: &[&str] = &[
    "a", "b", "bz", "c", "cs", "eb", "eg", "i", "j", "jd", "jm", "l", "lg", "lh", "m", "p", "pg",
    "pp", "v", "y",
];
const GFEX_PRODUCTS: &[&str] = &["lc", "pd", "ps", "pt", "si"];
const CZCE_PRODUCTS: &[&str] = &[
    "AP", "CF", "CJ", "FG", "MA", "OI", "PF", "PK", "PL", "PR", "PX", "RM", "SA", "SF", "SH", "SM",
    "SO", "SR", "TA", "UR", "ZC",
];
const SHFE_INE_PRODUCTS: &[&str] = &[
    "ad", "ag", "al", "ao", "au", "br", "bu", "cu", "fu", "hc", "ni", "op", "pb", "rb", "ru", "sn",
    "sp", "ss", "zn", "bc", "lu", "nr", "sc",
];

const BUY_VERTICAL_X: Decimal = Decimal::from_parts(2, 0, 0, false, 1);
const DCE_BUY_OPTION_FUTURES_X: Decimal = Decimal::from_parts(8, 0, 0, false, 1);

/// 期货合约代码的品种部分(开头的字母).
pub fn product_of(symbol: &str) -> &str {
    let end = symbol
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(symbol.len());
    &symbol[..end]
}

/// 标的期货所属的期权组合保证金交易所; 不在对照表中为 `None`.
pub fn combination_exchange(underlying: &str) -> Option<ComboExchange> {
    let product = product_of(underlying);
    if DCE_PRODUCTS.contains(&product) {
        Some(ComboExchange::Dce)
    } else if GFEX_PRODUCTS.contains(&product) {
        Some(ComboExchange::Gfex)
    } else if CZCE_PRODUCTS.contains(&product) {
        Some(ComboExchange::Czce)
    } else if SHFE_INE_PRODUCTS.contains(&product) {
        Some(ComboExchange::Shfe)
    } else {
        None
    }
}

/// 标的期货所属的期权组合保证金交易所代码(`DCE` / `GFEX` / `CZCE` / `SHFE`), 不在对照表中
/// 为 None; 供 Python 侧校验 `CHINA_COMMODITY_COMBINATION` 配置.
#[pyo3::pyfunction]
pub fn _option_combination_exchange(underlying: &str) -> Option<&'static str> {
    combination_exchange(underlying).map(|exchange| match exchange {
        ComboExchange::Dce => "DCE",
        ComboExchange::Gfex => "GFEX",
        ComboExchange::Czce => "CZCE",
        ComboExchange::Shfe => "SHFE",
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    BuyVertical,
    SellOptionFutures,
    Straddle,
    Strangle,
    SellVertical,
    BuyOptionFutures,
}

fn priority(exchange: ComboExchange) -> &'static [Strategy] {
    use Strategy::*;
    match exchange {
        ComboExchange::Dce => &[
            BuyVertical,
            SellOptionFutures,
            Straddle,
            Strangle,
            SellVertical,
            BuyOptionFutures,
        ],
        ComboExchange::Gfex => &[
            SellOptionFutures,
            Straddle,
            Strangle,
            BuyVertical,
            SellVertical,
        ],
        ComboExchange::Czce => &[SellOptionFutures, Straddle, Strangle],
        ComboExchange::Shfe => &[],
    }
}

/// 一条期权腿(同一合约的持仓), 数量为剩余可配对的手数.
#[derive(Debug, Clone)]
struct OptionLeg {
    symbol: String,
    short: bool,
    call: bool,
    strike: Decimal,
    expiry: Option<u32>,
    multiplier: Decimal,
    /// 每手权利金
    premium: Decimal,
    /// 每手卖方保证金(含权利金); 多头为 0
    margin: Decimal,
    remaining: Decimal,
}

/// 同一标的期货下的持仓: 期权腿与期货净持仓.
#[derive(Debug, Default)]
struct UnderlyingBook {
    exchange: Option<ComboExchange>,
    options: Vec<OptionLeg>,
    /// 期货净持仓(多正空负)中剩余可配对的手数
    futures_long: Decimal,
    futures_short: Decimal,
    /// 每手期货保证金
    futures_margin: Decimal,
}

fn per_lot_margin(
    quantity: Decimal,
    price: Decimal,
    instrument: &Instrument,
    prices: &HashMap<String, Decimal>,
    instruments: &HashMap<String, Instrument>,
) -> Option<Decimal> {
    let margin =
        MarginEngine::position_margin(quantity, price, instrument, prices, instruments, None);
    (margin != Decimal::MAX).then_some(margin)
}

fn collect_books(
    positions: &HashMap<String, Decimal>,
    prices: &HashMap<String, Decimal>,
    instruments: &HashMap<String, Instrument>,
) -> HashMap<String, UnderlyingBook> {
    let mut books: HashMap<String, UnderlyingBook> = HashMap::new();
    for (symbol, &quantity) in positions {
        if quantity.is_zero() {
            continue;
        }
        let Some(instrument) = instruments.get(symbol) else {
            continue;
        };
        if instrument.asset_type != AssetType::Option
            || instrument.option_margin_model()
                != Some(OptionMarginModel::ChinaCommodityCombination)
        {
            continue;
        }
        let (Some(option_type), Some(strike), Some(underlying), Some(&price)) = (
            instrument.option_type(),
            instrument.strike_price(),
            instrument.underlying_symbol(),
            prices.get(symbol),
        ) else {
            continue;
        };
        let margin = if quantity < Decimal::ZERO {
            match per_lot_margin(-Decimal::ONE, price, instrument, prices, instruments) {
                Some(m) => m,
                None => continue,
            }
        } else {
            Decimal::ZERO
        };
        let book = books.entry(underlying.clone()).or_default();
        book.exchange = combination_exchange(underlying);
        book.options.push(OptionLeg {
            symbol: symbol.clone(),
            short: quantity < Decimal::ZERO,
            call: option_type == OptionType::Call,
            strike,
            expiry: instrument.expiry_date(),
            multiplier: instrument.multiplier(),
            premium: price.abs() * instrument.multiplier(),
            margin,
            remaining: quantity.abs(),
        });
    }
    for (underlying, book) in books.iter_mut() {
        let quantity = positions.get(underlying).copied().unwrap_or(Decimal::ZERO);
        let (Some(instrument), Some(&price)) =
            (instruments.get(underlying), prices.get(underlying))
        else {
            continue;
        };
        if quantity.is_zero() || instrument.asset_type != AssetType::Futures {
            continue;
        }
        let Some(margin) = per_lot_margin(Decimal::ONE, price, instrument, prices, instruments)
        else {
            continue;
        };
        book.futures_margin = margin;
        if quantity > Decimal::ZERO {
            book.futures_long = quantity;
        } else {
            book.futures_short = -quantity;
        }
    }
    books
}

/// 跨式、宽跨式每组保证金.
fn straddle_margin(call: &OptionLeg, put: &OptionLeg) -> Decimal {
    if call.margin == put.margin {
        call.margin + call.premium.max(put.premium)
    } else if call.margin > put.margin {
        call.margin + put.premium
    } else {
        put.margin + call.premium
    }
}

/// 两条期权腿组成 `strategy` 时每组节省的保证金; 不构成该策略为 `None`.
fn option_pair_saving(strategy: Strategy, a: &OptionLeg, b: &OptionLeg) -> Option<Decimal> {
    if a.expiry != b.expiry {
        return None;
    }
    match strategy {
        Strategy::Straddle | Strategy::Strangle => {
            let (call, put) = match (a.call, b.call) {
                (true, false) => (a, b),
                (false, true) => (b, a),
                _ => return None,
            };
            let shape = if strategy == Strategy::Straddle {
                call.strike == put.strike
            } else {
                put.strike < call.strike
            };
            if !(call.short && put.short && shape) {
                return None;
            }
            Some(call.margin + put.margin - straddle_margin(call, put))
        }
        Strategy::BuyVertical | Strategy::SellVertical => {
            if a.call != b.call || a.short == b.short {
                return None;
            }
            let (short, long) = if a.short { (a, b) } else { (b, a) };
            // 认购: 买入价差为买低卖高; 认沽: 买入价差为买高卖低
            let bullish_call = short.call && long.strike < short.strike;
            let bearish_put = !short.call && long.strike > short.strike;
            let buy_spread = bullish_call || bearish_put;
            let sell_spread = (short.call && long.strike > short.strike)
                || (!short.call && long.strike < short.strike);
            let combo = match strategy {
                Strategy::BuyVertical if buy_spread => BUY_VERTICAL_X * short.margin,
                Strategy::SellVertical if sell_spread => {
                    ((short.strike - long.strike).abs() * short.multiplier).min(short.margin)
                }
                _ => return None,
            };
            Some(short.margin - combo)
        }
        Strategy::SellOptionFutures | Strategy::BuyOptionFutures => None,
    }
}

/// 期权腿与期货组成 `strategy` 时每组节省的保证金及所用期货方向(多为 true).
fn futures_pair_saving(
    strategy: Strategy,
    leg: &OptionLeg,
    futures_margin: Decimal,
) -> Option<(Decimal, bool)> {
    match strategy {
        // 卖认购 + 买期货; 卖认沽 + 卖期货
        Strategy::SellOptionFutures if leg.short => Some((leg.margin - leg.premium, leg.call)),
        // 买认购 + 卖期货; 买认沽 + 买期货
        Strategy::BuyOptionFutures if !leg.short => Some((
            futures_margin - DCE_BUY_OPTION_FUTURES_X * futures_margin,
            !leg.call,
        )),
        _ => None,
    }
}

fn match_book(book: &mut UnderlyingBook) -> Decimal {
    let Some(exchange) = book.exchange else {
        return Decimal::ZERO;
    };
    let mut discount = Decimal::ZERO;
    for &strategy in priority(exchange) {
        if matches!(
            strategy,
            Strategy::SellOptionFutures | Strategy::BuyOptionFutures
        ) {
            let mut candidates: Vec<(Decimal, usize, bool)> = book
                .options
                .iter()
                .enumerate()
                .filter_map(|(i, leg)| {
                    futures_pair_saving(strategy, leg, book.futures_margin)
                        .map(|(saving, long)| (saving, i, long))
                })
                .filter(|(saving, _, _)| *saving > Decimal::ZERO)
                .collect();
            candidates.sort_by(|x, y| {
                y.0.cmp(&x.0)
                    .then_with(|| book.options[x.1].symbol.cmp(&book.options[y.1].symbol))
            });
            for (saving, i, long) in candidates {
                let futures = if long {
                    &mut book.futures_long
                } else {
                    &mut book.futures_short
                };
                let units = book.options[i].remaining.min(*futures);
                if units <= Decimal::ZERO {
                    continue;
                }
                book.options[i].remaining -= units;
                *futures -= units;
                discount += saving * units;
            }
            continue;
        }
        let mut candidates: Vec<(Decimal, usize, usize)> = Vec::new();
        for i in 0..book.options.len() {
            for j in (i + 1)..book.options.len() {
                if let Some(saving) =
                    option_pair_saving(strategy, &book.options[i], &book.options[j])
                    && saving > Decimal::ZERO
                {
                    candidates.push((saving, i, j));
                }
            }
        }
        candidates.sort_by(|x, y| {
            y.0.cmp(&x.0).then_with(|| {
                (&book.options[x.1].symbol, &book.options[x.2].symbol)
                    .cmp(&(&book.options[y.1].symbol, &book.options[y.2].symbol))
            })
        });
        for (saving, i, j) in candidates {
            let units = book.options[i].remaining.min(book.options[j].remaining);
            if units <= Decimal::ZERO {
                continue;
            }
            book.options[i].remaining -= units;
            book.options[j].remaining -= units;
            discount += saving * units;
        }
    }
    discount
}

/// 组合保证金优惠: 按交易所组合策略配对后, 各组节省的保证金之和(不小于 0).
pub fn combination_discount(
    positions: &HashMap<String, Decimal>,
    prices: &HashMap<String, Decimal>,
    instruments: &HashMap<String, Instrument>,
) -> Decimal {
    let mut books = collect_books(positions, prices, instruments);
    let mut underlyings: Vec<String> = books.keys().cloned().collect();
    underlyings.sort();
    underlyings
        .iter()
        .map(|u| match_book(books.get_mut(u).expect("key from map")))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::instrument::{FuturesInstrument, InstrumentEnum, OptionInstrument};
    use rust_decimal_macros::dec;

    fn futures(symbol: &str) -> Instrument {
        Instrument {
            asset_type: AssetType::Futures,
            inner: InstrumentEnum::Futures(FuturesInstrument {
                symbol: symbol.to_string(),
                multiplier: dec!(10),
                margin_ratio: dec!(0.1),
                tick_size: dec!(1),
                expiry_date: None,
                settlement_type: None,
                settlement_price: None,
            }),
        }
    }

    fn option(
        underlying: &str,
        call: bool,
        strike: Decimal,
        model: OptionMarginModel,
    ) -> Instrument {
        let flag = if call { "C" } else { "P" };
        Instrument {
            asset_type: AssetType::Option,
            inner: InstrumentEnum::Option(OptionInstrument {
                symbol: format!("{underlying}-{flag}-{strike}"),
                multiplier: dec!(10),
                margin_ratio: dec!(1),
                tick_size: dec!(0.5),
                option_margin_model: model,
                option_type: if call {
                    OptionType::Call
                } else {
                    OptionType::Put
                },
                strike_price: strike,
                expiry_date: 20251207,
                underlying_symbol: underlying.to_string(),
                settlement_type: None,
                settlement_price: None,
                implied_volatility: None,
                reference_volatility: None,
            }),
        }
    }

    /// 标的 3000(每手期货保证金 3000); 单腿卖方保证金: C3000 价 100 -> 4000,
    /// P3000 价 80 -> 3800, C3100 价 60 -> 3100, P2900 价 40 -> 2900。
    fn used(underlying: &str, model: OptionMarginModel, holdings: &[(&str, Decimal)]) -> Decimal {
        let mut instruments = HashMap::new();
        let mut prices = HashMap::new();
        instruments.insert(underlying.to_string(), futures(underlying));
        prices.insert(underlying.to_string(), dec!(3000));
        for (call, strike, price) in [
            (true, dec!(3000), dec!(100)),
            (false, dec!(3000), dec!(80)),
            (true, dec!(3100), dec!(60)),
            (false, dec!(2900), dec!(40)),
        ] {
            let inst = option(underlying, call, strike, model);
            prices.insert(inst.symbol().to_string(), price);
            instruments.insert(inst.symbol().to_string(), inst);
        }
        let positions: HashMap<String, Decimal> = holdings
            .iter()
            .map(|(leg, qty)| {
                let symbol = if *leg == "F" {
                    underlying.to_string()
                } else {
                    format!("{underlying}-{leg}")
                };
                (symbol, *qty)
            })
            .collect();
        MarginEngine::used_margin(&positions, &prices, &instruments, None)
    }

    const COMBO: OptionMarginModel = OptionMarginModel::ChinaCommodityCombination;

    #[test]
    fn single_leg_margins_match_commodity_formula() {
        assert_eq!(used("m2601", COMBO, &[("C-3000", dec!(-1))]), dec!(4000));
        assert_eq!(used("m2601", COMBO, &[("P-3000", dec!(-1))]), dec!(3800));
        assert_eq!(used("m2601", COMBO, &[("C-3100", dec!(-1))]), dec!(3100));
        assert_eq!(used("m2601", COMBO, &[("P-2900", dec!(-1))]), dec!(2900));
    }

    #[test]
    fn dce_strategy_formulas() {
        // 跨式: max(4000, 3800) + 认沽权利金 800
        let straddle = [("C-3000", dec!(-2)), ("P-3000", dec!(-2))];
        assert_eq!(used("m2601", COMBO, &straddle), dec!(9600));
        // 宽跨式: max(3100, 2900) + 认沽权利金 400
        let strangle = [("C-3100", dec!(-1)), ("P-2900", dec!(-1))];
        assert_eq!(used("m2601", COMBO, &strangle), dec!(3500));
        // 卖出垂直价差: min(执行价差 100 × 10, 4000)
        let sell_vertical = [("C-3000", dec!(-1)), ("C-3100", dec!(1))];
        assert_eq!(used("m2601", COMBO, &sell_vertical), dec!(1000));
        // 买入垂直价差: 0.2 × 3100
        let buy_vertical = [("C-3000", dec!(1)), ("C-3100", dec!(-1))];
        assert_eq!(used("m2601", COMBO, &buy_vertical), dec!(620));
        // 卖出期权期货组合: 期货 3000 + 认购权利金 1000
        let covered = [("C-3000", dec!(-1)), ("F", dec!(1))];
        assert_eq!(used("m2601", COMBO, &covered), dec!(4000));
        // 买入期权期货组合: 0.8 × 3000
        let protective = [("C-3000", dec!(1)), ("F", dec!(-1))];
        assert_eq!(used("m2601", COMBO, &protective), dec!(2400));
        // 数量不等: 1 组跨式 + 1 手认购单腿
        let partial = [("C-3000", dec!(-2)), ("P-3000", dec!(-1))];
        assert_eq!(used("m2601", COMBO, &partial), dec!(8800));
    }

    #[test]
    fn dce_follows_priority_not_global_optimum() {
        // 买入垂直价差(优先级高)先配: 620 + 期货单腿 3000;
        // 若先配卖出期权期货组合会是 3000 + 600 = 3600, 交易所不这样配
        let holdings = [("C-3000", dec!(1)), ("C-3100", dec!(-1)), ("F", dec!(1))];
        assert_eq!(used("m2601", COMBO, &holdings), dec!(3620));
    }

    #[test]
    fn exchange_specific_strategies() {
        // 广期所没有买入期权期货组合
        let protective = [("C-3000", dec!(1)), ("F", dec!(-1))];
        assert_eq!(used("lc2601", COMBO, &protective), dec!(3000));
        // 郑商所有跨式, 没有垂直价差
        let straddle = [("C-3000", dec!(-1)), ("P-3000", dec!(-1))];
        assert_eq!(used("SR601", COMBO, &straddle), dec!(4800));
        let sell_vertical = [("C-3000", dec!(-1)), ("C-3100", dec!(1))];
        assert_eq!(used("SR601", COMBO, &sell_vertical), dec!(4000));
        // 上期所期权没有组合优惠
        assert_eq!(used("cu2601", COMBO, &straddle), dec!(7800));
        // 未开启组合保证金的期权不参与
        let plain = OptionMarginModel::ChinaCommodity;
        assert_eq!(used("m2601", plain, &straddle), dec!(7800));
    }

    #[test]
    fn exchange_lookup_by_product() {
        assert_eq!(product_of("SR601"), "SR");
        assert_eq!(combination_exchange("m2601"), Some(ComboExchange::Dce));
        assert_eq!(combination_exchange("ps2606"), Some(ComboExchange::Gfex));
        assert_eq!(combination_exchange("TA601"), Some(ComboExchange::Czce));
        assert_eq!(combination_exchange("sc2601"), Some(ComboExchange::Shfe));
        assert_eq!(combination_exchange("xx2601"), None);
    }
}
