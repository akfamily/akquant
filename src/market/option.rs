use crate::model::OrderSide;
use rust_decimal::Decimal;
use std::collections::HashMap;

/// 期权费用配置(按张计费)。缺省值取沪深 ETF 期权现行口径:
/// 交易经手费 1.3 元/张、交易结算费 0.3 元/张(均双向收取, 卖出开仓含备兑开仓暂免),
/// 行权结算费 0.6 元/张(向行权方收取)。券商佣金由券商与客户约定, 缺省 5 元/张。
#[derive(Clone, Debug)]
pub struct OptionConfig {
    /// 券商佣金(元/张), 卖出开仓照收
    pub commission_per_contract: Decimal,
    /// 交易所交易经手费(元/张)
    pub exchange_fee_per_contract: Decimal,
    /// 中国结算交易结算费(元/张)
    pub clearing_fee_per_contract: Decimal,
    /// 行权结算费(元/张), 到期结算时向实值被行权的多头收取
    pub exercise_fee_per_contract: Decimal,
    /// 卖出开仓免收经手费与结算费。交易所目前是"暂免", 恢复收费时关掉即可
    pub sell_open_exempt: bool,
}

impl Default for OptionConfig {
    fn default() -> Self {
        Self {
            commission_per_contract: Decimal::from(5),
            exchange_fee_per_contract: Decimal::new(13, 1),
            clearing_fee_per_contract: Decimal::new(3, 1),
            exercise_fee_per_contract: Decimal::new(6, 1),
            sell_open_exempt: true,
        }
    }
}

/// 计算期权交易费用(按张收取)。
///
/// `position_before` 是这笔成交之前的带符号持仓, 用来区分卖出开仓: 卖单先平掉多头
/// (`min(quantity, max(position_before, 0))` 张, 全额计费), 剩余部分是卖出开仓。
pub fn calculate_commission(
    config: &OptionConfig,
    side: OrderSide,
    quantity: Decimal,
    position_before: Decimal,
) -> Decimal {
    let full_per_contract = config.commission_per_contract
        + config.exchange_fee_per_contract
        + config.clearing_fee_per_contract;
    let sell_open_qty = if side == OrderSide::Sell && config.sell_open_exempt {
        (quantity - position_before.max(Decimal::ZERO)).max(Decimal::ZERO)
    } else {
        Decimal::ZERO
    };
    (quantity - sell_open_qty) * full_per_contract + sell_open_qty * config.commission_per_contract
}

/// 更新期权可用持仓 (T+0)
pub fn update_available_position(
    _config: &OptionConfig,
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
            if let Some(pos) = available_positions.get_mut(symbol) {
                *pos += quantity;
            }
        }
        OrderSide::Sell => {
            available_positions
                .entry(symbol.to_string())
                .or_insert(Decimal::ZERO);
            if let Some(pos) = available_positions.get_mut(symbol) {
                *pos -= quantity;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn cfg() -> OptionConfig {
        OptionConfig::default()
    }

    #[test]
    fn default_fees_follow_sse_etf_option_schedule() {
        let c = cfg();
        assert_eq!(c.commission_per_contract, dec!(5));
        assert_eq!(c.exchange_fee_per_contract, dec!(1.3));
        assert_eq!(c.clearing_fee_per_contract, dec!(0.3));
        assert_eq!(c.exercise_fee_per_contract, dec!(0.6));
        assert!(c.sell_open_exempt);
    }

    #[test]
    fn buy_open_pays_full_fee() {
        // 2 张 × (5 + 1.3 + 0.3)
        assert_eq!(
            calculate_commission(&cfg(), OrderSide::Buy, dec!(2), dec!(0)),
            dec!(13.2)
        );
    }

    #[test]
    fn buy_to_close_short_pays_full_fee() {
        assert_eq!(
            calculate_commission(&cfg(), OrderSide::Buy, dec!(3), dec!(-3)),
            dec!(19.8)
        );
    }

    #[test]
    fn sell_open_only_pays_commission() {
        assert_eq!(
            calculate_commission(&cfg(), OrderSide::Sell, dec!(3), dec!(0)),
            dec!(15)
        );
    }

    #[test]
    fn sell_to_close_long_pays_full_fee() {
        assert_eq!(
            calculate_commission(&cfg(), OrderSide::Sell, dec!(2), dec!(5)),
            dec!(13.2)
        );
    }

    #[test]
    fn sell_crossing_zero_splits_close_and_open() {
        // 持多 2 张卖 5 张: 2 张平仓全额 + 3 张卖开只收佣金 = 13.2 + 15
        assert_eq!(
            calculate_commission(&cfg(), OrderSide::Sell, dec!(5), dec!(2)),
            dec!(28.2)
        );
    }

    #[test]
    fn sell_open_from_existing_short_is_exempt() {
        // 已持空 4 张再卖 1 张, 仍是卖出开仓
        assert_eq!(
            calculate_commission(&cfg(), OrderSide::Sell, dec!(1), dec!(-4)),
            dec!(5)
        );
    }

    #[test]
    fn sell_open_exempt_disabled_charges_full_fee() {
        let c = OptionConfig {
            sell_open_exempt: false,
            ..OptionConfig::default()
        };
        assert_eq!(
            calculate_commission(&c, OrderSide::Sell, dec!(3), dec!(0)),
            dec!(19.8)
        );
    }
}
