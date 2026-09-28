//! 按品种覆盖费用规则, 对应 `InstrumentConfig` 的 commission_rate / min_commission /
//! stamp_tax_rate / transfer_fee_rate。未设置的项沿用市场配置。

use rust_decimal::Decimal;

use super::fund::FundConfig;
use super::futures::FuturesConfig;
use super::simple::SimpleMarketConfig;
use super::stock::StockConfig;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeeOverride {
    pub commission_rate: Option<Decimal>,
    pub min_commission: Option<Decimal>,
    pub stamp_tax: Option<Decimal>,
    pub transfer_fee: Option<Decimal>,
}

impl FeeOverride {
    pub fn is_empty(&self) -> bool {
        self.commission_rate.is_none()
            && self.min_commission.is_none()
            && self.stamp_tax.is_none()
            && self.transfer_fee.is_none()
    }

    pub fn apply_stock(&self, base: &StockConfig) -> StockConfig {
        let mut c = base.clone();
        if let Some(v) = self.commission_rate {
            c.commission_rate = v;
        }
        if let Some(v) = self.min_commission {
            c.min_commission = v;
        }
        if let Some(v) = self.stamp_tax {
            c.stamp_tax = v;
        }
        if let Some(v) = self.transfer_fee {
            c.transfer_fee = v;
        }
        c
    }

    pub fn apply_fund(&self, base: &FundConfig) -> FundConfig {
        let mut c = base.clone();
        if let Some(v) = self.commission_rate {
            c.commission_rate = v;
        }
        if let Some(v) = self.min_commission {
            c.min_commission = v;
        }
        if let Some(v) = self.stamp_tax {
            c.stamp_tax = v;
        }
        if let Some(v) = self.transfer_fee {
            c.transfer_fee = v;
        }
        c
    }

    pub fn apply_futures(&self, base: &FuturesConfig) -> FuturesConfig {
        let mut c = base.clone();
        if let Some(v) = self.commission_rate {
            c.commission_rate = v;
        }
        c
    }

    pub fn apply_simple(&self, base: &SimpleMarketConfig) -> SimpleMarketConfig {
        let mut c = base.clone();
        if let Some(v) = self.commission_rate {
            c.commission_rate = v;
        }
        if let Some(v) = self.min_commission {
            c.min_commission = v;
        }
        if let Some(v) = self.stamp_tax {
            c.stamp_tax = v;
        }
        if let Some(v) = self.transfer_fee {
            c.transfer_fee = v;
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn unset_fields_keep_base_values() {
        let base = StockConfig::default();
        let ov = FeeOverride {
            commission_rate: Some(dec!(0.0001)),
            ..Default::default()
        };
        let c = ov.apply_stock(&base);
        assert_eq!(c.commission_rate, dec!(0.0001));
        assert_eq!(c.min_commission, base.min_commission);
        assert_eq!(c.stamp_tax, base.stamp_tax);
    }

    #[test]
    fn fund_override_sets_min_commission() {
        let ov = FeeOverride {
            min_commission: Some(dec!(0)),
            ..Default::default()
        };
        assert_eq!(
            ov.apply_fund(&FundConfig::default()).min_commission,
            dec!(0)
        );
    }
}
