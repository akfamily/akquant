use super::market_data::extract_decimal;
use super::types::{AssetType, OptionType};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 交易标的
///
/// :ivar symbol: 代码
/// :ivar asset_type: 资产类型
/// :ivar multiplier: 合约乘数
/// :ivar margin_ratio: 保证金比率
/// :ivar tick_size: 最小变动价位
pub struct Instrument {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub asset_type: AssetType,
    pub multiplier: Decimal,   // 合约乘数 (股票为1.0)
    pub margin_ratio: Decimal, // 保证金比率 (股票为1.0)
    pub tick_size: Decimal,    // 最小变动价位
    pub option_type: Option<OptionType>,
    pub strike_price: Option<Decimal>,
    pub expiry_date: Option<u32>,
    pub lot_size: Decimal,
}

#[gen_stub_pymethods]
#[pymethods]
impl Instrument {
    /// 创建交易标的
    ///
    /// :param symbol: 代码
    /// :param asset_type: 资产类型
    /// :param multiplier: 合约乘数
    /// :param margin_ratio: 保证金比率
    /// :param tick_size: 最小变动价位
    /// :param option_type: 期权类型 (可选)
    /// :param strike_price: 行权价 (可选)
    /// :param expiry_date: 到期日 (可选)
    /// :param lot_size: 最小交易单位 (可选, 默认为1)
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (symbol, asset_type, multiplier=None, margin_ratio=None, tick_size=None, option_type=None, strike_price=None, expiry_date=None, lot_size=None))]
    pub fn new(
        symbol: String,
        asset_type: AssetType,
        multiplier: Option<&Bound<'_, PyAny>>,
        margin_ratio: Option<&Bound<'_, PyAny>>,
        tick_size: Option<&Bound<'_, PyAny>>,
        option_type: Option<OptionType>,
        strike_price: Option<&Bound<'_, PyAny>>,
        expiry_date: Option<u32>,
        lot_size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mult = if let Some(m) = multiplier {
            extract_decimal(m)?
        } else {
            Decimal::from(1)
        };

        let margin = if let Some(m) = margin_ratio {
            extract_decimal(m)?
        } else {
            Decimal::from(1) // Default 100% margin (no leverage)
        };

        let tick = if let Some(t) = tick_size {
            extract_decimal(t)?
        } else {
            Decimal::new(1, 2) // Default 0.01
        };

        let strike = if let Some(s) = strike_price {
            Some(extract_decimal(s)?)
        } else {
            None
        };

        let lot = if let Some(l) = lot_size {
            extract_decimal(l)?
        } else {
            Decimal::from(1)
        };

        Ok(Instrument {
            symbol,
            asset_type,
            multiplier: mult,
            margin_ratio: margin,
            tick_size: tick,
            option_type,
            strike_price: strike,
            expiry_date,
            lot_size: lot,
        })
    }

    #[getter]
    fn get_multiplier(&self) -> f64 {
        self.multiplier.to_f64().unwrap_or_default()
    }

    #[getter]
    fn get_margin_ratio(&self) -> f64 {
        self.margin_ratio.to_f64().unwrap_or_default()
    }

    #[getter]
    fn get_tick_size(&self) -> f64 {
        self.tick_size.to_f64().unwrap_or_default()
    }

    #[getter]
    fn get_lot_size(&self) -> f64 {
        self.lot_size.to_f64().unwrap_or_default()
    }

    #[setter]
    fn set_lot_size(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.lot_size = extract_decimal(value)?;
        Ok(())
    }
}
