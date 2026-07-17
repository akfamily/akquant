use super::market_data::extract_decimal;
use super::types::{AssetType, OptionMarginModel, OptionType, SettlementType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};

const INSTRUMENT_VALIDATION_PREFIX: &str = "AKQ-INSTRUMENT-VALIDATION";

fn validation_err(message: &str) -> PyErr {
    PyValueError::new_err(format!("[{INSTRUMENT_VALIDATION_PREFIX}] {message}"))
}

fn ensure_positive(value: Decimal, field: &str) -> PyResult<()> {
    if value <= Decimal::ZERO {
        return Err(validation_err(&format!("{field} must be > 0")));
    }
    Ok(())
}

fn ensure_non_negative(value: Decimal, field: &str) -> PyResult<()> {
    if value < Decimal::ZERO {
        return Err(validation_err(&format!("{field} must be >= 0")));
    }
    Ok(())
}

fn validate_instrument_inputs(
    symbol: &str,
    asset_type: AssetType,
    multiplier: Decimal,
    margin_ratio: Decimal,
    tick_size: Decimal,
    lot_size: Decimal,
    strike_price: Option<Decimal>,
    underlying_symbol: Option<&str>,
    settlement_price: Option<Decimal>,
    reference_volatility: Option<Decimal>,
) -> PyResult<()> {
    if symbol.trim().is_empty() {
        return Err(validation_err("symbol must not be empty"));
    }
    ensure_positive(tick_size, "tick_size")?;
    ensure_positive(lot_size, "lot_size")?;
    ensure_positive(multiplier, "multiplier")?;
    ensure_non_negative(margin_ratio, "margin_ratio")?;
    if let Some(v) = settlement_price {
        ensure_positive(v, "settlement_price")?;
    }
    if let Some(v) = reference_volatility {
        ensure_positive(v, "reference_volatility")?;
    }
    if asset_type == AssetType::Option {
        if let Some(v) = strike_price {
            ensure_non_negative(v, "strike_price")?;
        }
        if let Some(us) = underlying_symbol {
            if us.trim().is_empty() {
                return Err(validation_err(
                    "underlying_symbol must not be empty for option",
                ));
            }
        } else {
            return Err(validation_err(
                "underlying_symbol must not be empty for option",
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockInstrument {
    pub symbol: String,
    pub lot_size: Decimal,
    pub tick_size: Decimal,
    pub expiry_date: Option<u32>,
    /// 买入份额需 N 个交易日后可卖(0=T+0,1=T+1)。
    #[serde(default)]
    pub sellable_after_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundInstrument {
    pub symbol: String,
    pub lot_size: Decimal,
    pub tick_size: Decimal,
    /// 买入份额需 N 个交易日后可卖(0=T+0,1=T+1)。
    #[serde(default)]
    pub sellable_after_days: u32,
    // Add other fund-specific fields if needed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuturesInstrument {
    pub symbol: String,
    pub multiplier: Decimal,
    pub margin_ratio: Decimal,
    pub tick_size: Decimal,
    pub expiry_date: Option<u32>,
    pub settlement_type: Option<SettlementType>,
    pub settlement_price: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionInstrument {
    pub symbol: String,
    pub multiplier: Decimal,
    pub margin_ratio: Decimal,
    pub tick_size: Decimal,
    pub option_margin_model: OptionMarginModel,
    pub option_type: OptionType,
    pub strike_price: Decimal,
    pub expiry_date: u32,
    pub underlying_symbol: String,
    pub settlement_type: Option<SettlementType>,
    pub implied_volatility: Option<Decimal>,
    pub reference_volatility: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoInstrument {
    pub symbol: String,
    pub lot_size: Decimal, // Usually small (e.g. 0.0001)
    pub tick_size: Decimal,
    pub multiplier: Decimal, // Usually 1.0 for Spot, but contract size for futures
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForexInstrument {
    pub symbol: String,
    pub lot_size: Decimal,   // Standard lot is 100,000 units
    pub tick_size: Decimal,  // Pip size (e.g. 0.0001)
    pub multiplier: Decimal, // Usually 1.0
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
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
    pub asset_type: AssetType,
    pub inner: InstrumentEnum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstrumentEnum {
    Stock(StockInstrument),
    Fund(FundInstrument),
    Futures(FuturesInstrument),
    Option(OptionInstrument),
    Crypto(CryptoInstrument),
    Forex(ForexInstrument),
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
    /// :param underlying_symbol: 标的代码 (可选)
    /// :param settlement_type: 结算方式 (可选)
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (symbol, asset_type, multiplier=None, margin_ratio=None, tick_size=None, option_type=None, strike_price=None, expiry_date=None, lot_size=None, underlying_symbol=None, settlement_type=None, settlement_price=None, option_margin_model=None, implied_volatility=None, reference_volatility=None, sellable_after_days=None))]
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
        underlying_symbol: Option<String>,
        settlement_type: Option<SettlementType>,
        settlement_price: Option<&Bound<'_, PyAny>>,
        option_margin_model: Option<OptionMarginModel>,
        implied_volatility: Option<&Bound<'_, PyAny>>,
        reference_volatility: Option<&Bound<'_, PyAny>>,
        sellable_after_days: Option<u32>,
    ) -> PyResult<Self> {
        let clean_symbol = symbol.trim().to_string();
        let sellable_after_days_val = sellable_after_days.unwrap_or(1);
        let multiplier_val = multiplier
            .map(extract_decimal)
            .transpose()?
            .unwrap_or(Decimal::ONE);
        let margin_val = margin_ratio
            .map(extract_decimal)
            .transpose()?
            .unwrap_or(Decimal::ONE);
        let tick_val = tick_size
            .map(extract_decimal)
            .transpose()?
            .unwrap_or(Decimal::new(1, 2));
        let lot_val = lot_size
            .map(extract_decimal)
            .transpose()?
            .unwrap_or(Decimal::ONE);
        let settlement_price_val = settlement_price.map(extract_decimal).transpose()?;
        let strike_price_val = strike_price.map(extract_decimal).transpose()?;
        let implied_volatility_val = implied_volatility.map(extract_decimal).transpose()?;
        let reference_volatility_val = reference_volatility.map(extract_decimal).transpose()?;
        let underlying_symbol_value = underlying_symbol.as_deref();
        validate_instrument_inputs(
            &clean_symbol,
            asset_type,
            multiplier_val,
            margin_val,
            tick_val,
            lot_val,
            strike_price_val,
            underlying_symbol_value,
            settlement_price_val,
            reference_volatility_val,
        )?;

        let inner = match asset_type {
            AssetType::Stock => InstrumentEnum::Stock(StockInstrument {
                symbol: clean_symbol.clone(),
                lot_size: lot_val,
                tick_size: tick_val,
                expiry_date,
                sellable_after_days: sellable_after_days_val,
            }),
            AssetType::Fund => InstrumentEnum::Fund(FundInstrument {
                symbol: clean_symbol.clone(),
                lot_size: lot_val,
                tick_size: tick_val,
                sellable_after_days: sellable_after_days_val,
            }),
            AssetType::Futures => InstrumentEnum::Futures(FuturesInstrument {
                symbol: clean_symbol.clone(),
                multiplier: multiplier_val,
                margin_ratio: margin_val,
                tick_size: tick_val,
                expiry_date,
                settlement_type,
                settlement_price: settlement_price_val,
            }),
            AssetType::Option => InstrumentEnum::Option(OptionInstrument {
                symbol: clean_symbol.clone(),
                multiplier: multiplier_val,
                margin_ratio: margin_val,
                tick_size: tick_val,
                option_margin_model: option_margin_model.unwrap_or_default(),
                option_type: option_type.unwrap_or(OptionType::Call),
                strike_price: strike_price_val.unwrap_or(Decimal::ZERO),
                expiry_date: expiry_date.unwrap_or(0),
                underlying_symbol: underlying_symbol.unwrap_or_default(),
                settlement_type,
                implied_volatility: implied_volatility_val,
                reference_volatility: reference_volatility_val,
            }),
            AssetType::Crypto => InstrumentEnum::Crypto(CryptoInstrument {
                symbol: clean_symbol.clone(),
                lot_size: lot_val,
                tick_size: tick_val,
                multiplier: multiplier_val,
            }),
            AssetType::Forex => InstrumentEnum::Forex(ForexInstrument {
                symbol: clean_symbol.clone(),
                lot_size: lot_val,
                tick_size: tick_val,
                multiplier: multiplier_val,
            }),
        };

        Ok(Instrument { asset_type, inner })
    }

    #[getter]
    pub fn get_symbol(&self) -> String {
        self.symbol().to_string()
    }

    #[getter]
    pub fn get_sellable_after_days(&self) -> u32 {
        self.sellable_after_days()
    }

    #[getter]
    pub fn get_multiplier(&self) -> f64 {
        self.multiplier().to_f64().unwrap_or(1.0)
    }

    #[getter]
    pub fn get_margin_ratio(&self) -> f64 {
        self.margin_ratio().to_f64().unwrap_or(1.0)
    }

    #[getter]
    pub fn get_lot_size(&self) -> f64 {
        self.lot_size().to_f64().unwrap_or(1.0)
    }

    #[getter]
    pub fn get_tick_size(&self) -> f64 {
        self.tick_size().to_f64().unwrap_or(0.01)
    }

    #[getter]
    pub fn get_option_margin_model(&self) -> Option<OptionMarginModel> {
        self.option_margin_model()
    }

    #[getter]
    pub fn get_implied_volatility(&self) -> Option<f64> {
        self.implied_volatility().and_then(|v| v.to_f64())
    }

    #[getter]
    pub fn get_reference_volatility(&self) -> Option<f64> {
        self.reference_volatility().and_then(|v| v.to_f64())
    }
}

// Add public accessors for internal Rust usage to avoid breaking changes everywhere immediately
impl Instrument {
    pub fn symbol(&self) -> &str {
        match &self.inner {
            InstrumentEnum::Stock(s) => &s.symbol,
            InstrumentEnum::Fund(f) => &f.symbol,
            InstrumentEnum::Futures(f) => &f.symbol,
            InstrumentEnum::Option(o) => &o.symbol,
            InstrumentEnum::Crypto(c) => &c.symbol,
            InstrumentEnum::Forex(f) => &f.symbol,
        }
    }

    pub fn multiplier(&self) -> Decimal {
        match &self.inner {
            InstrumentEnum::Futures(f) => f.multiplier,
            InstrumentEnum::Option(o) => o.multiplier,
            InstrumentEnum::Crypto(c) => c.multiplier,
            InstrumentEnum::Forex(f) => f.multiplier,
            _ => Decimal::ONE,
        }
    }

    pub fn margin_ratio(&self) -> Decimal {
        match &self.inner {
            InstrumentEnum::Futures(f) => f.margin_ratio,
            InstrumentEnum::Option(o) => o.margin_ratio,
            InstrumentEnum::Forex(_) => Decimal::new(1, 2), // 0.01 default for Forex
            _ => Decimal::ONE,
        }
    }

    pub fn lot_size(&self) -> Decimal {
        match &self.inner {
            InstrumentEnum::Stock(s) => s.lot_size,
            InstrumentEnum::Fund(f) => f.lot_size,
            InstrumentEnum::Crypto(c) => c.lot_size,
            InstrumentEnum::Forex(f) => f.lot_size,
            _ => Decimal::ONE,
        }
    }

    pub fn tick_size(&self) -> Decimal {
        match &self.inner {
            InstrumentEnum::Stock(s) => s.tick_size,
            InstrumentEnum::Fund(f) => f.tick_size,
            InstrumentEnum::Futures(f) => f.tick_size,
            InstrumentEnum::Option(o) => o.tick_size,
            InstrumentEnum::Crypto(c) => c.tick_size,
            InstrumentEnum::Forex(f) => f.tick_size,
        }
    }

    pub fn expiry_date(&self) -> Option<u32> {
        match &self.inner {
            InstrumentEnum::Stock(s) => s.expiry_date,
            InstrumentEnum::Futures(f) => f.expiry_date,
            InstrumentEnum::Option(o) => Some(o.expiry_date),
            _ => None,
        }
    }

    pub fn underlying_symbol(&self) -> Option<&String> {
        match &self.inner {
            InstrumentEnum::Option(o) => Some(&o.underlying_symbol),
            _ => None,
        }
    }

    pub fn strike_price(&self) -> Option<Decimal> {
        match &self.inner {
            InstrumentEnum::Option(o) => Some(o.strike_price),
            _ => None,
        }
    }

    pub fn option_type(&self) -> Option<OptionType> {
        match &self.inner {
            InstrumentEnum::Option(o) => Some(o.option_type),
            _ => None,
        }
    }

    pub fn option_margin_model(&self) -> Option<OptionMarginModel> {
        match &self.inner {
            InstrumentEnum::Option(o) => Some(o.option_margin_model),
            _ => None,
        }
    }

    pub fn implied_volatility(&self) -> Option<Decimal> {
        match &self.inner {
            InstrumentEnum::Option(o) => o.implied_volatility,
            _ => None,
        }
    }

    pub fn reference_volatility(&self) -> Option<Decimal> {
        match &self.inner {
            InstrumentEnum::Option(o) => o.reference_volatility,
            _ => None,
        }
    }

    pub fn settlement_type(&self) -> Option<SettlementType> {
        match &self.inner {
            InstrumentEnum::Futures(f) => f.settlement_type,
            InstrumentEnum::Option(o) => o.settlement_type,
            _ => None,
        }
    }

    pub fn settlement_price(&self) -> Option<Decimal> {
        match &self.inner {
            InstrumentEnum::Futures(f) => f.settlement_price,
            _ => None,
        }
    }

    /// 买入份额需 N 个交易日后可卖(0=T+0,1=T+1)。非股票/基金资产返回 0(T+0)。
    pub fn sellable_after_days(&self) -> u32 {
        match &self.inner {
            InstrumentEnum::Stock(s) => s.sellable_after_days,
            InstrumentEnum::Fund(f) => f.sellable_after_days,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instrument_new_rejects_empty_symbol() {
        let result = validate_instrument_inputs(
            "   ",
            AssetType::Stock,
            Decimal::ONE,
            Decimal::ONE,
            Decimal::new(1, 2),
            Decimal::ONE,
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_instrument_new_rejects_non_positive_tick_size() {
        let result = validate_instrument_inputs(
            "AAPL",
            AssetType::Stock,
            Decimal::ONE,
            Decimal::ONE,
            Decimal::ZERO,
            Decimal::ONE,
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_instrument_new_rejects_option_without_underlying() {
        let result = Instrument::new(
            "OPT".to_string(),
            AssetType::Option,
            None,
            None,
            None,
            None,
            None,
            Some(20260101),
            None,
            Some("".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }
}
