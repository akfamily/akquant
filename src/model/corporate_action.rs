use chrono::NaiveDate;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

const CORP_ACTION_VALIDATION_PREFIX: &str = "AKQ-CORP-ACTION-VALIDATION";
const MIN_SPLIT_RATIO_SCALE: i64 = 1;
const MIN_SPLIT_RATIO_DP: u32 = 4;
const MAX_SPLIT_RATIO_SCALE: i64 = 10000;
const MAX_SPLIT_RATIO_DP: u32 = 0;

fn validation_err(message: &str) -> PyErr {
    PyValueError::new_err(format!("[{CORP_ACTION_VALIDATION_PREFIX}] {message}"))
}

fn split_ratio_bounds() -> (Decimal, Decimal) {
    (
        Decimal::new(MIN_SPLIT_RATIO_SCALE, MIN_SPLIT_RATIO_DP),
        Decimal::new(MAX_SPLIT_RATIO_SCALE, MAX_SPLIT_RATIO_DP),
    )
}

fn validate_corporate_action_value(
    symbol: &str,
    action_type: CorporateActionType,
    value: Decimal,
) -> PyResult<()> {
    if symbol.trim().is_empty() {
        return Err(validation_err("symbol must not be empty"));
    }
    match action_type {
        CorporateActionType::Split => {
            if value <= Decimal::ZERO {
                return Err(validation_err("split ratio must be > 0"));
            }
            let (min_ratio, max_ratio) = split_ratio_bounds();
            if value < min_ratio || value > max_ratio {
                return Err(validation_err(&format!(
                    "split ratio must be within [{min_ratio}, {max_ratio}]",
                )));
            }
        }
        CorporateActionType::Dividend => {
            if value < Decimal::ZERO {
                return Err(validation_err("dividend value must be >= 0"));
            }
        }
    }
    Ok(())
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorporateActionType {
    Split,    // 拆股/合股
    Dividend, // 分红
}

impl pyo3_stub_gen::PyStubType for CorporateActionType {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo::with_module("akquant.CorporateActionType", "akquant".into())
    }
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorporateAction {
    #[pyo3(get, set)]
    pub symbol: String,
    #[pyo3(get, set)]
    pub date: NaiveDate, // 除权除息日 (Ex-Date)
    #[pyo3(get, set)]
    pub action_type: CorporateActionType,
    pub value: Decimal, // Split: ratio (e.g., 2.0 for 1-to-2 split), Dividend: amount per share
}

use pyo3_stub_gen::derive::*;

#[gen_stub_pymethods]
#[pymethods]
impl CorporateAction {
    #[new]
    pub fn new(
        symbol: String,
        date: NaiveDate,
        action_type: CorporateActionType,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let val = extract_action_value(value)?;
        let symbol = symbol.trim().to_string();
        validate_corporate_action_value(&symbol, action_type.clone(), val)?;
        Ok(Self {
            symbol,
            date,
            action_type,
            value: val,
        })
    }

    #[getter]
    fn get_value(&self) -> f64 {
        use rust_decimal::prelude::ToPrimitive;
        self.value.to_f64().unwrap_or(0.0)
    }

    #[setter]
    fn set_value(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let val = extract_action_value(value)?;
        validate_corporate_action_value(&self.symbol, self.action_type.clone(), val)?;
        self.value = val;
        Ok(())
    }
}

/// 把浮点数按它最短的十进制写法转成 Decimal: `1.44` 就是 1.44。
///
/// `Decimal::from_f64_retain` 会保留浮点数的二进制尾数(1.44 变成
/// 1.43999999999999994...)。比例带着这个尾数去乘持仓, 拆股后的持仓就比应有的整数少
/// 一点点, 全仓卖出时因"可用持仓不足"被拒(#460)。
fn decimal_from_f64_shortest(value: f64) -> Option<Decimal> {
    use std::str::FromStr;
    if !value.is_finite() {
        return None;
    }
    // f64 的 Display 给出能还原回同一个浮点数的最短十进制写法, 不用科学计数法。
    Decimal::from_str(&value.to_string())
        .ok()
        .or_else(|| Decimal::from_f64_retain(value))
}

/// 公司行动的数值(拆股比例、每股分红): 浮点数按十进制写法取值, 其余同 `extract_decimal`。
fn extract_action_value(value: &Bound<'_, PyAny>) -> PyResult<Decimal> {
    if let Ok(float_value) = value.extract::<f64>() {
        return decimal_from_f64_shortest(float_value)
            .ok_or_else(|| PyValueError::new_err("Invalid float"));
    }
    crate::model::market_data::extract_decimal(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corporate_action_rejects_invalid_split_ratio() {
        let result =
            validate_corporate_action_value("AAPL", CorporateActionType::Split, Decimal::ZERO);
        assert!(result.is_err());
    }

    #[test]
    fn test_corporate_action_rejects_negative_dividend() {
        let result = validate_corporate_action_value(
            "AAPL",
            CorporateActionType::Dividend,
            Decimal::new(-1, 2),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_corporate_action_rejects_out_of_range_split_ratio() {
        let result =
            validate_corporate_action_value("AAPL", CorporateActionType::Split, Decimal::new(1, 5));
        assert!(result.is_err());
    }

    #[test]
    fn float_ratio_is_taken_at_its_decimal_spelling() {
        use std::str::FromStr;
        for (value, expected) in [(1.44, "1.44"), (1.45, "1.45"), (0.35, "0.35"), (2.0, "2")] {
            assert_eq!(
                decimal_from_f64_shortest(value),
                Some(Decimal::from_str(expected).unwrap())
            );
        }
        // 14500 股按 1.44 拆股得到整 20880 股, 不带尾数
        assert_eq!(
            Decimal::from(14500) * decimal_from_f64_shortest(1.44).unwrap(),
            Decimal::from(20880)
        );
        assert_eq!(decimal_from_f64_shortest(f64::NAN), None);
        assert_eq!(decimal_from_f64_shortest(f64::INFINITY), None);
    }
}
