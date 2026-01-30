use crate::model::Instrument;
use crate::model::market_data::extract_decimal;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone)]
/// 投资组合管理
///
/// :ivar cash: 当前现金余额
/// :ivar positions: 当前持仓 (symbol -> quantity)
/// :ivar available_positions: 可用持仓 (symbol -> quantity)
pub struct Portfolio {
    pub cash: Decimal,
    pub positions: HashMap<String, Decimal>,
    pub available_positions: HashMap<String, Decimal>,
}

#[pymethods]
impl Portfolio {
    /// 创建投资组合
    ///
    /// :param cash: 初始资金
    #[new]
    pub fn new(cash: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Portfolio {
            cash: extract_decimal(cash)?,
            positions: HashMap::new(),
            available_positions: HashMap::new(),
        })
    }

    #[getter]
    fn get_cash(&self) -> f64 {
        self.cash.to_f64().unwrap_or_default()
    }

    #[getter]
    fn get_positions(&self) -> HashMap<String, f64> {
        self.positions
            .iter()
            .map(|(k, v)| (k.clone(), v.to_f64().unwrap_or_default()))
            .collect()
    }

    #[getter]
    fn get_available_positions(&self) -> HashMap<String, f64> {
        self.available_positions
            .iter()
            .map(|(k, v)| (k.clone(), v.to_f64().unwrap_or_default()))
            .collect()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Portfolio(cash={:.2}, positions_count={})",
            self.cash,
            self.positions.len()
        )
    }

    /// 获取持仓数量
    pub fn get_position(&self, symbol: &str) -> f64 {
        self.positions
            .get(symbol)
            .unwrap_or(&Decimal::ZERO)
            .to_f64()
            .unwrap_or_default()
    }

    /// 获取可用持仓数量
    pub fn get_available_position(&self, symbol: &str) -> f64 {
        self.available_positions
            .get(symbol)
            .unwrap_or(&Decimal::ZERO)
            .to_f64()
            .unwrap_or_default()
    }
}

impl Portfolio {
    pub fn adjust_cash(&mut self, amount: Decimal) {
        self.cash += amount;
    }

    pub fn adjust_position(&mut self, symbol: &str, quantity: Decimal) {
        let entry = self
            .positions
            .entry(symbol.to_string())
            .or_insert(Decimal::ZERO);
        *entry += quantity;
    }

    pub fn calculate_equity(
        &self,
        prices: &HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
    ) -> Decimal {
        let mut equity = self.cash;
        for (symbol, quantity) in &self.positions {
            if !quantity.is_zero() {
                if let Some(price) = prices.get(symbol) {
                    let multiplier = if let Some(instr) = instruments.get(symbol) {
                        instr.multiplier
                    } else {
                        Decimal::ONE
                    };
                    equity += quantity * price * multiplier;
                }
            }
        }
        equity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_adjust_cash() {
        let mut portfolio = Portfolio {
            cash: Decimal::from(10000),
            positions: HashMap::new(),
            available_positions: HashMap::new(),
        };

        portfolio.adjust_cash(Decimal::from(500));
        assert_eq!(portfolio.cash, Decimal::from(10500));

        portfolio.adjust_cash(Decimal::from(-1000));
        assert_eq!(portfolio.cash, Decimal::from(9500));
    }

    #[test]
    fn test_portfolio_adjust_position() {
        let mut portfolio = Portfolio {
            cash: Decimal::from(10000),
            positions: HashMap::new(),
            available_positions: HashMap::new(),
        };

        // Buy 100
        portfolio.adjust_position("AAPL", Decimal::from(100));
        assert_eq!(portfolio.get_position("AAPL"), 100.0);

        // Buy 50 more
        portfolio.adjust_position("AAPL", Decimal::from(50));
        assert_eq!(portfolio.get_position("AAPL"), 150.0);

        // Sell 200 (Short 50)
        portfolio.adjust_position("AAPL", Decimal::from(-200));
        assert_eq!(portfolio.get_position("AAPL"), -50.0);
    }

    #[test]
    fn test_portfolio_getters() {
        let mut positions = HashMap::new();
        positions.insert("AAPL".to_string(), Decimal::from(100));

        let mut available = HashMap::new();
        available.insert("AAPL".to_string(), Decimal::from(100));

        let portfolio = Portfolio {
            cash: Decimal::from(10000),
            positions,
            available_positions: available,
        };

        assert_eq!(portfolio.get_cash(), 10000.0);
        assert_eq!(portfolio.get_position("AAPL"), 100.0);
        assert_eq!(portfolio.get_available_position("AAPL"), 100.0);
        assert_eq!(portfolio.get_position("MSFT"), 0.0);
    }

    #[test]
    fn test_portfolio_calculate_equity() {
        use crate::model::Instrument;
        use crate::model::types::AssetType;

        let mut positions = HashMap::new();
        positions.insert("AAPL".to_string(), Decimal::from(100));

        let portfolio = Portfolio {
            cash: Decimal::from(10000),
            positions,
            available_positions: HashMap::new(),
        };

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), Decimal::from(150)); // 100 * 150 = 15000

        let mut instruments = HashMap::new();
        let instr = Instrument {
            symbol: "AAPL".to_string(),
            asset_type: AssetType::Stock,
            multiplier: Decimal::ONE,
            margin_ratio: Decimal::ONE,
            tick_size: Decimal::new(1, 2), // 0.01
            option_type: None,
            strike_price: None,
            expiry_date: None,
            lot_size: Decimal::from(100),
        };
        instruments.insert("AAPL".to_string(), instr);

        let equity = portfolio.calculate_equity(&prices, &instruments);
        // Cash 10000 + Value 15000 = 25000
        assert_eq!(equity, Decimal::from(25000));
    }

    #[test]
    fn test_portfolio_calculate_equity_with_multiplier() {
        use crate::model::Instrument;
        use crate::model::types::AssetType;

        let mut positions = HashMap::new();
        positions.insert("FUT".to_string(), Decimal::from(10)); // 10 contracts

        let portfolio = Portfolio {
            cash: Decimal::from(100000),
            positions,
            available_positions: HashMap::new(),
        };

        let mut prices = HashMap::new();
        prices.insert("FUT".to_string(), Decimal::from(2000));

        let mut instruments = HashMap::new();
        let instr = Instrument {
            symbol: "FUT".to_string(),
            asset_type: AssetType::Futures,
            multiplier: Decimal::from(10), // Multiplier 10
            margin_ratio: Decimal::new(1, 1), // 0.1
            tick_size: Decimal::new(2, 1), // 0.2
            option_type: None,
            strike_price: None,
            expiry_date: None,
            lot_size: Decimal::from(1),
        };
        instruments.insert("FUT".to_string(), instr);

        let equity = portfolio.calculate_equity(&prices, &instruments);
        // Cash 100000
        // Value = 10 (qty) * 2000 (price) * 10 (mult) = 200,000
        // Total = 300,000
        assert_eq!(equity, Decimal::from(300000));
    }
}
