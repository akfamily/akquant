use crate::margin::MarginEngine;
use crate::model::Instrument;
use crate::model::market_data::extract_decimal;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use std::sync::Arc;

fn checked_mul_or_cap(lhs: Decimal, rhs: Decimal) -> Decimal {
    lhs.checked_mul(rhs).unwrap_or(Decimal::MAX)
}

fn checked_add_or_cap(lhs: Decimal, rhs: Decimal) -> Decimal {
    lhs.checked_add(rhs).unwrap_or(Decimal::MAX)
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 投资组合管理.
///
/// :ivar cash: 当前现金余额
/// :ivar positions: 当前持仓 (symbol -> quantity)
/// :ivar available_positions: 可用持仓 (symbol -> quantity)
/// :ivar total_equity: 总权益 (动态计算)
pub struct Portfolio {
    pub cash: Decimal,
    pub positions: Arc<HashMap<String, Decimal>>,
    pub available_positions: Arc<HashMap<String, Decimal>>,
}

#[test]
fn test_portfolio_calculate_margin() {
    use crate::model::Instrument;
    use crate::model::instrument::{FuturesInstrument, InstrumentEnum, OptionInstrument};
    use crate::model::types::{AssetType, OptionMarginModel, OptionType};
    use std::str::FromStr;
    use std::sync::Arc;

    let mut positions = HashMap::new();
    positions.insert("FUT".to_string(), Decimal::from(10)); // 10 contracts
    positions.insert("OPT_LONG".to_string(), Decimal::from(5)); // 5 Long Option
    positions.insert("OPT_SHORT".to_string(), Decimal::from(-2)); // 2 Short Option

    let portfolio = Portfolio {
        cash: Decimal::from(100000),
        positions: Arc::new(positions),
        available_positions: Arc::new(HashMap::new()),
    };

    let mut prices = HashMap::new();
    prices.insert("FUT".to_string(), Decimal::from(3000));
    prices.insert("OPT_LONG".to_string(), Decimal::from(50));
    prices.insert("OPT_SHORT".to_string(), Decimal::from(50));
    prices.insert("UNDERLYING".to_string(), Decimal::from(100));

    let mut instruments = HashMap::new();

    // Futures: Multiplier 10, Margin 0.1
    // Margin = 10 * 3000 * 10 * 0.1 = 30,000
    let fut_instr = Instrument {
        asset_type: AssetType::Futures,
        inner: InstrumentEnum::Futures(FuturesInstrument {
            symbol: "FUT".to_string(),
            multiplier: Decimal::from(10),
            margin_ratio: Decimal::from_str("0.1").unwrap(),
            tick_size: Decimal::from(1),
            expiry_date: None,
            settlement_type: None,
            settlement_price: None,
        }),
    };
    instruments.insert("FUT".to_string(), fut_instr);

    // Option Long: Margin 0
    let opt_long = Instrument {
        asset_type: AssetType::Option,
        inner: InstrumentEnum::Option(OptionInstrument {
            symbol: "OPT_LONG".to_string(),
            multiplier: Decimal::from(100),
            margin_ratio: Decimal::from_str("0.2").unwrap(),
            tick_size: Decimal::from_str("0.01").unwrap(),
            option_margin_model: OptionMarginModel::ChinaSingleLeg,
            option_type: OptionType::Call,
            strike_price: Decimal::from(100),
            expiry_date: 20240101,
            underlying_symbol: "UNDERLYING".to_string(),
            settlement_type: None,
            implied_volatility: None,
            reference_volatility: None,
        }),
    };
    instruments.insert("OPT_LONG".to_string(), opt_long);

    // Short call: (50 + max(100 * 0.12, 100 * 0.07)) * 100 * 2 = 12,400
    let opt_short = Instrument {
        asset_type: AssetType::Option,
        inner: InstrumentEnum::Option(OptionInstrument {
            symbol: "OPT_SHORT".to_string(),
            multiplier: Decimal::from(100),
            margin_ratio: Decimal::from_str("0.2").unwrap(),
            tick_size: Decimal::from_str("0.01").unwrap(),
            option_margin_model: OptionMarginModel::ChinaSingleLeg,
            option_type: OptionType::Call,
            strike_price: Decimal::from(100),
            expiry_date: 20240101,
            underlying_symbol: "UNDERLYING".to_string(),
            settlement_type: None,
            implied_volatility: None,
            reference_volatility: None,
        }),
    };
    instruments.insert("OPT_SHORT".to_string(), opt_short);

    let used_margin = portfolio.calculate_used_margin(&prices, &instruments);

    // Futures: 30,000
    // Long Option: 0
    // Short Option: 12,400
    assert_eq!(used_margin, Decimal::from(42400));
}

#[pymethods]
impl Portfolio {
    /// 创建投资组合.
    ///
    /// :param cash: 初始资金
    #[new]
    pub fn new(cash: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Portfolio {
            cash: extract_decimal(cash)?,
            positions: Arc::new(HashMap::new()),
            available_positions: Arc::new(HashMap::new()),
        })
    }

    #[getter]
    /// 获取当前现金余额.
    /// :return: 现金余额
    fn get_cash(&self) -> f64 {
        self.cash.to_f64().unwrap_or_default()
    }

    #[getter]
    /// 获取当前持仓字典.
    /// :return: 持仓字典 {symbol: quantity}
    fn get_positions(&self) -> HashMap<String, f64> {
        self.positions
            .iter()
            .filter(|(_, v)| !v.is_zero())
            .map(|(k, v)| (k.clone(), v.to_f64().unwrap_or_default()))
            .collect()
    }

    #[getter]
    /// 获取可用持仓字典.
    /// :return: 可用持仓字典 {symbol: quantity}
    fn get_available_positions(&self) -> HashMap<String, f64> {
        self.available_positions
            .iter()
            .filter(|(_, v)| !v.is_zero())
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

    /// 获取持仓数量.
    ///
    /// :param symbol: 标的代码
    /// :return: 持仓数量
    pub fn get_position(&self, symbol: &str) -> f64 {
        self.positions
            .get(symbol)
            .unwrap_or(&Decimal::ZERO)
            .to_f64()
            .unwrap_or_default()
    }

    /// 获取可用持仓数量.
    ///
    /// :param symbol: 标的代码
    /// :return: 可用持仓数量
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
        let positions = Arc::make_mut(&mut self.positions);
        let entry = positions.entry(symbol.to_string()).or_insert(Decimal::ZERO);
        *entry += quantity;
    }

    pub fn calculate_equity(
        &self,
        prices: &HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
    ) -> Decimal {
        let mut equity = self.cash;
        for (symbol, quantity) in self.positions.iter() {
            if !quantity.is_zero()
                && let Some(price) = prices.get(symbol)
            {
                let multiplier = if let Some(instr) = instruments.get(symbol) {
                    instr.multiplier()
                } else {
                    Decimal::ONE
                };
                let exposure =
                    checked_mul_or_cap(checked_mul_or_cap(*quantity, *price), multiplier);
                equity = checked_add_or_cap(equity, exposure);
            }
        }
        equity
    }

    /// Calculate total used margin based on current positions
    pub fn calculate_used_margin(
        &self,
        prices: &HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
    ) -> Decimal {
        MarginEngine::used_margin(&self.positions, prices, instruments, None)
    }

    pub fn calculate_used_margin_with_stock_ratio(
        &self,
        prices: &HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
        stock_margin_ratio_override: Option<Decimal>,
    ) -> Decimal {
        MarginEngine::used_margin(
            &self.positions,
            prices,
            instruments,
            stock_margin_ratio_override,
        )
    }

    /// Calculate free margin (Equity - Used Margin)
    pub fn calculate_free_margin(
        &self,
        prices: &HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
    ) -> Decimal {
        let equity = self.calculate_equity(prices, instruments);
        let used_margin = self.calculate_used_margin(prices, instruments);
        equity - used_margin
    }

    pub fn calculate_free_margin_with_stock_ratio(
        &self,
        prices: &HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
        stock_margin_ratio_override: Option<Decimal>,
    ) -> Decimal {
        let equity = self.calculate_equity(prices, instruments);
        let used_margin = self.calculate_used_margin_with_stock_ratio(
            prices,
            instruments,
            stock_margin_ratio_override,
        );
        equity - used_margin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_adjust_cash() {
        let mut portfolio = Portfolio {
            cash: Decimal::from(10000),
            positions: Arc::new(HashMap::new()),
            available_positions: Arc::new(HashMap::new()),
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
            positions: Arc::new(HashMap::new()),
            available_positions: Arc::new(HashMap::new()),
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
        use std::sync::Arc;
        let mut positions = HashMap::new();
        positions.insert("AAPL".to_string(), Decimal::from(100));

        let mut available = HashMap::new();
        available.insert("AAPL".to_string(), Decimal::from(100));

        let portfolio = Portfolio {
            cash: Decimal::from(10000),
            positions: Arc::new(positions),
            available_positions: Arc::new(available),
        };

        assert_eq!(portfolio.get_cash(), 10000.0);
        assert_eq!(portfolio.get_position("AAPL"), 100.0);
        assert_eq!(portfolio.get_available_position("AAPL"), 100.0);
        assert_eq!(portfolio.get_position("MSFT"), 0.0);
    }

    #[test]
    fn test_portfolio_calculate_equity() {
        use crate::model::Instrument;
        use crate::model::instrument::{InstrumentEnum, StockInstrument};
        use crate::model::types::AssetType;
        use std::sync::Arc;

        let mut positions = HashMap::new();
        positions.insert("AAPL".to_string(), Decimal::from(100));

        let portfolio = Portfolio {
            cash: Decimal::from(10000),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), Decimal::from(150)); // 100 * 150 = 15000

        let mut instruments = HashMap::new();
        let instr = Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: "AAPL".to_string(),
                lot_size: Decimal::from(100),
                tick_size: Decimal::new(1, 2),
                expiry_date: None,
                sellable_after_days: 1,
            }),
        };
        instruments.insert("AAPL".to_string(), instr);

        let equity = portfolio.calculate_equity(&prices, &instruments);
        // Cash 10000 + Value 15000 = 25000
        assert_eq!(equity, Decimal::from(25000));
    }

    #[test]
    fn test_portfolio_calculate_equity_with_multiplier() {
        use crate::model::Instrument;
        use crate::model::instrument::{FuturesInstrument, InstrumentEnum};
        use crate::model::types::AssetType;
        use std::sync::Arc;

        let mut positions = HashMap::new();
        positions.insert("FUT".to_string(), Decimal::from(10)); // 10 contracts

        let portfolio = Portfolio {
            cash: Decimal::from(100000),
            positions: Arc::new(positions),
            available_positions: Arc::new(HashMap::new()),
        };

        let mut prices = HashMap::new();
        prices.insert("FUT".to_string(), Decimal::from(2000));

        let mut instruments = HashMap::new();
        let instr = Instrument {
            asset_type: AssetType::Futures,
            inner: InstrumentEnum::Futures(FuturesInstrument {
                symbol: "FUT".to_string(),
                multiplier: Decimal::from(10),
                margin_ratio: Decimal::new(1, 1), // 0.1
                tick_size: Decimal::new(2, 1),    // 0.2
                expiry_date: None,
                settlement_type: None,
                settlement_price: None,
            }),
        };
        instruments.insert("FUT".to_string(), instr);

        let equity = portfolio.calculate_equity(&prices, &instruments);
        // Cash 100000
        // Value = 10 (qty) * 2000 (price) * 10 (mult) = 200,000
        // Total = 300,000
        assert_eq!(equity, Decimal::from(300000));
    }
}
