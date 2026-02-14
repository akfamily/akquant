use crate::error::AkQuantError;
use crate::model::market_data::extract_decimal;
use crate::model::{Instrument, Order, OrderSide};
use crate::portfolio::Portfolio;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Default)]
/// 风控配置.
///
/// :ivar max_order_size: 单笔最大下单数量
/// :ivar max_order_value: 单笔最大下单金额
/// :ivar max_position_size: 最大持仓数量 (绝对值)
/// :ivar restricted_list: 限制交易标的列表
/// :ivar active: 是否启用风控
pub struct RiskConfig {
    pub max_order_size: Option<Decimal>,
    pub max_order_value: Option<Decimal>,
    pub max_position_size: Option<Decimal>,
    #[pyo3(get, set)]
    pub restricted_list: Vec<String>,
    #[pyo3(get, set)]
    pub active: bool,
    #[pyo3(get, set)]
    pub check_cash: bool,
    #[pyo3(get, set)]
    pub safety_margin: f64,
}

#[pymethods]
impl RiskConfig {
    #[new]
    pub fn new() -> Self {
        Self {
            max_order_size: None,
            max_order_value: None,
            max_position_size: None,
            restricted_list: Vec::new(),
            active: true,
            check_cash: true,
            safety_margin: 0.0001,
        }
    }

    #[getter]
    /// 获取单笔最大下单数量.
    /// :return: 单笔最大下单数量
    pub fn get_max_order_size(&self) -> Option<f64> {
        self.max_order_size.map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
    /// 设置单笔最大下单数量.
    /// :param value: 单笔最大下单数量
    pub fn set_max_order_size(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.max_order_size = match value {
            Some(v) => Some(extract_decimal(v)?),
            None => None,
        };
        Ok(())
    }

    #[getter]
    /// 获取单笔最大下单金额.
    /// :return: 单笔最大下单金额
    pub fn get_max_order_value(&self) -> Option<f64> {
        self.max_order_value.map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
    /// 设置单笔最大下单金额.
    /// :param value: 单笔最大下单金额
    pub fn set_max_order_value(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.max_order_value = match value {
            Some(v) => Some(extract_decimal(v)?),
            None => None,
        };
        Ok(())
    }

    #[getter]
    /// 获取最大持仓数量.
    /// :return: 最大持仓数量
    pub fn get_max_position_size(&self) -> Option<f64> {
        self.max_position_size
            .map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
    /// 设置最大持仓数量.
    /// :param value: 最大持仓数量
    pub fn set_max_position_size(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.max_position_size = match value {
            Some(v) => Some(extract_decimal(v)?),
            None => None,
        };
        Ok(())
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone)]
/// 风控管理器.
///
/// :ivar config: 风控配置
pub struct RiskManager {
    #[pyo3(get, set)]
    pub config: RiskConfig,
}

#[pymethods]
impl RiskManager {
    /// 创建风控管理器.
    #[new]
    pub fn new() -> Self {
        Self {
            config: RiskConfig::new(),
        }
    }

    /// 检查订单是否符合风控规则.
    ///
    /// :param order: 待检查订单
    /// :param portfolio: 当前投资组合
    /// :param instruments: 交易标的信息
    /// :param active_orders: 当前活动订单
    /// :param current_prices: 当前最新价格 (可选)
    /// :return: 如果检查通过返回 None，否则返回错误信息
    pub fn check(
        &self,
        order: &Order,
        portfolio: &Portfolio,
        instruments: HashMap<String, Instrument>,
        active_orders: Vec<Order>,
        current_prices: Option<HashMap<String, f64>>,
    ) -> Option<String> {
        let prices_dec: HashMap<String, Decimal> = if let Some(cp) = current_prices {
            cp.into_iter()
                .map(|(k, v)| (k, Decimal::from_f64(v).unwrap_or(Decimal::ZERO)))
                .collect()
        } else {
            HashMap::new()
        };
        match self.check_internal(order, portfolio, &instruments, &active_orders, &prices_dec) {
            Ok(_) => None,
            Err(e) => Some(e.to_string()),
        }
    }
}

impl RiskManager {
    /// Internal check method for Rust usage
    pub fn check_internal(
        &self,
        order: &Order,
        portfolio: &Portfolio,
        instruments: &HashMap<String, Instrument>,
        active_orders: &[Order],
        current_prices: &HashMap<String, Decimal>,
    ) -> Result<(), AkQuantError> {
        if !self.config.active {
            return Ok(());
        }

        // 1. Check Restricted List
        if self.config.restricted_list.contains(&order.symbol) {
            return Err(AkQuantError::OrderError(format!(
                "Risk: Symbol {} is restricted",
                order.symbol
            )));
        }

        // 2. Check Max Order Size
        if let Some(max_size) = self.config.max_order_size
            && order.quantity > max_size
        {
            return Err(AkQuantError::OrderError(format!(
                "Risk: Order quantity {} exceeds limit {}",
                order.quantity, max_size
            )));
        }

        // 3. Check Max Order Value
        if let Some(max_value) = self.config.max_order_value {
            let price = if let Some(p) = order.price {
                Some(p)
            } else {
                current_prices.get(&order.symbol).cloned()
            };

            if let Some(p) = price {
                let value = p * order.quantity;
                if value > max_value {
                    return Err(AkQuantError::OrderError(format!(
                        "Risk: Order value {} exceeds limit {}",
                        value, max_value
                    )));
                }
            }
        }

        // 4. Check Cash Sufficiency (New)
        if self.config.check_cash && order.side == OrderSide::Buy {
            let mut required_cash = Decimal::ZERO;
            let mut price_found = false;

            // Determine price for current order
            if let Some(p) = order.price {
                required_cash = p * order.quantity;
                price_found = true;
            } else if let Some(p) = current_prices.get(&order.symbol) {
                required_cash = *p * order.quantity;
                price_found = true;
            }

            if price_found {
                // Check Active Buy Orders for committed cash
                let mut committed_cash = Decimal::ZERO;
                for o in active_orders {
                    if o.side == OrderSide::Buy && o.status == crate::model::OrderStatus::New {
                        if let Some(p) = o.price {
                             committed_cash += p * o.quantity;
                        } else if let Some(p) = current_prices.get(&o.symbol) {
                             committed_cash += *p * o.quantity;
                        }
                    }
                }

                // Apply Safety Margin (default 0.0001 or user config)
                let safety_margin = self.config.safety_margin;
                // Safety factor = 1.0 - margin (e.g., 0.9999)
                let safety_factor = Decimal::from_f64(1.0 - safety_margin).unwrap_or(Decimal::from_f64(0.9999).unwrap());

                // Available Cash = (Total Cash - Committed) * Safety Factor
                let available_cash = (portfolio.cash - committed_cash) * safety_factor;

                if required_cash > available_cash {
                     return Err(AkQuantError::OrderError(format!(
                        "Risk: Insufficient cash. Required: {}, Available: {} (Safety Margin: {})",
                        required_cash, available_cash, safety_margin
                    )));
                }
            }
        }

        // 5. Check Max Position Size
        if let Some(max_pos) = self.config.max_position_size {
            let current_pos = portfolio
                .positions
                .get(&order.symbol)
                .cloned()
                .unwrap_or(Decimal::ZERO);
            let new_pos = match order.side {
                OrderSide::Buy => current_pos + order.quantity,
                OrderSide::Sell => current_pos - order.quantity,
            };
            if new_pos.abs() > max_pos {
                return Err(AkQuantError::OrderError(format!(
                    "Risk: Resulting position {} exceeds limit {}",
                    new_pos, max_pos
                )));
            }
        }

        // 5. Check Available Position (For Stocks/Funds - Long Only)
        if let Some(instr) = instruments.get(&order.symbol) {
            match instr.asset_type {
                crate::model::AssetType::Stock | crate::model::AssetType::Fund => {
                    if order.side == OrderSide::Sell {
                        let available = portfolio
                            .available_positions
                            .get(&order.symbol)
                            .cloned()
                            .unwrap_or(Decimal::ZERO);

                        let pending_sell: Decimal = active_orders
                            .iter()
                            .filter(|o| o.symbol == order.symbol && o.side == OrderSide::Sell)
                            .map(|o| o.quantity - o.filled_quantity)
                            .sum();

                        if available - pending_sell < order.quantity {
                            return Err(AkQuantError::OrderError(format!(
                                "Risk: Insufficient available position for {}. Available: {}, Pending Sell: {}, Required: {}",
                                order.symbol, available, pending_sell, order.quantity
                            )));
                        }
                    }
                }
                _ => {}
            }
        } else if order.side == OrderSide::Sell {
            // Warn if selling without instrument metadata (can't check T+1/Short logic)
             println!("Warning: Risk check skipping T+1/Short validation for {} due to missing instrument metadata.", order.symbol);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::OrderType;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn create_dummy_order(symbol: &str, quantity: Decimal, price: Option<Decimal>) -> Order {
        Order {
            id: Uuid::new_v4().to_string(),
            symbol: symbol.to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity,
            price,
            time_in_force: crate::model::TimeInForce::Day,
            trigger_price: None,
            status: crate::model::OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: 0,
            updated_at: 0,
            commission: Decimal::ZERO,
            tag: String::new(),
            reject_reason: String::new(),
        }
    }

    fn create_dummy_portfolio() -> Portfolio {
        Portfolio {
            cash: Decimal::from(100000),
            positions: HashMap::new(),
            available_positions: HashMap::new(),
        }
    }

    #[test]
    fn test_risk_restricted_list() {
        let mut risk = RiskManager::new();
        risk.config.restricted_list.push("BAD".to_string());

        let order = create_dummy_order("BAD", Decimal::from(100), Some(Decimal::from(10)));
        let portfolio = create_dummy_portfolio();
        let instruments = HashMap::new();
        let active_orders = Vec::new();
        let current_prices = HashMap::new();

        let result = risk.check_internal(&order, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("restricted"));

        let order_ok = create_dummy_order("GOOD", Decimal::from(100), Some(Decimal::from(10)));
        let result_ok = risk.check_internal(&order_ok, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result_ok.is_ok());
    }

    #[test]
    fn test_risk_max_order_size() {
        let mut risk = RiskManager::new();
        risk.config.max_order_size = Some(Decimal::from(100));

        let order_fail = create_dummy_order("AAPL", Decimal::from(101), Some(Decimal::from(10)));
        let portfolio = create_dummy_portfolio();
        let instruments = HashMap::new();
        let active_orders = Vec::new();
        let current_prices = HashMap::new();

        let result = risk.check_internal(&order_fail, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("quantity"));

        let order_ok = create_dummy_order("AAPL", Decimal::from(100), Some(Decimal::from(10)));
        let result_ok = risk.check_internal(&order_ok, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result_ok.is_ok());
    }

    #[test]
    fn test_risk_max_order_value() {
        let mut risk = RiskManager::new();
        risk.config.max_order_value = Some(Decimal::from(1000));

        // 100 * 11 = 1100 > 1000
        let order_fail = create_dummy_order("AAPL", Decimal::from(100), Some(Decimal::from(11)));
        let portfolio = create_dummy_portfolio();
        let instruments = HashMap::new();
        let active_orders = Vec::new();
        let current_prices = HashMap::new();

        let result = risk.check_internal(&order_fail, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("value"));

        // 100 * 10 = 1000 <= 1000
        let order_ok = create_dummy_order("AAPL", Decimal::from(100), Some(Decimal::from(10)));
        let result_ok = risk.check_internal(&order_ok, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result_ok.is_ok());
    }

    #[test]
    fn test_risk_max_position_size() {
        let mut risk = RiskManager::new();
        risk.config.max_position_size = Some(Decimal::from(500));

        let mut portfolio = create_dummy_portfolio();
        portfolio.adjust_position("AAPL", Decimal::from(400));

        // 400 + 101 = 501 > 500
        let order_fail = create_dummy_order("AAPL", Decimal::from(101), Some(Decimal::from(10)));
        let instruments = HashMap::new();
        let active_orders = Vec::new();
        let current_prices = HashMap::new();

        let result = risk.check_internal(&order_fail, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("position"));

        // 400 + 100 = 500 <= 500
        let order_ok = create_dummy_order("AAPL", Decimal::from(100), Some(Decimal::from(10)));
        let result_ok = risk.check_internal(&order_ok, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result_ok.is_ok());
    }

    #[test]
    fn test_risk_available_position() {
        use crate::model::{AssetType, Instrument};

        let risk = RiskManager::new();
        // Config doesn't need specific setting, check is always active for Stock/Fund

        let mut portfolio = create_dummy_portfolio();
        // Available: 100
        portfolio
            .available_positions
            .insert("AAPL".to_string(), Decimal::from(100));

        let mut instruments = HashMap::new();
        instruments.insert(
            "AAPL".to_string(),
            Instrument {
                symbol: "AAPL".to_string(),
                asset_type: AssetType::Stock,
                multiplier: Decimal::ONE,
                tick_size: Decimal::from_str("0.01").unwrap(),
                lot_size: Decimal::from(100),
                margin_ratio: Decimal::ONE,
                option_type: None,
                strike_price: None,
                expiry_date: None,
            },
        );

        // Sell 101 -> Fail
        let mut order_fail =
            create_dummy_order("AAPL", Decimal::from(101), Some(Decimal::from(10)));
        order_fail.side = OrderSide::Sell;

        let active_orders = Vec::new();
        let current_prices = HashMap::new();

        let result = risk.check_internal(&order_fail, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Insufficient available"));

        // Sell 100 -> OK
        let mut order_ok = create_dummy_order("AAPL", Decimal::from(100), Some(Decimal::from(10)));
        order_ok.side = OrderSide::Sell;
        let result_ok = risk.check_internal(&order_ok, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result_ok.is_ok());

        // Sell 50 with Pending Sell 50 -> OK (Total 100)
        let mut order_ok2 = create_dummy_order("AAPL", Decimal::from(50), Some(Decimal::from(10)));
        order_ok2.side = OrderSide::Sell;

        let mut pending_order =
            create_dummy_order("AAPL", Decimal::from(50), Some(Decimal::from(10)));
        pending_order.side = OrderSide::Sell;
        let active_orders_2 = vec![pending_order];

        let result_ok2 =
            risk.check_internal(&order_ok2, &portfolio, &instruments, &active_orders_2, &current_prices);
        assert!(result_ok2.is_ok());

        // Sell 51 with Pending Sell 50 -> Fail (Total 101)
        let mut order_fail2 =
            create_dummy_order("AAPL", Decimal::from(51), Some(Decimal::from(10)));
        order_fail2.side = OrderSide::Sell;

        let result_fail2 =
            risk.check_internal(&order_fail2, &portfolio, &instruments, &active_orders_2, &current_prices);
        assert!(result_fail2.is_err());
    }

    #[test]
    fn test_risk_cash_sufficiency() {
        let risk = RiskManager::new();
        // Default check_cash is true

        let mut portfolio = create_dummy_portfolio();
        portfolio.cash = Decimal::from(1001); // 1001 Cash (to pass safety margin check for 1000)

        let instruments = HashMap::new();
        let active_orders = Vec::new();
        let current_prices = HashMap::new();

        // 1. Buy 100 @ 10 = 1000 (OK)
        let order_ok = create_dummy_order("AAPL", Decimal::from(100), Some(Decimal::from(10)));
        let result_ok = risk.check_internal(&order_ok, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result_ok.is_ok());

        // 2. Buy 101 @ 10 = 1010 (Fail)
        let order_fail = create_dummy_order("AAPL", Decimal::from(101), Some(Decimal::from(10)));
        let result_fail = risk.check_internal(&order_fail, &portfolio, &instruments, &active_orders, &current_prices);
        assert!(result_fail.is_err());
        assert!(result_fail.unwrap_err().to_string().contains("Insufficient cash"));

        // 3. Buy with pending orders
        // Pending: Buy 50 @ 10 = 500
        let mut pending = create_dummy_order("AAPL", Decimal::from(50), Some(Decimal::from(10)));
        pending.side = OrderSide::Buy;
        let active_orders_2 = vec![pending];

        // Try Buy 50 @ 10 = 500. Total 1000. Available ~1000.9. OK.
        let order_ok2 = create_dummy_order("AAPL", Decimal::from(50), Some(Decimal::from(10)));
        let result_ok2 = risk.check_internal(&order_ok2, &portfolio, &instruments, &active_orders_2, &current_prices);
        assert!(result_ok2.is_ok());

        // Try Buy 51 @ 10 = 510. Total 1010. Available ~1000.9. Fail.
        let order_fail2 = create_dummy_order("AAPL", Decimal::from(51), Some(Decimal::from(10)));
        let result_fail2 = risk.check_internal(&order_fail2, &portfolio, &instruments, &active_orders_2, &current_prices);
        assert!(result_fail2.is_err());
    }
}
