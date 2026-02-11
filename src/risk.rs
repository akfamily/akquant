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
        }
    }

    #[getter]
    pub fn get_max_order_size(&self) -> Option<f64> {
        self.max_order_size.map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
    pub fn set_max_order_size(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.max_order_size = match value {
            Some(v) => Some(extract_decimal(v)?),
            None => None,
        };
        Ok(())
    }

    #[getter]
    pub fn get_max_order_value(&self) -> Option<f64> {
        self.max_order_value.map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
    pub fn set_max_order_value(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.max_order_value = match value {
            Some(v) => Some(extract_decimal(v)?),
            None => None,
        };
        Ok(())
    }

    #[getter]
    pub fn get_max_position_size(&self) -> Option<f64> {
        self.max_position_size
            .map(|d| d.to_f64().unwrap_or_default())
    }

    #[setter]
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
    /// :return: 如果检查通过返回 None，否则返回错误信息
    pub fn check(
        &self,
        order: &Order,
        portfolio: &Portfolio,
        instruments: HashMap<String, Instrument>,
        active_orders: Vec<Order>,
    ) -> Option<String> {
        self.check_internal(order, portfolio, &instruments, &active_orders)
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
    ) -> Option<String> {
        if !self.config.active {
            return None;
        }

        // 1. Check Restricted List
        if self.config.restricted_list.contains(&order.symbol) {
            return Some(format!("Risk: Symbol {} is restricted", order.symbol));
        }

        // 2. Check Max Order Size
        if let Some(max_size) = self.config.max_order_size
            && order.quantity > max_size
        {
            return Some(format!(
                "Risk: Order quantity {} exceeds limit {}",
                order.quantity, max_size
            ));
        }

        // 3. Check Max Order Value
        if let Some(max_value) = self.config.max_order_value
            && let Some(price) = order.price
        {
            let value = price * order.quantity;
            if value > max_value {
                return Some(format!(
                    "Risk: Order value {} exceeds limit {}",
                    value, max_value
                ));
            }
        }

        // 4. Check Max Position Size
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
                return Some(format!(
                    "Risk: Resulting position {} exceeds limit {}",
                    new_pos, max_pos
                ));
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
                            return Some(format!(
                                "Risk: Insufficient available position for {}. Available: {}, Pending Sell: {}, Required: {}",
                                order.symbol, available, pending_sell, order.quantity
                            ));
                        }
                    }
                }
                _ => {}
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::OrderType;
    use std::collections::HashMap;

    fn create_dummy_order(symbol: &str, quantity: Decimal, price: Option<Decimal>) -> Order {
        Order {
            id: uuid::Uuid::new_v4().to_string(),
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
            commission: Decimal::ZERO,
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

        let result = risk.check_internal(&order, &portfolio, &instruments, &active_orders);
        assert!(result.is_some());
        assert!(result.unwrap().contains("restricted"));

        let order_ok = create_dummy_order("GOOD", Decimal::from(100), Some(Decimal::from(10)));
        let result_ok = risk.check_internal(&order_ok, &portfolio, &instruments, &active_orders);
        assert!(result_ok.is_none());
    }

    #[test]
    fn test_risk_max_order_size() {
        let mut risk = RiskManager::new();
        risk.config.max_order_size = Some(Decimal::from(100));

        let order_fail = create_dummy_order("AAPL", Decimal::from(101), Some(Decimal::from(10)));
        let portfolio = create_dummy_portfolio();
        let instruments = HashMap::new();
        let active_orders = Vec::new();

        let result = risk.check_internal(&order_fail, &portfolio, &instruments, &active_orders);
        assert!(result.is_some());
        assert!(result.unwrap().contains("quantity"));

        let order_ok = create_dummy_order("AAPL", Decimal::from(100), Some(Decimal::from(10)));
        let result_ok = risk.check_internal(&order_ok, &portfolio, &instruments, &active_orders);
        assert!(result_ok.is_none());
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

        let result = risk.check_internal(&order_fail, &portfolio, &instruments, &active_orders);
        assert!(result.is_some());
        assert!(result.unwrap().contains("value"));

        // 100 * 10 = 1000 <= 1000
        let order_ok = create_dummy_order("AAPL", Decimal::from(100), Some(Decimal::from(10)));
        let result_ok = risk.check_internal(&order_ok, &portfolio, &instruments, &active_orders);
        assert!(result_ok.is_none());
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

        let result = risk.check_internal(&order_fail, &portfolio, &instruments, &active_orders);
        assert!(result.is_some());
        assert!(result.unwrap().contains("position"));

        // 400 + 100 = 500 <= 500
        let order_ok = create_dummy_order("AAPL", Decimal::from(100), Some(Decimal::from(10)));
        let result_ok = risk.check_internal(&order_ok, &portfolio, &instruments, &active_orders);
        assert!(result_ok.is_none());
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

        let result = risk.check_internal(&order_fail, &portfolio, &instruments, &active_orders);
        assert!(result.is_some());
        assert!(result.unwrap().contains("Insufficient available"));

        // Sell 100 -> OK
        let mut order_ok = create_dummy_order("AAPL", Decimal::from(100), Some(Decimal::from(10)));
        order_ok.side = OrderSide::Sell;
        let result_ok = risk.check_internal(&order_ok, &portfolio, &instruments, &active_orders);
        assert!(result_ok.is_none());

        // Sell 50 with Pending Sell 50 -> OK (Total 100)
        let mut order_ok2 = create_dummy_order("AAPL", Decimal::from(50), Some(Decimal::from(10)));
        order_ok2.side = OrderSide::Sell;

        let mut pending_order =
            create_dummy_order("AAPL", Decimal::from(50), Some(Decimal::from(10)));
        pending_order.side = OrderSide::Sell;
        let active_orders_2 = vec![pending_order];

        let result_ok2 =
            risk.check_internal(&order_ok2, &portfolio, &instruments, &active_orders_2);
        assert!(result_ok2.is_none());

        // Sell 51 with Pending Sell 50 -> Fail (Total 101)
        let mut order_fail2 =
            create_dummy_order("AAPL", Decimal::from(51), Some(Decimal::from(10)));
        order_fail2.side = OrderSide::Sell;

        let result_fail2 =
            risk.check_internal(&order_fail2, &portfolio, &instruments, &active_orders_2);
        assert!(result_fail2.is_some());
    }
}
