use crate::model::{AssetType, Instrument, OrderSide, TradingSession};
use chrono::NaiveTime;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MarketType {
    China,
    Simple,
}

/// 简单市场模型 (如加密货币/外汇)
/// 24/7 交易, T+0, 简单佣金结构
pub struct SimpleMarket {
    pub commission_rate: Decimal,
}

impl SimpleMarket {
    pub fn new(commission_rate: f64) -> Self {
        Self {
            commission_rate: Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO),
        }
    }
}

impl MarketModel for SimpleMarket {
    fn get_session_status(&self, _time: NaiveTime) -> TradingSession {
        TradingSession::Continuous // 24/7 交易
    }

    fn calculate_commission(
        &self,
        instrument: &Instrument,
        _side: OrderSide,
        price: Decimal,
        quantity: Decimal,
    ) -> Decimal {
        price * quantity * instrument.multiplier * self.commission_rate
    }

    fn update_available_position(
        &self,
        available_positions: &mut HashMap<String, Decimal>,
        instrument: &Instrument,
        quantity: Decimal,
        side: OrderSide,
    ) {
        let symbol = &instrument.symbol;
        match side {
            OrderSide::Buy => {
                available_positions
                    .entry(symbol.clone())
                    .or_insert(Decimal::ZERO);
                if let Some(pos) = available_positions.get_mut(symbol) {
                    *pos += quantity;
                }
            }
            OrderSide::Sell => {
                if let Some(pos) = available_positions.get_mut(symbol) {
                    *pos -= quantity;
                }
            }
        }
    }

    fn on_day_close(
        &self,
        _positions: &HashMap<String, Decimal>,
        _available_positions: &mut HashMap<String, Decimal>,
        _instruments: &HashMap<String, Instrument>,
    ) {
        // T+0 无需每日结算解锁持仓
    }
}

#[derive(Clone)]
pub struct SessionRange {
    pub start: NaiveTime,
    pub end: NaiveTime,
    pub session: TradingSession,
}

#[derive(Clone)]
pub struct ChinaMarketConfig {
    pub t_plus_one: bool,
    pub stock_commission_rate: Decimal,
    pub stock_stamp_tax: Decimal,
    pub stock_transfer_fee: Decimal,
    pub stock_min_commission: Decimal,
    pub fund_commission_rate: Decimal,
    pub fund_transfer_fee: Decimal,
    pub fund_min_commission: Decimal,
    pub future_commission_rate: Decimal,
    pub option_commission_per_contract: Decimal,
    pub sessions: Vec<SessionRange>,
}

fn default_sessions() -> Vec<SessionRange> {
    let t_9_15 = NaiveTime::from_hms_opt(9, 15, 0).unwrap();
    let t_9_25 = NaiveTime::from_hms_opt(9, 25, 0).unwrap();
    let t_9_30 = NaiveTime::from_hms_opt(9, 30, 0).unwrap();
    let t_11_30 = NaiveTime::from_hms_opt(11, 30, 0).unwrap();
    let t_13_00 = NaiveTime::from_hms_opt(13, 0, 0).unwrap();
    let t_14_57 = NaiveTime::from_hms_opt(14, 57, 0).unwrap();
    // 延长闭市时间到 15:00:01 以包含 15:00:00 的数据点
    let t_15_00 = NaiveTime::from_hms_opt(15, 0, 1).unwrap();
    vec![
        SessionRange {
            start: t_9_15,
            end: t_9_25,
            session: TradingSession::CallAuction,
        },
        SessionRange {
            start: t_9_25,
            end: t_9_30,
            session: TradingSession::PreOpen,
        },
        SessionRange {
            start: t_9_30,
            end: t_11_30,
            session: TradingSession::Continuous,
        },
        SessionRange {
            start: t_11_30,
            end: t_13_00,
            session: TradingSession::Break,
        },
        SessionRange {
            start: t_13_00,
            end: t_14_57,
            session: TradingSession::Continuous,
        },
        SessionRange {
            start: t_14_57,
            end: t_15_00,
            session: TradingSession::CallAuction,
        },
    ]
}

impl Default for ChinaMarketConfig {
    fn default() -> Self {
        Self {
            t_plus_one: true,
            stock_commission_rate: Decimal::from_str("0.0003").unwrap(),
            stock_stamp_tax: Decimal::from_str("0.0005").unwrap(),
            stock_transfer_fee: Decimal::from_str("0.00001").unwrap(),
            stock_min_commission: Decimal::from(5),
            fund_commission_rate: Decimal::from_str("0.0003").unwrap(),
            fund_transfer_fee: Decimal::from_str("0.00001").unwrap(),
            fund_min_commission: Decimal::from(5),
            future_commission_rate: Decimal::from_str("0.000023").unwrap(),
            option_commission_per_contract: Decimal::from(5),
            sessions: default_sessions(),
        }
    }
}
/// 市场模型特征
/// 定义了不同市场（如A股、美股、期货等）的特定规则
pub trait MarketModel: Send + Sync {
    /// 计算交易费用（佣金、印花税等）
    fn calculate_commission(
        &self,
        instrument: &Instrument,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
    ) -> Decimal;

    /// 检查并更新可用持仓（处理 T+1, T+0 等规则）
    ///
    /// # Arguments
    /// * `available_positions` - 可用持仓（可变引用，用于更新）
    /// * `instrument` - 交易标的
    /// * `quantity` - 交易数量
    /// * `side` - 交易方向
    fn update_available_position(
        &self,
        available_positions: &mut HashMap<String, Decimal>,
        instrument: &Instrument,
        quantity: Decimal,
        side: OrderSide,
    );

    /// 日终结算处理（如 T+1 资产解锁）
    fn on_day_close(
        &self,
        positions: &HashMap<String, Decimal>,
        available_positions: &mut HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
    );

    /// 获取当前交易时段状态
    fn get_session_status(&self, time: NaiveTime) -> TradingSession;
}

/// 中国市场模型 (A股 + 期货)
/// A股: T+1 卖出 (可选 T+0), 买入收取佣金, 卖出收取佣金+印花税
/// 期货: T+0, 双边收取佣金
pub struct ChinaMarket {
    pub config: ChinaMarketConfig,
}

impl ChinaMarket {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            config: ChinaMarketConfig::default(),
        }
    }

    pub fn from_config(config: ChinaMarketConfig) -> Self {
        Self { config }
    }

    #[allow(dead_code)]
    pub fn config(&self) -> ChinaMarketConfig {
        self.config.clone()
    }
}

impl MarketModel for ChinaMarket {
    fn get_session_status(&self, time: NaiveTime) -> TradingSession {
        for range in &self.config.sessions {
            if time >= range.start && time < range.end {
                return range.session;
            }
        }
        TradingSession::Closed
    }

    fn calculate_commission(
        &self,
        instrument: &Instrument,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
    ) -> Decimal {
        let transaction_value = price * quantity * instrument.multiplier;
        let mut commission = Decimal::ZERO;

        match instrument.asset_type {
            AssetType::Stock => {
                // 佣金: 万分之三，最低 5 元
                let mut brokerage = transaction_value * self.config.stock_commission_rate;
                let min_brokerage = self.config.stock_min_commission;
                if brokerage < min_brokerage {
                    brokerage = min_brokerage;
                }
                commission += brokerage;

                // 印花税: 万分之五 (仅卖出)
                if side == OrderSide::Sell {
                    commission += transaction_value * self.config.stock_stamp_tax;
                }

                // 过户费: 十万分之一
                commission += transaction_value * self.config.stock_transfer_fee;
            }
            AssetType::Fund => {
                // 基金佣金
                let mut brokerage = transaction_value * self.config.fund_commission_rate;
                let min_brokerage = self.config.fund_min_commission;
                if brokerage < min_brokerage {
                    brokerage = min_brokerage;
                }
                commission += brokerage;

                // 基金过户费
                commission += transaction_value * self.config.fund_transfer_fee;
            }
            AssetType::Futures => {
                // 期货佣金: 假设按金额的万分之0.23 (股指期货示例)
                commission += transaction_value * self.config.future_commission_rate;
            }
            AssetType::Option => {
                // 期权佣金: 按张收取
                commission += quantity * self.config.option_commission_per_contract;
            }
        }

        commission
    }

    fn update_available_position(
        &self,
        available_positions: &mut HashMap<String, Decimal>,
        instrument: &Instrument,
        quantity: Decimal,
        side: OrderSide,
    ) {
        let symbol = &instrument.symbol;

        match side {
            OrderSide::Buy => {
                // 初始化可用持仓记录（如果不存在）
                available_positions
                    .entry(symbol.clone())
                    .or_insert(Decimal::ZERO);

                match instrument.asset_type {
                    AssetType::Stock | AssetType::Fund => {
                        if self.config.t_plus_one {
                            // T+1: 买入不增加今日可用持仓
                        } else {
                            // T+0: 买入立即增加可用持仓
                            if let Some(pos) = available_positions.get_mut(symbol) {
                                *pos += quantity;
                            }
                        }
                    }
                    AssetType::Futures | AssetType::Option => {
                        // T+0: 买入立即增加可用持仓（假设是开仓）
                        // 注意：这里简化了开平仓逻辑，假设都是开仓或平仓导致净持仓变化
                        if let Some(pos) = available_positions.get_mut(symbol) {
                            *pos += quantity;
                        }
                    }
                }
            }
            OrderSide::Sell => {
                match instrument.asset_type {
                    AssetType::Stock | AssetType::Fund => {
                        // 卖出：减少可用持仓 (股票必须先有持仓)
                        if let Some(pos) = available_positions.get_mut(symbol) {
                            *pos -= quantity;
                        }
                    }
                    AssetType::Futures | AssetType::Option => {
                        // 卖出：减少可用持仓 (平仓或开空)
                        available_positions
                            .entry(symbol.clone())
                            .or_insert(Decimal::ZERO);
                        if let Some(pos) = available_positions.get_mut(symbol) {
                            *pos -= quantity;
                        }
                    }
                }
            }
        }
    }

    fn on_day_close(
        &self,
        positions: &HashMap<String, Decimal>,
        available_positions: &mut HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
    ) {
        // T+1 结算：将所有持仓更新为可用
        // 遍历当前持仓，如果是股票/基金且为 T+1 模式，则可用持仓 = 总持仓
        for (symbol, quantity) in positions {
            let is_t_plus_one_asset = if let Some(instr) = instruments.get(symbol) {
                matches!(instr.asset_type, AssetType::Stock | AssetType::Fund)
            } else {
                false
            };

            if is_t_plus_one_asset && self.config.t_plus_one {
                available_positions.insert(symbol.clone(), *quantity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AssetType, Instrument};

    fn create_stock_instrument(symbol: &str) -> Instrument {
        Instrument {
            symbol: symbol.to_string(),
            asset_type: AssetType::Stock,
            multiplier: Decimal::ONE,
            tick_size: Decimal::from_str("0.01").unwrap(),
            margin_ratio: Decimal::ONE,
            option_type: None,
            strike_price: None,
            expiry_date: None,
            lot_size: Decimal::from(100),
        }
    }

    fn create_future_instrument(symbol: &str) -> Instrument {
        Instrument {
            symbol: symbol.to_string(),
            asset_type: AssetType::Futures,
            multiplier: Decimal::from(300),
            tick_size: Decimal::from_str("0.2").unwrap(),
            margin_ratio: Decimal::from_str("0.1").unwrap(),
            option_type: None,
            strike_price: None,
            expiry_date: None,
            lot_size: Decimal::from(1),
        }
    }

    #[test]
    fn test_simple_market_commission() {
        let market = SimpleMarket::new(0.001); // 0.1%
        let instr = create_stock_instrument("AAPL");

        // Buy 100 @ 100. Value = 10000. Comm = 10.
        let comm = market.calculate_commission(
            &instr,
            OrderSide::Buy,
            Decimal::from(100),
            Decimal::from(100),
        );
        assert_eq!(comm, Decimal::from(10));
    }

    #[test]
    fn test_china_market_stock_commission() {
        let market = ChinaMarket::new();
        let instr = create_stock_instrument("600000");

        // Case 1: Minimum Commission (5 RMB)
        // Buy 100 @ 10. Value = 1000.
        // Brokerage = 1000 * 0.0003 = 0.3 < 5 -> 5.
        // Transfer = 1000 * 0.00001 = 0.01.
        // Total = 5.01.
        let comm_buy = market.calculate_commission(
            &instr,
            OrderSide::Buy,
            Decimal::from(10),
            Decimal::from(100),
        );
        assert_eq!(comm_buy, Decimal::from_str("5.01").unwrap());

        // Case 2: Sell (Add Stamp Tax)
        // Sell 100 @ 10. Value = 1000.
        // Brokerage = 5.
        // Stamp Tax = 1000 * 0.0005 = 0.5.
        // Transfer = 0.01.
        // Total = 5.51.
        let comm_sell = market.calculate_commission(
            &instr,
            OrderSide::Sell,
            Decimal::from(10),
            Decimal::from(100),
        );
        assert_eq!(comm_sell, Decimal::from_str("5.51").unwrap());
    }

    #[test]
    fn test_china_market_t_plus_one() {
        let market = ChinaMarket::new();
        let instr = create_stock_instrument("600000");
        let mut available = HashMap::new();

        // Buy 100. T+1 means available shouldn't increase immediately.
        market.update_available_position(
            &mut available,
            &instr,
            Decimal::from(100),
            OrderSide::Buy,
        );
        assert!(
            available.get("600000").is_none() || *available.get("600000").unwrap() == Decimal::ZERO
        );

        // Day Close. Positions (Inventory) has 100. Available should update.
        let mut positions = HashMap::new();
        positions.insert("600000".to_string(), Decimal::from(100));

        let mut instruments = HashMap::new();
        instruments.insert("600000".to_string(), instr.clone());

        market.on_day_close(&positions, &mut available, &instruments);
        assert_eq!(*available.get("600000").unwrap(), Decimal::from(100));
    }

    #[test]
    fn test_china_market_futures_t_plus_zero() {
        let market = ChinaMarket::new();
        let instr = create_future_instrument("IF2206");
        let mut available = HashMap::new();

        // Buy 1. T+0 means available increases immediately.
        market.update_available_position(&mut available, &instr, Decimal::from(1), OrderSide::Buy);
        assert_eq!(*available.get("IF2206").unwrap(), Decimal::from(1));
    }

    #[test]
    fn test_china_market_session() {
        let market = ChinaMarket::new();

        // 9:15 -> CallAuction
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(9, 15, 0).unwrap()),
            TradingSession::CallAuction
        );
        // 9:30 -> Continuous
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(9, 30, 0).unwrap()),
            TradingSession::Continuous
        );
        // 12:00 -> Break
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(12, 0, 0).unwrap()),
            TradingSession::Break
        );
        // 18:00 -> Closed
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(18, 0, 0).unwrap()),
            TradingSession::Closed
        );
    }
}
