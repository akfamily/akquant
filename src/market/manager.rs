use chrono::NaiveTime;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

use crate::market::{
    ChinaMarketConfig, MarketConfig, MarketModel, SessionRange, SimpleMarketConfig, fund, futures,
    option, stock,
};
use crate::market::fee_override::FeeOverride;
use crate::market::stock::CommissionMode;
use crate::model::{Instrument, TradingSession};

/// 从 Python 传入的 f64 构造期权费率配置, 非法值按 0 处理(与其它 setter 一致)。
pub fn option_fee_config(
    commission_per_contract: f64,
    exchange_fee_per_contract: f64,
    clearing_fee_per_contract: f64,
    exercise_fee_per_contract: f64,
    sell_open_exempt: bool,
) -> option::OptionConfig {
    let d = |v: f64| Decimal::from_f64(v).unwrap_or(Decimal::ZERO);
    option::OptionConfig {
        commission_per_contract: d(commission_per_contract),
        exchange_fee_per_contract: d(exchange_fee_per_contract),
        clearing_fee_per_contract: d(clearing_fee_per_contract),
        exercise_fee_per_contract: d(exercise_fee_per_contract),
        sell_open_exempt,
    }
}

/// 市场管理器
/// 负责管理市场配置、市场模型以及相关的费率和交易时段设置
pub struct MarketManager {
    pub config: MarketConfig,
    pub model: Box<dyn MarketModel>,
}

impl MarketManager {
    /// 创建新的市场管理器
    pub fn new() -> Self {
        // 默认初始化所有市场配置，保持向后兼容
        let config = ChinaMarketConfig {
            stock: Some(stock::StockConfig::default()),
            futures: Some(futures::FuturesConfig::default()),
            fund: Some(fund::FundConfig::default()),
            option: Some(option::OptionConfig::default()),
            ..Default::default()
        };

        let config = MarketConfig::China(config);
        Self {
            config: config.clone(),
            model: config.create_model(),
        }
    }

    /// 期权行权结算费(元/张)。只有 ChinaMarket 配置了期权费率; SimpleMarket 下为 0。
    pub fn option_exercise_fee_per_contract(&self, symbol: &str) -> Decimal {
        match &self.config {
            MarketConfig::China(c) => crate::market::china::resolve_option_config(c, symbol)
                .map_or(Decimal::ZERO, |o| o.exercise_fee_per_contract),
            MarketConfig::Simple(_) => Decimal::ZERO,
        }
    }

    /// 设置某个标的的费用覆盖; 空覆盖等于删除。
    pub fn set_instrument_fee_override(&mut self, symbol: &str, fee: FeeOverride) {
        let key = crate::model::instrument::normalize_symbol_suffix(symbol.trim());
        let overrides = match &mut self.config {
            MarketConfig::China(c) => &mut c.fee_overrides,
            MarketConfig::Simple(c) => &mut c.fee_overrides,
        };
        if fee.is_empty() {
            overrides.remove(&key);
        } else {
            overrides.insert(key, fee);
        }
        self.model = self.config.create_model();
    }

    /// 当前配置里的费用覆盖表, 供整体替换市场配置时带过去。
    fn fee_overrides(&self) -> HashMap<String, FeeOverride> {
        match &self.config {
            MarketConfig::China(c) => c.fee_overrides.clone(),
            MarketConfig::Simple(c) => c.fee_overrides.clone(),
        }
    }

    /// 启用 SimpleMarket (7x24小时, T+0, 无税, 简单佣金)
    ///
    /// :param commission_rate: 佣金率
    pub fn use_simple_market(&mut self, commission_rate: f64) {
        let fee_overrides = self.fee_overrides();
        let config = SimpleMarketConfig {
            commission_rate: Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO),
            fee_overrides,
            ..Default::default()
        };
        self.config = MarketConfig::Simple(config);
        self.model = self.config.create_model();
    }

    pub fn use_simple_market_policy(&mut self, commission_type: String, commission_value: f64) {
        let fee_overrides = self.fee_overrides();
        let config = SimpleMarketConfig {
            commission_mode: parse_commission_mode(&commission_type),
            commission_rate: Decimal::from_f64(commission_value).unwrap_or(Decimal::ZERO),
            fee_overrides,
            ..Default::default()
        };
        self.config = MarketConfig::Simple(config);
        self.model = self.config.create_model();
    }

    /// 启用 ChinaMarket (支持 T+1/T+0, 印花税, 过户费, 交易时段等)
    pub fn use_china_market(&mut self) {
        let fee_overrides = self.fee_overrides();
        let config = ChinaMarketConfig {
            stock: Some(stock::StockConfig::default()),
            futures: Some(futures::FuturesConfig::default()),
            fund: Some(fund::FundConfig::default()),
            option: Some(option::OptionConfig::default()),
            fee_overrides,
            ..Default::default()
        };
        self.config = MarketConfig::China(config);
        self.model = self.config.create_model();
    }

    /// 启用/禁用 T+1 交易规则 (仅针对 ChinaMarket)
    ///
    /// :param enabled: 是否启用 T+1
    pub fn set_t_plus_one(&mut self, enabled: bool) {
        if let MarketConfig::China(ref mut c) = self.config {
            c.stock
                .get_or_insert_with(stock::StockConfig::default)
                .t_plus_one = enabled;
            c.fund
                .get_or_insert_with(fund::FundConfig::default)
                .t_plus_one = enabled;
            self.model = self.config.create_model();
        }
    }

    /// 启用中国期货市场默认配置
    /// - 切换到 ChinaMarket
    /// - 仅启用期货配置
    /// - 保持当前交易时段配置 (需手动设置 set_market_sessions 以匹配特定品种)
    pub fn use_china_futures_market(&mut self) {
        let fee_overrides = self.fee_overrides();
        let config = ChinaMarketConfig {
            futures: Some(futures::FuturesConfig::default()),
            fee_overrides,
            ..Default::default()
        };
        self.config = MarketConfig::China(config);
        self.model = self.config.create_model();
    }

    /// 设置股票费率规则
    ///
    /// :param commission_rate: 佣金率 (如 0.0003)
    /// :param stamp_tax: 印花税率 (如 0.001)
    /// :param transfer_fee: 过户费率 (如 0.00002)
    /// :param min_commission: 最低佣金 (如 5.0)
    pub fn set_stock_fee_rules(
        &mut self,
        commission_rate: f64,
        stamp_tax: f64,
        transfer_fee: f64,
        min_commission: f64,
    ) {
        self.set_stock_fee_policy(
            "percent".to_string(),
            commission_rate,
            stamp_tax,
            transfer_fee,
            min_commission,
        );
    }

    pub fn set_stock_fee_policy(
        &mut self,
        commission_type: String,
        commission_value: f64,
        stamp_tax: f64,
        transfer_fee: f64,
        min_commission: f64,
    ) {
        let commission_mode = parse_commission_mode(&commission_type);
        match &mut self.config {
            MarketConfig::China(c) => {
                let stock = c.stock.get_or_insert_with(stock::StockConfig::default);
                stock.commission_mode = commission_mode.clone();
                stock.commission_rate = Decimal::from_f64(commission_value).unwrap_or(Decimal::ZERO);
                stock.stamp_tax = Decimal::from_f64(stamp_tax).unwrap_or(Decimal::ZERO);
                stock.transfer_fee = Decimal::from_f64(transfer_fee).unwrap_or(Decimal::ZERO);
                stock.min_commission = Decimal::from_f64(min_commission).unwrap_or(Decimal::ZERO);
            }
            MarketConfig::Simple(c) => {
                c.commission_mode = commission_mode;
                c.commission_rate = Decimal::from_f64(commission_value).unwrap_or(Decimal::ZERO);
                c.stamp_tax = Decimal::from_f64(stamp_tax).unwrap_or(Decimal::ZERO);
                c.transfer_fee = Decimal::from_f64(transfer_fee).unwrap_or(Decimal::ZERO);
                c.min_commission = Decimal::from_f64(min_commission).unwrap_or(Decimal::ZERO);
            }
        }
        self.model = self.config.create_model();
    }

    /// 设置期货费率规则
    ///
    /// :param commission_rate: 佣金率 (如 0.0001)
    pub fn set_futures_fee_rules(&mut self, commission_rate: f64) {
        if let MarketConfig::China(ref mut c) = self.config {
            let futures = c
                .futures
                .get_or_insert_with(futures::FuturesConfig::default);
            futures.commission_rate = Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO);
            self.model = self.config.create_model();
        }
    }

    pub fn set_futures_fee_rules_by_prefix(&mut self, symbol_prefix: String, commission_rate: f64) {
        if let MarketConfig::China(ref mut c) = self.config {
            let prefix = symbol_prefix.trim().to_uppercase();
            if prefix.is_empty() {
                return;
            }
            let mut updated = false;
            for (existing_prefix, cfg) in &mut c.futures_fee_by_prefix {
                if existing_prefix == &prefix {
                    cfg.commission_rate =
                        Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO);
                    updated = true;
                    break;
                }
            }
            if !updated {
                let cfg = futures::FuturesConfig {
                    commission_rate: Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO),
                };
                c.futures_fee_by_prefix.push((prefix, cfg));
            }
            self.model = self.config.create_model();
        }
    }

    /// 设置基金费率规则
    ///
    /// :param commission_rate: 佣金率
    /// :param transfer_fee: 过户费率
    /// :param min_commission: 最低佣金
    pub fn set_fund_fee_rules(
        &mut self,
        commission_rate: f64,
        transfer_fee: f64,
        min_commission: f64,
    ) {
        if let MarketConfig::China(ref mut c) = self.config {
            let fund = c.fund.get_or_insert_with(fund::FundConfig::default);
            fund.commission_rate = Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO);
            fund.transfer_fee = Decimal::from_f64(transfer_fee).unwrap_or(Decimal::ZERO);
            fund.min_commission = Decimal::from_f64(min_commission).unwrap_or(Decimal::ZERO);
            self.model = self.config.create_model();
        }
    }

    /// 设置全局期权费率规则
    pub fn set_option_fee_rules(&mut self, fees: option::OptionConfig) {
        if let MarketConfig::China(ref mut c) = self.config {
            c.option = Some(fees);
            self.model = self.config.create_model();
        }
    }

    /// 设置按品种前缀的期权费率规则(同一前缀重复设置时覆盖)
    pub fn set_options_fee_rules_by_prefix(
        &mut self,
        symbol_prefix: String,
        fees: option::OptionConfig,
    ) {
        if let MarketConfig::China(ref mut c) = self.config {
            let prefix = symbol_prefix.trim().to_uppercase();
            if prefix.is_empty() {
                return;
            }
            if let Some((_, cfg)) = c
                .options_fee_by_prefix
                .iter_mut()
                .find(|(p, _)| *p == prefix)
            {
                *cfg = fees;
            } else {
                c.options_fee_by_prefix.push((prefix, fees));
            }
            self.model = self.config.create_model();
        }
    }

    /// 设置市场交易时段
    ///
    /// :param sessions: 交易时段列表，每个元素为 (开始时间, 结束时间, 时段类型)
    pub fn set_market_sessions(&mut self, sessions: Vec<(NaiveTime, NaiveTime, TradingSession)>) {
        let mut ranges = Vec::with_capacity(sessions.len());
        for (start, end, session) in sessions {
            ranges.push(SessionRange {
                start,
                end,
                session,
            });
        }
        if let MarketConfig::China(ref mut c) = self.config {
            c.sessions = ranges;
            self.model = self.config.create_model();
        }
    }

    /// 获取当前交易时段状态
    pub fn get_session_status(&self, time: NaiveTime) -> TradingSession {
        self.model.get_session_status(time)
    }

    /// 处理日终逻辑 (T+1 等)
    pub fn on_day_close(
        &self,
        positions: &HashMap<String, Decimal>,
        available_positions: &mut HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
    ) {
        self.model
            .on_day_close(positions, available_positions, instruments);
    }
}

fn parse_commission_mode(raw_type: &str) -> CommissionMode {
    match raw_type.trim().to_ascii_lowercase().as_str() {
        "fixed" => CommissionMode::Fixed,
        "per_unit" => CommissionMode::PerUnit,
        _ => CommissionMode::Percent,
    }
}

impl Default for MarketManager {
    fn default() -> Self {
        Self::new()
    }
}
