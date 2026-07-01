use rust_decimal::prelude::*;
use std::collections::HashMap;

use crate::account::calculate_account_metrics;
use crate::analysis::{BacktestResult, LiquidationAudit, PositionSnapshot};
use crate::model::Instrument;
use crate::portfolio::Portfolio;
use crate::risk::RiskConfig;

/// 统计管理器
///
/// 负责维护回测过程中的统计数据，包括：
/// - 权益曲线 (Equity Curve)
/// - 现金曲线 (Cash Curve)
/// - 持仓快照 (Position Snapshots)
/// - 生成最终的回测结果 (BacktestResult)
pub struct StatisticsManager {
    /// 权益曲线 [(timestamp, equity)]
    equity_curve: Vec<(i64, Decimal)>,
    /// 现金曲线 [(timestamp, cash)]
    cash_curve: Vec<(i64, Decimal)>,
    /// 保证金曲线 [(timestamp, used_margin)]
    margin_curve: Vec<(i64, Decimal)>,
    /// 持仓快照 [(timestamp, snapshots)]
    pub snapshots: Vec<(i64, Vec<PositionSnapshot>)>,
    /// 强平审计记录
    pub liquidation_audits: Vec<LiquidationAudit>,
}

impl Default for StatisticsManager {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticsManager {
    fn upsert_timestamped_value<T>(series: &mut Vec<(i64, T)>, timestamp: i64, value: T) {
        if let Some((last_ts, last_value)) = series.last_mut()
            && *last_ts == timestamp
        {
            *last_value = value;
            return;
        }
        series.push((timestamp, value));
    }

    /// 创建新的统计管理器
    pub fn new() -> Self {
        Self {
            equity_curve: Vec::new(),
            cash_curve: Vec::new(),
            margin_curve: Vec::new(),
            snapshots: Vec::new(),
            liquidation_audits: Vec::new(),
        }
    }

    /// 更新权益和现金曲线
    pub fn update(&mut self, timestamp: i64, equity: Decimal, cash: Decimal, margin: Decimal) {
        Self::upsert_timestamped_value(&mut self.equity_curve, timestamp, equity);
        Self::upsert_timestamped_value(&mut self.cash_curve, timestamp, cash);
        Self::upsert_timestamped_value(&mut self.margin_curve, timestamp, margin);
    }

    /// 记录持仓快照
    pub fn record_snapshot(
        &mut self,
        timestamp: i64,
        portfolio: &Portfolio,
        instruments: &HashMap<String, Instrument>,
        last_prices: &HashMap<String, Decimal>,
        trade_tracker: &crate::analysis::TradeTracker,
        risk_config: &RiskConfig,
    ) {
        let snapshots = Self::create_snapshot(
            portfolio,
            instruments,
            last_prices,
            trade_tracker,
            risk_config,
        );
        Self::upsert_timestamped_value(&mut self.snapshots, timestamp, snapshots);
    }

    pub fn record_liquidation_audit(&mut self, audit: LiquidationAudit) {
        self.liquidation_audits.push(audit);
    }

    /// 创建持仓快照 (静态方法/无状态)
    pub fn create_snapshot(
        portfolio: &Portfolio,
        instruments: &HashMap<String, Instrument>,
        last_prices: &HashMap<String, Decimal>,
        trade_tracker: &crate::analysis::TradeTracker,
        risk_config: &RiskConfig,
    ) -> Vec<PositionSnapshot> {
        let account_equity = calculate_account_metrics(
            portfolio,
            last_prices,
            instruments,
            trade_tracker,
            risk_config,
        )
        .equity;
        let account_equity_f64 = account_equity.to_f64().unwrap_or(0.0);

        let mut current_snapshots = Vec::new();
        for (symbol, &qty) in portfolio.positions.iter() {
            if qty == Decimal::ZERO {
                continue;
            }

            let price = last_prices.get(symbol).copied().unwrap_or(Decimal::ZERO);
            let instr = instruments.get(symbol);
            let multiplier = instr.map_or(Decimal::ONE, Instrument::multiplier);
            let margin_ratio = instr.map_or(Decimal::ZERO, Instrument::margin_ratio);

            // Convert to f64 for snapshot
            let qty_f64 = qty.to_f64().unwrap_or(0.0);
            let price_f64 = price.to_f64().unwrap_or(0.0);

            let market_value = qty.abs() * price * multiplier;
            let market_value_f64 = market_value.to_f64().unwrap_or(0.0);

            let (long_shares, short_shares) = if qty > Decimal::ZERO {
                (qty_f64, 0.0)
            } else {
                (0.0, qty.abs().to_f64().unwrap_or(0.0))
            };

            let margin_dec = market_value * margin_ratio;
            let margin_f64 = margin_dec.to_f64().unwrap_or(0.0);

            let unrealized_pnl = trade_tracker.get_unrealized_pnl(symbol, price);
            let unrealized_pnl_f64 = unrealized_pnl.to_f64().unwrap_or(0.0);

            let entry_price = trade_tracker.get_average_price(symbol);
            let entry_price_f64 = entry_price.to_f64().unwrap_or(0.0);

            current_snapshots.push(PositionSnapshot {
                symbol: symbol.clone(),
                quantity: qty_f64,
                entry_price: entry_price_f64,
                long_shares,
                short_shares,
                close: price_f64,
                equity: account_equity_f64,
                market_value: market_value_f64,
                margin: margin_f64,
                unrealized_pnl: unrealized_pnl_f64,
            });
        }
        current_snapshots
    }

    /// 生成最终的回测结果
    pub fn generate_backtest_result(
        &self,
        portfolio: &Portfolio,
        instruments: &HashMap<String, Instrument>,
        last_prices: &HashMap<String, Decimal>,
        order_manager: &crate::order_manager::OrderManager,
        risk_config: &RiskConfig,
        initial_cash: Decimal,
        now_ns: Option<i64>,
        timezone_name: Option<String>,
        timezone_offset: i32,
        days_per_year: f64,
        risk_free_rate: f64,
    ) -> BacktestResult {
        // Calculate final PnL
        let trade_pnl = order_manager
            .trade_tracker
            .calculate_pnl(Some(last_prices.clone()));

        // Prepare data for result creation
        let mut equity_curve = self.equity_curve.clone();
        let mut cash_curve = self.cash_curve.clone();
        let mut margin_curve = self.margin_curve.clone();
        let mut snapshots = self.snapshots.clone();

        // Add final snapshot if needed
        if let Some(ts) = now_ns {
            let equity = calculate_account_metrics(
                portfolio,
                last_prices,
                instruments,
                &order_manager.trade_tracker,
                risk_config,
            )
            .equity;
            let margin = portfolio.calculate_used_margin(last_prices, instruments);
            let snap = Self::create_snapshot(
                portfolio,
                instruments,
                last_prices,
                &order_manager.trade_tracker,
                risk_config,
            );

            // Always overwrite the terminal point at the same timestamp so the
            // final result reflects the fully updated portfolio state.
            Self::upsert_timestamped_value(&mut equity_curve, ts, equity);
            Self::upsert_timestamped_value(&mut cash_curve, ts, portfolio.cash);
            Self::upsert_timestamped_value(&mut margin_curve, ts, margin);
            Self::upsert_timestamped_value(&mut snapshots, ts, snap);
        }

        BacktestResult::calculate(crate::analysis::CalculatorInput {
            equity_curve_decimal: equity_curve,
            cash_curve_decimal: cash_curve,
            margin_curve_decimal: margin_curve,
            snapshots,
            timezone_name,
            timezone_offset,
            trade_pnl,
            trades: order_manager.trade_tracker.closed_trades.to_vec(),
            initial_cash,
            orders: order_manager.get_all_orders(),
            executions: order_manager.trades.clone(),
            liquidation_audits: self.liquidation_audits.clone(),
            days_per_year,
            risk_free_rate,
        })
    }

    // Removed create_backtest_result as it is merged into generate_backtest_result

    /// 获取当前的权益曲线（只读）
    pub fn equity_curve(&self) -> &Vec<(i64, Decimal)> {
        &self.equity_curve
    }

    /// 获取当前的现金曲线（只读）
    pub fn cash_curve(&self) -> &Vec<(i64, Decimal)> {
        &self.cash_curve
    }

    /// 获取当前的保证金曲线（只读）
    pub fn margin_curve(&self) -> &Vec<(i64, Decimal)> {
        &self.margin_curve
    }
}
