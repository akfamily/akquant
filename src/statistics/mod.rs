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
    /// 回撤折叠缓存：(折叠用的初始资金, 已折叠到的下标, 状态)。最后一个点可能被 upsert
    /// 覆盖，所以只折叠到 len-1，最后一个点在查询时临时 push。
    dd_cache: Option<(Decimal, usize, crate::analysis::DrawdownState)>,
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
            dd_cache: None,
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
    ///
    /// `now_ns` 处补一个期末点（权益/现金/保证金曲线与持仓快照）。
    /// 已有晚于 `now_ns` 的持仓快照时（最后交易日在收盘做了日终结算，收盘快照晚于
    /// 末个行情事件）不补持仓快照：收盘快照之后组合不再变化，它已是期末状态；再在
    /// 末个行情事件时刻补一条会让结算后的状态出现在收盘之前，且排在收盘快照之后、
    /// 时间倒流。权益类曲线仍在末个行情事件时刻覆盖为期末状态，使期末权益包含收盘
    /// 结算（利息、到期、强平）的影响。
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
        trading_day_ends: Vec<(chrono::NaiveDate, i64)>,
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

            // Always overwrite the terminal point at the same timestamp so the
            // final result reflects the fully updated portfolio state.
            Self::upsert_timestamped_value(&mut equity_curve, ts, equity);
            Self::upsert_timestamped_value(&mut cash_curve, ts, portfolio.cash);
            Self::upsert_timestamped_value(&mut margin_curve, ts, margin);
            let settled_after = snapshots.last().is_some_and(|(last_ts, _)| *last_ts > ts);
            if !settled_after {
                let snap = Self::create_snapshot(
                    portfolio,
                    instruments,
                    last_prices,
                    &order_manager.trade_tracker,
                    risk_config,
                );
                Self::upsert_timestamped_value(&mut snapshots, ts, snap);
            }
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
            trading_day_ends,
        })
    }

    /// 当前权益曲线的回撤状态（摊还 O(1)）。初始资金变了，或曲线被截短时重建。
    ///
    /// 缓存只折叠到倒数第二个点：最后一个点在下一个时间戳到来之前可能被 upsert 覆盖，
    /// 每次查询时临时 push 到返回值上。曲线相邻点时间戳必不相同（upsert 保证），
    /// 所以临时 push 的点不会与缓存里的 pending 点合并。
    pub fn drawdown_state(&mut self, initial: Decimal) -> crate::analysis::DrawdownState {
        let len = self.equity_curve.len();
        let stop = len.saturating_sub(1);
        let (init, mut folded, mut state) = match self.dd_cache.take() {
            Some((init, k, st)) if init == initial && k <= stop => (init, k, st),
            _ => (initial, 0, crate::analysis::DrawdownState::new(initial)),
        };
        for (ts, eq) in &self.equity_curve[folded..stop] {
            state.push(*ts, *eq);
        }
        folded = stop.max(folded);
        let mut out = state;
        if let Some((ts, eq)) = self.equity_curve.last() {
            out.push(*ts, *eq);
        }
        self.dd_cache = Some((init, folded, state));
        out
    }

    /// on_stop 用：与 generate_backtest_result 补期末点的方式相同，结果与 metrics 逐位相等。
    #[allow(clippy::too_many_arguments)]
    pub fn stats_snapshot(
        &mut self,
        portfolio: &Portfolio,
        instruments: &HashMap<String, Instrument>,
        last_prices: &HashMap<String, Decimal>,
        order_manager: &crate::order_manager::OrderManager,
        risk_config: &RiskConfig,
        initial_cash: Decimal,
        now_ns: Option<i64>,
    ) -> crate::analysis::StrategyStatsValues {
        let trade_pnl = order_manager
            .trade_tracker
            .calculate_pnl(Some(last_prices.clone()));
        let state = crate::analysis::StatsState {
            trades: order_manager.trade_tracker.summary(),
            drawdown: self.drawdown_state(initial_cash),
            initial_cash,
        };
        // generate_backtest_result：有 now_ns 时在该时刻 upsert 期末权益，否则取曲线末点
        // （空曲线时 metrics 的 end_market_value 为 initial_cash）。
        let equity = match now_ns {
            Some(_) => {
                calculate_account_metrics(
                    portfolio,
                    last_prices,
                    instruments,
                    &order_manager.trade_tracker,
                    risk_config,
                )
                .equity
            }
            None => self
                .equity_curve
                .last()
                .map(|(_, e)| *e)
                .unwrap_or(initial_cash),
        };
        crate::analysis::compute_strategy_stats(&state, now_ns, equity, trade_pnl.unrealized_pnl)
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

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn fold(
        curve: &[(i64, Decimal)],
        initial: Decimal,
    ) -> crate::analysis::strategy_stats::DrawdownSummary {
        let mut st = crate::analysis::DrawdownState::new(initial);
        for (ts, eq) in curve {
            st.push(*ts, *eq);
        }
        st.summary()
    }

    /// 增量缓存（含末点被同一时间戳覆盖、初始资金变化）必须与整条曲线一次折叠逐位相同。
    #[test]
    fn drawdown_state_cache_equals_full_fold() {
        let mut sm = StatisticsManager::new();
        let initial = dec!(100);
        assert_eq!(sm.drawdown_state(initial).summary(), fold(&[], initial));
        let steps = [
            (1, dec!(110)),
            (2, dec!(70)), // 随后被同一时间戳覆盖，不能留下痕迹
            (2, dec!(105)),
            (3, dec!(90)),
            (3, dec!(95)),
            (4, dec!(120)),
            (5, dec!(100)),
        ];
        for (ts, eq) in steps {
            sm.update(ts, eq, Decimal::ZERO, Decimal::ZERO);
            let got = sm.drawdown_state(initial).summary();
            assert_eq!(got, fold(sm.equity_curve(), initial), "ts={ts}");
            // 重复查询不改变结果
            assert_eq!(sm.drawdown_state(initial).summary(), got);
        }
        assert_eq!(
            sm.drawdown_state(initial).summary().max_drawdown_value,
            dec!(20)
        );
        // 初始资金变了要重建
        assert_eq!(
            sm.drawdown_state(dec!(200)).summary(),
            fold(sm.equity_curve(), dec!(200))
        );
    }
}
