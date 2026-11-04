//! 运行时绩效统计：`result.metrics`、`StrategyContext::strategy_stats`、
//! `Engine::strategy_stats`、`LiveTradeStats` 共用的唯一算法。
//!
//! 公式逐项复刻 `analysis/result.rs`（回撤、收益）与 `analysis/tracker.rs::calculate_pnl`
//! （交易汇总）、`analysis/python.rs`（metrics_df 的 total_pnl），任何口径改动必须同时改这里。

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};

/// 交易汇总：`TradeTracker` 增量维护，O(1) 复制进 ctx。
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TradeSummary {
    pub won_count: usize,
    pub lost_count: usize,
    pub won_pnl: Decimal,
    pub lost_pnl: Decimal,
    pub sum_closed_pnl: Decimal,
    pub largest_win: Decimal,
    pub largest_loss: Decimal,
    pub total_commission: Decimal,
    pub max_wins: usize,
    pub max_losses: usize,
}

/// 权益曲线回撤的增量折叠，语义与 `StatisticsManager::upsert_timestamped_value` 一致：
/// 同一时间戳的新点覆盖旧点，所以最后一个点在被下一个时间戳取代之前都只是 `pending`。
#[derive(Clone, Copy, Debug, Default)]
pub struct DrawdownState {
    initial: Decimal,
    peak: Decimal,
    max_drawdown: Decimal,
    max_drawdown_value: Decimal,
    pending: Option<(i64, Decimal)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawdownSummary {
    pub max_drawdown: Decimal,
    pub max_drawdown_value: Decimal,
    pub current_drawdown: Decimal,
    pub last_equity: Option<Decimal>,
}

impl DrawdownState {
    pub fn new(initial: Decimal) -> Self {
        Self {
            initial,
            peak: initial,
            max_drawdown: Decimal::ZERO,
            max_drawdown_value: Decimal::ZERO,
            pending: None,
        }
    }

    pub fn initial(&self) -> Decimal {
        self.initial
    }

    /// 与 result.rs 原循环体逐行相同（峰值、回撤金额、回撤比例分别取最大）。
    fn commit(&mut self, equity: Decimal) {
        if equity > self.peak {
            self.peak = equity;
        }
        let drawdown_val = self.peak - equity;
        if drawdown_val > self.max_drawdown_value {
            self.max_drawdown_value = drawdown_val;
        }
        let drawdown = if !self.peak.is_zero() {
            drawdown_val / self.peak
        } else {
            Decimal::ZERO
        };
        if drawdown > self.max_drawdown {
            self.max_drawdown = drawdown;
        }
    }

    pub fn push(&mut self, ts: i64, equity: Decimal) {
        if let Some((last_ts, _)) = self.pending
            && last_ts == ts
        {
            self.pending = Some((ts, equity));
            return;
        }
        if let Some((_, last_eq)) = self.pending.take() {
            self.commit(last_eq);
        }
        self.pending = Some((ts, equity));
    }

    pub fn summary(&self) -> DrawdownSummary {
        let mut s = *self;
        let last_equity = s.pending.take().map(|(_, eq)| {
            s.commit(eq);
            eq
        });
        let current_drawdown = match last_equity {
            Some(eq) if !s.peak.is_zero() => (s.peak - eq) / s.peak,
            _ => Decimal::ZERO,
        };
        DrawdownSummary {
            max_drawdown: s.max_drawdown,
            max_drawdown_value: s.max_drawdown_value,
            current_drawdown,
            last_equity,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StatsState {
    pub trades: TradeSummary,
    pub drawdown: DrawdownState,
    pub initial_cash: Decimal,
}

/// 字段名与 `result.metrics_df` 一一对应（额外字段见各自注释）。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StrategyStatsValues {
    pub total_return_pct: f64,
    pub total_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_commission: f64,
    pub max_drawdown: f64,
    pub max_drawdown_pct: f64,
    pub max_drawdown_value: f64,
    pub win_rate: f64,
    pub loss_rate: f64,
    pub winning_trades: f64,
    pub losing_trades: f64,
    pub profit_factor: f64,
    pub avg_pnl: f64,
    pub avg_profit: f64,
    pub avg_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub max_wins: f64,
    pub max_losses: f64,
    /// metrics 中没有：winning_trades + losing_trades
    pub trade_count: f64,
    /// metrics 中没有：当前权益相对峰值的回撤（%）
    pub current_drawdown_pct: f64,
    /// metrics 中没有：当前权益
    pub equity: f64,
}

pub fn compute_strategy_stats(
    state: &StatsState,
    current_ts: Option<i64>,
    current_equity: Decimal,
    unrealized_pnl: f64,
) -> StrategyStatsValues {
    let to_f64 = |d: Decimal| d.to_f64().unwrap_or_default();
    let t = &state.trades;
    let total = t.won_count + t.lost_count;

    let mut dd = state.drawdown;
    if let Some(ts) = current_ts {
        dd.push(ts, current_equity);
    }
    let dd = dd.summary();

    let initial = state.initial_cash;
    // result.rs: total_return_dec = (final - initial) / initial; pct = to_f64 * 100
    let total_return = if !initial.is_zero() {
        (current_equity - initial) / initial
    } else {
        Decimal::ZERO
    };
    let ratio = |n: usize| {
        if total > 0 {
            (n as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    };
    let profit_factor = if t.lost_pnl.abs() > Decimal::ZERO {
        to_f64(t.won_pnl / t.lost_pnl.abs())
    } else if t.won_pnl > Decimal::ZERO {
        f64::INFINITY
    } else {
        0.0
    };
    let max_dd = to_f64(dd.max_drawdown);
    StrategyStatsValues {
        total_return_pct: to_f64(total_return) * 100.0,
        // python.rs: end_market_value - initial_market_value，两者都是先转 f64 再相减
        total_pnl: to_f64(current_equity) - to_f64(initial),
        unrealized_pnl,
        total_commission: to_f64(t.total_commission),
        max_drawdown: max_dd,
        max_drawdown_pct: max_dd * 100.0,
        max_drawdown_value: to_f64(dd.max_drawdown_value),
        win_rate: ratio(t.won_count),
        loss_rate: ratio(t.lost_count),
        winning_trades: t.won_count as f64,
        losing_trades: t.lost_count as f64,
        profit_factor,
        avg_pnl: if total > 0 {
            to_f64(t.sum_closed_pnl) / total as f64
        } else {
            0.0
        },
        avg_profit: if t.won_count > 0 {
            to_f64(t.won_pnl) / t.won_count as f64
        } else {
            0.0
        },
        avg_loss: if t.lost_count > 0 {
            to_f64(t.lost_pnl) / t.lost_count as f64
        } else {
            0.0
        },
        largest_win: to_f64(t.largest_win),
        largest_loss: to_f64(t.largest_loss),
        max_wins: t.max_wins as f64,
        max_losses: t.max_losses as f64,
        trade_count: total as f64,
        current_drawdown_pct: to_f64(dd.current_drawdown) * 100.0,
        equity: to_f64(current_equity),
    }
}

impl StrategyStatsValues {
    pub fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        macro_rules! put {
            ($($f:ident),*) => { $( d.set_item(stringify!($f), self.$f)?; )* };
        }
        put!(
            total_return_pct,
            total_pnl,
            unrealized_pnl,
            total_commission,
            max_drawdown,
            max_drawdown_pct,
            max_drawdown_value,
            win_rate,
            loss_rate,
            winning_trades,
            losing_trades,
            profit_factor,
            avg_pnl,
            avg_profit,
            avg_loss,
            largest_win,
            largest_loss,
            max_wins,
            max_losses,
            trade_count,
            current_drawdown_pct,
            equity
        );
        Ok(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    /// 逐点 upsert（同一时间戳覆盖）后的结果，必须等于对“覆盖后的最终曲线”做一次折叠。
    /// 这条性质保证 ctx 增量路径与 result.rs 的整条曲线路径数值完全一致。
    #[test]
    fn incremental_upsert_equals_fold_of_final_curve() {
        let mut inc = DrawdownState::new(dec!(100));
        inc.push(1, dec!(110));
        inc.push(2, dec!(90)); // 之后被同一时间戳覆盖
        inc.push(2, dec!(105));
        inc.push(3, dec!(80));
        inc.push(4, dec!(120));

        let mut fold = DrawdownState::new(dec!(100));
        for (ts, eq) in [
            (1, dec!(110)),
            (2, dec!(105)),
            (3, dec!(80)),
            (4, dec!(120)),
        ] {
            fold.push(ts, eq);
        }
        let a = inc.summary();
        let b = fold.summary();
        assert_eq!(a.max_drawdown, b.max_drawdown);
        assert_eq!(a.max_drawdown_value, b.max_drawdown_value);
        // 被覆盖掉的 90 不能留下痕迹：峰值 110 → 最低 80
        assert_eq!(a.max_drawdown_value, dec!(30));
        assert_eq!(a.max_drawdown, dec!(30) / dec!(110));
        assert_eq!(a.current_drawdown, Decimal::ZERO);
    }

    #[test]
    fn peak_starts_at_initial_cash() {
        let mut s = DrawdownState::new(dec!(100));
        s.push(1, dec!(95));
        let sm = s.summary();
        assert_eq!(sm.max_drawdown_value, dec!(5));
        assert_eq!(sm.current_drawdown, dec!(5) / dec!(100));
    }

    #[test]
    fn compute_stats_matches_metrics_formulas() {
        let state = StatsState {
            trades: TradeSummary {
                won_count: 2,
                lost_count: 1,
                won_pnl: dec!(30),
                lost_pnl: dec!(-10),
                sum_closed_pnl: dec!(20),
                largest_win: dec!(20),
                largest_loss: dec!(-10),
                total_commission: dec!(3),
                max_wins: 2,
                max_losses: 1,
            },
            drawdown: DrawdownState::new(dec!(1000)),
            initial_cash: dec!(1000),
        };
        let v = compute_strategy_stats(&state, Some(5), dec!(1017), 1.5);
        assert_eq!(v.win_rate, 2.0 / 3.0 * 100.0);
        assert_eq!(v.loss_rate, 1.0 / 3.0 * 100.0);
        assert_eq!(v.profit_factor, 3.0);
        assert_eq!(v.avg_pnl, 20.0 / 3.0);
        assert_eq!(v.total_pnl, 1017.0 - 1000.0);
        assert_eq!(v.total_return_pct, 0.017 * 100.0);
        assert_eq!(v.trade_count, 3.0);
        assert_eq!(v.equity, 1017.0);
        assert_eq!(v.unrealized_pnl, 1.5);
    }

    #[test]
    fn profit_factor_edge_cases() {
        let mut state = StatsState::default();
        state.initial_cash = dec!(1);
        state.drawdown = DrawdownState::new(dec!(1));
        assert_eq!(
            compute_strategy_stats(&state, None, dec!(1), 0.0).profit_factor,
            0.0
        );
        state.trades.won_pnl = dec!(5);
        state.trades.won_count = 1;
        assert!(
            compute_strategy_stats(&state, None, dec!(1), 0.0)
                .profit_factor
                .is_infinite()
        );
    }
}
