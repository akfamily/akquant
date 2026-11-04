//! broker_live 会话级统计：柜台成交驱动 TradeTracker，算法与回测共用（spec 5.5）。

use super::strategy_stats::{
    DrawdownState, StatsState, StrategyStatsValues, compute_strategy_stats,
};
use super::tracker::{TradeEntry, TradeTracker};
use crate::model::{OrderSide, PositionEffect, Trade};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::{HashMap, VecDeque};

/// 实盘会话级统计累加器.
///
/// 由柜台真实成交驱动 `TradeTracker`，回撤与汇总公式和回测共用同一套实现。
#[gen_stub_pyclass]
#[pyclass(module = "akquant.akquant")]
pub struct LiveTradeStats {
    tracker: TradeTracker,
    drawdown: DrawdownState,
    initial: Decimal,
    seq: usize,
}

fn dec(v: f64, name: &str) -> PyResult<Decimal> {
    Decimal::from_f64(v).ok_or_else(|| PyValueError::new_err(format!("{name} 不是有限数: {v}")))
}

fn build_trade(
    symbol: &str,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    commission: Decimal,
    timestamp_ns: i64,
    seq: usize,
) -> Trade {
    Trade {
        id: format!("live-{seq}"),
        order_id: format!("live-{seq}"),
        symbol: symbol.to_string(),
        side,
        position_effect: PositionEffect::Auto,
        quantity,
        price,
        commission,
        timestamp: timestamp_ns,
        bar_index: seq,
        owner_strategy_id: None,
    }
}

impl LiveTradeStats {
    #[allow(clippy::too_many_arguments)]
    pub fn on_fill_inner(
        &mut self,
        symbol: &str,
        side: &str,
        quantity: f64,
        price: f64,
        timestamp_ns: i64,
        multiplier: f64,
        commission: f64,
    ) -> PyResult<()> {
        let side = match side.to_ascii_lowercase().as_str() {
            "buy" => OrderSide::Buy,
            "sell" => OrderSide::Sell,
            other => {
                return Err(PyValueError::new_err(format!(
                    "side 须为 buy/sell: {other}"
                )));
            }
        };
        let trade = build_trade(
            symbol,
            side,
            dec(quantity, "quantity")?,
            dec(price, "price")?,
            dec(commission, "commission")?,
            timestamp_ns,
            self.seq + 1,
        );
        let multiplier = dec(multiplier, "multiplier")?;
        self.seq += 1;
        let equity = self.drawdown.summary().last_equity.unwrap_or(self.initial);
        self.tracker
            .process_trade(&trade, multiplier, None, None, equity);
        Ok(())
    }

    /// 带符号净持仓：多头库存减空头库存（TradeEntry.quantity 两侧都为正）。
    pub fn position_inner(&self, symbol: &str) -> f64 {
        let sum = |m: &HashMap<String, VecDeque<TradeEntry>>| {
            m.get(symbol)
                .map(|q| q.iter().map(|e| e.quantity).sum::<Decimal>())
                .unwrap_or_default()
        };
        let net = sum(&self.tracker.long_inventory) - sum(&self.tracker.short_inventory);
        net.to_f64().unwrap_or_default()
    }

    pub fn snapshot_inner(
        &self,
        timestamp_ns: i64,
        equity: f64,
        last_prices: &HashMap<String, f64>,
    ) -> StrategyStatsValues {
        let prices: HashMap<String, Decimal> = last_prices
            .iter()
            .filter_map(|(k, v)| Decimal::from_f64(*v).map(|d| (k.clone(), d)))
            .collect();
        let unrealized = self.tracker.calculate_pnl(Some(prices)).unrealized_pnl;
        let state = StatsState {
            trades: self.tracker.summary(),
            drawdown: self.drawdown,
            initial_cash: self.initial,
        };
        compute_strategy_stats(
            &state,
            Some(timestamp_ns),
            Decimal::from_f64(equity).unwrap_or(self.initial),
            unrealized,
        )
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl LiveTradeStats {
    #[new]
    pub fn new(initial_equity: f64) -> Self {
        let initial = Decimal::from_f64(initial_equity).unwrap_or_default();
        Self {
            tracker: TradeTracker::new(),
            drawdown: DrawdownState::new(initial),
            initial,
            seq: 0,
        }
    }

    /// 记录一笔成交（side 取 "buy" 或 "sell"）.
    #[pyo3(signature = (symbol, side, quantity, price, timestamp_ns, multiplier=1.0, commission=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn on_fill(
        &mut self,
        symbol: &str,
        side: &str,
        quantity: f64,
        price: f64,
        timestamp_ns: i64,
        multiplier: f64,
        commission: f64,
    ) -> PyResult<()> {
        self.on_fill_inner(
            symbol,
            side,
            quantity,
            price,
            timestamp_ns,
            multiplier,
            commission,
        )
    }

    /// 记录一个权益点（用于回撤）.
    fn on_equity(&mut self, timestamp_ns: i64, equity: f64) -> PyResult<()> {
        self.drawdown.push(timestamp_ns, dec(equity, "equity")?);
        Ok(())
    }

    /// tracker 库存的带符号净持仓.
    fn position(&self, symbol: &str) -> f64 {
        self.position_inner(symbol)
    }

    /// 当前统计快照，键与 `ctx.strategy_stats()` 相同.
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        timestamp_ns: i64,
        equity: f64,
        last_prices: HashMap<String, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.snapshot_inner(timestamp_ns, equity, &last_prices)
            .to_pydict(py)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_counts_one_win() {
        let mut s = LiveTradeStats::new(100_000.0);
        s.on_fill_inner("AAA", "buy", 100.0, 10.0, 1, 1.0, 0.0)
            .unwrap();
        s.on_fill_inner("AAA", "sell", 100.0, 11.0, 2, 1.0, 0.0)
            .unwrap();
        let v = s.snapshot_inner(3, 100_100.0, &Default::default());
        assert_eq!(v.winning_trades, 1.0);
        assert_eq!(v.win_rate, 100.0);
        assert_eq!(s.position_inner("AAA"), 0.0);
    }

    #[test]
    fn multiplier_scales_pnl() {
        let mut s = LiveTradeStats::new(1_000_000.0);
        s.on_fill_inner("IF", "buy", 1.0, 3800.0, 1, 300.0, 0.0)
            .unwrap();
        s.on_fill_inner("IF", "sell", 1.0, 3810.0, 2, 300.0, 0.0)
            .unwrap();
        let v = s.snapshot_inner(3, 1_003_000.0, &Default::default());
        assert_eq!(v.largest_win, 3000.0);
    }

    #[test]
    fn bad_side_rejected() {
        let mut s = LiveTradeStats::new(1.0);
        assert!(
            s.on_fill_inner("AAA", "hold", 1.0, 1.0, 1, 1.0, 0.0)
                .is_err()
        );
    }

    #[test]
    fn short_position_is_negative() {
        let mut s = LiveTradeStats::new(1.0);
        s.on_fill_inner("AAA", "sell", 5.0, 10.0, 1, 1.0, 0.0)
            .unwrap();
        assert_eq!(s.position_inner("AAA"), -5.0);
    }

    #[test]
    fn same_summary_as_plain_tracker() {
        let fills = [
            ("AAA", "buy", 100.0, 10.0, 1, 0.5),
            ("AAA", "sell", 60.0, 12.0, 2, 0.3),
            ("AAA", "sell", 80.0, 11.0, 3, 0.4),
            ("AAA", "buy", 20.0, 13.0, 4, 0.1),
            ("BBB", "sell", 10.0, 50.0, 5, 0.2),
            ("BBB", "buy", 10.0, 55.0, 6, 0.2),
        ];
        let mut live = LiveTradeStats::new(100_000.0);
        let mut plain = TradeTracker::new();
        for (i, (sym, side, q, p, ts, c)) in fills.iter().enumerate() {
            live.on_fill_inner(sym, side, *q, *p, *ts, 1.0, *c).unwrap();
            let sd = if *side == "buy" {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            };
            let t = build_trade(
                sym,
                sd,
                Decimal::from_f64(*q).unwrap(),
                Decimal::from_f64(*p).unwrap(),
                Decimal::from_f64(*c).unwrap(),
                *ts,
                i + 1,
            );
            plain.process_trade(&t, Decimal::ONE, None, None, Decimal::from(100_000));
        }
        assert_eq!(live.tracker.summary(), plain.summary());
        let v = live.snapshot_inner(7, 100_000.0, &Default::default());
        let s = plain.summary();
        assert_eq!(v.winning_trades, s.won_count as f64);
        assert_eq!(v.losing_trades, s.lost_count as f64);
        assert_eq!(v.total_commission, s.total_commission.to_f64().unwrap());
        assert_eq!(v.largest_win, s.largest_win.to_f64().unwrap());
        assert_eq!(v.largest_loss, s.largest_loss.to_f64().unwrap());
    }
}
