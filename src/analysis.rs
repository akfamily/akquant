use crate::model::{OrderSide, Trade};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 平仓交易记录.
pub struct ClosedTrade {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub entry_time: i64,
    #[pyo3(get)]
    pub exit_time: i64,
    #[pyo3(get)]
    pub entry_price: f64,
    #[pyo3(get)]
    pub exit_price: f64,
    #[pyo3(get)]
    pub quantity: f64,
    #[pyo3(get)]
    pub direction: String, // "Long" or "Short"
    #[pyo3(get)]
    pub pnl: f64, // Gross PnL (aligned with Backtrader)
    #[pyo3(get)]
    pub net_pnl: f64, // Net PnL (pnl - commission)
    #[pyo3(get)]
    pub return_pct: f64,
    #[pyo3(get)]
    pub commission: f64,
    #[pyo3(get)]
    pub duration_bars: usize,
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 绩效指标.
pub struct PerformanceMetrics {
    #[pyo3(get)]
    pub total_return: f64,
    #[pyo3(get)]
    pub annualized_return: f64,
    #[pyo3(get)]
    pub max_drawdown: f64,
    #[pyo3(get)]
    pub max_drawdown_pct: f64, // Same as max_drawdown but explicit
    #[pyo3(get)]
    pub sharpe_ratio: f64,
    #[pyo3(get)]
    pub sortino_ratio: f64,
    #[pyo3(get)]
    pub volatility: f64,
    #[pyo3(get)]
    pub ulcer_index: f64,
    #[pyo3(get)]
    pub upi: f64,
    #[pyo3(get)]
    pub equity_r2: f64,
    #[pyo3(get)]
    pub std_error: f64,
    #[pyo3(get)]
    pub win_rate: f64,
    #[pyo3(get)]
    pub initial_market_value: f64,
    #[pyo3(get)]
    pub end_market_value: f64,
    #[pyo3(get)]
    pub total_return_pct: f64,
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 交易盈亏统计 (FIFO).
pub struct TradePnL {
    #[pyo3(get)]
    pub gross_pnl: f64,
    #[pyo3(get)]
    pub net_pnl: f64,
    #[pyo3(get)]
    pub total_commission: f64,
    #[pyo3(get)]
    pub total_closed_trades: usize,
    #[pyo3(get)]
    pub won_count: usize,
    #[pyo3(get)]
    pub lost_count: usize,
    #[pyo3(get)]
    pub won_pnl: f64,
    #[pyo3(get)]
    pub lost_pnl: f64,
    #[pyo3(get)]
    pub win_rate: f64,
    #[pyo3(get)]
    pub loss_rate: f64,
    #[pyo3(get)]
    pub unrealized_pnl: f64,

    // New fields
    #[pyo3(get)]
    pub avg_pnl: f64,
    #[pyo3(get)]
    pub avg_return_pct: f64,
    #[pyo3(get)]
    pub avg_trade_bars: f64,
    #[pyo3(get)]
    pub avg_profit: f64,
    #[pyo3(get)]
    pub avg_profit_pct: f64,
    #[pyo3(get)]
    pub avg_winning_trade_bars: f64,
    #[pyo3(get)]
    pub avg_loss: f64,
    #[pyo3(get)]
    pub avg_loss_pct: f64,
    #[pyo3(get)]
    pub avg_losing_trade_bars: f64,
    #[pyo3(get)]
    pub largest_win: f64,
    #[pyo3(get)]
    pub largest_win_pct: f64,
    #[pyo3(get)]
    pub largest_win_bars: f64,
    #[pyo3(get)]
    pub largest_loss: f64,
    #[pyo3(get)]
    pub largest_loss_pct: f64,
    #[pyo3(get)]
    pub largest_loss_bars: f64,
    #[pyo3(get)]
    pub max_wins: usize,
    #[pyo3(get)]
    pub max_losses: usize,
    #[pyo3(get)]
    pub profit_factor: f64,
    #[pyo3(get)]
    pub total_profit: f64, // Same as won_pnl
    #[pyo3(get)]
    pub total_loss: f64, // Same as lost_pnl
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 回测结果.
pub struct BacktestResult {
    #[pyo3(get)]
    pub equity_curve: Vec<(i64, f64)>,
    #[pyo3(get)]
    pub metrics: PerformanceMetrics,
    #[pyo3(get)]
    pub trade_metrics: TradePnL,
    #[pyo3(get)]
    pub trades: Vec<ClosedTrade>,
    #[pyo3(get)]
    pub daily_positions: Vec<(i64, HashMap<String, f64>)>,
}

impl BacktestResult {
    pub fn calculate(
        equity_curve_decimal: Vec<(i64, Decimal)>,
        daily_positions_decimal: Vec<(i64, HashMap<String, Decimal>)>,
        trade_pnl: TradePnL,
        trades: Vec<ClosedTrade>,
    ) -> Self {
        // Convert equity_curve to f64 for storage/python
        let equity_curve: Vec<(i64, f64)> = equity_curve_decimal
            .iter()
            .map(|(t, d)| (*t, d.to_f64().unwrap_or_default()))
            .collect();

        // Convert daily_positions to f64
        let daily_positions: Vec<(i64, HashMap<String, f64>)> = daily_positions_decimal
            .into_iter()
            .map(|(t, positions)| {
                let pos_f64 = positions
                    .into_iter()
                    .map(|(k, v)| (k, v.to_f64().unwrap_or_default()))
                    .collect();
                (t, pos_f64)
            })
            .collect();

        if equity_curve_decimal.is_empty() {
            return BacktestResult {
                equity_curve,
                metrics: PerformanceMetrics {
                    total_return: 0.0,
                    annualized_return: 0.0,
                    max_drawdown: 0.0,
                    max_drawdown_pct: 0.0,
                    sharpe_ratio: 0.0,
                    sortino_ratio: 0.0,
                    volatility: 0.0,
                    ulcer_index: 0.0,
                    upi: 0.0,
                    equity_r2: 0.0,
                    std_error: 0.0,
                    win_rate: 0.0,
                    initial_market_value: 0.0,
                    end_market_value: 0.0,
                    total_return_pct: 0.0,
                },
                trade_metrics: trade_pnl,
                trades,
                daily_positions,
            };
        }

        let initial_equity = equity_curve_decimal.first().unwrap().1;
        let final_equity = equity_curve_decimal.last().unwrap().1;

        // 1. Total Return (Decimal)
        let total_return_dec = if !initial_equity.is_zero() {
            (final_equity - initial_equity) / initial_equity
        } else {
            Decimal::ZERO
        };

        // 2. Max Drawdown & Ulcer Index
        let mut max_drawdown_dec = Decimal::ZERO;
        let mut peak = initial_equity;
        let mut sum_sq_drawdown = 0.0;

        for (_, equity) in &equity_curve_decimal {
            if *equity > peak {
                peak = *equity;
            }
            let drawdown_dec = if !peak.is_zero() {
                (peak - *equity) / peak
            } else {
                Decimal::ZERO
            };
            if drawdown_dec > max_drawdown_dec {
                max_drawdown_dec = drawdown_dec;
            }

            let dd_f64 = drawdown_dec.to_f64().unwrap_or_default();
            sum_sq_drawdown += dd_f64 * dd_f64;
        }

        let ulcer_index = (sum_sq_drawdown / equity_curve.len() as f64).sqrt();

        // 3. Returns Series for Volatility & Sharpe
        let mut returns = Vec::new();
        let mut downside_returns = Vec::new();
        for i in 1..equity_curve.len() {
            let prev = equity_curve[i - 1].1;
            let curr = equity_curve[i].1;
            if prev != 0.0 {
                let r = (curr - prev) / prev;
                returns.push(r);
                if r < 0.0 {
                    downside_returns.push(r);
                } else {
                    downside_returns.push(0.0);
                }
            }
        }

        // 4. Annualized Return & Volatility
        let start_ts = equity_curve.first().unwrap().0;
        let end_ts = equity_curve.last().unwrap().0;
        let duration_seconds = (end_ts - start_ts) as f64 / 1_000_000_000.0;
        let years = duration_seconds / (365.0 * 24.0 * 3600.0);

        let total_return_f64 = total_return_dec.to_f64().unwrap_or_default();

        let annualized_return = if years > 0.0 {
            (1.0 + total_return_f64).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        let mean_return = if !returns.is_empty() {
            returns.iter().sum::<f64>() / returns.len() as f64
        } else {
            0.0
        };

        let variance = if returns.len() > 1 {
            returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / (returns.len() - 1) as f64
        } else {
            0.0
        };

        let std_dev = variance.sqrt();
        let annualized_volatility = std_dev * (252.0f64).sqrt();

        // 5. Sharpe Ratio
        let risk_free_rate = 0.0; // Assume 0 for simplicity or pass config
        let sharpe_ratio = if annualized_volatility != 0.0 {
            (annualized_return - risk_free_rate) / annualized_volatility
        } else {
            0.0
        };

        // 6. Sortino Ratio
        // Downside deviation
        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_std_dev = downside_variance.sqrt();
        let annualized_downside_volatility = downside_std_dev * (252.0f64).sqrt();

        let sortino_ratio = if annualized_downside_volatility != 0.0 {
            (annualized_return - risk_free_rate) / annualized_downside_volatility
        } else {
            0.0
        };

        // 7. UPI
        let upi = if ulcer_index != 0.0 {
            (annualized_return - risk_free_rate) / ulcer_index
        } else {
            0.0
        };

        // 8. R2 and Std Error (Linear Regression of Equity)
        // X = index, Y = Equity
        let n = equity_curve.len() as f64;
        let sum_x = (0..equity_curve.len()).map(|i| i as f64).sum::<f64>();
        let sum_y = equity_curve.iter().map(|(_, y)| *y).sum::<f64>();
        let sum_xy = equity_curve
            .iter()
            .enumerate()
            .map(|(i, (_, y))| i as f64 * *y)
            .sum::<f64>();
        let sum_xx = (0..equity_curve.len()).map(|i| (i * i) as f64).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // R2
        let ss_tot = equity_curve
            .iter()
            .map(|(_, y)| (y - (sum_y / n)).powi(2))
            .sum::<f64>();
        let ss_res = equity_curve
            .iter()
            .enumerate()
            .map(|(i, (_, y))| (y - (slope * i as f64 + intercept)).powi(2))
            .sum::<f64>();

        let equity_r2 = if ss_tot != 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Standard Error of Estimate
        let std_error = if n > 2.0 {
            (ss_res / (n - 2.0)).sqrt()
        } else {
            0.0
        };

        BacktestResult {
            equity_curve,
            metrics: PerformanceMetrics {
                total_return: total_return_f64,
                annualized_return,
                max_drawdown: max_drawdown_dec.to_f64().unwrap_or_default(),
                max_drawdown_pct: max_drawdown_dec.to_f64().unwrap_or_default() * 100.0,
                sharpe_ratio,
                sortino_ratio,
                volatility: annualized_volatility,
                ulcer_index,
                upi,
                equity_r2,
                std_error,
                win_rate: trade_pnl.win_rate,
                initial_market_value: initial_equity.to_f64().unwrap_or_default(),
                end_market_value: final_equity.to_f64().unwrap_or_default(),
                total_return_pct: total_return_f64 * 100.0,
            },
            trade_metrics: trade_pnl,
            trades,
            daily_positions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::OrderSide;
    use rust_decimal::Decimal;
    use std::collections::HashMap;

    fn create_trade(
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
    ) -> Trade {
        Trade {
            id: "trade_id".to_string(),
            order_id: "order_id".to_string(),
            symbol: symbol.to_string(),
            side,
            quantity,
            price,
            commission,
            timestamp: 0,
            bar_index: 0,
        }
    }

    fn analyze_trades(
        trades: Vec<Trade>,
        current_prices: Option<HashMap<String, Decimal>>,
    ) -> (TradePnL, Vec<ClosedTrade>) {
        let mut tracker = TradeTracker::new();
        for trade in trades {
            tracker.process_trade(&trade);
        }
        (tracker.calculate_pnl(current_prices), tracker.closed_trades.to_vec())
    }

    #[test]
    fn test_trade_analyzer_long_profit() {
        // Buy 100 @ 10, Sell 100 @ 12
        let t1 = create_trade("AAPL", OrderSide::Buy, Decimal::from(100), Decimal::from(10), Decimal::ZERO);
        let t2 = create_trade("AAPL", OrderSide::Sell, Decimal::from(100), Decimal::from(12), Decimal::ZERO);

        let (pnl, _) = analyze_trades(vec![t1, t2], None);

        assert_eq!(pnl.total_closed_trades, 1);
        assert_eq!(pnl.won_count, 1);
        assert_eq!(pnl.gross_pnl, 200.0); // (12-10)*100
        assert_eq!(pnl.win_rate, 1.0);
    }

    #[test]
    fn test_trade_analyzer_short_loss() {
        // Sell 100 @ 10, Buy 100 @ 12
        let t1 = create_trade("AAPL", OrderSide::Sell, Decimal::from(100), Decimal::from(10), Decimal::ZERO);
        let t2 = create_trade("AAPL", OrderSide::Buy, Decimal::from(100), Decimal::from(12), Decimal::ZERO);

        let (pnl, _) = analyze_trades(vec![t1, t2], None);

        assert_eq!(pnl.total_closed_trades, 1);
        assert_eq!(pnl.lost_count, 1);
        assert_eq!(pnl.gross_pnl, -200.0); // (10-12)*100
        assert_eq!(pnl.loss_rate, 1.0);
    }

    #[test]
    fn test_trade_analyzer_fifo() {
        // Buy 100 @ 10
        // Buy 100 @ 12
        // Sell 150 @ 11
        let t1 = create_trade("AAPL", OrderSide::Buy, Decimal::from(100), Decimal::from(10), Decimal::ZERO);
        let t2 = create_trade("AAPL", OrderSide::Buy, Decimal::from(100), Decimal::from(12), Decimal::ZERO);
        let t3 = create_trade("AAPL", OrderSide::Sell, Decimal::from(150), Decimal::from(11), Decimal::ZERO);

        let (pnl, _) = analyze_trades(vec![t1, t2, t3], None);

        assert_eq!(pnl.gross_pnl, 50.0);
        assert_eq!(pnl.total_closed_trades, 2);
    }

    #[test]
    fn test_unrealized_pnl() {
        // Buy 100 @ 10
        let t1 = create_trade("AAPL", OrderSide::Buy, Decimal::from(100), Decimal::from(10), Decimal::ZERO);

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), Decimal::from(15));

        let (pnl, _) = analyze_trades(vec![t1], Some(prices));

        assert_eq!(pnl.total_closed_trades, 0);
        assert_eq!(pnl.unrealized_pnl, 500.0); // (15-10)*100
    }
}

#[derive(Debug, Clone)]
pub struct TradeTracker {
    pub long_inventory: HashMap<String, VecDeque<(Decimal, Decimal, Decimal, usize, i64)>>,
    pub short_inventory: HashMap<String, VecDeque<(Decimal, Decimal, Decimal, usize, i64)>>,
    pub closed_trades: Arc<Vec<ClosedTrade>>,
    pub closed_trades_stats: Vec<(Decimal, Decimal, Decimal, bool)>, // (pnl, return_pct, bars, is_win)

    // Aggregate stats
    pub total_pnl: Decimal,
    pub total_commission: Decimal,
    pub won_count: usize,
    pub lost_count: usize,
    pub won_pnl: Decimal,
    pub lost_pnl: Decimal,
    pub max_wins: usize,
    pub max_losses: usize,
    pub current_wins: usize,
    pub current_losses: usize,
}

impl TradeTracker {
    pub fn new() -> Self {
        Self {
            long_inventory: HashMap::new(),
            short_inventory: HashMap::new(),
            closed_trades: Arc::new(Vec::new()),
            closed_trades_stats: Vec::new(),
            total_pnl: Decimal::ZERO,
            total_commission: Decimal::ZERO,
            won_count: 0,
            lost_count: 0,
            won_pnl: Decimal::ZERO,
            lost_pnl: Decimal::ZERO,
            max_wins: 0,
            max_losses: 0,
            current_wins: 0,
            current_losses: 0,
        }
    }

    pub fn process_trade(&mut self, trade: &Trade) {
        let symbol = trade.symbol.clone();
        let side = trade.side;
        let qty = trade.quantity;
        let price = trade.price;
        let comm = trade.commission;
        let bar_idx = trade.bar_index;
        let timestamp = trade.timestamp;

        self.total_commission += comm;

        let mut remaining_qty = qty;

        match side {
            OrderSide::Buy => {
                // Try to cover shorts
                if let Some(inventory) = self.short_inventory.get_mut(&symbol) {
                    while remaining_qty > Decimal::ZERO && !inventory.is_empty() {
                        let (match_qty, match_price, match_comm, match_bar_idx, match_timestamp) =
                            inventory.front_mut().unwrap();
                        let covered_qty = remaining_qty.min(*match_qty);

                        // Short PnL = (Entry Price - Exit Price) * Qty
                        let pnl = (*match_price - price) * covered_qty;
                        self.total_pnl += pnl;

                        let entry_val = *match_price * covered_qty;
                        let ret_pct = if !entry_val.is_zero() {
                            pnl / entry_val
                        } else {
                            Decimal::ZERO
                        };
                        let bars = if bar_idx >= *match_bar_idx {
                            Decimal::from((bar_idx - *match_bar_idx) as i64)
                        } else {
                            Decimal::ZERO
                        };

                        // Pro-rate commission for entry and exit
                        // Entry comm (partial)
                        let entry_comm_part = if *match_qty > Decimal::ZERO {
                            *match_comm * (covered_qty / *match_qty)
                        } else {
                            Decimal::ZERO
                        };
                        // Exit comm (partial of current trade)
                        let exit_comm_part = if qty > Decimal::ZERO {
                            comm * (covered_qty / qty)
                        } else {
                            Decimal::ZERO
                        };
                        let total_trade_comm = entry_comm_part + exit_comm_part;

                        let to_f64 = |d: Decimal| d.to_f64().unwrap_or_default();

                        Arc::make_mut(&mut self.closed_trades).push(ClosedTrade {
                            symbol: symbol.clone(),
                            entry_time: *match_timestamp,
                            exit_time: timestamp,
                            entry_price: to_f64(*match_price),
                            exit_price: to_f64(price),
                            quantity: to_f64(covered_qty),
                            direction: "Short".to_string(),
                            pnl: to_f64(pnl),
                            net_pnl: to_f64(pnl - total_trade_comm),
                            return_pct: to_f64(ret_pct),
                            commission: to_f64(total_trade_comm),
                            duration_bars: bars.to_usize().unwrap_or(0),
                        });

                        if pnl > Decimal::ZERO {
                            self.won_count += 1;
                            self.won_pnl += pnl;
                            self.closed_trades_stats.push((pnl, ret_pct, bars, true));

                            self.current_wins += 1;
                            self.current_losses = 0;
                            if self.current_wins > self.max_wins {
                                self.max_wins = self.current_wins;
                            }
                        } else {
                            self.lost_count += 1;
                            self.lost_pnl += pnl;
                            self.closed_trades_stats.push((pnl, ret_pct, bars, false));

                            self.current_losses += 1;
                            self.current_wins = 0;
                            if self.current_losses > self.max_losses {
                                self.max_losses = self.current_losses;
                            }
                        }

                        remaining_qty -= covered_qty;
                        *match_qty -= covered_qty;
                        // Reduce remaining commission in inventory proportionally
                        *match_comm -= entry_comm_part;

                        if *match_qty <= Decimal::new(1, 6) {
                            inventory.pop_front();
                        }
                    }
                }

                if remaining_qty > Decimal::ZERO {
                    // Calculate remaining commission for this part
                    let remaining_comm = if qty > Decimal::ZERO {
                        comm * (remaining_qty / qty)
                    } else {
                        Decimal::ZERO
                    };
                    self.long_inventory
                        .entry(symbol.clone())
                        .or_default()
                        .push_back((remaining_qty, price, remaining_comm, bar_idx, timestamp));
                }
            }
            OrderSide::Sell => {
                // Try to close longs
                if let Some(inventory) = self.long_inventory.get_mut(&symbol) {
                    while remaining_qty > Decimal::ZERO && !inventory.is_empty() {
                        let (match_qty, match_price, match_comm, match_bar_idx, match_timestamp) =
                            inventory.front_mut().unwrap();
                        let covered_qty = remaining_qty.min(*match_qty);

                        // Long PnL = (Exit Price - Entry Price) * Qty
                        let pnl = (price - *match_price) * covered_qty;
                        self.total_pnl += pnl;

                        let entry_val = *match_price * covered_qty;
                        let ret_pct = if !entry_val.is_zero() {
                            pnl / entry_val
                        } else {
                            Decimal::ZERO
                        };
                        let bars = if bar_idx >= *match_bar_idx {
                            Decimal::from((bar_idx - *match_bar_idx) as i64)
                        } else {
                            Decimal::ZERO
                        };

                        // Pro-rate commission
                        let entry_comm_part = if *match_qty > Decimal::ZERO {
                            *match_comm * (covered_qty / *match_qty)
                        } else {
                            Decimal::ZERO
                        };
                        let exit_comm_part = if qty > Decimal::ZERO {
                            comm * (covered_qty / qty)
                        } else {
                            Decimal::ZERO
                        };
                        let total_trade_comm = entry_comm_part + exit_comm_part;

                        let to_f64 = |d: Decimal| d.to_f64().unwrap_or_default();

                        Arc::make_mut(&mut self.closed_trades).push(ClosedTrade {
                            symbol: symbol.clone(),
                            entry_time: *match_timestamp,
                            exit_time: timestamp,
                            entry_price: to_f64(*match_price),
                            exit_price: to_f64(price),
                            quantity: to_f64(covered_qty),
                            direction: "Long".to_string(),
                            pnl: to_f64(pnl),
                            net_pnl: to_f64(pnl - total_trade_comm),
                            return_pct: to_f64(ret_pct),
                            commission: to_f64(total_trade_comm),
                            duration_bars: bars.to_usize().unwrap_or(0),
                        });

                        if pnl > Decimal::ZERO {
                            self.won_count += 1;
                            self.won_pnl += pnl;
                            self.closed_trades_stats.push((pnl, ret_pct, bars, true));

                            self.current_wins += 1;
                            self.current_losses = 0;
                            if self.current_wins > self.max_wins {
                                self.max_wins = self.current_wins;
                            }
                        } else {
                            self.lost_count += 1;
                            self.lost_pnl += pnl;
                            self.closed_trades_stats.push((pnl, ret_pct, bars, false));

                            self.current_losses += 1;
                            self.current_wins = 0;
                            if self.current_losses > self.max_losses {
                                self.max_losses = self.current_losses;
                            }
                        }

                        remaining_qty -= covered_qty;
                        *match_qty -= covered_qty;
                        *match_comm -= entry_comm_part;

                        if *match_qty <= Decimal::new(1, 6) {
                            inventory.pop_front();
                        }
                    }
                }

                if remaining_qty > Decimal::ZERO {
                    let remaining_comm = if qty > Decimal::ZERO {
                        comm * (remaining_qty / qty)
                    } else {
                        Decimal::ZERO
                    };
                    self.short_inventory
                        .entry(symbol.clone())
                        .or_default()
                        .push_back((remaining_qty, price, remaining_comm, bar_idx, timestamp));
                }
            }
        }
    }

    pub fn calculate_pnl(&self, current_prices: Option<HashMap<String, Decimal>>) -> TradePnL {
        let mut unrealized_pnl = Decimal::ZERO;

        if let Some(prices) = current_prices {
            for (symbol, inventory) in &self.long_inventory {
                if let Some(price) = prices.get(symbol) {
                    for (qty, entry_price, _, _, _) in inventory {
                        unrealized_pnl += (*price - *entry_price) * *qty;
                    }
                }
            }
            for (symbol, inventory) in &self.short_inventory {
                if let Some(price) = prices.get(symbol) {
                    for (qty, entry_price, _, _, _) in inventory {
                        unrealized_pnl += (*entry_price - *price) * *qty;
                    }
                }
            }
        }

        let total_closed_trades = self.closed_trades.len();
        let win_rate = if total_closed_trades > 0 {
            self.won_count as f64 / total_closed_trades as f64
        } else {
            0.0
        };
        let loss_rate = if total_closed_trades > 0 {
            self.lost_count as f64 / total_closed_trades as f64
        } else {
            0.0
        };

        // Avg stats
        let mut sum_pnl = Decimal::ZERO;
        let mut sum_ret = Decimal::ZERO;
        let mut sum_bars = Decimal::ZERO;
        let mut sum_win_bars = Decimal::ZERO;
        let mut sum_loss_bars = Decimal::ZERO;

        let mut largest_win = Decimal::ZERO;
        let mut largest_win_pct = Decimal::ZERO;
        let mut largest_win_bars = Decimal::ZERO;
        let mut largest_loss = Decimal::ZERO;
        let mut largest_loss_pct = Decimal::ZERO;
        let mut largest_loss_bars = Decimal::ZERO;

        for (pnl, ret, bars, is_win) in &self.closed_trades_stats {
            sum_pnl += pnl;
            sum_ret += ret;
            sum_bars += bars;

            if *is_win {
                sum_win_bars += bars;
                if *pnl > largest_win {
                    largest_win = *pnl;
                    largest_win_pct = *ret;
                    largest_win_bars = *bars;
                }
            } else {
                sum_loss_bars += bars;
                if *pnl < largest_loss {
                    largest_loss = *pnl;
                    largest_loss_pct = *ret;
                    largest_loss_bars = *bars;
                }
            }
        }

        let to_f64 = |d: Decimal| d.to_f64().unwrap_or_default();

        let avg_pnl = if total_closed_trades > 0 {
            to_f64(sum_pnl) / total_closed_trades as f64
        } else {
            0.0
        };
        let avg_return_pct = if total_closed_trades > 0 {
            to_f64(sum_ret) / total_closed_trades as f64
        } else {
            0.0
        };
        let avg_trade_bars = if total_closed_trades > 0 {
            to_f64(sum_bars) / total_closed_trades as f64
        } else {
            0.0
        };

        let avg_profit = if self.won_count > 0 {
            to_f64(self.won_pnl) / self.won_count as f64
        } else {
            0.0
        };
        let avg_profit_pct = if self.won_count > 0 {
            let sum_win_ret: Decimal = self.closed_trades_stats.iter().filter(|(_, _, _, w)| *w).map(|(_, r, _, _)| *r).sum();
             to_f64(sum_win_ret) / self.won_count as f64
        } else {
            0.0
        };
        let avg_winning_trade_bars = if self.won_count > 0 {
            to_f64(sum_win_bars) / self.won_count as f64
        } else {
            0.0
        };

        let avg_loss = if self.lost_count > 0 {
            to_f64(self.lost_pnl) / self.lost_count as f64
        } else {
            0.0
        };
        let avg_loss_pct = if self.lost_count > 0 {
             let sum_loss_ret: Decimal = self.closed_trades_stats.iter().filter(|(_, _, _, w)| !*w).map(|(_, r, _, _)| *r).sum();
             to_f64(sum_loss_ret) / self.lost_count as f64
        } else {
            0.0
        };
        let avg_losing_trade_bars = if self.lost_count > 0 {
            to_f64(sum_loss_bars) / self.lost_count as f64
        } else {
            0.0
        };

        let profit_factor = if self.lost_pnl.abs() > Decimal::ZERO {
            to_f64(self.won_pnl / self.lost_pnl.abs())
        } else if self.won_pnl > Decimal::ZERO {
            f64::INFINITY
        } else {
            0.0
        };

        TradePnL {
            gross_pnl: to_f64(self.total_pnl),
            net_pnl: to_f64(self.total_pnl - self.total_commission),
            total_commission: to_f64(self.total_commission),
            total_closed_trades,
            won_count: self.won_count,
            lost_count: self.lost_count,
            won_pnl: to_f64(self.won_pnl),
            lost_pnl: to_f64(self.lost_pnl),
            win_rate,
            loss_rate,
            unrealized_pnl: to_f64(unrealized_pnl),
            avg_pnl,
            avg_return_pct,
            avg_trade_bars,
            avg_profit,
            avg_profit_pct,
            avg_winning_trade_bars,
            avg_loss,
            avg_loss_pct,
            avg_losing_trade_bars,
            largest_win: to_f64(largest_win),
            largest_win_pct: to_f64(largest_win_pct),
            largest_win_bars: to_f64(largest_win_bars),
            largest_loss: to_f64(largest_loss),
            largest_loss_pct: to_f64(largest_loss_pct),
            largest_loss_bars: to_f64(largest_loss_bars),
            max_wins: self.max_wins,
            max_losses: self.max_losses,
            profit_factor,
            total_profit: to_f64(self.won_pnl),
            total_loss: to_f64(self.lost_pnl),
        }
    }
}
