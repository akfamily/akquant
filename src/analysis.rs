use crate::model::{Order, OrderSide, Trade};
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
    pub side: String, // "long" or "short"
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
    #[pyo3(get)]
    pub duration: i64, // Duration in nanoseconds
}

#[gen_stub_pymethods]
#[pymethods]
impl ClosedTrade {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
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
    pub max_drawdown_value: f64,
    #[pyo3(get)]
    pub max_drawdown_pct: f64, // Same as max_drawdown but explicit
    #[pyo3(get)]
    pub sharpe_ratio: f64,
    #[pyo3(get)]
    pub sortino_ratio: f64,
    #[pyo3(get)]
    pub calmar_ratio: f64,
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
    #[pyo3(get)]
    pub start_time: i64,
    #[pyo3(get)]
    pub end_time: i64,
    #[pyo3(get)]
    pub duration: i64,
    #[pyo3(get)]
    pub total_bars: usize,

    // New risk metrics
    #[pyo3(get)]
    pub exposure_time_pct: f64,
    #[pyo3(get)]
    pub var_95: f64,
    #[pyo3(get)]
    pub var_99: f64,
    #[pyo3(get)]
    pub cvar_95: f64,
    #[pyo3(get)]
    pub cvar_99: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PerformanceMetrics {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    pub fn __getitem__(&self, py: Python, key: &str) -> PyResult<Py<PyAny>> {
        let v = match key {
            "total_return" => self.total_return.into_pyobject(py).unwrap().into_any().unbind(),
            "annualized_return" => self.annualized_return.into_pyobject(py).unwrap().into_any().unbind(),
            "max_drawdown" => self.max_drawdown.into_pyobject(py).unwrap().into_any().unbind(),
            "max_drawdown_value" => self.max_drawdown_value.into_pyobject(py).unwrap().into_any().unbind(),
            "max_drawdown_pct" => self.max_drawdown_pct.into_pyobject(py).unwrap().into_any().unbind(),
            "sharpe_ratio" => self.sharpe_ratio.into_pyobject(py).unwrap().into_any().unbind(),
            "sortino_ratio" => self.sortino_ratio.into_pyobject(py).unwrap().into_any().unbind(),
            "calmar_ratio" => self.calmar_ratio.into_pyobject(py).unwrap().into_any().unbind(),
            "volatility" => self.volatility.into_pyobject(py).unwrap().into_any().unbind(),
            "ulcer_index" => self.ulcer_index.into_pyobject(py).unwrap().into_any().unbind(),
            "upi" => self.upi.into_pyobject(py).unwrap().into_any().unbind(),
            "equity_r2" => self.equity_r2.into_pyobject(py).unwrap().into_any().unbind(),
            "std_error" => self.std_error.into_pyobject(py).unwrap().into_any().unbind(),
            "win_rate" => self.win_rate.into_pyobject(py).unwrap().into_any().unbind(),
            "initial_market_value" => self.initial_market_value.into_pyobject(py).unwrap().into_any().unbind(),
            "end_market_value" => self.end_market_value.into_pyobject(py).unwrap().into_any().unbind(),
            "total_return_pct" => self.total_return_pct.into_pyobject(py).unwrap().into_any().unbind(),
            "start_time" => self.start_time.into_pyobject(py).unwrap().into_any().unbind(),
            "end_time" => self.end_time.into_pyobject(py).unwrap().into_any().unbind(),
            "duration" => self.duration.into_pyobject(py).unwrap().into_any().unbind(),
            "total_bars" => self.total_bars.into_pyobject(py).unwrap().into_any().unbind(),
            "exposure_time_pct" => self.exposure_time_pct.into_pyobject(py).unwrap().into_any().unbind(),
            "var_95" => self.var_95.into_pyobject(py).unwrap().into_any().unbind(),
            "var_99" => self.var_99.into_pyobject(py).unwrap().into_any().unbind(),
            "cvar_95" => self.cvar_95.into_pyobject(py).unwrap().into_any().unbind(),
            "cvar_99" => self.cvar_99.into_pyobject(py).unwrap().into_any().unbind(),
            _ => return Err(pyo3::exceptions::PyKeyError::new_err(format!("Key '{}' not found", key))),
        };
        Ok(v)
    }
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
    #[pyo3(get)]
    pub sqn: f64,
    #[pyo3(get)]
    pub kelly_criterion: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl TradePnL {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 每日持仓快照.
pub struct PositionSnapshot {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub quantity: f64,
    #[pyo3(get)]
    pub long_shares: f64,
    #[pyo3(get)]
    pub short_shares: f64,
    #[pyo3(get)]
    pub close: f64,
    #[pyo3(get)]
    pub equity: f64,
    #[pyo3(get)]
    pub market_value: f64,
    #[pyo3(get)]
    pub margin: f64,
    #[pyo3(get)]
    pub unrealized_pnl: f64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PositionSnapshot {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
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
    pub snapshots: Vec<(i64, Vec<PositionSnapshot>)>,
    #[pyo3(get)]
    pub orders: Vec<Order>,
    #[pyo3(get)]
    pub executions: Vec<Trade>,
}

impl BacktestResult {
    pub fn calculate(
        equity_curve_decimal: Vec<(i64, Decimal)>,
        snapshots: Vec<(i64, Vec<PositionSnapshot>)>,
        trade_pnl: TradePnL,
        trades: Vec<ClosedTrade>,
        initial_capital: Decimal,
        orders: Vec<Order>,
        executions: Vec<Trade>,
    ) -> Self {
        // Convert equity_curve to f64 for storage/python
        let equity_curve: Vec<(i64, f64)> = equity_curve_decimal
            .iter()
            .map(|(t, d)| (*t, d.to_f64().unwrap_or_default()))
            .collect();

        if equity_curve_decimal.is_empty() {
            return BacktestResult {
                equity_curve,
                metrics: PerformanceMetrics {
                    total_return: 0.0,
                    annualized_return: 0.0,
                    max_drawdown: 0.0,
                    max_drawdown_value: 0.0,
                    max_drawdown_pct: 0.0,
                    sharpe_ratio: 0.0,
                    sortino_ratio: 0.0,
                    calmar_ratio: 0.0,
                    volatility: 0.0,
                    ulcer_index: 0.0,
                    upi: 0.0,
                    equity_r2: 0.0,
                    std_error: 0.0,
                    win_rate: 0.0,
                    initial_market_value: initial_capital.to_f64().unwrap_or_default(),
                    end_market_value: initial_capital.to_f64().unwrap_or_default(),
                    total_return_pct: 0.0,
                    start_time: 0,
                    end_time: 0,
                    duration: 0,
                    total_bars: 0,
                    exposure_time_pct: 0.0,
                    var_95: 0.0,
                    var_99: 0.0,
                    cvar_95: 0.0,
                    cvar_99: 0.0,
                },
                trade_metrics: trade_pnl,
                trades,
                snapshots,
                orders,
                executions,
            };
        }

        let initial_equity = initial_capital;
        let final_equity = equity_curve_decimal.last().unwrap().1;

        // 1. Total Return (Decimal)
        let total_return_dec = if !initial_equity.is_zero() {
            (final_equity - initial_equity) / initial_equity
        } else {
            Decimal::ZERO
        };

        // 2. Max Drawdown & Ulcer Index
        let mut max_drawdown_dec = Decimal::ZERO;
        let mut max_drawdown_val = Decimal::ZERO;
        let mut peak = initial_equity;
        let mut sum_sq_drawdown = 0.0;

        for (_, equity) in &equity_curve_decimal {
            if *equity > peak {
                peak = *equity;
            }
            let drawdown_val = peak - *equity;
            if drawdown_val > max_drawdown_val {
                max_drawdown_val = drawdown_val;
            }

            let drawdown_dec = if !peak.is_zero() {
                drawdown_val / peak
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

        // 8. Calmar Ratio
        let max_dd_f64 = max_drawdown_dec.to_f64().unwrap_or_default();
        let calmar_ratio = if max_dd_f64 != 0.0 {
            annualized_return / max_dd_f64
        } else {
            0.0
        };

        // 9. R2 and Std Error (Linear Regression of Equity)
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

        let total_bars = equity_curve.len();

        // 10. Exposure Time %
        let exposure_count = snapshots.iter().filter(|(_, positions)| {
             positions.iter().any(|p| p.quantity != 0.0)
        }).count();
        let exposure_time_pct = if !snapshots.is_empty() {
             (exposure_count as f64 / snapshots.len() as f64) * 100.0
        } else {
             0.0
        };

        // 11. VaR and CVaR
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let calculate_risk_metrics = |alpha: f64, sorted_rets: &Vec<f64>| -> (f64, f64) {
             if sorted_rets.is_empty() {
                 return (0.0, 0.0);
             }
             let idx = ((sorted_rets.len() as f64) * alpha).floor() as usize;
             let idx = idx.min(sorted_rets.len() - 1);
             let var = sorted_rets[idx];

             let cvar_sum: f64 = sorted_rets[0..=idx].iter().sum();
             let cvar = cvar_sum / (idx + 1) as f64;
             (var, cvar)
        };

        let (var_95, cvar_95) = calculate_risk_metrics(0.05, &sorted_returns);
        let (var_99, cvar_99) = calculate_risk_metrics(0.01, &sorted_returns);

        BacktestResult {
            equity_curve,
            metrics: PerformanceMetrics {
                total_return: total_return_f64,
                annualized_return,
                max_drawdown: max_drawdown_dec.to_f64().unwrap_or_default(),
                max_drawdown_value: max_drawdown_val.to_f64().unwrap_or_default(),
                max_drawdown_pct: max_drawdown_dec.to_f64().unwrap_or_default() * 100.0,
                sharpe_ratio,
                sortino_ratio,
                calmar_ratio,
                volatility: annualized_volatility,
                ulcer_index,
                upi,
                equity_r2,
                std_error,
                win_rate: trade_pnl.win_rate,
                initial_market_value: initial_equity.to_f64().unwrap_or_default(),
                end_market_value: final_equity.to_f64().unwrap_or_default(),
                total_return_pct: total_return_f64 * 100.0,
                start_time: start_ts,
                end_time: end_ts,
                duration: (end_ts - start_ts),
                total_bars,
                exposure_time_pct,
                var_95,
                var_99,
                cvar_95,
                cvar_99,
            },
            trade_metrics: trade_pnl,
            trades,
            snapshots,
            orders,
            executions,
        }
    }
}

#[pymethods]
impl BacktestResult {
    /// Get trades as a dictionary of columns for fast DataFrame creation.
    pub fn get_trades_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let n = self.trades.len();
        let mut symbols = Vec::with_capacity(n);
        let mut entry_times = Vec::with_capacity(n);
        let mut exit_times = Vec::with_capacity(n);
        let mut entry_prices = Vec::with_capacity(n);
        let mut exit_prices = Vec::with_capacity(n);
        let mut quantities = Vec::with_capacity(n);
        let mut sides = Vec::with_capacity(n);
        let mut pnls = Vec::with_capacity(n);
        let mut net_pnls = Vec::with_capacity(n);
        let mut return_pcts = Vec::with_capacity(n);
        let mut commissions = Vec::with_capacity(n);
        let mut duration_bars = Vec::with_capacity(n);
        let mut durations = Vec::with_capacity(n);

        for t in &self.trades {
            symbols.push(t.symbol.clone());
            entry_times.push(t.entry_time);
            exit_times.push(t.exit_time);
            entry_prices.push(t.entry_price);
            exit_prices.push(t.exit_price);
            quantities.push(t.quantity);
            sides.push(t.side.clone());
            pnls.push(t.pnl);
            net_pnls.push(t.net_pnl);
            return_pcts.push(t.return_pct);
            commissions.push(t.commission);
            duration_bars.push(t.duration_bars);
            durations.push(t.duration);
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("symbol", symbols)?;
        dict.set_item("entry_time", entry_times)?;
        dict.set_item("exit_time", exit_times)?;
        dict.set_item("entry_price", entry_prices)?;
        dict.set_item("exit_price", exit_prices)?;
        dict.set_item("quantity", quantities)?;
        dict.set_item("side", sides)?;
        dict.set_item("pnl", pnls)?;
        dict.set_item("net_pnl", net_pnls)?;
        dict.set_item("return_pct", return_pcts)?;
        dict.set_item("commission", commissions)?;
        dict.set_item("duration_bars", duration_bars)?;
        dict.set_item("duration", durations)?;

        Ok(dict.into())
    }

    /// Get positions history as a dictionary of columns.
    pub fn get_positions_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let mut symbols = Vec::new();
        let mut dates = Vec::new(); // int64 timestamps
        let mut long_shares = Vec::new();
        let mut short_shares = Vec::new();
        let mut closes = Vec::new();
        let mut equities = Vec::new();
        let mut market_values = Vec::new();
        let mut margins = Vec::new();
        let mut unrealized_pnls = Vec::new();

        for (ts, snapshots) in &self.snapshots {
            for s in snapshots {
                symbols.push(s.symbol.clone());
                dates.push(*ts);
                long_shares.push(s.long_shares);
                short_shares.push(s.short_shares);
                closes.push(s.close);
                equities.push(s.equity);
                market_values.push(s.market_value);
                margins.push(s.margin);
                unrealized_pnls.push(s.unrealized_pnl);
            }
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("symbol", symbols)?;
        dict.set_item("date", dates)?;
        dict.set_item("long_shares", long_shares)?;
        dict.set_item("short_shares", short_shares)?;
        dict.set_item("close", closes)?;
        dict.set_item("equity", equities)?;
        dict.set_item("market_value", market_values)?;
        dict.set_item("margin", margins)?;
        dict.set_item("unrealized_pnl", unrealized_pnls)?;

        Ok(dict.into())
    }

    /// Get metrics as a DataFrame similar to PyBroker's format.
    /// Returns a DataFrame with 'name' and 'value' columns.
    #[getter]
    pub fn metrics_df(&self, py: Python) -> PyResult<Py<PyAny>> {
        let metrics = &self.metrics;
        let t_metrics = &self.trade_metrics;

        let names = vec![
            "start_time",
            "end_time",
            "duration",
            "total_bars",
            "trade_count",
            "initial_market_value",
            "end_market_value",
            "total_pnl",
            "unrealized_pnl",
            "total_return_pct",
            "annualized_return",
            "volatility",
            "total_profit",
            "total_loss",
            "total_commission",
            "max_drawdown",
            "max_drawdown_pct",
            "win_rate",
            "loss_rate",
            "winning_trades",
            "losing_trades",
            "avg_pnl",
            "avg_return_pct",
            "avg_trade_bars",
            "avg_profit",
            "avg_profit_pct",
            "avg_winning_trade_bars",
            "avg_loss",
            "avg_loss_pct",
            "avg_losing_trade_bars",
            "largest_win",
            "largest_win_pct",
            "largest_win_bars",
            "largest_loss",
            "largest_loss_pct",
            "largest_loss_bars",
            "max_wins",
            "max_losses",
            "sharpe_ratio",
            "sortino_ratio",
            "profit_factor",
            "ulcer_index",
            "upi",
            "equity_r2",
            "std_error",
            "calmar_ratio",
            "exposure_time_pct",
            "var_95",
            "var_99",
            "cvar_95",
            "cvar_99",
            "sqn",
            "kelly_criterion",
        ];

        let mut values = Vec::<Py<PyAny>>::new();

        // Push start/end time as datetime objects
        // Convert nanoseconds to DateTime
        use chrono::{TimeZone, Utc};
        let start_dt = Utc.timestamp_nanos(metrics.start_time);
        let end_dt = Utc.timestamp_nanos(metrics.end_time);

        if let Ok(obj) = start_dt.into_pyobject(py) {
            values.push(obj.into_any().unbind());
        }
        if let Ok(obj) = end_dt.into_pyobject(py) {
            values.push(obj.into_any().unbind());
        }

        // Push duration as timedelta
        let duration = chrono::Duration::nanoseconds(metrics.duration);
        if let Ok(obj) = duration.into_pyobject(py) {
            values.push(obj.into_any().unbind());
        }

        // Push total_bars
        let obj = metrics.total_bars.into_pyobject(py).unwrap();
        values.push(obj.into_any().unbind());

        // Helper to push f64
        let mut push_f64 = |v: f64| {
            let obj = v.into_pyobject(py).unwrap();
            values.push(obj.into_any().unbind());
        };

        push_f64(t_metrics.total_closed_trades as f64);
        push_f64(metrics.initial_market_value);
        push_f64(metrics.end_market_value);
        push_f64(t_metrics.gross_pnl);
        push_f64(t_metrics.unrealized_pnl);
        push_f64(metrics.total_return_pct);
        push_f64(metrics.annualized_return);
        push_f64(metrics.volatility);
        push_f64(t_metrics.total_profit);
        push_f64(t_metrics.total_loss);
        push_f64(t_metrics.total_commission);
        push_f64(metrics.max_drawdown_value);
        push_f64(metrics.max_drawdown_pct);
        push_f64(t_metrics.win_rate);
        push_f64(t_metrics.loss_rate);
        push_f64(t_metrics.won_count as f64);
        push_f64(t_metrics.lost_count as f64);
        push_f64(t_metrics.avg_pnl);
        push_f64(t_metrics.avg_return_pct);
        push_f64(t_metrics.avg_trade_bars);
        push_f64(t_metrics.avg_profit);
        push_f64(t_metrics.avg_profit_pct);
        push_f64(t_metrics.avg_winning_trade_bars);
        push_f64(t_metrics.avg_loss);
        push_f64(t_metrics.avg_loss_pct);
        push_f64(t_metrics.avg_losing_trade_bars);
        push_f64(t_metrics.largest_win);
        push_f64(t_metrics.largest_win_pct);
        push_f64(t_metrics.largest_win_bars);
        push_f64(t_metrics.largest_loss);
        push_f64(t_metrics.largest_loss_pct);
        push_f64(t_metrics.largest_loss_bars);
        push_f64(t_metrics.max_wins as f64);
        push_f64(t_metrics.max_losses as f64);
        push_f64(metrics.sharpe_ratio);
        push_f64(metrics.sortino_ratio);
        push_f64(t_metrics.profit_factor);
        push_f64(metrics.ulcer_index);
        push_f64(metrics.upi);
        push_f64(metrics.equity_r2);
        push_f64(metrics.std_error);
        push_f64(metrics.calmar_ratio);
        push_f64(metrics.exposure_time_pct);
        push_f64(metrics.var_95);
        push_f64(metrics.var_99);
        push_f64(metrics.cvar_95);
        push_f64(metrics.cvar_99);
        push_f64(t_metrics.sqn);
        push_f64(t_metrics.kelly_criterion);

        // Try to import pandas
        match py.import("pandas") {
            Ok(pandas) => {
                let data = pyo3::types::PyDict::new(py);
                data.set_item("name", names)?;
                data.set_item("value", values)?;

                // Create DataFrame and set 'name' as index
                let df = pandas.call_method1("DataFrame", (data,))?;
                let df_indexed = df.call_method1("set_index", ("name",))?;

                Ok(df_indexed.into())
            }
            Err(_) => {
                // Fallback to dict if pandas is not available
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("name", names)?;
                dict.set_item("value", values)?;
                Ok(dict.into())
            }
        }
    }
    /// Get orders as a dictionary of columns for fast DataFrame creation.
    pub fn get_orders_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let n = self.orders.len();
        let mut ids = Vec::with_capacity(n);
        let mut sides = Vec::with_capacity(n); // buy/sell
        let mut order_types = Vec::with_capacity(n); // market/limit/stop
        let mut symbols = Vec::with_capacity(n);
        let mut dates = Vec::with_capacity(n);
        let mut quantities = Vec::with_capacity(n);
        let mut filled_quantities = Vec::with_capacity(n);
        let mut limit_prices = Vec::with_capacity(n);
        let mut trigger_prices = Vec::with_capacity(n);
        let mut avg_prices = Vec::with_capacity(n);
        let mut commissions = Vec::with_capacity(n);
        let mut status = Vec::with_capacity(n);
        let mut time_in_force = Vec::with_capacity(n);

        for o in &self.orders {
            ids.push(o.id.clone());
            sides.push(format!("{:?}", o.side).to_lowercase());
            order_types.push(format!("{:?}", o.order_type).to_lowercase());
            symbols.push(o.symbol.clone());
            dates.push(o.created_at); // i64 timestamp (ns)
            quantities.push(o.quantity.to_f64().unwrap_or(0.0));
            filled_quantities.push(o.filled_quantity.to_f64().unwrap_or(0.0));
            limit_prices.push(
                o.price
                    .map(|p| p.to_f64().unwrap_or(f64::NAN))
                    .unwrap_or(f64::NAN),
            );
            trigger_prices.push(
                o.trigger_price
                    .map(|p| p.to_f64().unwrap_or(f64::NAN))
                    .unwrap_or(f64::NAN),
            );
            avg_prices.push(
                o.average_filled_price
                    .map(|p| p.to_f64().unwrap_or(f64::NAN))
                    .unwrap_or(f64::NAN),
            );
            commissions.push(o.commission.to_f64().unwrap_or(0.0));
            status.push(format!("{:?}", o.status).to_lowercase());
            time_in_force.push(format!("{:?}", o.time_in_force).to_lowercase());
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("id", ids)?;
        dict.set_item("symbol", symbols)?;
        dict.set_item("side", sides)?;
        dict.set_item("order_type", order_types)?;
        dict.set_item("quantity", quantities)?;
        dict.set_item("filled_quantity", filled_quantities)?;
        dict.set_item("limit_price", limit_prices)?;
        dict.set_item("stop_price", trigger_prices)?;
        dict.set_item("avg_price", avg_prices)?;
        dict.set_item("commission", commissions)?;
        dict.set_item("status", status)?;
        dict.set_item("time_in_force", time_in_force)?;
        dict.set_item("created_at", dates)?; // Renamed date -> created_at for consistency

        Ok(dict.into())
    }

    /// Get orders as a DataFrame similar to PyBroker's format.
    #[getter]
    pub fn orders_df(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = self.get_orders_dict(py)?;

        match py.import("pandas") {
            Ok(pandas) => {
                let df = pandas.call_method1("DataFrame", (dict,))?;
                Ok(df.into())
            }
            Err(_) => Ok(dict),
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
        (
            tracker.calculate_pnl(current_prices),
            tracker.closed_trades.to_vec(),
        )
    }

    #[test]
    fn test_trade_analyzer_long_profit() {
        // Buy 100 @ 10, Sell 100 @ 12
        let t1 = create_trade(
            "AAPL",
            OrderSide::Buy,
            Decimal::from(100),
            Decimal::from(10),
            Decimal::ZERO,
        );
        let t2 = create_trade(
            "AAPL",
            OrderSide::Sell,
            Decimal::from(100),
            Decimal::from(12),
            Decimal::ZERO,
        );

        let (pnl, _) = analyze_trades(vec![t1, t2], None);

        assert_eq!(pnl.total_closed_trades, 1);
        assert_eq!(pnl.won_count, 1);
        assert_eq!(pnl.gross_pnl, 200.0); // (12-10)*100
        assert_eq!(pnl.win_rate, 100.0);
    }

    #[test]
    fn test_trade_analyzer_short_loss() {
        // Sell 100 @ 10, Buy 100 @ 12
        let t1 = create_trade(
            "AAPL",
            OrderSide::Sell,
            Decimal::from(100),
            Decimal::from(10),
            Decimal::ZERO,
        );
        let t2 = create_trade(
            "AAPL",
            OrderSide::Buy,
            Decimal::from(100),
            Decimal::from(12),
            Decimal::ZERO,
        );

        let (pnl, _) = analyze_trades(vec![t1, t2], None);

        assert_eq!(pnl.total_closed_trades, 1);
        assert_eq!(pnl.lost_count, 1);
        assert_eq!(pnl.gross_pnl, -200.0); // (10-12)*100
        assert_eq!(pnl.loss_rate, 100.0);
    }

    #[test]
    fn test_trade_analyzer_fifo() {
        // Buy 100 @ 10
        // Buy 100 @ 12
        // Sell 150 @ 11
        let t1 = create_trade(
            "AAPL",
            OrderSide::Buy,
            Decimal::from(100),
            Decimal::from(10),
            Decimal::ZERO,
        );
        let t2 = create_trade(
            "AAPL",
            OrderSide::Buy,
            Decimal::from(100),
            Decimal::from(12),
            Decimal::ZERO,
        );
        let t3 = create_trade(
            "AAPL",
            OrderSide::Sell,
            Decimal::from(150),
            Decimal::from(11),
            Decimal::ZERO,
        );

        let (pnl, _) = analyze_trades(vec![t1, t2, t3], None);

        assert_eq!(pnl.gross_pnl, 50.0);
        assert_eq!(pnl.total_closed_trades, 2);
    }

    #[test]
    fn test_unrealized_pnl() {
        // Buy 100 @ 10
        let t1 = create_trade(
            "AAPL",
            OrderSide::Buy,
            Decimal::from(100),
            Decimal::from(10),
            Decimal::ZERO,
        );

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), Decimal::from(15));

        let (pnl, _) = analyze_trades(vec![t1], Some(prices));

        assert_eq!(pnl.total_closed_trades, 0);
        assert_eq!(pnl.unrealized_pnl, 500.0); // (15-10)*100
    }

    #[test]
    fn test_max_drawdown_logic() {
        let empty_pnl = TradeTracker::new().calculate_pnl(None);

        // Case 1: Standard Drawdown
        // 100 -> 120 -> 90 -> 110
        let equity_curve = vec![
            (1, Decimal::from(100)),
            (2, Decimal::from(120)), // Peak
            (3, Decimal::from(90)),  // Drawdown: (120-90)/120 = 0.25
            (4, Decimal::from(110)),
        ];

        let result = BacktestResult::calculate(
            equity_curve,
            vec![],
            empty_pnl.clone(),
            vec![],
            Decimal::from(100),
            vec![],
            vec![],
        );
        assert_eq!(result.metrics.max_drawdown, 0.25);

        // Case 2: No Drawdown (Monotonic Increase)
        // 100 -> 110 -> 120
        let equity_curve_2 = vec![
            (1, Decimal::from(100)),
            (2, Decimal::from(110)),
            (3, Decimal::from(120)),
        ];
        let result_2 = BacktestResult::calculate(
            equity_curve_2,
            vec![],
            empty_pnl.clone(),
            vec![],
            Decimal::from(100),
            vec![],
            vec![],
        );
        assert_eq!(result_2.metrics.max_drawdown, 0.0);

        // Case 3: Immediate Drawdown
        // 100 -> 80 -> 90
        let equity_curve_3 = vec![
            (1, Decimal::from(100)), // Peak
            (2, Decimal::from(80)),  // Drawdown: (100-80)/100 = 0.2
            (3, Decimal::from(90)),
        ];
        let result_3 = BacktestResult::calculate(
            equity_curve_3,
            vec![],
            empty_pnl.clone(),
            vec![],
            Decimal::from(100),
            vec![],
            vec![],
        );
        assert_eq!(result_3.metrics.max_drawdown, 0.2);

        // Case 4: Multiple Peaks
        // 100 -> 90 (0.1) -> 100 -> 110 -> 55 (0.5) -> 110
        let equity_curve_4 = vec![
            (1, Decimal::from(100)),
            (2, Decimal::from(90)), // DD 0.1
            (3, Decimal::from(100)),
            (4, Decimal::from(110)), // New Peak
            (5, Decimal::from(55)),  // DD (110-55)/110 = 0.5
            (6, Decimal::from(110)),
        ];
        let result_4 = BacktestResult::calculate(
            equity_curve_4,
            vec![],
            empty_pnl.clone(),
            vec![],
            Decimal::from(100),
            vec![],
            vec![],
        );
        assert_eq!(result_4.metrics.max_drawdown, 0.5);
    }

    #[test]
    fn test_ulcer_index_logic() {
        let empty_pnl = TradeTracker::new().calculate_pnl(None);

        // Equity Curve: [100, 100, 90, 90, 100]
        // DDs: [0, 0, 0.1, 0.1, 0]
        // Squares: [0, 0, 0.01, 0.01, 0]
        // Sum = 0.02
        // Mean = 0.02 / 5 = 0.004
        // UI = sqrt(0.004) = 0.0632455532

        let equity_curve = vec![
            (1, Decimal::from(100)),
            (2, Decimal::from(100)),
            (3, Decimal::from(90)),
            (4, Decimal::from(90)),
            (5, Decimal::from(100)),
        ];

        let result = BacktestResult::calculate(
            equity_curve,
            vec![],
            empty_pnl.clone(),
            vec![],
            Decimal::from(100),
            vec![],
            vec![],
        );
        let expected_ui = 0.004f64.sqrt();
        assert!((result.metrics.ulcer_index - expected_ui).abs() < 1e-9);
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

    pub fn get_unrealized_pnl(
        &self,
        symbol: &str,
        current_price: Decimal,
        multiplier: Decimal,
    ) -> Decimal {
        let mut pnl = Decimal::ZERO;

        // Long positions: (Price - Entry) * Qty * Multiplier
        if let Some(queue) = self.long_inventory.get(symbol) {
            for (qty, price, _, _, _) in queue {
                pnl += (current_price - price) * qty * multiplier;
            }
        }

        // Short positions: (Entry - Price) * Qty * Multiplier
        if let Some(queue) = self.short_inventory.get(symbol) {
            for (qty, price, _, _, _) in queue {
                pnl += (price - current_price) * qty * multiplier;
            }
        }

        pnl
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
                            side: "Short".to_string(),
                            pnl: to_f64(pnl),
                            net_pnl: to_f64(pnl - total_trade_comm),
                            return_pct: to_f64(ret_pct) * 100.0,
                            commission: to_f64(total_trade_comm),
                            duration_bars: bars.to_usize().unwrap_or(0),
                            duration: timestamp - *match_timestamp,
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
                            side: "Long".to_string(),
                            pnl: to_f64(pnl),
                            net_pnl: to_f64(pnl - total_trade_comm),
                            return_pct: to_f64(ret_pct) * 100.0,
                            commission: to_f64(total_trade_comm),
                            duration_bars: bars.to_usize().unwrap_or(0),
                            duration: timestamp - *match_timestamp,
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
            (self.won_count as f64 / total_closed_trades as f64) * 100.0
        } else {
            0.0
        };
        let loss_rate = if total_closed_trades > 0 {
            (self.lost_count as f64 / total_closed_trades as f64) * 100.0
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
            let sum_win_ret: Decimal = self
                .closed_trades_stats
                .iter()
                .filter(|(_, _, _, w)| *w)
                .map(|(_, r, _, _)| *r)
                .sum();
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
            let sum_loss_ret: Decimal = self
                .closed_trades_stats
                .iter()
                .filter(|(_, _, _, w)| !*w)
                .map(|(_, r, _, _)| *r)
                .sum();
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

        // SQN
        let sqn = if total_closed_trades > 0 {
            let avg_pnl_val = to_f64(sum_pnl) / total_closed_trades as f64;
            let variance = self.closed_trades_stats.iter()
                .map(|(pnl, _, _, _)| {
                    let diff = to_f64(*pnl) - avg_pnl_val;
                    diff * diff
                })
                .sum::<f64>() / total_closed_trades as f64;
            let std_dev = variance.sqrt();
            if std_dev != 0.0 {
                (avg_pnl_val / std_dev) * (total_closed_trades as f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Kelly Criterion
        let kelly_criterion = if win_rate > 0.0 && avg_loss.abs() > 0.0 {
            let w = win_rate / 100.0;
            let r = avg_profit / avg_loss.abs();
            if r > 0.0 {
                 w - (1.0 - w) / r
            } else {
                0.0
            }
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
            avg_return_pct: avg_return_pct * 100.0,
            avg_trade_bars,
            avg_profit,
            avg_profit_pct: avg_profit_pct * 100.0,
            avg_winning_trade_bars,
            avg_loss,
            avg_loss_pct: avg_loss_pct * 100.0,
            avg_losing_trade_bars,
            largest_win: to_f64(largest_win),
            largest_win_pct: to_f64(largest_win_pct) * 100.0,
            largest_win_bars: to_f64(largest_win_bars),
            largest_loss: to_f64(largest_loss),
            largest_loss_pct: to_f64(largest_loss_pct) * 100.0,
            largest_loss_bars: to_f64(largest_loss_bars),
            max_wins: self.max_wins,
            max_losses: self.max_losses,
            profit_factor,
            total_profit: to_f64(self.won_pnl),
            total_loss: to_f64(self.lost_pnl),
            sqn,
            kelly_criterion,
        }
    }
}
