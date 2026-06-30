use super::result::BacktestResult;
use chrono::{TimeZone, Utc};
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;

#[gen_stub_pymethods]
#[pymethods]
impl BacktestResult {
    /// Get trades as Arrow IPC bytes (Zero-Copy-ish to Python)
    pub fn get_trades_ipc(&self, py: Python) -> PyResult<Py<PyBytes>> {
        if self.trades.is_empty() {
            return Ok(PyBytes::new(py, &[]).into());
        }

        let trades = &self.trades;
        let s_symbol = Series::new(
            "symbol".into(),
            trades.iter().map(|t| t.symbol.clone()).collect::<Vec<_>>(),
        );
        let s_entry_time = Series::new(
            "entry_time".into(),
            trades.iter().map(|t| t.entry_time).collect::<Vec<_>>(),
        );
        let s_exit_time = Series::new(
            "exit_time".into(),
            trades.iter().map(|t| t.exit_time).collect::<Vec<_>>(),
        );
        let s_entry_price = Series::new(
            "entry_price".into(),
            trades.iter().map(|t| t.entry_price).collect::<Vec<_>>(),
        );
        let s_exit_price = Series::new(
            "exit_price".into(),
            trades.iter().map(|t| t.exit_price).collect::<Vec<_>>(),
        );
        let s_quantity = Series::new(
            "quantity".into(),
            trades.iter().map(|t| t.quantity).collect::<Vec<_>>(),
        );
        let s_side = Series::new(
            "side".into(),
            trades.iter().map(|t| t.side.clone()).collect::<Vec<_>>(),
        );
        let s_pnl = Series::new(
            "pnl".into(),
            trades.iter().map(|t| t.pnl).collect::<Vec<_>>(),
        );
        let s_net_pnl = Series::new(
            "net_pnl".into(),
            trades.iter().map(|t| t.net_pnl).collect::<Vec<_>>(),
        );
        let s_return_pct = Series::new(
            "return_pct".into(),
            trades.iter().map(|t| t.return_pct).collect::<Vec<_>>(),
        );
        let s_commission = Series::new(
            "commission".into(),
            trades.iter().map(|t| t.commission).collect::<Vec<_>>(),
        );
        let s_duration_bars = Series::new(
            "duration_bars".into(),
            trades
                .iter()
                .map(|t| t.duration_bars as u64)
                .collect::<Vec<_>>(),
        );
        let s_duration = Series::new(
            "duration".into(),
            trades.iter().map(|t| t.duration).collect::<Vec<_>>(),
        );
        let s_mae = Series::new(
            "mae".into(),
            trades.iter().map(|t| t.mae).collect::<Vec<_>>(),
        );
        let s_mfe = Series::new(
            "mfe".into(),
            trades.iter().map(|t| t.mfe).collect::<Vec<_>>(),
        );
        let s_entry_tag = Series::new(
            "entry_tag".into(),
            trades
                .iter()
                .map(|t| t.entry_tag.clone())
                .collect::<Vec<_>>(),
        );
        let s_exit_tag = Series::new(
            "exit_tag".into(),
            trades
                .iter()
                .map(|t| t.exit_tag.clone())
                .collect::<Vec<_>>(),
        );
        let s_entry_portfolio_value = Series::new(
            "entry_portfolio_value".into(),
            trades
                .iter()
                .map(|t| t.entry_portfolio_value)
                .collect::<Vec<_>>(),
        );
        let s_max_drawdown_pct = Series::new(
            "max_drawdown_pct".into(),
            trades
                .iter()
                .map(|t| t.max_drawdown_pct)
                .collect::<Vec<_>>(),
        );

        let mut df = DataFrame::new(vec![
            s_symbol.into(),
            s_entry_time.into(),
            s_exit_time.into(),
            s_entry_price.into(),
            s_exit_price.into(),
            s_quantity.into(),
            s_side.into(),
            s_pnl.into(),
            s_net_pnl.into(),
            s_return_pct.into(),
            s_commission.into(),
            s_duration_bars.into(),
            s_duration.into(),
            s_mae.into(),
            s_mfe.into(),
            s_entry_tag.into(),
            s_exit_tag.into(),
            s_entry_portfolio_value.into(),
            s_max_drawdown_pct.into(),
        ])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut buf = Vec::new();
        IpcWriter::new(&mut buf)
            .finish(&mut df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyBytes::new(py, &buf).into())
    }

    /// Get positions history as Arrow IPC bytes
    pub fn get_positions_ipc(&self, py: Python) -> PyResult<Py<PyBytes>> {
        let mut symbols = Vec::new();
        let mut dates = Vec::new();
        let mut long_shares = Vec::new();
        let mut short_shares = Vec::new();
        let mut closes = Vec::new();
        let mut equities = Vec::new();
        let mut market_values = Vec::new();
        let mut margins = Vec::new();
        let mut unrealized_pnls = Vec::new();
        let mut entry_prices = Vec::new();

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
                entry_prices.push(s.entry_price);
            }
        }

        if symbols.is_empty() {
            return Ok(PyBytes::new(py, &[]).into());
        }

        let s_symbol = Series::new("symbol".into(), symbols);
        let s_date = Series::new("date".into(), dates);
        let s_long = Series::new("long_shares".into(), long_shares);
        let s_short = Series::new("short_shares".into(), short_shares);
        let s_close = Series::new("close".into(), closes);
        let s_equity = Series::new("equity".into(), equities);
        let s_mv = Series::new("market_value".into(), market_values);
        let s_margin = Series::new("margin".into(), margins);
        let s_upnl = Series::new("unrealized_pnl".into(), unrealized_pnls);
        let s_entry_price = Series::new("entry_price".into(), entry_prices);

        let mut df = DataFrame::new(vec![
            s_symbol.into(),
            s_date.into(),
            s_long.into(),
            s_short.into(),
            s_close.into(),
            s_equity.into(),
            s_mv.into(),
            s_margin.into(),
            s_upnl.into(),
            s_entry_price.into(),
        ])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut buf = Vec::new();
        IpcWriter::new(&mut buf)
            .finish(&mut df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyBytes::new(py, &buf).into())
    }

    pub fn get_executions_ipc(&self, py: Python) -> PyResult<Py<PyBytes>> {
        if self.executions.is_empty() {
            return Ok(PyBytes::new(py, &[]).into());
        }

        let executions = &self.executions;
        let s_id = Series::new(
            "id".into(),
            executions.iter().map(|t| t.id.clone()).collect::<Vec<_>>(),
        );
        let s_order_id = Series::new(
            "order_id".into(),
            executions
                .iter()
                .map(|t| t.order_id.clone())
                .collect::<Vec<_>>(),
        );
        let s_symbol = Series::new(
            "symbol".into(),
            executions
                .iter()
                .map(|t| t.symbol.clone())
                .collect::<Vec<_>>(),
        );
        let s_side = Series::new(
            "side".into(),
            executions
                .iter()
                .map(|t| format!("{:?}", t.side).to_lowercase())
                .collect::<Vec<_>>(),
        );
        let s_quantity = Series::new(
            "quantity".into(),
            executions
                .iter()
                .map(|t| t.quantity.to_f64().unwrap_or(0.0))
                .collect::<Vec<_>>(),
        );
        let s_price = Series::new(
            "price".into(),
            executions
                .iter()
                .map(|t| t.price.to_f64().unwrap_or(0.0))
                .collect::<Vec<_>>(),
        );
        let s_commission = Series::new(
            "commission".into(),
            executions
                .iter()
                .map(|t| t.commission.to_f64().unwrap_or(0.0))
                .collect::<Vec<_>>(),
        );
        let s_timestamp = Series::new(
            "timestamp".into(),
            executions.iter().map(|t| t.timestamp).collect::<Vec<_>>(),
        );
        let s_bar_index = Series::new(
            "bar_index".into(),
            executions
                .iter()
                .map(|t| t.bar_index as u64)
                .collect::<Vec<_>>(),
        );
        let s_owner_strategy_id = Series::new(
            "owner_strategy_id".into(),
            executions
                .iter()
                .map(|t| t.owner_strategy_id.clone())
                .collect::<Vec<_>>(),
        );

        let mut df = DataFrame::new(vec![
            s_id.into(),
            s_order_id.into(),
            s_symbol.into(),
            s_side.into(),
            s_quantity.into(),
            s_price.into(),
            s_commission.into(),
            s_timestamp.into(),
            s_bar_index.into(),
            s_owner_strategy_id.into(),
        ])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut buf = Vec::new();
        IpcWriter::new(&mut buf)
            .finish(&mut df)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyBytes::new(py, &buf).into())
    }

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
        let mut maes = Vec::with_capacity(n);
        let mut mfes = Vec::with_capacity(n);
        let mut entry_tags = Vec::with_capacity(n);
        let mut exit_tags = Vec::with_capacity(n);
        let mut entry_portfolio_values = Vec::with_capacity(n);
        let mut max_drawdown_pcts = Vec::with_capacity(n);

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
            maes.push(t.mae);
            mfes.push(t.mfe);
            entry_tags.push(t.entry_tag.clone());
            exit_tags.push(t.exit_tag.clone());
            entry_portfolio_values.push(t.entry_portfolio_value);
            max_drawdown_pcts.push(t.max_drawdown_pct);
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
        dict.set_item("mae", maes)?;
        dict.set_item("mfe", mfes)?;
        dict.set_item("entry_tag", entry_tags)?;
        dict.set_item("exit_tag", exit_tags)?;
        dict.set_item("entry_portfolio_value", entry_portfolio_values)?;
        dict.set_item("max_drawdown_pct", max_drawdown_pcts)?;

        Ok(dict.into())
    }

    pub fn get_executions_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let n = self.executions.len();
        let mut ids = Vec::with_capacity(n);
        let mut order_ids = Vec::with_capacity(n);
        let mut symbols = Vec::with_capacity(n);
        let mut sides = Vec::with_capacity(n);
        let mut quantities = Vec::with_capacity(n);
        let mut prices = Vec::with_capacity(n);
        let mut commissions = Vec::with_capacity(n);
        let mut timestamps = Vec::with_capacity(n);
        let mut bar_indexes = Vec::with_capacity(n);
        let mut owner_strategy_ids = Vec::with_capacity(n);

        for t in &self.executions {
            ids.push(t.id.clone());
            order_ids.push(t.order_id.clone());
            symbols.push(t.symbol.clone());
            sides.push(format!("{:?}", t.side).to_lowercase());
            quantities.push(t.quantity.to_f64().unwrap_or(0.0));
            prices.push(t.price.to_f64().unwrap_or(0.0));
            commissions.push(t.commission.to_f64().unwrap_or(0.0));
            timestamps.push(t.timestamp);
            bar_indexes.push(t.bar_index);
            owner_strategy_ids.push(t.owner_strategy_id.clone());
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("id", ids)?;
        dict.set_item("order_id", order_ids)?;
        dict.set_item("symbol", symbols)?;
        dict.set_item("side", sides)?;
        dict.set_item("quantity", quantities)?;
        dict.set_item("price", prices)?;
        dict.set_item("commission", commissions)?;
        dict.set_item("timestamp", timestamps)?;
        dict.set_item("bar_index", bar_indexes)?;
        dict.set_item("owner_strategy_id", owner_strategy_ids)?;
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
        let mut entry_prices = Vec::new();

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
                entry_prices.push(s.entry_price);
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
        dict.set_item("entry_price", entry_prices)?;

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
            "max_drawdown_value",
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
        let start_dt = Utc.timestamp_nanos(metrics.start_time);
        let end_dt = Utc.timestamp_nanos(metrics.end_time);

        if let Ok(obj) = start_dt.into_pyobject(py) {
            values.push(obj.into_any().unbind());
        }
        if let Ok(obj) = end_dt.into_pyobject(py) {
            values.push(obj.into_any().unbind());
        }

        // Push duration as timedelta
        let nanos = metrics.duration;
        let days = (nanos / 86_400_000_000_000) as i32;
        let rem_ns = nanos % 86_400_000_000_000;
        let seconds = (rem_ns / 1_000_000_000) as i32;
        let rem_ns = rem_ns % 1_000_000_000;
        let microseconds = (rem_ns / 1_000) as i32;

        if let Ok(delta) = pyo3::types::PyDelta::new(py, days, seconds, microseconds, true) {
            values.push(delta.into_any().unbind());
        } else {
            values.push(py.None());
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
        push_f64(metrics.max_drawdown);
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
        let mut updated_dates = Vec::with_capacity(n);
        let mut quantities = Vec::with_capacity(n);
        let mut filled_quantities = Vec::with_capacity(n);
        let mut limit_prices = Vec::with_capacity(n);
        let mut trigger_prices = Vec::with_capacity(n);
        let mut avg_prices = Vec::with_capacity(n);
        let mut commissions = Vec::with_capacity(n);
        let mut status = Vec::with_capacity(n);
        let mut time_in_force = Vec::with_capacity(n);
        let mut tags = Vec::with_capacity(n);
        let mut reject_reasons = Vec::with_capacity(n);
        let mut owner_strategy_ids = Vec::with_capacity(n);

        for o in &self.orders {
            ids.push(o.id.clone());
            sides.push(format!("{:?}", o.side).to_lowercase());
            order_types.push(format!("{:?}", o.order_type).to_lowercase());
            symbols.push(o.symbol.clone());
            dates.push(o.created_at); // i64 timestamp (ns)
            updated_dates.push(o.updated_at);
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
            tags.push(o.tag.clone());
            reject_reasons.push(o.reject_reason.clone());
            owner_strategy_ids.push(o.owner_strategy_id.clone());
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
        dict.set_item("updated_at", updated_dates)?;
        dict.set_item("tag", tags)?;
        dict.set_item("reject_reason", reject_reasons)?;
        dict.set_item("owner_strategy_id", owner_strategy_ids)?;

        Ok(dict.into())
    }

    pub fn get_liquidation_audits_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let n = self.liquidation_audits.len();
        let mut timestamps = Vec::with_capacity(n);
        let mut dates = Vec::with_capacity(n);
        let mut daily_interests = Vec::with_capacity(n);
        let mut liquidated_counts = Vec::with_capacity(n);
        let mut liquidated_symbols = Vec::with_capacity(n);
        let mut priorities = Vec::with_capacity(n);

        for row in &self.liquidation_audits {
            timestamps.push(row.timestamp);
            dates.push(row.date.clone());
            daily_interests.push(row.daily_interest);
            liquidated_counts.push(row.liquidated_count as u64);
            liquidated_symbols.push(row.liquidated_symbols.join(","));
            priorities.push(row.priority.clone());
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("timestamp", timestamps)?;
        dict.set_item("date", dates)?;
        dict.set_item("daily_interest", daily_interests)?;
        dict.set_item("liquidated_count", liquidated_counts)?;
        dict.set_item("liquidated_symbols", liquidated_symbols)?;
        dict.set_item("priority", priorities)?;
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
