# Backtest Results & Metrics

This document details the performance metrics in AKQuant backtest results (`metrics_df`), including their meanings, units, and calculation methods. It also covers the visualization capabilities for analyzing these results.

## Visualization (Plotting)

AKQuant provides a comprehensive visualization module to analyze backtest results. You can generate professional-grade interactive HTML reports or individual plots directly from the `BacktestResult` object.

### Quick Start

The easiest way to visualize your backtest results is using the `viz.report()` method:

```python
# Generate a full HTML report
result.viz.report(
    title="My Strategy Report",
    filename="report.html",
    show=True,  # Set to True to open in browser automatically (default is False)
    compact_currency=True,  # Format amount columns as K/M/B in report tables
    curve_freq="D",  # Default uses end-of-day points; "raw" keeps all bars
)
```

If you prefer raw amount precision in report tables:

```python
result.viz.report(
    title="My Strategy Report",
    filename="report_raw_amount.html",
    compact_currency=False,
    curve_freq="D",
)
```

### Structured Benchmark Analysis

If you need benchmark comparison outputs for a web frontend, notebook workflow, or offline pipeline, use the structured API instead of reading values from the HTML report:

```python
benchmark_returns = (
    benchmark_df.set_index("date")["close"].pct_change().fillna(0.0)
)

benchmark_analysis = result.benchmark_analysis(
    benchmark=benchmark_returns,
    curve_freq="D",
)

summary = benchmark_analysis["summary"]
series = benchmark_analysis["series"]

print(summary["information_ratio"])
print(series[:2])
```

The structured benchmark analysis shares the same alignment and calculation logic as `result.viz.report(..., benchmark=...)`, making it suitable as a long-term frontend/API contract:

- `summary`: aggregate relative metrics
- `series`: aligned daily time series
- `meta`: sample count and date range
- `reason`: explanation when the benchmark input is invalid or cannot be aligned

#### Summary Field Units

All return-based metrics in `summary` are returned as **Ratio (decimal)**, matching the convention of `annualized_return` and `max_drawdown`. Frontend clients must multiply by 100 when displaying as percentages. Dimensionless ratio metrics should **not** be multiplied.

| Field | Description | Unit/Type | Display Format |
| :--- | :--- | :--- | :--- |
| `total_excess` | Total excess return | Ratio (decimal) | `× 100` with `%` |
| `annual_excess` | Annualized excess return | Ratio (decimal) | `× 100` with `%` |
| `tracking_error` | Tracking error | Ratio (decimal) | `× 100` with `%` |
| `alpha` | Alpha (annualized) | Ratio (decimal) | `× 100` with `%` |
| `information_ratio` | Information ratio | Ratio (dimensionless) | Show as-is, no scaling |
| `beta` | Beta | Ratio (dimensionless) | Show as-is, no scaling |

Any field may be `None` when the benchmark cannot be aligned or variance is zero; check before rendering.

Returns in `series` (`strategy_return`, `benchmark_return`, `excess_return`, and all `*_cum_return` fields) are also decimals.

The built-in HTML report follows this convention: `total_excess` / `annual_excess` / `tracking_error` / `alpha` are formatted as percentages, while `information_ratio` / `beta` are displayed as 4-decimal floats. Use the information ratio as a sanity check — it always equals `annual_excess / tracking_error`. If your frontend's calculated ratio doesn't match the `information_ratio` field, one of the operands was scaled incorrectly.

To persist the analysis:

```python
result.export_benchmark_analysis(
    path="artifacts/benchmark_analysis.json",
    benchmark=benchmark_returns,
    format="json",
    curve_freq="D",
)
```

This generates a consolidated dashboard including:

- **Equity Curve**: Interactive chart of account equity over time.
- **Drawdown**: Historical drawdown analysis.
- **Monthly Heatmap**: Monthly return performance grid.
- **Key Metrics**: A summary of important performance statistics.

### Intraday Support (New)

The plotting module automatically detects and adapts to intraday (minute-level) backtests:

- **Smart Downsampling**: For large datasets (>10k points), it switches to WebGL rendering (`Scattergl`) for high performance.
- **Adaptive X-Axis**: Automatically formats time labels (e.g., `%Y-%m-%d %H:%M`) and prevents label overlap.
- **Adaptive Duration Units**: Trade duration analysis automatically switches units (Days, Hours, or Minutes) based on the strategy's average holding period.

### Advanced Plotting

You can also access individual plotting functions for more granular control:

```python
import akquant.plot as aqp

# 1. Plot Dashboard (Equity, Drawdown, Heatmap)
aqp.plot_dashboard(result)

# 2. Analyze Trade Distribution (PnL vs Duration)
aqp.plot_trades_distribution(result.trades_df)

# 3. Analyze PnL vs Duration
aqp.plot_pnl_vs_duration(result.trades_df)
```

## Metrics Overview

| Name | Description | Unit/Type | Calculation |
| :--- | :--- | :--- | :--- |
| `start_time` | Backtest Start Time | Datetime | Time of the first bar. |
| `end_time` | Backtest End Time | Datetime | Time of the last bar. |
| `duration` | Backtest Duration | Timedelta | `end_time - start_time`. |
| `total_bars` | Total Bars | Int | Total number of bars in backtest. |
| `closed_trade_count` | Closed Trade Count | Int | Total number of closed trades (round-trip). |
| `execution_count` | Execution Count | Int | Total number of execution fills. Usually greater than or equal to closed trade count. |
| `open_position_count` | Open Position Count | Int | Number of symbols still carrying open positions at the end of the backtest. |
| `initial_market_value` | Initial Market Value | Float | Initial capital (usually Cash). |
| `end_market_value` | End Market Value | Float | Total asset value at the end (Cash + Position Value). |
| `total_pnl` | Total PnL | Float | `end_market_value - initial_market_value` (portfolio-level: realized net PnL + unrealized floating PnL). Note it is NOT equal to `total_profit + total_loss` (which is the trade-level realized gross PnL, before commission and excluding floating PnL); for the trade-level realized gross PnL use `result.trade_metrics.gross_pnl`. |
| `unrealized_pnl` | Unrealized PnL | Float | Floating PnL of open positions at the end. |
| `total_return_pct` | Total Return | **%** | `(End MV - Initial MV) / Initial MV * 100`. |
| `annualized_return` | Annualized Return | Ratio | `(1 + Total Return)^(1/Years) - 1`. |
| `volatility` | Annualized Volatility | Ratio | Return Std Dev * sqrt(252). |
| `total_profit` | Total Profit | Float | Sum of profits from winning trades. |
| `total_loss` | Total Loss | Float | Sum of losses from losing trades. |
| `total_commission` | Total Commission | Float | Total commission paid. |
| `max_drawdown` | Max Drawdown | Ratio | Max drawdown magnitude (e.g., 0.1 for 10%). |
| `max_drawdown_value` | Max Drawdown Value | Float | Absolute value of max drawdown. |
| `max_drawdown_pct` | Max Drawdown % | **%** | `max_drawdown * 100`. |
| `win_rate` | Win Rate | **%** | `Winning Trades / Total Trades * 100`. |
| `loss_rate` | Loss Rate | **%** | `Losing Trades / Total Trades * 100`. |
| `winning_trades` | Winning Trades | Int | Count of trades with PnL > 0. |
| `losing_trades` | Losing Trades | Int | Count of trades with PnL < 0. |
| `avg_pnl` | Avg PnL | Float | Average net PnL per trade. |
| `avg_return_pct` | Avg Return % | **%** | Average return percentage per trade. |
| `avg_trade_bars` | Avg Trade Bars | Float | Average bars held per trade. |
| `avg_profit` | Avg Profit | Float | Average profit of winning trades. |
| `avg_profit_pct` | Avg Profit % | **%** | Average return of winning trades. |
| `avg_winning_trade_bars`| Avg Win Trade Bars | Float | Average bars held for winning trades. |
| `avg_loss` | Avg Loss | Float | Average loss of losing trades. |
| `avg_loss_pct` | Avg Loss % | **%** | Average return of losing trades. |
| `avg_losing_trade_bars` | Avg Loss Trade Bars | Float | Average bars held for losing trades. |
| `largest_win` | Largest Win | Float | Max profit in a single trade. |
| `largest_win_pct` | Largest Win % | **%** | Max return in a single trade. |
| `largest_win_bars` | Largest Win Bars | Float | Duration (bars) of the largest win trade. |
| `largest_loss` | Largest Loss | Float | Max loss in a single trade. |
| `largest_loss_pct` | Largest Loss % | **%** | Max loss rate in a single trade. |
| `largest_loss_bars` | Largest Loss Bars | Float | Duration (bars) of the largest loss trade. |
| `max_wins` | Max Consecutive Wins | Int | Max number of consecutive winning trades. |
| `max_losses` | Max Consecutive Losses | Int | Max number of consecutive losing trades. |
| `sharpe_ratio` | Sharpe Ratio | Ratio | `(Ann. Return - Risk Free) / Ann. Volatility`. |
| `sortino_ratio` | Sortino Ratio | Ratio | `(Ann. Return - Risk Free) / Downside Volatility`. |
| `profit_factor` | Profit Factor | Ratio | `Total Profit / abs(Total Loss)`. |
| `ulcer_index` | Ulcer Index | Ratio | Measure of drawdown depth and duration. |
| `upi` | Ulcer Performance Index | Ratio | `(Ann. Return - Risk Free) / Ulcer Index`. |
| `equity_r2` | Equity R² | Ratio | Linear regression fit of equity curve (0-1). |
| `std_error` | Standard Error | Float | Standard error of equity curve regression. |
| `calmar_ratio` | Calmar Ratio | Ratio | `Annualized Return / Max Drawdown`. |
| `exposure_time_pct` | Exposure Time % | **%** | Percentage of time with open positions. |
| `var_95` | VaR 95% | Ratio | Value at Risk at 95% confidence (daily). |
| `var_99` | VaR 99% | Ratio | Value at Risk at 99% confidence (daily). |
| `cvar_95` | CVaR 95% | Ratio | Conditional VaR at 95% (Expected Shortfall). |
| `cvar_99` | CVaR 99% | Ratio | Conditional VaR at 99% (Expected Shortfall). |
| `sqn` | SQN | Float | System Quality Number. |
| `kelly_criterion` | Kelly Criterion | Ratio | `Win Rate - (Loss Rate / Profit Factor)`. |

## Key Metrics Explained

### Risk Metrics

*   **Max Drawdown**: Measures the worst-case scenario. E.g., 30% means buying at the peak results in a 30% loss at the trough.
*   **Volatility**: Measures uncertainty of returns. Higher volatility means larger price swings.
*   **VaR & CVaR**:
    *   **VaR (Value at Risk)**: Max expected loss over a day at a given confidence level (e.g., 95%).
    *   **CVaR (Conditional VaR)**: Average loss exceeding VaR (Expected Shortfall).

### Risk-Reward Metrics

*   **Sharpe Ratio**: Excess return per unit of total risk. >1 is good, >2 is excellent.
*   **Sortino Ratio**: Similar to Sharpe but considers only downside volatility.
*   **Calmar Ratio**: Annualized Return / Max Drawdown.
*   **SQN**: System Quality Number. Measures system stability.
*   **Kelly Criterion**: Optimal position size based on win rate and payoff ratio.

## Equity, Cash & Margin Curves

The `result` object provides equity/cash/margin curves over time, useful for plotting and risk analysis.

| Property | Description | Type | Explanation |
| :--- | :--- | :--- | :--- |
| `equity_curve` | Equity Curve | `pandas.Series` | Index is `Datetime`, values are Total Equity. Shows the trend of net asset value. |
| `cash_curve` | Cash Curve | `pandas.Series` | Index is `Datetime`, values are Available Cash. Shows the trend of liquid capital, useful for money management analysis. |
| `margin_curve` | Margin Curve | `pandas.Series` | Index is `Datetime`, values are Used Margin. Useful for leveraged/margin account monitoring. |
| `equity_curve_daily` | Daily Equity Curve | `pandas.Series` | End-of-day `equity_curve` values, useful for fast reporting and long-horizon comparisons. |
| `cash_curve_daily` | Daily Cash Curve | `pandas.Series` | End-of-day `cash_curve` values. |
| `margin_curve_daily` | Daily Margin Curve | `pandas.Series` | End-of-day `margin_curve` values. |

For long intraday backtests, you can speed up HTML report rendering by using daily curve mode:

```python
result.viz.report(filename="report_daily.html", curve_freq="D")
```

## Trades

`result.trades_df` contains details of every closed trade.

| Name | Description | Unit/Type | Calculation |
| :--- | :--- | :--- | :--- |
| `symbol` | Symbol | String | Trading symbol. |
| `entry_time` | Entry Time | Datetime | Time of entry. |
| `exit_time` | Exit Time | Datetime | Time of exit. |
| `entry_price` | Entry Price | Float | Average entry price. |

| `exit_price` | Exit Price | Float | Average exit price. |
| `quantity` | Quantity | Float | Traded quantity. |
| `side` | Side | String | `long` or `short`. |
| `pnl` | Gross PnL | Float | PnL before commission. |
| `net_pnl` | Net PnL | Float | `pnl - commission`. |
| `return_pct` | Return | Float | Trade return (decimal). |
| `commission` | Commission | Float | Commission paid. |
| `duration_bars` | Duration (Bars) | Int | Number of bars held. |
| `duration` | Duration | Timedelta | `exit_time - entry_time`. |
| `mae` | MAE | **%** | Maximum Adverse Excursion (max loss during trade). |
| `mfe` | MFE | **%** | Maximum Favorable Excursion (max profit during trade). |
| `entry_tag` | Entry Tag | String | Tag of the entry order. |
| `exit_tag` | Exit Tag | String | Tag of the exit order. |
| `entry_portfolio_value` | Entry Portfolio Value | Float | Total account equity at entry. |
| `max_drawdown_pct` | Max Drawdown % | **%** | Max drawdown percentage during the trade. |

## Orders

`result.orders_df` contains all order history.

| Name | Description | Unit/Type | Calculation |
| :--- | :--- | :--- | :--- |
| `id` | Order ID | String | Unique identifier. |
| `symbol` | Symbol | String | Trading symbol. |
| `side` | Side | String | `buy` or `sell`. |
| `order_type` | Type | String | `market`, `limit`, `stop`. |
| `quantity` | Quantity | Float | Order quantity. |
| `filled_quantity` | Filled Qty | Float | Executed quantity. |
| `limit_price` | Limit Price | Float | Price for limit orders. |
| `stop_price` | Stop Price | Float | Trigger price for stop orders. |
| `avg_price` | Avg Price | Float | Average execution price. |
| `commission` | Commission | Float | Commission paid. |
| `status` | Status | String | `filled`, `cancelled`, `rejected`, etc. |
| `time_in_force` | TIF | String | `gtc`, `day`, `ioc`, etc. |
| `created_at` | Created At | Datetime | Creation time. |
| `updated_at` | Updated At | Datetime | Last update time. |
| `duration` | Duration | Timedelta | `updated_at - created_at`. |
| `filled_value` | Filled Value | Float | `filled_quantity * avg_price`. |
| `tag` | Tag | String | User defined tag. |
| `reject_reason` | Reject Reason | String | Reason for rejection (if any). |

## Positions

`result.positions_df` contains daily (or per-bar) position snapshots.

| Name | Description | Unit/Type | Calculation |
| :--- | :--- | :--- | :--- |
| `date` | Date | Datetime | Snapshot time. |
| `symbol` | Symbol | String | Trading symbol. |
| `long_shares` | Long Shares | Float | Long position quantity. |
| `short_shares` | Short Shares | Float | Short position quantity. |
| `close` | Close Price | Float | Closing price. |
| `equity` | Equity | Float | Total account equity. |
| `market_value` | Market Value | Float | Market value of positions. |
| `margin` | Margin | Float | Margin used. |
| `unrealized_pnl` | Unrealized PnL | Float | Floating PnL. |
| `entry_price` | Entry Price | Float | Average entry price. |

## Attribution & Capacity

`BacktestResult` also exposes structured analysis outputs that are easy to reuse:

```python
exposure = result.exposure_df()  # net/gross exposure and leverage
attr_by_symbol = result.attribution_df(by="symbol")
attr_by_tag = result.attribution_df(by="tag")
capacity = result.capacity_df()  # order count, fill rates, turnover
orders_by_strategy = result.orders_by_strategy()  # strategy-level order summary
exec_by_strategy = result.executions_by_strategy()  # strategy-level execution summary
risk_by_strategy = result.risk_rejections_by_strategy()  # strategy-level risk rejection summary
risk_trend = result.risk_rejections_trend(freq="D")  # daily trend of risk rejections
risk_trend_by_strategy = result.risk_rejections_trend_by_strategy(freq="D")

# margin-account forced liquidation audit (when enabled)
liquidation_audit = result.liquidation_audit_df
```

### Attribution & Capacity Field Units

All ratio and percentage fields in these structured outputs follow the same convention as `annualized_return` and `max_drawdown`, returning **Ratio (decimal)**. Frontend clients must multiply by 100 when displaying as percentages.

**`exposure_df` Fields**

| Field | Description | Unit/Type |
| :--- | :--- | :--- |
| `date` | Date | Datetime |
| `equity` | Account equity | Float |
| `long_exposure` | Long market value | Float |
| `short_exposure` | Short market value | Float |
| `net_exposure` | Net exposure | Float |
| `gross_exposure` | Gross exposure | Float |
| `net_exposure_pct` | Net exposure ratio | Ratio (decimal) |
| `gross_exposure_pct` | Gross exposure ratio | Ratio (decimal) |
| `leverage` | Leverage multiple | Ratio (decimal) |

**`attribution_df` Fields**

| Field | Description | Unit/Type |
| :--- | :--- | :--- |
| `group` | Group identifier (symbol / tag) | String |
| `trade_count` | Number of trades | Int |
| `total_pnl` | Total P&L | Float |
| `avg_return_pct` | Average return | Ratio (decimal) |
| `total_commission` | Total commission | Float |
| `contribution_pct` | Contribution ratio | Ratio (decimal) |
| `abs_contribution_pct` | Absolute contribution ratio | Ratio (decimal) |

Note: Despite the `_pct` suffix, all three percentage fields in `attribution_df` return **decimals**, not percentages. Multiply by 100 for display.

**`capacity_df` Fields**

| Field | Description | Unit/Type |
| :--- | :--- | :--- |
| `date` | Date | Datetime |
| `order_count` | Order count | Int |
| `filled_order_count` | Filled order count | Int |
| `ordered_quantity` | Total ordered quantity | Float |
| `filled_quantity` | Total filled quantity | Float |
| `ordered_value` | Total ordered value | Float |
| `filled_value` | Total filled value | Float |
| `fill_rate_qty` | Fill rate (quantity) | Ratio (decimal) |
| `fill_rate_value` | Fill rate (value) | Ratio (decimal) |
| `equity` | Account equity | Float |
| `turnover` | Turnover rate | Ratio (decimal) |

**`orders_by_strategy` Fields**

| Field | Description | Unit/Type |
| :--- | :--- | :--- |
| `owner_strategy_id` | Strategy identifier | String |
| `order_count` | Order count | Int |
| `filled_order_count` | Filled order count | Int |
| `ordered_quantity` | Total ordered quantity | Float |
| `filled_quantity` | Total filled quantity | Float |
| `ordered_value` | Total ordered value | Float |
| `filled_value` | Total filled value | Float |
| `fill_rate_qty` | Fill rate (quantity) | Ratio (decimal) |
| `fill_rate_value` | Fill rate (value) | Ratio (decimal) |

**`executions_by_strategy` Fields**

| Field | Description | Unit/Type |
| :--- | :--- | :--- |
| `owner_strategy_id` | Strategy identifier | String |
| `execution_count` | Execution count | Int |
| `total_quantity` | Total executed quantity | Float |
| `total_notional` | Total executed value | Float |
| `total_commission` | Total commission | Float |
| `avg_fill_price` | Average fill price | Float |

When margin mode is enabled and forced liquidation occurs, `result.viz.report(...)` automatically includes:

- a forced liquidation audit table (date, daily interest, symbols, priority)
- daily liquidation charts in the risk chart section (shown when data exists)
