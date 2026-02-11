# Backtest Results & Metrics

This document details the performance metrics in AKQuant backtest results (`metrics_df`), including their meanings, units, and calculation methods.

## Metrics Overview

| Name | Description | Unit/Type | Calculation |
| :--- | :--- | :--- | :--- |
| `start_time` | Backtest Start Time | Datetime | Time of the first bar. |
| `end_time` | Backtest End Time | Datetime | Time of the last bar. |
| `duration` | Backtest Duration | Timedelta | `end_time - start_time`. |
| `total_bars` | Total Bars | Int | Total number of bars in backtest. |
| `trade_count` | Trade Count | Int | Total number of closed trades (Round-trip). |
| `initial_market_value` | Initial Market Value | Float | Initial capital (usually Cash). |
| `end_market_value` | End Market Value | Float | Total asset value at the end (Cash + Position Value). |
| `total_pnl` | Total PnL | Float | `end_market_value - initial_market_value`. |
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
