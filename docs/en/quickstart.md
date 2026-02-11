# Quick Start

Welcome to AKQuant! Let's run the simplest strategy as quickly as possible.

## 1. Installation

Open your terminal and run:

```bash
pip install akquant
```

## 2. Minimal Example

This strategy is very simple:

1. When Close > Open (Bullish) -> Buy
2. When Close < Open (Bearish) -> Sell

```python
import akshare as ak
from akquant import Strategy, run_backtest

# 1. Prepare Data (Using AKShare to get data, install via `pip install akshare`)
df = ak.stock_zh_a_daily(symbol="sh600000", start_date="20250101", end_date="20260212")


# 2. Define Strategy
class MyStrategy(Strategy):

    # 3. Strategy Logic
    def on_bar(self, bar):
        # Get current position
        current_pos = self.get_position(bar.symbol)
        # 1. When Close > Open (Bullish) -> Buy
        if current_pos == 0 and bar.close > bar.open:
            self.buy(bar.symbol, 100)  # Buy 100 shares

        # 2. When Close < Open (Bearish) -> Sell
        elif current_pos > 0 and bar.close < bar.open:
            self.close_position(bar.symbol)  # Sell all shares


# 3. Run Backtest
print("Starting backtest...")
result = run_backtest(
    data=df,  # Input data
    strategy=MyStrategy,  # Input strategy
    cash=100000.0,  # Initial cash
    symbol="sh600000"  # Trading symbol
)

# 4. View Results
print(result)  # Equivalent to print(result.metrics_df)
```

In the output, you will see the complete performance metrics of the strategy:

```text
Starting backtest...
2026-02-12 00:58:53 | INFO | Running backtest via run_backtest()...
BacktestResult:
                                            Value
name
start_time              2025-01-02 00:00:00+08:00
end_time                2026-02-11 00:00:00+08:00
duration                        405 days, 0:00:00
total_bars                                    271
trade_count                                  67.0
initial_market_value                     100000.0
end_market_value                      99100.68204
total_pnl                                  -188.0
unrealized_pnl                                0.0
total_return_pct                        -0.899318
annualized_return                       -0.008109
volatility                               0.002453
total_profit                                584.0
total_loss                                 -772.0
total_commission                        711.31796
max_drawdown                            913.30785
max_drawdown_pct                          0.91318
win_rate                                26.865672
loss_rate                               73.134328
winning_trades                               18.0
losing_trades                                49.0
avg_pnl                                  -2.80597
avg_return_pct                          -0.172318
avg_trade_bars                           2.014925
avg_profit                              32.444444
avg_profit_pct                           2.818291
avg_winning_trade_bars                   4.055556
avg_loss                               -15.755102
avg_loss_pct                            -1.270909
avg_losing_trade_bars                    1.265306
largest_win                                 120.0
largest_win_pct                         10.178117
largest_win_bars                              7.0
largest_loss                                -70.0
largest_loss_pct                        -5.380477
largest_loss_bars                             1.0
max_wins                                      2.0
max_losses                                    9.0
sharpe_ratio                            -3.305093
sortino_ratio                            -3.92213
profit_factor                            0.756477
ulcer_index                              0.004666
upi                                     -1.737695
equity_r2                                0.932224
std_error                               70.552942
calmar_ratio                            -0.887949
exposure_time_pct                       49.815498
var_95                                  -0.000281
var_99                                  -0.000625
cvar_95                                 -0.000434
cvar_99                                  -0.00071
sqn                                     -0.708177
kelly_criterion                         -0.086485
```

You can view detailed position metrics via `print(result.positions_df)`.

```text
     long_shares  short_shares  close  equity  market_value  margin  \
0          100.0           0.0  10.27  1027.0        1027.0  1027.0
1          100.0           0.0  10.30  1030.0        1030.0  1030.0
2          100.0           0.0  10.19  1019.0        1019.0  1019.0
3          100.0           0.0  10.21  1021.0        1021.0  1021.0
4          100.0           0.0  10.21  1021.0        1021.0  1021.0
..           ...           ...    ...     ...           ...     ...
130        100.0           0.0  11.03  1103.0        1103.0  1103.0
131        100.0           0.0  10.04  1004.0        1004.0  1004.0
132        100.0           0.0  10.23  1023.0        1023.0  1023.0
133        100.0           0.0  10.12  1012.0        1012.0  1012.0
134        100.0           0.0  10.18  1018.0        1018.0  1018.0
     unrealized_pnl    symbol                      date
0              16.0  sh600000 2025-01-07 00:00:00+08:00
1              19.0  sh600000 2025-01-08 00:00:00+08:00
2               8.0  sh600000 2025-01-09 00:00:00+08:00
3               5.0  sh600000 2025-01-15 00:00:00+08:00
4               5.0  sh600000 2025-01-16 00:00:00+08:00
..              ...       ...                       ...
130            -7.0  sh600000 2026-01-20 00:00:00+08:00
131           -10.0  sh600000 2026-01-30 00:00:00+08:00
132             8.0  sh600000 2026-02-05 00:00:00+08:00
133            -3.0  sh600000 2026-02-06 00:00:00+08:00
134            -1.0  sh600000 2026-02-10 00:00:00+08:00
[135 rows x 9 columns]
```

You can view detailed order metrics via `print(result.orders_df)`.

```text
                                       id    symbol  side order_type  \
0    fe400570-5971-4307-afff-13d47b627148  sh600000   buy     market
1    111b9549-e363-439d-a79e-85c7e9f70295  sh600000  sell     market
2    27967cc5-7e67-4f4a-af9e-78263d25a6e8  sh600000   buy     market
3    c69e6a33-f157-464f-8dba-237abafa1dde  sh600000  sell     market
4    1e3ed424-eebc-49bd-ae9d-62b2cb779347  sh600000   buy     market
..                                    ...       ...   ...        ...
129  f4d338e6-ef74-4658-868a-f19f0e57c3b9  sh600000  sell     market
130  419e9504-c6bf-4708-9b62-f9cebac62876  sh600000   buy     market
131  0f682e75-fb18-4336-9b40-c821c385ec70  sh600000  sell     market
132  52f97d2d-741e-4511-b104-81e99123f870  sh600000   buy     market
133  95408a55-9b3d-4620-a752-f74b3272db25  sh600000  sell     market
     quantity  filled_quantity  limit_price  stop_price  avg_price  \
0       100.0            100.0          NaN         NaN      10.11
1       100.0            100.0          NaN         NaN      10.23
2       100.0            100.0          NaN         NaN      10.16
3       100.0            100.0          NaN         NaN      10.25
4       100.0            100.0          NaN         NaN      10.32
..        ...              ...          ...         ...        ...
129     100.0            100.0          NaN         NaN      10.07
130     100.0            100.0          NaN         NaN      10.15
131     100.0            100.0          NaN         NaN      10.11
132     100.0            100.0          NaN         NaN      10.19
133     100.0            100.0          NaN         NaN      10.18
     commission  status time_in_force                created_at
0       5.01011  filled           gtc 2025-01-06 00:00:00+08:00
1       5.52173  filled           gtc 2025-01-09 00:00:00+08:00
2       5.01016  filled           gtc 2025-01-14 00:00:00+08:00
3       5.52275  filled           gtc 2025-01-16 00:00:00+08:00
4       5.01032  filled           gtc 2025-01-17 00:00:00+08:00
..          ...     ...           ...                       ...
129     5.51357  filled           gtc 2026-01-30 00:00:00+08:00
130     5.01015  filled           gtc 2026-02-04 00:00:00+08:00
131     5.51561  filled           gtc 2026-02-06 00:00:00+08:00
132     5.01019  filled           gtc 2026-02-09 00:00:00+08:00
133     5.51918  filled           gtc 2026-02-10 00:00:00+08:00
[134 rows x 13 columns]
```

You can view detailed trade metrics via `print(result.trades_df)`.

```text
      symbol                entry_time                 exit_time  entry_price  \
0   sh600000 2025-01-07 00:00:00+08:00 2025-01-10 00:00:00+08:00        10.11
1   sh600000 2025-01-15 00:00:00+08:00 2025-01-17 00:00:00+08:00        10.16
2   sh600000 2025-01-20 00:00:00+08:00 2025-01-23 00:00:00+08:00        10.32
3   sh600000 2025-01-24 00:00:00+08:00 2025-02-06 00:00:00+08:00        10.26
4   sh600000 2025-02-07 00:00:00+08:00 2025-02-10 00:00:00+08:00        10.43
..       ...                       ...                       ...          ...
62  sh600000 2026-01-13 00:00:00+08:00 2026-01-14 00:00:00+08:00        11.72
63  sh600000 2026-01-20 00:00:00+08:00 2026-01-21 00:00:00+08:00        11.10
64  sh600000 2026-01-30 00:00:00+08:00 2026-02-02 00:00:00+08:00        10.14
65  sh600000 2026-02-05 00:00:00+08:00 2026-02-09 00:00:00+08:00        10.15
66  sh600000 2026-02-10 00:00:00+08:00 2026-02-11 00:00:00+08:00        10.19
    exit_price  quantity  side   pnl   net_pnl  return_pct  commission  \
0        10.23     100.0  Long  12.0   1.46816    1.186944    10.53184
1        10.25     100.0  Long   9.0  -1.53291    0.885827    10.53291
2        10.11     100.0  Long -21.0 -31.52593   -2.034884    10.52593
3        10.38     100.0  Long  12.0   1.46036    1.169591    10.53964
4        10.32     100.0  Long -11.0 -21.53675   -1.054650    10.53675
..         ...       ...   ...   ...       ...         ...         ...
62       11.61     100.0  Long -11.0 -21.60383   -0.938567    10.60383
63       11.03     100.0  Long  -7.0 -17.57363   -0.630631    10.57363
64       10.07     100.0  Long  -7.0 -17.52371   -0.690335    10.52371
65       10.11     100.0  Long  -4.0 -14.52576   -0.394089    10.52576
66       10.18     100.0  Long  -1.0 -11.52937   -0.098135    10.52937
    duration_bars duration
0               3   3 days
1               2   2 days
2               3   3 days
3               3  13 days
4               1   3 days
..            ...      ...
62              1   1 days
63              1   1 days
64              1   3 days
65              2   4 days
66              1   1 days
[67 rows x 13 columns]
```

## 3. Advanced Learning

Too simple? Want to learn how to write real quantitative strategies (like Dual Moving Average, MACD, etc.)?

👉 **Please read [Tutorial: Writing Your First Strategy](tutorial.md)**

This tutorial will cover:

*   How to get historical data (`get_history`)
*   How to calculate technical indicators (MA, RSI)
*   How to implement stop-loss and take-profit
