# Quick Start

Welcome to AKQuant! Let's run the simplest strategy as quickly as possible.

## 1. Installation

Open your terminal and run:

```bash
uv pip install akquant akshare
```

## 2. Minimal Example

This strategy is very simple:

1. When Close > Open (Bullish) -> Buy
2. When Close < Open (Bearish) -> Sell

> Note:
> If you later see fields such as `timestamp_iso` or `event_time_iso` ending with `Z` in logs, JSON output, or object properties, that is expected.
> AKQuant stores the factual event time in UTC, while many human-facing outputs still render time in the local market timezone.
> The one-line rule is: **AKQuant stores facts in UTC and renders local time for humans.**
> For the full explanation, see [AKQuant Time and Timezones](../guide/quant_basics.md) and the [Timezone Handling Guide](../advanced/timezone.md).

```python
import akshare as ak
from akquant import Strategy, run_backtest

# 1. Prepare Data (Using AKShare to get data; if needed, install it first via `uv pip install akshare`)
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
    initial_cash=100000.0,  # Initial cash
    symbols="sh600000"  # Trading symbol
)

# 4. View Results
print(result)  # Equivalent to print(result.metrics_df)
```

## Benchmark Quick View

After the minimal backtest, you can generate an HTML report with benchmark comparison:

```python
benchmark_returns = (
    df.set_index("date")["close"].pct_change().fillna(0.0).rename("SIMPLE_BENCH")
)

result.report(
    filename="quickstart_report.html",
    show=False,
    benchmark=benchmark_returns,
)
```

The report will add a “Benchmark Comparison” section with strategy/benchmark/excess cumulative curves and relative metrics, including total excess return, annual excess return, tracking error, information ratio, beta, and alpha.

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
closed_trade_count                                  67.0
execution_count                                    134.0
open_position_count                                  0.0
initial_market_value                     100000.0
end_market_value                      99100.68204
total_pnl                              -899.31796
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

### Why may these timestamps differ from `timestamp_iso`?

In the examples above, outputs such as `result`, `positions_df`, `orders_df`, and `trades_df` often show local timestamps like `+08:00`. Those views are mainly designed for human reading.

But if you explicitly print:

```python
print(bar.timestamp_iso)
```

you may see a UTC ISO 8601 string such as:

```text
2025-01-06T16:00:00Z
```

That does not mean the time is wrong. It is the same instant expressed in two ways:

*   Local market display: better for reading logs and backtest outputs
*   UTC structured field: better for ordering, cross-language transport, JSON logs, and multi-market consistency

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

## Report & Analysis Outputs

After the backtest, you can generate an interactive report directly:

```python
result.report(
    show=True,
    filename="report.html",
    compact_currency=True,
    curve_freq="D",
)

result.report(
    show=False,
    filename="report_raw_amount.html",
    compact_currency=False,
    curve_freq="D",
)
```

Curve outputs are also directly reusable:

```python
equity = result.equity_curve
cash = result.cash_curve
margin = result.margin_curve

equity_daily = result.equity_curve_daily
cash_daily = result.cash_curve_daily
margin_daily = result.margin_curve_daily
```

You can also reuse structured analysis outputs for downstream research:

```python
exposure = result.exposure_df()
attr_by_symbol = result.attribution_df(by="symbol")
attr_by_tag = result.attribution_df(by="tag")
capacity = result.capacity_df()
orders_by_strategy = result.orders_by_strategy()
executions_by_strategy = result.executions_by_strategy()
```

## 2.1 Quick Instrument Metadata Access

When strategy logic needs static contract fields (for example expiry, strike, multiplier), prefer Strategy APIs:

- `self.get_instrument(symbol)`
- `self.get_instrument_field(symbol, field)`
- `self.get_instrument_config(symbol, fields=None)`

```python
import akquant
from akquant import BacktestConfig, InstrumentConfig, Strategy, StrategyConfig, run_backtest


class MetaQuickStrategy(Strategy):
    def on_start(self):
        self.subscribe("OPTION_A")
        expiry = self.get_instrument_field("OPTION_A", "expiry_date")
        print(expiry)

    def on_bar(self, bar):
        meta = self.get_instrument_config(
            bar.symbol, fields=["asset_type", "option_type", "multiplier"]
        )
        if meta["asset_type"] == "OPTION":
            pass


config = BacktestConfig(
    strategy_config=StrategyConfig(),
    instruments_config=[
        InstrumentConfig(
            symbol="OPTION_A",
            asset_type=akquant.InstrumentAssetTypeEnum.OPTION,
            option_type=akquant.InstrumentOptionTypeEnum.CALL,
            strike_price=100.0,
            expiry_date=20260131,
            multiplier=100.0,
        )
    ]
)

result = run_backtest(data=df, strategy=MetaQuickStrategy, config=config)
```

## Complex Orders in 30 Seconds

If you need an entry + stop-loss + take-profit flow with automatic OCO linkage, use `place_bracket` directly:

```python
from akquant import OrderStatus, Strategy

class BracketQuickStrategy(Strategy):
    def __init__(self):
        self.entry_order_id = ""

    def on_bar(self, bar):
        if self.get_position(bar.symbol) > 0 or self.entry_order_id:
            return
        self.entry_order_id = self.place_bracket(
            symbol=bar.symbol,
            quantity=100,
            stop_trigger_price=bar.close * 0.98,
            take_profit_price=bar.close * 1.04,
            entry_tag="entry",
            stop_tag="stop",
            take_profit_tag="take",
        )

    def on_order(self, order):
        if order.id == self.entry_order_id and order.status in (
            OrderStatus.Cancelled,
            OrderStatus.Rejected,
        ):
            self.entry_order_id = ""
```

For a full runnable script, see `examples/06_complex_orders.py`.

## 3. Streaming Backtest

If you want real-time events during a backtest (progress, equity, orders, trades,
and final status), use `run_backtest` with `on_event`.

```python
from akquant import run_backtest

def on_event(event):
    if event["event_type"] == "finished":
        payload = event["payload"]
        print("status:", payload.get("status"))
        print("callback_error_count:", payload.get("callback_error_count"))

result = run_backtest(
    data=df,
    strategy=MyStrategy,
    symbols="sh600000",
    on_event=on_event,
    show_progress=False,
    stream_progress_interval=10,
    stream_equity_interval=10,
    stream_batch_size=32,
    stream_max_buffer=256,
    stream_error_mode="continue",
)

print("event_stats:", result.get_event_stats())
```

If you want to react inside the strategy after an actual expiry settlement/removal, implement `on_expiry(event)`. The callback fires only after the engine executes an `expiry_date` driven settlement task. Runnable example: `examples/49_on_expiry_demo.py`.

Common streaming options:

*   `stream_progress_interval`: sampling interval for `progress` events (positive int)
*   `stream_equity_interval`: sampling interval for `equity` events (positive int)
*   `stream_batch_size`: flush threshold for buffered events (positive int)
*   `stream_max_buffer`: maximum buffered event count (positive int)
*   `stream_error_mode`: callback exception policy
    *   `"continue"`: continue backtest on callback exceptions and report summary in
        `finished.payload`
    *   `"fail_fast"`: stop immediately and raise an exception on callback errors

Stream event common fields:

*   `run_id`: stream run id
*   `seq`: monotonic event sequence
*   `ts`: event timestamp in nanoseconds
*   `event_type`: event type
*   `symbol`: related symbol (nullable for some events)
*   `level`: event level
*   `payload`: event payload

Phase-5 Migration FAQ:

*   Is `run_backtest` renamed? No, the public entry name stays unchanged.
*   Can `run_backtest` still be called without `on_event`? Yes, and result-return semantics stay the same.
*   How do we roll back? Since Phase 5, `_engine_mode` runtime fallback is removed; use release-level rollback.
*   Home navigation entry: see [Quick Links in docs home](../index.md#quick-links).

## 4. Margin Account & Liquidation Audit (Backtest)

To validate financing/short-selling behavior in backtests, enable `margin` mode in `RiskConfig`:

```python
from akquant import run_backtest
from akquant.config import RiskConfig

result = run_backtest(
    data=df,
    strategy=MyStrategy,
    symbols="sh600000",
    risk_config=RiskConfig(
        account_mode="margin",
        enable_short_sell=True,
        initial_margin_ratio=0.5,
        maintenance_margin_ratio=0.3,
        financing_rate_annual=0.08,
        borrow_rate_annual=0.10,
        allow_force_liquidation=True,
        liquidation_priority="short_first",
    ),
)
```

After the run, inspect the audit output directly:

```python
print(result.liquidation_audit_df)
result.report(filename="report_margin.html", show=False)
```

The generated report includes:

*   Forced liquidation audit table (date, symbols, priority, daily interest)
*   Daily liquidation charts in the risk chart area (shown when data exists)

## 5. Advanced Learning

Too simple? Want to learn how to write real quantitative strategies (like Dual Moving Average, MACD, etc.)?

👉 **Please read [Tutorial: Writing Your First Strategy](first_strategy.md)**

This tutorial will cover:

*   How to get historical data (`get_history`)
*   How to calculate technical indicators (MA, RSI)
*   How to implement stop-loss and take-profit
