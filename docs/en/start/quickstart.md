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
# Trading calendar: without it, every weekday (holidays included) is treated as a trading day
trade_dates = ak.tool_trade_date_hist_sina()["trade_date"]


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
    symbols="sh600000",  # Trading symbol
    broker_profile="cn_stock_t1_low_fee",  # China A-share fees, T+1, 100-share lots
    trading_calendar=trade_dates,  # Trading calendar
)

# 4. View Results
print(result)  # Equivalent to print(result.metrics_df)
```

> **Note: by default `run_backtest` charges no costs and does not enforce T+1.** Without `broker_profile` or explicit fee arguments, commission, stamp tax and transfer fee are all 0, same-day round trips are allowed and the lot size is 1 share — an idealised result that flatters high-turnover strategies. For China A-share backtests pass a `broker_profile` as above, or set `commission_rate`, `stamp_tax_rate`, `transfer_fee_rate`, `min_commission`, `t_plus_one=True` and `lot_size=100` explicitly. The templates use the stamp tax rate in force since 2023-08-28 (0.05%); for earlier data pass `stamp_tax_rate=0.001`.

## Benchmark Quick View

After the minimal backtest, you can generate an HTML report with benchmark comparison:

```python
benchmark_returns = (
    df.set_index("date")["close"].pct_change().fillna(0.0).rename("SIMPLE_BENCH")
)

result.viz.report(
    filename="quickstart_report.html",
    show=False,
    benchmark=benchmark_returns,
)
```

The report will add a “Benchmark Comparison” section with strategy/benchmark/excess cumulative curves and relative metrics, including total excess return, annual excess return, tracking error, information ratio, beta, and alpha.

In the output, you will see the complete performance metrics of the strategy:

```text
Starting backtest...
2026-10-29 19:24:06 | INFO | akquant.backtest | Running backtest via run_backtest()...
BacktestResult:
                                            Value
name
start_time              2025-01-02 15:00:00+08:00
end_time                2026-02-12 15:00:00+08:00
duration                        406 days, 0:00:00
total_bars                                    272
closed_trade_count                           67.0
execution_count                             134.0
open_position_count                           0.0
initial_market_value                     100000.0
end_market_value                     99353.571893
total_pnl                             -646.428107
unrealized_pnl                                0.0
total_return_pct                        -0.646428
annualized_return                       -0.005813
volatility                                0.00241
total_profit                             579.8034
total_loss                               -783.713
total_commission                       442.518507
max_drawdown                             0.006622
max_drawdown_value                     662.321952
max_drawdown_pct                         0.662217
win_rate                                26.865672
loss_rate                               73.134328
winning_trades                               18.0
losing_trades                                49.0
avg_pnl                                 -3.043427
avg_return_pct                          -0.192281
avg_trade_bars                           2.014925
avg_profit                                32.2113
avg_profit_pct                           2.797729
avg_winning_trade_bars                   4.055556
avg_loss                               -15.994143
avg_loss_pct                            -1.290653
avg_losing_trade_bars                    1.265306
largest_win                              119.7522
largest_win_pct                         10.156084
largest_win_bars                              7.0
largest_loss                             -70.2532
largest_loss_pct                        -5.399399
largest_loss_bars                             1.0
max_wins                                      2.0
max_losses                                    9.0
sharpe_ratio                            -2.500877
sortino_ratio                            -3.07497
profit_factor                            0.739816
ulcer_index                              0.003265
upi                                     -1.780317
equity_r2                                0.878604
std_error                               69.866917
calmar_ratio                            -0.877868
exposure_time_pct                       49.632353
var_95                                  -0.000262
var_99                                  -0.000623
cvar_95                                 -0.000421
cvar_99                                 -0.000702
sqn                                     -0.768037
kelly_criterion                         -0.094483
max_leverage                             0.014608
min_margin_level                        68.456055
```

`print(result.positions_df)` shows the daily positions; the main columns:

```python
print(result.positions_df[["long_shares", "close", "market_value", "unrealized_pnl", "date"]])
```

```text
     long_shares  close  market_value  unrealized_pnl                      date
0          100.0  10.27        1027.0         15.8989 2025-01-07 15:00:00+08:00
1          100.0  10.30        1030.0         18.8989 2025-01-08 15:00:00+08:00
2          100.0  10.19        1019.0          7.8989 2025-01-09 15:00:00+08:00
3          100.0  10.21        1021.0          4.8984 2025-01-15 15:00:00+08:00
4          100.0  10.21        1021.0          4.8984 2025-01-16 15:00:00+08:00
..           ...    ...           ...             ...                       ...
130        100.0  11.03        1103.0         -7.1110 2026-01-20 15:00:00+08:00
131        100.0  10.04        1004.0        -10.1014 2026-01-30 15:00:00+08:00
132        100.0  10.23        1023.0          7.8985 2026-02-05 15:00:00+08:00
133        100.0  10.12        1012.0         -3.1015 2026-02-06 15:00:00+08:00
134        100.0  10.18        1018.0         -1.1019 2026-02-10 15:00:00+08:00

[135 rows x 5 columns]
```

### Why may these timestamps differ from `timestamp_iso`?

In the examples above, outputs such as `result`, `positions_df`, `orders_df`, and `trades_df` often show local timestamps like `+08:00`. Those views are mainly designed for human reading.

But if you explicitly print:

```python
print(bar.timestamp_iso)
```

you may see a UTC ISO 8601 string such as:

```text
2025-01-06T07:00:00Z
```

That does not mean the time is wrong. It is the same instant expressed in two ways:

*   Local market display: better for reading logs and backtest outputs
*   UTC structured field: better for ordering, cross-language transport, JSON logs, and multi-market consistency

`print(result.orders_df)` shows every order (fees are in the `commission` column); the main columns:

```python
print(result.orders_df[["side", "quantity", "filled_quantity", "avg_price", "commission", "status", "created_at"]])
```

```text
     side  quantity  filled_quantity  avg_price  commission  status                created_at
0     buy     100.0            100.0  10.111011    3.005056  filled 2025-01-06 15:00:00+08:00
1    sell     100.0            100.0  10.228977    3.516563  filled 2025-01-09 15:00:00+08:00
2     buy     100.0            100.0  10.161016    3.005081  filled 2025-01-14 15:00:00+08:00
3    sell     100.0            100.0  10.248975    3.517573  filled 2025-01-16 15:00:00+08:00
4     buy     100.0            100.0  10.321032    3.005161  filled 2025-01-17 15:00:00+08:00
..    ...       ...              ...        ...         ...     ...                       ...
129  sell     100.0            100.0  10.068993    3.508484  filled 2026-01-30 15:00:00+08:00
130   buy     100.0            100.0  10.151015    3.005076  filled 2026-02-04 15:00:00+08:00
131  sell     100.0            100.0  10.108989    3.510504  filled 2026-02-06 15:00:00+08:00
132   buy     100.0            100.0  10.191019    3.005096  filled 2026-02-09 15:00:00+08:00
133  sell     100.0            100.0  10.178982    3.514039  filled 2026-02-10 15:00:00+08:00

[134 rows x 7 columns]
```

`print(result.trades_df)` shows each round trip (entry to exit); the main columns:

```python
print(result.trades_df[["entry_time", "exit_time", "entry_price", "exit_price", "quantity", "pnl", "net_pnl", "commission"]])
```

```text
                  entry_time                 exit_time  entry_price  exit_price  quantity      pnl    net_pnl  commission
0  2025-01-07 15:00:00+08:00 2025-01-10 15:00:00+08:00    10.111011   10.228977     100.0  11.7966   5.274981    6.521619
1  2025-01-15 15:00:00+08:00 2025-01-17 15:00:00+08:00    10.161016   10.248975     100.0   8.7959   2.273246    6.522654
2  2025-01-20 15:00:00+08:00 2025-01-23 15:00:00+08:00    10.321032   10.108989     100.0 -21.2043 -27.719964    6.515664
3  2025-01-24 15:00:00+08:00 2025-02-06 15:00:00+08:00    10.261026   10.378962     100.0  11.7936   5.264332    6.529268
4  2025-02-07 15:00:00+08:00 2025-02-10 15:00:00+08:00    10.431043   10.318968     100.0 -11.2075 -17.733823    6.526323
..                       ...                       ...          ...         ...       ...      ...        ...         ...
62 2026-01-13 15:00:00+08:00 2026-01-14 15:00:00+08:00    11.721172   11.608839     100.0 -11.2333 -17.825407    6.592107
63 2026-01-20 15:00:00+08:00 2026-01-21 15:00:00+08:00    11.101110   11.028897     100.0  -7.2213 -13.783810    6.562510
64 2026-01-30 15:00:00+08:00 2026-02-02 15:00:00+08:00    10.141014   10.068993     100.0  -7.2021 -13.715655    6.513555
65 2026-02-05 15:00:00+08:00 2026-02-09 15:00:00+08:00    10.151015   10.108989     100.0  -4.2026 -10.718179    6.515579
66 2026-02-10 15:00:00+08:00 2026-02-11 15:00:00+08:00    10.191019   10.178982     100.0  -1.2037  -7.722834    6.519134

[67 rows x 8 columns]
```

## Report & Analysis Outputs

After the backtest, you can generate an interactive report directly:

```python
result.viz.report(
    show=True,
    filename="report.html",
    compact_currency=True,
    curve_freq="D",
)

result.viz.report(
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
result.viz.report(filename="report_margin.html", show=False)
```

The generated report includes:

*   Forced liquidation audit table (date, symbols, priority, interest of this settlement, accrued over the calendar days between two settlements, e.g. 3 days from Friday to Monday)
*   Daily liquidation charts in the risk chart area (shown when data exists)

## 5. Advanced Learning

Too simple? Want to learn how to write real quantitative strategies (like Dual Moving Average, MACD, etc.)?

👉 **Please read [Tutorial: Writing Your First Strategy](first_strategy.md)**

This tutorial will cover:

*   How to get historical data (`get_history`)
*   How to calculate technical indicators (MA, RSI)
*   How to implement stop-loss and take-profit
