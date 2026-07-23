# 快速开始

欢迎使用 AKQuant！让我们用最快的速度运行一个最简单的策略。

## 1. 安装

打开终端，运行：

```bash
uv pip install akquant akshare
```

## 2. 极简示例

这个策略非常简单：

1. 当收盘价 > 开盘价 (阳线) -> 买入
2. 当收盘价 < 开盘价 (阴线) -> 卖出

> 提示：
> 如果你后面在日志、JSON 输出或对象属性里看到 `timestamp_iso`、`event_time_iso` 这类字段是 `Z` 结尾的 UTC 字符串，不要慌。
> AKQuant 统一用 UTC 保存“事件真实发生的时刻”，而终端里很多面向阅读的时间仍可能按本地时区显示。
> 记住一句话就够了：**AKQuant 用 UTC 保存事实时间，用本地时区展示给人看。**
> 想深入理解，请阅读 [AKQuant 的时间与时区](../guide/quant_basics.md) 和 [时区处理指南](../advanced/timezone.md)。

```python
import akshare as ak
from akquant import Strategy, run_backtest

# 1. 准备数据 (这里利用 AKShare 来获取数据；若尚未安装，可先执行 `uv pip install akshare`)
df = ak.stock_zh_a_daily(symbol="sh600000", start_date="20250101", end_date="20260212")


# 2. 策略定义
class MyStrategy(Strategy):

    # 3. 策略逻辑
    def on_bar(self, bar):
        # 获取当前持仓
        current_pos = self.get_position(bar.symbol)
        # 1. 当收盘价 > 开盘价 (阳线) -> 买入
        if current_pos == 0 and bar.close > bar.open:
            self.buy(bar.symbol, 100)  # 买入 100 股

        # 2. 当收盘价 < 开盘价 (阴线) -> 卖出
        elif current_pos > 0 and bar.close < bar.open:
            self.close_position(bar.symbol)  # 卖出所有持仓


# 3. 运行回测
print("开始回测...")
result = run_backtest(
    data=df,  # 输入数据
    strategy=MyStrategy,  # 输入策略
    initial_cash=100000.0,  # 初始资金
    symbols="sh600000"  # 交易的股票代码
)

# 4. 查看结果
print(result)  # 跟 print(result.metrics_df) 等效
```

## 基准对比快览

完成最小回测后，可以立即生成带基准对比的 HTML 报告：

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

报告会新增“基准对比 (Benchmark Comparison)”区块，展示策略/基准/超额累计收益曲线，以及累计超额收益、年化超额收益、跟踪误差、信息比率、Beta、Alpha 等指标。

运行结果中，你会看到该策略的完整绩效评价指标：

```text
开始回测...
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

可以通过 `print(result.positions_df)` 来查看详细的持仓指标。

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

### 结果里的时间为什么和 `timestamp_iso` 可能不一样？

你在上面的 `result`、`positions_df`、`orders_df`、`trades_df` 输出里，会经常看到类似 `+08:00` 的本地时间。这些结果主要面向人类阅读。

但如果你在策略里主动打印：

```python
print(bar.timestamp_iso)
```

你看到的可能会是 UTC ISO 8601，例如：

```text
2025-01-06T16:00:00Z
```

这并不表示时间错了，而是同一个时刻的不同表达：

- 本地市场展示：更适合读日志、看回测结果
- UTC 结构化字段：更适合排序、跨语言传递、JSON 日志和多市场统一处理

可以通过 `print(result.orders_df)` 来查看详细的订单指标。

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

可以通过 `print(result.trades_df)` 来查看详细的成交指标。

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

## 复杂订单 30 秒示例

如果你需要“进场 + 止损 + 止盈 + 自动 OCO 联动”，可以直接使用 `place_bracket`：

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

完整可运行脚本见：`examples/06_complex_orders.py`。

## 报告与分析输出

回测完成后，你可以直接生成交互式报告：

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

你也可以直接复用曲线数据做二次分析：

```python
equity = result.equity_curve
cash = result.cash_curve
margin = result.margin_curve

equity_daily = result.equity_curve_daily
cash_daily = result.cash_curve_daily
margin_daily = result.margin_curve_daily
```

你也可以直接获取结构化分析结果用于二次研究：

```python
exposure = result.exposure_df()
attr_by_symbol = result.attribution_df(by="symbol")
attr_by_tag = result.attribution_df(by="tag")
capacity = result.capacity_df()
orders_by_strategy = result.orders_by_strategy()
executions_by_strategy = result.executions_by_strategy()
```

## 2.1 快速读取标的静态属性

当策略需要读取合约属性（如到期日、行权价、乘数）时，推荐使用策略 API：

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

## 3. 进阶学习

刚才的例子太简单了？想要学习如何编写真正的量化策略（如双均线、MACD 等）？

👉 **请阅读 [手把手教程：编写你的第一个量化策略](first_strategy.md)**

该教程将详细讲解：

*   如何获取历史数据 (`get_history`)
*   如何计算技术指标 (MA, RSI)
*   如何止盈止损

## 4. 流式回测

如果你希望在回测过程中实时接收事件（进度、权益、订单、成交、结束状态），可以使用
`run_backtest` 并传入 `on_event`。

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

如果你还想在策略内部直接处理到期结算后的动作，可以实现 `on_expiry(event)`；当引擎实际执行 `expiry_date` 驱动的结算/移除后会触发该回调。最小示例见：`examples/49_on_expiry_demo.py`。

常用流式参数说明：

- `stream_progress_interval`: `progress` 事件采样间隔（正整数）
- `stream_equity_interval`: `equity` 事件采样间隔（正整数）
- `stream_batch_size`: 事件批量刷新阈值（正整数）
- `stream_max_buffer`: 最大缓冲事件数（正整数）
- `stream_error_mode`: 回调异常策略
  - `"continue"`：回调报错后继续回测，并在 `finished.payload` 给出错误统计
  - `"fail_fast"`：回调报错后立即终止并抛出异常

### 4.1 PyCharm 下进度条不显示的排查与解决

在 PyCharm 的 Run 窗口中，如果未开启终端仿真，`stderr` 可能不是 TTY，导致原生进度条不可见。

推荐按以下顺序处理：

1. 在 `Run/Debug Configurations` 中勾选 `Emulate terminal in output console`。
2. 确认回测参数中未关闭进度条（即 `show_progress=True`）。
3. 若你更希望跨 IDE 稳定显示进度，建议使用 `on_event` 消费 `progress` 事件。

示例（不依赖原生进度条）：

```python
from akquant import run_backtest

progress_events = 0


def on_event(event):
    global progress_events
    if event.get("event_type") == "progress":
        progress_events += 1
        if progress_events % 10 == 0:
            print(f"[progress] events={progress_events}", flush=True)


result = run_backtest(
    data=df,
    strategy=MyStrategy,
    symbols="sh600000",
    on_event=on_event,
    show_progress=False,
    stream_progress_interval=5,
)
```

流式事件公共字段：

- `run_id`: 本次流式回测 ID
- `seq`: 事件序号（单调递增）
- `ts`: 事件时间戳（纳秒）
- `event_type`: 事件类型
- `symbol`: 关联标的（部分事件为空）
- `level`: 事件级别
- `payload`: 事件内容

阶段 5 迁移 FAQ：

- `run_backtest` 是否改名？不改名，调用方式保持不变。
- `run_backtest` 是否还能不传 `on_event`？可以，不传时仍返回同样的结果对象语义。
- PyCharm 看不到进度条怎么办？开启 `Emulate terminal in output console`，或使用 `on_event + progress` 事件输出文本进度。
- 如何回滚？阶段 5 后不再支持 `_engine_mode` 参数级回滚，建议使用版本级回滚。
- 首页导航入口：见 [文档首页的阶段 5 迁移入口](../index.md)。

## 5. 信用账户与强平审计（回测）

如果你要在回测中验证融资/融券场景，可在 `RiskConfig` 中开启 `margin` 模式：

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

回测完成后可以直接查看审计数据：

```python
print(result.liquidation_audit_df)
result.viz.report(filename="report_margin.html", show=False)
```

`report_margin.html` 会包含：

- 强平审计明细（日期、强平标的、优先顺序、当日利息）
- 风险图表区中的按日强平统计图（有数据时显示）
