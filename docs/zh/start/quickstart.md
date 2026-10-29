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
# 交易日历：不传时按工作日推算，节假日会被当作交易日
trade_dates = ak.tool_trade_date_hist_sina()["trade_date"]


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
    symbols="sh600000",  # 交易的股票代码
    broker_profile="cn_stock_t1_low_fee",  # A 股佣金/印花税/过户费、T+1、一手 100 股
    trading_calendar=trade_dates,  # 交易日历
)

# 4. 查看结果
print(result)  # 跟 print(result.metrics_df) 等效
```

> **注意：`run_backtest` 默认不计交易成本、不开 T+1。** 不传 `broker_profile` 也不传费率时，佣金、印花税、过户费都是 0，可以当日买卖，最小交易单位是 1 股——这样得到的是理想化的结果，换手越高越偏乐观。A 股回测请像上面这样传 `broker_profile`，或者显式传 `commission_rate`、`stamp_tax_rate`、`transfer_fee_rate`、`min_commission`、`t_plus_one=True`、`lot_size=100`。模板的印花税按 2023-08-28 起的现行税率 0.05% 计，回测此前的行情请显式传 `stamp_tax_rate=0.001`。

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

可以通过 `print(result.positions_df)` 查看每日持仓明细，下面只列出主要几列：

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

### 结果里的时间为什么和 `timestamp_iso` 可能不一样？

你在上面的 `result`、`positions_df`、`orders_df`、`trades_df` 输出里，会经常看到类似 `+08:00` 的本地时间。这些结果主要面向人类阅读。

但如果你在策略里主动打印：

```python
print(bar.timestamp_iso)
```

你看到的可能会是 UTC ISO 8601，例如：

```text
2025-01-06T07:00:00Z
```

这并不表示时间错了，而是同一个时刻的不同表达：

- 本地市场展示：更适合读日志、看回测结果
- UTC 结构化字段：更适合排序、跨语言传递、JSON 日志和多市场统一处理

可以通过 `print(result.orders_df)` 查看订单明细（手续费在 `commission` 列），下面只列出主要几列：

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

可以通过 `print(result.trades_df)` 查看每笔完整交易（开仓到平仓），下面只列出主要几列：

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

- 强平审计明细（日期、强平标的、优先顺序、本次结算利息；按两次结算之间的自然日天数计提，周五到周一计 3 天）
- 风险图表区中的按日强平统计图（有数据时显示）
