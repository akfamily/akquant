# 可视化与报告

以下是 AKQuant 生成的交互式回测报告示例。您可以在此页面直接与图表进行交互，查看详细的回测数据。

<iframe src="../../assets/reports/akquant_report.html" width="100%" height="1000px" frameborder="0" style="border: 1px solid #eee; border-radius: 4px;"></iframe>

## 基准对比

`BacktestResult.viz.report` 支持直接传入基准收益率序列：

```python
benchmark_returns = (
    benchmark_df.set_index("date")["close"].pct_change().fillna(0.0)
)
result.viz.report(
    filename="akquant_report.html",
    benchmark=benchmark_returns,
    show=False,
)
```

报告会新增“基准对比 (Benchmark Comparison)”区块，提供累计超额收益、年化超额收益、跟踪误差、信息比率、Beta、Alpha 等指标，并展示策略/基准/超额三条累计收益曲线。

## 结构化 Benchmark Analysis

从当前版本开始，AKQuant 不再只有 HTML 报告里的基准对比区块，还提供可直接给前端、API 或离线分析复用的结构化 benchmark analysis：

```python
benchmark_returns = (
    benchmark_df.set_index("date")["close"].pct_change().fillna(0.0)
)

payload = result.benchmark_analysis(
    benchmark=benchmark_returns,
    curve_freq="D",
)

print(payload["schema_version"])
print(payload["summary"]["annual_excess"])
print(payload["series"][0])
```

返回 payload 主要包含：

- `schema_version`: 数据契约版本
- `available`: 当前 benchmark analysis 是否可用
- `reason`: 当 benchmark 无法对齐或输入非法时的原因
- `benchmark.label`: 基准显示名称
- `summary`: 汇总指标，如 `total_excess`、`annual_excess`、`tracking_error`、`information_ratio`、`beta`、`alpha`（收益/误差类字段为小数，前端展示百分比需 ×100；比率类字段无量纲不乘，详见 [回测结果与指标详解](./analysis.md#summary-字段单位)）
- `series`: 对齐后的逐日序列，包含策略收益、基准收益、超额收益及三条累计收益曲线
- `meta`: 对齐样本数、起止日期、年化因子等元信息

推荐实践：

- 后端负责准备 benchmark 收益率序列并调用 `result.benchmark_analysis(...)`
- 前端直接消费 `summary + series + meta`
- `result.viz.report(..., benchmark=...)` 与前端页面应复用同一份 benchmark analysis 逻辑，而不是各自重新计算

## 导出给前端或归档

如果需要把 benchmark analysis 固化为回测产物，可以直接导出：

```python
result.export_benchmark_analysis(
    path="artifacts/benchmark_analysis.json",
    benchmark=benchmark_returns,
    format="json",
    curve_freq="D",
)
```

也支持 `format="parquet"`，会输出：

- `series.parquet`: 逐点时间序列
- `metadata.json`: 汇总指标与元信息

## LWC 交互式交易复盘

`result.viz.review()` 基于 [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts) 生成**离线自包含**的单文件 HTML，在 K 线上标注买卖点，面向大数据量 / 日内的交易复盘。

它与 `result.viz.report()` 是**互补而非替代**：分析类图表（权益曲线、回撤、热力图、归因等）仍由 `report()` 的 plotly 负责；`review()` 只补足「交互式 K 线 + 买卖点」这一场景，适合逐笔复盘成交时机。

```python
# market_data 为单个 DataFrame 或 {symbol: df} 字典
path = result.viz.review(
    market_data=df,
    title="AKQuant 交易复盘",
    theme="dark",          # 初始主题 "light" / "dark"，页面内可即时切换
    filename="akquant_review.html",
    show=False,            # True 则自动打开浏览器
)
```

要点：

- 生成的 HTML 内联了 lightweight-charts，**无 CDN 依赖**，可离线打开与归档。
- 页面顶部有**明暗主题切换按钮**，`theme` 参数只决定初始主题；切换时即时重着色，无需重新生成文件。
- 多标的行情（`{symbol: df}`）会在页面顶部提供标的切换下拉，`initial_symbol` 可指定初始展示标的。
- 当前标的带 candle 指标（如平均 K 线）时，顶部会多一个“隐藏 K 线”按钮；`hide_price=True` 让页面打开时就隐藏价格 K 线，买卖点照常显示。
- 行情列名大小写不敏感，并兼容中文列名（`开盘/最高/最低/收盘/成交量/日期` 等）。
- 日频数据用 `YYYY-MM-DD` 时间轴，日内数据自动切换为带时分的时间轴。
- 面向**大数据量 / 日内**优化:payload 用向量化构建、时间戳自动去重,数万根 K 线也能流畅复盘(这正是相对 plotly 分析图的核心优势)。

完整示例见 `examples/67_lwc_trade_review.py`。


## 策略绘图与统计表

`self.I()` 指标表达的是“每根 bar 一个数值”的时间序列；`self.draw` 画的是**带 id、可覆盖、可删除的对象**：趋势线、区间框、文字标注、折线、斐波那契回撤，以及固定在图表角落的统计表（对标 Pine 的 `line.new` / `box.new` / `label.new` / `polyline.new` / `table.new`）。只保留“当前存活的对象”，回测结束时展示的是最终状态，与 TradingView 一致。主策略和 `aq.Indicator` 脚本都可以使用，身份按 `(owner, symbol, id)` 区分，互不串线。

```python
class MyStrategy(aq.Strategy):
    def on_bar(self, bar):
        self.draw.label(f"sig-{bar.timestamp}", (bar.timestamp, bar.high), "突破", style="down", color="#e03131")
        self.draw.box("zone", (t0, 3820.0), (bar.timestamp, 3780.0), border_color="#1971c2", fill_color="#1971c233")
        self.draw.table("perf", self.stats.as_rows(), header=["指标", "数值"], symbol="*")
```

完整示例见 `examples/78_chart_drawings.py`。

### 对象与参数

每个方法的第一个参数是非空的 `id`，返回归一后的 `id`。公共可选参数 `symbol` 缺省取当前回调的标的。

| 方法 | 参数 | 说明 |
|---|---|---|
| `line(id, p1, p2, *, extend="none"\|"right"\|"both", color, width=1, style="solid"\|"dashed"\|"dotted")` | 两点线段 | `extend` 控制向右或双向延伸 |
| `box(id, top_left, bottom_right, *, border_color, fill_color, width=1, text="")` | 矩形区间 | 可带居中文字 |
| `label(id, point, text, *, style="down"\|"up"\|"left"\|"right"\|"none", color, text_color, size="normal"\|"small"\|"large")` | 锚点文字标注 | `style` 是气泡指向，默认 `"down"` |
| `polyline(id, points, *, closed=False, color, width=1, fill_color=None)` | 折线 / 闭合多边形 | 2～1000 个点 |
| `fib(id, p1, p2, *, color)` | 斐波那契回撤 | 档位固定为 0 / 0.236 / 0.382 / 0.5 / 0.618 / 0.786 / 1 |
| `table(id, rows, *, header=None, position="top_right", precision=2, bg_color, border_color)` | 屏幕固定位置的表格 | 见下文 |
| `delete(id, *, symbol=None)` | 删除单个对象 | 不存在时静默忽略 |
| `clear(kind=None, *, symbol=None)` | 清除本 owner 的对象 | 可按 kind / symbol 过滤；每个被清除的对象各发一条删除 |

坐标点是 `(time, price)`：`time` 可以是 `int`（UTC 纳秒，即 `bar.timestamp`）、`pd.Timestamp` 或 `datetime`（无时区的按策略时区本地化），NumPy 标量也可以。时间与指标点用同一个转换函数归一为毫秒，因此标注和 K 线对齐。颜色只接受 `#RRGGBB` 或 `#RRGGBBAA`，缺省（空串）表示交给渲染端的默认色。

`table` 的细节：

- `precision`：数值显示的小数位数，整数 0～10，默认 2，超出范围抛 `ValueError`。
- `position` 共 9 个：`top_left` / `top_center` / `top_right` / `middle_left` / `middle_center` / `middle_right` / `bottom_left` / `bottom_center` / `bottom_right`。
- `rows` 是二维序列，单元格可以是 `str` / `int` / `float` / `bool` / `None`，也可以是 `{"value": ..., "text": ..., "bg_color": ..., "text_color": ...}`。
- 归一后每个单元格是 `{value, text, bg_color, text_color}`：`value` 保留原值，`text` 是显示文本（数值按 `precision` 格式化，`None` 和 `NaN` 显示为“—”，自己给了 `text` 则以自己的为准）。
- 各行列数不齐时右侧补空单元格；`header` 视为第 0 行，并带 `is_header` 标记。
- `symbol="*"` 是全局表，在每个标的的图上都显示，只有 `table` 允许。

### 覆盖、删除与上限

- **同 id 覆盖**：同一 `(owner, symbol, id)` 再次调用即整体覆盖（不是增量修改字段）。内容与已有对象完全相同时不产生任何事件，所以每根 bar 都调用也只在内容变化时才有开销。
- **不同 kind 不能共用 id**：用 `line` 画过的 id 再用 `box` 画会抛 `ValueError`，要先 `draw.delete(id)`。同一个标的下 `table` 也占用 id，别与线、框重名。
- **删除与清除**：`delete` 删单个，`clear(kind=..., symbol=...)` 批量清除，只影响本 owner。

| 项 | 上限 | 超出时 |
|---|---|---|
| 每个 `(owner, symbol, kind)` 的存活对象数 | 500（类属性 `Strategy.max_drawings_per_kind`，在策略类里覆盖即可调整） | 淘汰最早创建的对象，每个 owner + kind 只记一条 `warning` |
| 表格 | 50 行 × 10 列 | `ValueError` |
| polyline 点数 | 1000 | `ValueError` |
| 单段文字 | 500 字符 | 截断并 `warning` |

淘汰而不是报错：Pine 用户习惯“每个信号打一个 label”，按 `max_labels_count` 的行为自动淘汰最旧的对象更符合预期。

!!! note "时间对齐"
    坐标点的时间等于当前 bar 的 `bar.timestamp` 时，会和指标点一样映射到 bar 的结束时刻，自动对齐。引用**更早的 bar** 的时间戳时不做这个映射：在由 tick 聚合出来的 bar 上，这样的点可能比对应的指标点靠左一根 bar。要标某一根 bar，就用那一根 bar 自己的 `bar.timestamp`。

### 运行时统计 `self.stats`

`self.stats` 每次访问返回一个冻结的快照 `StrategyStats`，访问时才计算。数值与最终 `result.metrics` 同源同算法，口径如下：

- 权益曲线每根 bar 一个点（同一时刻覆盖）；`max_drawdown*` 的峰值从初始资金算起。
- `win_rate` 是百分比（0～100），按**毛利** `pnl > 0` 判定，不是扣手续费后的净盈亏。
- `total_pnl` 等于当前权益减初始资金，**包含浮动盈亏**，不是逐笔毛利之和。
- 在回调中读到的是**当前状态**；在 `on_stop` 中读到的与 `result.metrics` 的同名字段完全相等。
- 统计是**账户级**的：多 slot 回测里所有 slot 读到的是同一份统计，按策略拆分的统计不在范围内。
- 运行期间的 `unrealized_pnl` 取账户的浮动盈亏；在 `on_stop` 中取交易记录按库存计算的值（与 metrics 相同）。股票和期货两者一致，期权等需要保证金计价的品种在运行期间可能有差异。
- 不在回调里、也不在 `on_stop` 里访问，抛 `RuntimeError`。

| 字段 | 含义 |
|---|---|
| `total_return_pct` / `total_pnl` / `unrealized_pnl` / `total_commission` | 收益率、总盈亏、浮动盈亏、手续费 |
| `max_drawdown` / `max_drawdown_pct` / `max_drawdown_value` | 最大回撤（比例 / 百分比 / 金额） |
| `win_rate` / `loss_rate` / `winning_trades` / `losing_trades` | 胜率、败率及笔数 |
| `profit_factor` / `avg_pnl` / `avg_profit` / `avg_loss` | 盈亏比、平均盈亏 |
| `largest_win` / `largest_loss` / `max_wins` / `max_losses` | 最大单笔盈亏、最大连胜 / 连亏 |
| `trade_count` | 交易次数，等于 `winning_trades + losing_trades` |
| `current_drawdown_pct` | 当前权益相对峰值的回撤百分比（`result.metrics` 里没有这一项） |
| `equity` | 当前权益 |

不提供 Sharpe、Sortino、Calmar、年化收益等指标：它们依赖完整收益序列和年化假设，日内读到的值含义不清。

`self.stats.as_rows(fields=None, labels=None)` 返回可直接传给 `draw.table(rows=...)` 的二维列表（名称、数值两列），默认字段是 `total_return_pct` / `max_drawdown_pct` / `win_rate` / `profit_factor` / `trade_count` / `total_pnl`，默认中文标签；`*_pct` 和 `win_rate` 的显示文本带 `%`，`profit_factor` 为无穷大时显示 `∞`。

!!! warning "实盘接真实柜台时的口径与限制"
    `run_live` 不接交易柜台（如只接 klinedata 行情的纸面撮合）时，`self.stats` 全部字段可用，与回测同口径。

    接真实柜台（`broker_live`，如 `trader_broker="middleware"`）时，引擎本地收不到柜台成交，`self.stats` 改由会话级统计器提供：柜台成交按 `trade_id` 去重后逐笔计入，统计算法与回测相同；每根 bar 在 `on_bar` 之前用最近一次查到的柜台账户权益记一个权益点（用于回撤，不为此额外查询柜台），`total_return_pct` 是本次会话的收益；浮动盈亏按策略已收到的最新收盘价计算；合约乘数取 `run_live(instruments=...)` 的配置，未配置的标的按 1 计算并告警一次。多 slot 时各 slot 读到的是同一份账户级统计。柜台权益只在成交或委托回报之后、策略读取账户（含读取 `self.stats`）时重新查询，两次之间的价格波动不会反映到回撤里。会话启动时查询柜台权益或持仓失败，统计器不可用，除 `equity` 外的字段为 `None`。另有以下限制：

    - **手续费为 0**：柜台成交不带手续费，`total_commission` 恒为 0。胜率和盈亏比按毛利判定，不受影响。
    - **只统计本次会话**：只计入本会话策略下的单，前端手动单、其他会话的单不计入。会话重启后从柜台持仓重新初始化，之前已平仓的交易找不回来。
    - **会话启动前已有的持仓也会计入**：启动时用柜台持仓的数量、方向和均价初始化，即使它最初是手动交易建的仓；之后平掉这部分仓位，会计入本会话的统计。柜台持仓没有均价时，这部分持仓无法初始化，平掉它的成交不计入（会告警）。**接 middleware（`trader_broker="middleware"`）时目前就是这种情况**：middleware 插件的持仓不带均价和方向，会话启动前的持仓一律不初始化，之后卖出这部分持仓的成交按超出持仓截断为 0，不计入统计。
    - **超额平仓截断**：平仓数量超过统计器持仓的部分直接截掉，不当成反向开仓。开平的判定方式：`run_live(instruments=...)` 中配置为期货、期权的标的按成交的 `position_effect` 判断（`auto` 时按持仓方向推断）；配置为股票、基金等其他类型的标的，卖出一律视为平仓；未配置的标的，`position_effect` 明确为开仓或平仓时照此判断，为 `auto` 时卖出视为平仓。
    - **按净持仓记账**：统计器对每个标的只记一个净持仓。锁仓（例如持有空头时再买入开多）会被记为平掉空头，而不是同时持有多空两笔；会话启动时柜台上同一标的的多空持仓也先合并为净持仓再初始化。

### 回测中查看

```python
result.drawings                 # list[dict]：回测结束时仍存活的对象
result.drawings_df()            # DataFrame；points / style / table 保留为嵌套结构
result.export_indicators("out.json")   # 导出文件里带 drawings 键（format="parquet" 时单独写 drawings.json）
result.viz.review(df, include_drawings=True)   # 默认开启
```

- `viz.review` 把线、框、折线、斐波那契、标注画在主图 K 线上；表格是 HTML 浮层，只叠在主图 pane 上，按 9 个位置摆放，切换标的时显示该标的的表加全局表。顶部工具栏有“隐藏绘图”按钮。
- 时间不在数据范围内的部分由渲染端裁剪。
- `merge_results` 会把各段结果的 `drawings` 直接拼接，不去重：同一个策略跨多个滚动窗口时，同一个 id 可能重复出现。

### 实盘流消息（schema 1.7）

实盘时，绘图调用经 sink 发出 `indicator_drawing` 事件，用新增的两个函数转换：

```python
if aq.is_drawing_stream_event(event):
    message = aq.to_drawing_message(event)
    # {"channel": "indicator", "type": "drawing", "schema_version": "1.7",
    #  "op": "upsert" | "delete", "drawing": {...}, "symbol": ..., ...}
```

- `op="delete"` 时 `drawing` 只含身份字段 `owner_strategy_id` / `drawing_id` / `symbol` / `kind`；身份键是 `(owner_strategy_id, symbol, drawing_id)`，后到的 upsert 整体替换先到的。全局表的 `drawing.symbol` 为 `"*"`。
- **旧入口不会收到 drawing 消息**：`is_indicator_stream_event` 和 `to_indicator_message` 保持原样，只认 point 和 snapshot，这样只认这两种消息的旧版下游不会把 drawing 原样转发给前端。
- `schema_version` 小于 1.7 的一方不会产生 drawing 消息；消费方对未知 `type` 忽略即可。
- sink 没有 `record_drawing` 方法时，绘图调用静默丢弃，首次丢弃记一条 `info` 日志。
- `run_live(preload_history=)` 回放期间的绘图照常生效（`phase="preload"`），预加载期创建、运行期仍存活的对象就是同一个对象。
