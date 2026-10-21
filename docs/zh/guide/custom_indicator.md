# 自定义指标指南

本页聚焦一个问题：当 AKQuant 内置指标不够用时，如何安全地编写、注册并维护你自己的指标。

适用场景：

- 私有因子或策略专用信号；
- 需要在 pandas 上快速验证的原型指标；
- 需要在事件流里逐 Bar 更新状态的增量指标；
- 需要配合热启动一起恢复的状态型指标。

## 先做判断

在 AKQuant 中，常见有三种“指标扩展”需求，它们不是同一件事：

| 需求 | 推荐路径 | 典型用法 |
| :--- | :--- | :--- |
| 逐 Bar / 逐 Tick 维护状态 | 增量指标对象（有 `update()`） | `self.I(aq.EMA(20), ...)` |
| 用 pandas 一次性计算整段历史 | `PrecomputedIndicator(name, fn)` | `self.I(PrecomputedIndicator(...), ...)` |
| 给 `akquant.talib` 增加一个新函数名 | 修改 Python/Rust 兼容层源码 | 不属于运行时动态注册 |

如果你的目标只是“在策略里用一个自己的指标”，通常不需要修改 `akquant.talib`。

## 统一入口：`self.I()`

指标只有一个声明入口 `Strategy.I()`。声明一次，框架负责三件事：**每根 bar 自动推进**、
**支持 `ind[0]` / `ind[1]` 序列回溯**、**按需自动上报绘图点**。这对标 TradingView
Pine Script 的 `ta.*` + `plot()`。

```python
import akquant as aq
from akquant import Bar, Strategy


class MaCross(Strategy):
    def on_start(self) -> None:
        # 声明一次 = 自动更新 + 可回溯；写 plot=True 再加上自动绘图
        self.ema_fast = self.I(aq.EMA(10), pane=0, color="#e91e63", label="EMA10", plot=True)
        self.ema_slow = self.I(aq.EMA(30), pane=0, color="#3f51b5", plot=True)
        self.rsi = self.I(aq.RSI(14), pane=1, plot=True)

    def on_bar(self, bar: Bar) -> None:
        # [0] 是当前 bar，[1] 是上一根 —— 与 Pine 的 sma[0] / sma[1] 一致
        if self.ema_fast[1] is None or self.ema_slow[1] is None:
            return
        if self.ema_fast[1] <= self.ema_slow[1] and self.ema_fast[0] > self.ema_slow[0]:
            self.buy(bar.symbol, 100)
        # 无需再写 record_indicator —— 写了 plot=True 就会自动上报
```

!!! warning "0.3 起的破坏性变更"
    旧的 `indicator_mode` 开关与 `register_incremental_indicator(...)` /
    `register_precomputed_indicator(...)` 已**移除**，统一由 `self.I()` 按传入对象
    自动分派。副产品是两类指标现在可以**共存于同一个策略**——旧的模式开关是互斥的。

    | 旧写法 | 新写法 |
    | :--- | :--- |
    | `self.indicator_mode = "incremental"` + `register_incremental_indicator("x", obj, source="close")` | `self.x = self.I(obj, name="x", source="close")` |
    | `self.indicator_mode = "precompute"` + `register_precomputed_indicator("x", ind)` | `self.x = self.I(ind, name="x")` |
    | `register_incremental_indicator(..., indicator_factory=f)` | `self.I(factory=f, ...)` |

### 在哪里声明

`__init__` 与 `on_start` 都可以，但**推荐 `on_start`**：`self.params` 在 `__init__`
之后才注入，参数化指标（`run_grid_search` 要扫的那些）只有在 `on_start` 里才拿得到
参数值。`__init__` 只适合字面量参数。

`subscribe_bars` 是例外，它**只能**在 `__init__` 调用（引擎启动前要冻结订阅表）；
`self.I(freq="5min")` 会校验该周期已被订阅。

### 序列回溯

- `ind[0]`：当前 bar 的值（`.value` 是它的别名）
- `ind[1]`：上一根；`ind[n]`：前 n 根
- **越界返回 `None`，不抛异常**——预热期天然越界，抛错会逼每个策略都写 `try`
- 回溯深度由 `lookback` 界定（默认 128），底层是有界环形缓冲：实盘是无限流，
  无界缓冲必然泄漏

未就绪时取值为 `None`（原生指标未满窗、预计算指标 asof 落空都归一成 `None`），
因此判空统一写 `if value is None`，不必再区分 `NaN`。

## 路径一：增量指标

传入任何有 `update()` 的对象即走增量路径。AKQuant 内置了 100+ 个 Rust 实现的
有状态指标类（`aq.SMA` / `aq.EMA` / `aq.RSI` / `aq.MACD` / `aq.ATR` / `aq.BollingerBands` 等），
它们就是 Pine `ta.*` 的等价物，开箱即用。

### 自定义增量指标

```python
from collections import deque

import pandas as pd
from akquant import PrecomputedIndicator, Strategy


class MyMomentum(PrecomputedIndicator):
    def __init__(self, period: int = 10):
        super().__init__("my_momentum", lambda df: df["close"] - df["close"].shift(period))
        self.period = period
        self.buffer: deque[float] = deque(maxlen=period)
        self._current_value = float("nan")

    def update(self, value: float) -> float:
        if pd.isna(value):
            return self._current_value
        self.buffer.append(float(value))
        if len(self.buffer) < self.period:
            self._current_value = float("nan")
        else:
            self._current_value = self.buffer[-1] - self.buffer[0]
        return self._current_value

    @property
    def value(self) -> float:
        return self._current_value


class IncrementalMomentumStrategy(Strategy):
    def on_start(self):
        self.mom10 = self.I(
            factory=lambda: MyMomentum(period=10),
            name="mom10",
            source="close",
            symbols=["AAPL", "MSFT"],
            warmup_bars=10,
        )

    def on_bar(self, bar):
        value = self.mom10[0]
        if value is not None and value > 0:
            self.buy(bar.symbol, 100)
```

### 为什么推荐 `factory`

多标的策略里，增量指标通常都有内部状态。多个 `symbol` 共用同一个实例会串线，
因此传实例（而非 `factory`）时跨多标的使用会直接报错，不会静默共享状态。

推荐写法：

```python
self.mom10 = self.I(factory=lambda: MyMomentum(period=10), source="close")
```

而不是：

```python
self.mom10 = self.I(MyMomentum(period=10), source="close")  # 仅适合单标的
```

!!! warning "热启动下 `factory` 必须可 pickle"
    快照会序列化策略实例，`on_start` 里的 lambda 是局部对象，pickle 不了。
    要配合 `run_from_checkpoint` 时，请改用模块级函数 + `functools.partial`：

    ```python
    from functools import partial

    def _make_sma(window: int):
        return aq.SMA(window)

    # on_start 里：
    self.sma = self.I(factory=partial(_make_sma, 20), name="sma")
    ```

### `source` 代表什么

`source` 指定框架从行情对象里拿哪个字段喂给指标。最常见的是：

- `source="close"`
- `source="open"`
- `source="high"`
- `source="low"`
- `source="volume"`

需要多输入的指标（ATR 之类）用 `input_mode` 指定喂入形态：`"source"`（单值，默认）/
`"hl"` / `"hlc"` / `"ohlc"` / `"close_volume"`，框架会按该顺序把字段传给你的 `update(...)`。

### 多值指标：`outputs`

MACD 这类一次产出多个分量的指标，用 `outputs` 声明分量名，它们会被拆成多条
独立的线（`indicator_key` 形如 `macd.dif`），而不是把元组丢给前端去拆：

```python
self.macd = self.I(aq.MACD(12, 26, 9), name="macd", pane=2,
                   outputs=("dif", "dea", "hist"), plot=True)

# 读取：
self.macd[0]        # 整个元组 (dif, dea, hist)
self.macd.dif[0]    # 单个分量，同样支持 [1] 回溯
```

`outputs` 个数与指标实际分量数不符时会直接报错，不会静默错位。

### K 线形态指标：`render_type="candle"`

平均 K 线（Heikin-Ashi）这类指标一次产出一整根 K 线。对应 Pine 的 `plotcandle`，
AKQuant 的写法是四个分量的 `outputs` 加上 `render_type="candle"`：

```python
class HA(aq.Indicator, title="平均K线", overlay=True, replaces_price=True):
    def on_start(self):
        self.ha = self.I(aq.HeikinAshi(), name="ha", input_mode="ohlc",
                         outputs=("open", "high", "low", "close"),
                         render_type="candle", label="Heikin-Ashi")
```

- 契约不变：四个分量仍然各自上报单值点，`indicator_key` 为 `ha.open` / `ha.high` /
  `ha.low` / `ha.close`。定义行和流消息多了 `group`（组名，即 `self.I` 的 `name`）与
  `role`（`open`/`high`/`low`/`close`），非 candle 指标这两个字段为空串。
- 校验：`outputs` 必须恰好是这四个名字（顺序不限），否则声明时报 `ValueError`；四个分量
  共用 `pane`、`color`、`label`。`record_indicator` 不能直接写 `render_type="candle"`。
- `aq.HeikinAshi()` 是内置 Rust 指标：`update(open, high, low, close)` 返回
  `(ha_open, ha_high, ha_low, ha_close)`，第一根就有值。
- 类定义行的 `replaces_price=True` 只是给前端的提示（“这个指标替代价格 K 线”），只能写在
  `aq.Indicator` 上且必须同时 `overlay=True`；隐藏原 K 线由前端决定，框架不处理。
  `X.resolve_replaces_price()` 与 `aq.inspect_strategy` 的 `replaces_price` 字段可读出它。

各出口的画法：

| 出口 | 画法 |
| :--- | :--- |
| LWC 复盘（`result.viz.review`、`to_lwc_indicator_series`） | 每个 `(owner, group)` 一条 Candlestick，点位为 `{time, open, high, low, close}` |
| plotly（`plot_indicators`、报告指标区块） | 每组一个 `go.Candlestick` |
| d3Kline（`to_d3kline_options`） | **降级**：d3Kline 没有 K 线型指标，只输出 close 分量的一条 line，`name` 为组的 series 键，并带 `render_type="candle"` 标明降级 |
| `to_raw_panes` | 四个分量各占一个条目，带 `group` / `role` / `group_series_key`，可自行拼回 OHLC |
| 实时流 / `to_lwc_update` | 每个分量一条点，带 `group` / `role`；按 `(owner_strategy_id, group, symbol, timestamp)` 收齐四个分量即得一根 K 线 |

LWC 与 plotly 给 candle 指标配色时**不用**主题的涨跌色（否则与价格 K 线分不清）：只用一种颜色（`color`，未写时取指标色板），上涨为实心、下跌为空心（TradingView 空心 K 线的惯例）。某个时刻缺任一分量时，LWC 与 plotly 跳过该时刻，不会报错。组的 series 键见下文
「指标身份与 series 键」。

可运行示例：[77_heikin_ashi_indicator.py](https://github.com/akfamily/akquant/blob/main/examples/77_heikin_ashi_indicator.py)。

## 路径二：预计算指标

当你的指标更适合一次性对完整 `DataFrame` 计算时，传一个 `PrecomputedIndicator(name, fn)`
实例即可，`self.I()` 会自动走向量化预计算路径。

### 最小示例

```python
from akquant import PrecomputedIndicator, Strategy


class PrecomputeMomentumStrategy(Strategy):
    def on_start(self):
        self.mom10 = self.I(
            PrecomputedIndicator("mom10", lambda df: df["close"] - df["close"].shift(10)),
            name="mom10",
        )

    def on_bar(self, bar):
        value = self.mom10[0]
        if value is not None and value > 0:
            self.buy(bar.symbol, 100)
```

### 何时适合

- 指标天然可向量化；
- 主要依赖 pandas `rolling` / `shift` / `ewm`；
- 更关心回测开发效率，而不是在线增量更新；
- 同一个 `symbol` 的整段历史可以提前准备好。

### 限制

- **实盘用不了**：实盘没有完整的历史 `DataFrame`；
- **含 Tick 的输入会报错**：预计算需要完整 OHLC，tick 只有成交价；
- **不支持 `freq=`**：窗口周期请改用增量指标。
- **不支持 `warmup_bars=`**：它没有增量状态可预热，整段本来就算好了；
- **多标的天然可用**：`PrecomputedIndicator` 自己按 symbol 缓存整段结果，不需要 `factory=`。

## 自动绘图

要画的指标，框架会在每次更新后自动上报一个绘图点，**无需在 `on_bar` 里手写
`record_indicator`**。画不画只由两件事决定：**`self.I` 写在哪类脚本里**，以及有没有显式写
`plot=`：

- 在 `aq.Strategy` 里**默认不画**——与 Pine 一致（`ta.sma()` 只计算，`plot()` 才画）。这不只是
  风格：实盘的指标 sink 是每点一个事件，多标的 × 多指标默认全开会淹没前端链路。
- 在指标脚本 `aq.Indicator` 里**默认画**——指标脚本本来就是为了画图（见下文“指标脚本”一节）。
- `plot=True` / `plot=False` 显式覆盖默认值。
- `pane` / `color` / `label` / `render_type` / `reference_lines` / `scale_group` 只决定“怎么画”，
  **不再隐含 `plot=True`**。策略里只写了 `pane=` 想画图的，需补 `plot=True`。

| 写在哪里 | 写法 | 画不画 | pane |
| :--- | :--- | :--- | :--- |
| `aq.Strategy` | `self.I(x)` | 不画 | 0 |
| `aq.Strategy` | `self.I(x, color="red")` | 不画 | 0 |
| `aq.Strategy` | `self.I(x, pane=1)` | 不画 | 1 |
| `aq.Strategy` | `self.I(x, plot=True)` | 画 | 0 |
| `aq.Indicator`（`overlay=False`，默认） | `self.I(x)` | 画 | 1 |
| `aq.Indicator`（`overlay=False`，默认） | `self.I(x, pane=0)` | 画 | 0 |
| `aq.Indicator`（`overlay=False`，默认） | `self.I(x, plot=False)` | 不画 | 1 |
| `aq.Indicator`（`overlay=True`） | `self.I(x)` | 画 | 0 |
| `aq.Indicator`（`overlay=True`） | `self.I(x, pane=2)` | 画 | 2 |

未就绪的点（值为 `None`）不上报，对应 Pine 预热期的 `na`。

```python
# 只算不画
self.atr = self.I(aq.ATR(14), input_mode="hlc")

# 算且画（策略里必须显式写 plot=True）
self.rsi = self.I(aq.RSI(14), pane=1, plot=True, reference_lines=[
    {"value": 70, "label": "超买"},
    {"value": 30, "label": "超卖"},
])

# 只写 pane= 不会画——绘图参数不再隐含 plot=True
self.sma = self.I(aq.SMA(20), pane=0)
```

绘图参数的含义与 `record_indicator` 完全一致，详见下文「导出指标给前端」。
`record_indicator` 保留为底层逃生口，用于记录**非指标类**的自定义值（仓位、
信号强度、风控中间量等），那些是 `self.I()` 覆盖不了的。

## 实时值：未闭合窗口上的 intrabar

TradingView 的 realtime bar 会随每笔成交刷新指标，收盘后才确认（`barstate.isrealtime` /
`isconfirmed`）。AKQuant 对窗口周期指标提供同样的语义：声明 `intrabar=True` 后，每根基础
bar 闭合时，框架用 `ctx.current_window()` 的**未闭合窗口快照**在指标**副本**上试算一个临时值，
真实状态不受污染；窗口闭合时才真正 `update()`。

```python
class S(Strategy):
    def __init__(self):
        super().__init__()
        self.subscribe_bars("5min")

    def on_start(self):
        self.sma5 = self.I(aq.SMA(20), freq="5min", intrabar=True, pane=0, plot=True)

    def on_bar(self, bar):                 # 每根 1min bar
        v0 = self.sma5[0]                  # 有临时值时: 未闭合 5min 窗口上的试算值
        v1 = self.sma5[1]                  # 上一根已确认的 5min 值
        if not self.sma5.confirmed: ...    # 当前 [0] 是临时值
```

下标语义与 Pine 完全一致：有临时值时它占据 `[0]`，已确认序列整体后移一位。流事件里临时点带
`confirmed=false`，且**时间戳就是该窗口将来闭合的标签**，确认点随后以同一 `time` 到达——前端按
`(owner_strategy_id, indicator_key, symbol, timestamp)` 覆盖即可（`akquant.lwc.to_lwc_update()` 已按此处理）。
临时点**不进** `indicator_df()` / `export_indicators()`，那两个出口只含确认值。

三条限制：

- **基础周期指标（不给 `freq=`）在每笔 tick 上试算**：框架从 tick 自己累出形成中的 bar
  （真实的 open/high/low/close，ATR 之类 H/L 指标也能试算），tick **不再 `update()`** 该指标，
  只有 bar 闭合才推进状态。临时点的时间戳按聚合器的区间末打戳公式预测
  （`(ts // 间隔 + 1) * 间隔 − 1ns`），bar 真闭合时若与实际不符（行情源按起点打戳），该标的
  停发临时点并告警一次，`[0]` 仍可读。**默认 `intrabar=False` 时 tick 照旧 `update()` 基础
  指标**——纯 tick 策略依赖这个行为，未改动。
- **临时点只在基础周期已知时上报**（回测 `run_backtest(data=[Tick,...], freq=)`，实盘网关声明
  `metadata["freq"]`）。基础周期未知时窗口标签会随每根基础 bar 漂移，前端无法覆盖，此时只算
  `[0]` 不发流事件，告警一次。
- 试算靠复制指标状态：内建 Rust 指标走 `akquant.clone_indicator()`（pyo3 不为
  `#[derive(Clone)]` 暴露 `__copy__`，`copy` / `deepcopy` / `pickle` 对它们一律 TypeError），
  用户自写的 Python 指标走 `copy.deepcopy`。

可运行示例：[75_intrabar_indicators.py](https://github.com/akfamily/akquant/blob/main/examples/75_intrabar_indicators.py)（窗口周期）、
[76_tick_intrabar_indicators.py](https://github.com/akfamily/akquant/blob/main/examples/76_tick_intrabar_indicators.py)（基础周期，tick 驱动）。

## 指标脚本 `aq.Indicator`：只画图不交易

TradingView 里 `indicator()` 脚本与 `strategy()` 脚本是两类东西，一张图上可以叠 N 个只画图的
指标脚本。AKQuant 的对应物是 `aq.Indicator`（对标 Pine v5 的 `indicator()`）：继承它、在
`on_start` 里用 `self.I()` 声明指标，然后用 `indicators=[...]` 挂到任意一次回测或实盘上。

脚本元数据写在类定义行上，与 Pine 的 `indicator(title=..., overlay=...)` 对应：

| 参数 | 含义 | 省略时 |
| :--- | :--- | :--- |
| `title` | 显示名，只用于展示 | 类名 |
| `overlay` | `True` 画主图（叠在 K 线上），`False` 画本脚本的附图 | `False`（附图） |
| `id` | 机器标识，即指标点的 `owner_strategy_id` / slot key | 类名 snake_case |
| `replaces_price` | `True` 表示本脚本替代价格 K 线（如平均 K 线），只是给前端的提示；必须同时 `overlay=True` | `False` |

```python
import akquant as aq
from akquant import run_backtest


class DualMA(aq.Indicator, title="双均线", overlay=True, id="dual_ma"):
    def on_start(self):
        self.fast = self.I(aq.EMA(10), name="ema_fast")    # overlay=True：默认画主图 pane=0
        self.slow = self.I(aq.EMA(30), name="ema_slow")


class RsiIndicator(aq.Indicator, title="RSI"):            # id 省略为 "rsi_indicator"
    def on_start(self):
        self.rsi = self.I(aq.RSI(14), name="rsi")          # overlay=False：默认画附图 pane=1


class MacdIndicator(aq.Indicator, title="MACD", id="macd_view"):
    def on_start(self):
        self.macd = self.I(aq.MACD(12, 26, 9), pane=2, outputs=("dif", "dea", "hist"))


result = run_backtest(strategy=MyStrategy, data=data, symbols=["600000"],
                      strategy_id="main",
                      indicators=[DualMA, RsiIndicator, MacdIndicator])
result.indicator_df(owner="rsi_indicator")   # 只看这个指标脚本的点
result.viz.review(data)                      # 指标脚本的指标一样画到 LWC 图上
```

要点：

- **pane 编号只在脚本内有效**：`overlay=True` 时 `self.I` 默认 `pane=0`，否则默认 `pane=1`；
  需要多个附图时显式写 `pane=2`、`pane=3`……。显式写的 `pane=` 总是优先。
- **不新建管线**：每个指标脚本就是多策略拓扑里的一个 slot（`strategies_by_slot`），指标点带
  自己的 `owner_strategy_id`，窗口订阅跨 slot 合并，`run_live(indicators=...)` 同一套语义。
- **不交易是硬保证**：`aq.Indicator` 上所有交易 API（`buy` / `sell` / `place_*` /
  `order_target*` / `rebalance_*` / `cancel_*` ……）一律抛 `aq.IndicatorCannotTradeError`，
  在引擎回调里误写也会当场报错，不会静默成交。要下单请继承 `aq.Strategy`。
- **id 不许撞**：脚本 id 与 `strategy_id` 或已有 slot key 重复时直接 `ValueError`；
  `indicators=` 里塞进非 `aq.Indicator` 的类会 `TypeError`——否则“不交易”的保证就没了。
- **`aq.Strategy` 也能写 `title=` / `id=`**：`class Trend(aq.Strategy, title="趋势", id="trend")`。
  策略没有主图/附图之分，写 `overlay=` 会报 `TypeError`；类上的 `id` 与 `run_backtest/run_live`
  的 `strategy_id=` 不一致时报 `ValueError`。不写 `id` 的策略，`strategy_id` 仍默认 `"_default"`。

可运行示例：[74_indicator_scripts.py](https://github.com/akfamily/akquant/blob/main/examples/74_indicator_scripts.py)。

## 静态检查：`aq.inspect_strategy`

平台在列出用户上传的脚本时，往往需要先知道“这个文件里是指标还是策略、画主图还是附图、
有没有要画的线”，用来分类展示、把不适用的操作灰显——但又不能为此去执行用户代码。
`aq.inspect_strategy(path_or_source)` 只用 `ast` 解析源码，**不导入、不执行**，返回
`aq.InspectResult`（`to_dict()` 可直接序列化为 JSON）：

```python
report = aq.inspect_strategy("strategies/dual_ma.py")   # 也可以直接传源码字符串
report.to_dict()
```

```python
{
  "classes": [
    {
      "name": "DualMA",
      "kind": "indicator",          # "indicator" | "strategy" | "unknown"
      "title": "双均线",
      "id": "dual_ma",
      "overlay": True,              # kind != "indicator" 时为 None
      "draws_indicators": True,     # 至少有一项 plot=True，或者 plot 是动态的
      "indicators": [
        {"name": "ema_fast", "pane": 0, "plot": True,
         "outputs": None, "label": "EMA10", "dynamic": []},
      ],
      "has_dynamic": False
    }
  ],
  "error": None                     # 语法错误时填 "SyntaxError: ... (line N)"，classes 为空
}
```

使用约定：

- **只接受明文**：传路径时按 UTF-8 读取（带 BOM 也可以），加密的策略文件需由调用方先解密。
  以 `.py` 结尾但不存在的路径报 `FileNotFoundError`；其余问题都写进 `error` / `dynamic`，不抛异常。
- **`dynamic` 表示“静态看不出来”**：值来自参数（如 `pane=self.params["pane"]`）、写在循环或推导式里、
  用了 `factory=` 或 `**kwargs`、出现在条件分支或辅助方法里、调用了无法识别的可调用对象时，
  对应字段取 `None`，字段名记进 `dynamic`（循环与 `factory=` 记为 `"count"`，表示条数不确定），
  类上的 `has_dynamic` 为 `True`。平台据此提示“需运行后才能确定”，不要自行猜测。
- **`kind="unknown"`**：基类来自其他文件、无法在本文件内追到 `Indicator` / `Strategy`。
- **`akquant` 与 `qmfquant` 的导入都识别**：`import akquant as aq`、`from akquant import Indicator`
  （含 `as` 别名），以及平台重新打包的 `qmfquant` 同名写法。
- 类上另有 `replaces_price`：策略为 `None`，指标默认 `False`，值无法静态确定时为 `None` 并记入 `dynamic`。
- `pane` / `plot` 的解析规则与运行时完全一致（见上文“自动绘图”的规则表），两边共用同一个函数。

## `warmup_bars` 怎么用

`warmup_bars` 用于在正式事件流开始前，先使用 `start_time` 之前的历史 Bar 预热指标。

适合以下场景：

- 你希望第一根有效 Bar 就拿到完整指标值；
- 指标依赖窗口历史，如 `period=20`；
- 你不想在 `on_bar` 里手工跳过前 N 根。

推荐示例可参考：

- [58_incremental_bootstrap_demo.py](https://github.com/akfamily/akquant/blob/main/examples/58_incremental_bootstrap_demo.py)
- [60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py)
- [72_declarative_indicators.py](https://github.com/akfamily/akquant/blob/main/examples/72_declarative_indicators.py)

## 热启动与序列化

如果你的策略会使用 `run_from_checkpoint`，自定义指标需要考虑状态持久化。

原则如下：

- 纯 Python 简单对象通常可直接 `pickle`；
- 如果指标里持有文件句柄、网络连接、线程锁等对象，需自行处理；
- 必要时实现 `__getstate__` 和 `__setstate__`，只保存必要状态。

例如：

```python
def __getstate__(self):
    state = self.__dict__.copy()
    return state


def __setstate__(self, state):
    self.__dict__.update(state)
```

更多背景见：[热启动指南](../advanced/warm_start.md)。

## 与 `akquant.talib` 的边界

很多用户会把“自定义策略指标”和“扩展 `akquant.talib`”混在一起。建议按下面理解：

- `akquant.talib`：内置 TA-Lib 风格兼容层，主要服务于已有函数式指标调用；
- 自定义策略指标：服务于你的具体策略，可直接注册到 `Strategy`；
- 新增 Rust 高性能指标：需要改源码、重新编译，不是运行时热插拔。

如果你只是要一个策略内的私有信号，优先写自定义指标，而不是去扩展 `akquant.talib`。

## 导出指标给前端

如果你的目标不只是“在策略里使用指标”，而是要把指标结果进一步交给 Web 前端展示，建议把“计算指标”和“输出指标”分开处理：

- 指标计算仍然放在 `Strategy` / `Indicator` 里；
- 指标输出使用 `Strategy.record_indicator(...)` 记录标准化点位；
- 回测结束后，通过 `BacktestResult.indicator_df(...)` 或 `export_indicators(...)` 交给外部服务或前端。

### 最小示例

```python
from akquant import Bar, Strategy


class IndicatorExportStrategy(Strategy):
    def on_bar(self, bar: Bar) -> None:
        spread = bar.high - bar.low
        self.record_indicator(
            name="intrabar_spread",
            value=spread,
            display_name="Intra Bar Spread",
            pane=1,
            render_type="line",
            precision=4,
            meta={"source": ["high", "low"]},
        )
```

!!! note "关于 `symbol` 归属"
    每个指标点位都**必然归属于某一个标的**，`symbol` 决定这条指标画在哪只标的的图上。

    省略 `symbol` 时（推荐），自动取当前正在处理的 bar / tick 的标的。多标的回测下，
    同一句 `record_indicator` 会在每个标的下各自独立记录一条序列——这正是你想要的：
    每只标的都有自己的 MA5。

    ```python
    def on_bar(self, bar: Bar) -> None:
        # 省略 symbol：自动归属 bar.symbol
        # 多标的回测下，每个标的各得一条独立的 ma5 序列
        self.record_indicator(name="ma5", value=self.ma5.value, pane=0)
    ```

    显式传 `symbol` 只在"把指标记到另一个标的名下"时才需要（例如给基准或指数标的
    记录一条参考线）。注意此时**每个标的的 bar 都会触发一次记录**，若不加条件判断，
    同一时间戳会被重复写入多条：

    ```python
    def on_bar(self, bar: Bar) -> None:
        # 只在主标的的 bar 上记录一次，避免其它标的的 bar 重复触发
        if bar.symbol == "000300.SH":
            self.record_indicator(name="bench_ma", value=v, symbol="000300.SH")
    ```

    既拿不到当前 bar / tick、也没有显式传入时会抛 `ValueError`，不会静默落到某个
    占位标的上。消费端可依赖"`symbol` 一定非空"这个前提做分组。

!!! note "关于 `pane` 取值"
    `pane` 是**整数行索引**：`0` 表示主图（价格图），`1`..`N` 表示主图下方堆叠的副图。
    省略 `pane` 默认落在主图（`0`）。默认上限 `N` 为 `8`——这是一个基于屏幕可读性的
    **软上限**，而非硬性限制。多因子或衍生品场景若需要更多副图，可在运行前设置环境变量
    `AKQUANT_MAX_SUB_PANES` 抬高上限。取值超出 `0..N` 会直接报错（fail-fast），
    确保写错的索引不会被静默塞进错误的窗格。这与图表渲染器实际消费的形态一致，
    `record_indicator` 在纯回测导出与前端流式桥接两条路径上输出相同的整数 `pane`。

!!! warning "0.3 起的破坏性变更"
    早期版本 `pane` 曾支持 `"main"` / `"sub1"` / `"主图"` / `"signal"` 等字符串写法，现已移除。
    请改用整数索引；原先画交易信号用的 `pane="signal"` 语义，改为用 `render_type="signal"`
    在主图（`pane=0`）上渲染标记。

!!! note "关于 `render_type` 取值"
    `render_type` 是一个**封闭枚举**，共 8 个值，消费者可据此穷举渲染分支：

    | 值 | 渲染 |
    | :--- | :--- |
    | `line` | 折线（默认） |
    | `area` | 折线 + 向零填充 |
    | `bar` | 垂直柱 |
    | `column` | `bar` 的别名（语义化：分类柱） |
    | `histogram` | `bar` 的别名（语义化：分布柱） |
    | `scatter` | 离散点标记 |
    | `signal` | 交易信号标记，渲染在主图 |
    | `candle` | K 线（四个分量作一组，只能经 `self.I(outputs=...)` 声明，见上文） |

    传入枚举以外的值会直接抛 `ValueError`（fail-fast），不会静默退化成折线。

!!! note "关于 `reference_lines` 与 `scale_group`"
    - `reference_lines`：可选，静态参考线列表，每项 `{"value": 数值, "label": 文字, "color": 颜色}`；
      用于超买/超卖线、0 轴等固定横线。会动的线请用独立指标逐 bar 记录。
    - `scale_group`：可选，共享刻度分组的语义组名（如 `"percent"`），纯提示，前端据此判断
      同量纲指标；不改变 `pane` 决定的行布局。

带参考线与刻度分组的 RSI 示例：

```python
self.record_indicator(
    name="rsi",
    value=rsi_value,
    pane=1,
    reference_lines=[
        {"value": 70, "label": "超买", "color": "#ef4444"},
        {"value": 30, "label": "超卖", "color": "#22c55e"},
    ],
    scale_group="percent",
)
```

运行结束后：

```python
result = ...

# 1) 在 Python 里直接读取
indicator_df = result.indicator_df(name="intrabar_spread", symbol="AAPL")

# 2) 在本地做一个轻量预览
fig = result.viz.indicators(
    name="intrabar_spread",
    symbol="AAPL",
    show=False,
    filename="indicator_preview.html",
)

# 3) 导出给前端或外部服务
result.export_indicators("indicator_outputs.json", format="json")
result.export_indicators("indicator_outputs", format="parquet")
```

其中 JSON 导出在可用时会额外带上顶层 `run_id`，方便外部服务把离线导出与流式事件链路关联起来。

### 内置最小可视化

如果你只是想快速确认指标历史形态，而不是立刻接入完整前端，可以直接使用：

- `result.viz.indicators(...)`
- `from akquant.plot import plot_indicators`

这条内置路径的定位是“轻量 history preview”，特点是：

- 保持现有 `result.viz.dashboard()` 继续只做账户 dashboard；
- 按 `pane` 自动拆分子图；
- 复用 `render_type`，第一版支持常见的 `line` / `bar`；
- 支持 `name`、`symbol`、`include_warmup` 过滤；
- 可直接输出为本地 HTML，方便和导出的 JSON 一起联调。

例如：

```python
fig = result.viz.indicators(
    name="intrabar_spread",
    symbol="AAPL",
    include_warmup=False,
    show=False,
    filename="indicator_preview.html",
    title="Indicator Preview",
)
```

如果你需要的是企业级多图联动、权限、持久化和实时订阅，这些仍建议放在外部平台实现；AKQuant 内部只提供最小预览和标准化数据输出。

#### 在 LWC 复盘图上看指标

`result.viz.review()` 会把 `self.I()` / `record_indicator` 上报的指标**默认一并画出**
（`include_indicators=True`）：`pane=0` 叠在主图 K 线上，`pane=k≥1` 落到成交量之下的
第 `k+1` 个副图；`reference_lines` 成为虚线参考线；未指定 `color` 时按主题色板轮转。
`render_type` 的 8 个值全部有对应：`line`/`area`/`bar`/`column`/`histogram`/`scatter`
各画各的，`signal` 走 K 线上的标记而不建独立序列，`candle` 的四个分量合成一条 Candlestick。

```python
result.viz.review(market_data, filename="review.html")            # 带指标
result.viz.review(market_data, filename="review.html", include_indicators=False)
```

底层适配器是 `akquant.lwc.to_lwc_indicator_series()`，与 `akquant.chart.to_d3kline_options()`
是同一份指标契约面向两个前端的镜像实现；自建 LWC 页面可以直接复用。每条 series 带
`owner_strategy_id` 与 `series_key`（规则见下文「指标身份与 series 键」），`indicator_key`
保持原始值。

### 报告中的可选指标区块

如果你希望把指标预览放进内置 HTML 报告，而不是单独输出一个图，也可以显式开启：

```python
result.viz.report(
    filename="akquant_report.html",
    show=False,
    include_indicators=True,
    indicator_name="intrabar_spread",
    indicator_symbol="AAPL",
    indicator_include_warmup=False,
)
```

这条路径有几个约束：

- 默认关闭，不会改变现有 `report()` 的输出；
- 适合把“一个轻量指标区块”嵌进策略报告；
- 如果没有指标数据，会在报告里显示空状态提示；
- 如果你需要复杂交互布局，仍建议交给外部前端实现。

### 流式桥接到前端消息

如果你的外部服务是 WebSocket / SSE 网关，推荐不要把原始 `payload` 解析逻辑散落在业务代码里，可以直接使用：

- `akquant.is_indicator_stream_event(event)`
- `akquant.to_indicator_message(event)`
- `akquant.to_indicator_messages(events)`

例如：

```python
def on_event(event):
    if not aq.is_indicator_stream_event(event):
        return
    message = aq.to_indicator_message(event)
    if message is not None:
        websocket.broadcast_json(message)
```

这条 helper 的目标是：

- 只桥接 `indicator_point` / `indicator_snapshot`
- 把数值字段转成更适合前端消费的类型
- 自动解开 `meta_json` / `items_json`
- 保留 `run_id`、`seq`、`ts` 等外层流式语义
- 同时给出 `timestamp`（纳秒）与 `timestamp_ms`（毫秒）两个时间字段

当前 `snapshot` 桥接结果除了 `items` 之外，还会补充几组快捷字段，方便前端减少二次遍历：

- `indicator_keys`
- `series_keys`（与 `indicator_keys` 一一对应）
- `panes`
- `render_types`
- `value_by_key`（按 `series_key` 作键）
- `items_by_key`（按 `series_key` 作键）
- `warmup_count`
- `has_warmup`

`point` 消息的 `indicator` 带 `series_key`，`snapshot` 的每个 item 带 `owner_strategy_id`
与 `series_key`。`series_key` 与历史接口的规则相同（见下文「指标身份与 series 键」）：
单 owner 时就是 `indicator_key`，主策略与指标脚本同名时为 `"{owner}:{key}"`。前端按
`series_key` 建线即可，不必再自己拼接或先拉定义表建映射。

另外，bridge helper 也会把 `_unknown` / 空 `symbol` 规整为 `None`，并兼容已经预先解码成
`dict/list` 的 `meta_json` / `items_json` 值，便于网关层做二次封装。

`timestamp_ms` 在 `point` 与 `snapshot` 两种消息上都有，取值与同一次
`record_indicator` 在 `indicator_df()`、`export_indicators()` 两个出口上的值一致——
三个出口的字段口径统一，前端不必自己做纳秒到毫秒的换算。若事件来自早期版本或
第三方 `IndicatorSink`、payload 里只有纳秒 `timestamp`，桥接会自动换算补齐。

!!! tip "用 `schema_version` 协商可选字段"
    每条消息的信封里都有 `schema_version`（`MAJOR.MINOR`，见 `akquant.STREAM_SCHEMA_VERSION`）。
    向后兼容的新增字段会升 MINOR，前端据此判断某字段是否存在即可，不必做
    `try/except` 式的探测。`timestamp_ms` 自 `1.2` 起在指标流上提供；读不到
    该版本的旧消息时，回落到纳秒 `timestamp` 自行换算。`phase` 自 `1.4` 起提供
    （point、snapshot 及其每个 item 上都有），`"preload"` 表示来自
    `run_live(preload_history=)` 的回放，其余一律为 `"run"`；缺失时按 `"run"` 处理。
    `series_key`（point 与每个 snapshot item）及 item 上的 `owner_strategy_id`、snapshot 的
    `series_keys` 自 `1.5` 起提供，同时 `items_by_key` / `value_by_key` 改按 `series_key`
    作键；读到 `1.4` 的消息时桥接把 `series_key` 回落为 `indicator_key`、item owner 回落为
    snapshot 的 owner，单 owner 场景键名与此前一致。
    `group` 与 `role`（point 与每个 snapshot item）自 `1.6` 起提供：candle 指标的四个分量
    带组名和 open/high/low/close，其余指标为空串；读到更早的消息时按空串处理。

它不是新的传输层，只是把 AKQuant 的事件结构整理成更稳定的“前端消息对象”。

### 注入自定义指标采集器（IndicatorSink）

如果你想把指标点位直接接进自己的采集/转发逻辑（例如推进一个广播队列、写入时序库），
不必去 monkey-patch 策略的私有属性。`run_backtest` / `run_live` 都接受一个公开的
`indicator_recorder` 参数，它是一个实现 `akquant.IndicatorSink` 协议的对象：

```python
from akquant import IndicatorSink, run_backtest


class QueueSink:
    """把每个指标点位塞进自己的队列，而不在内存累积。"""

    def record(self, *, name, value, symbol, timestamp, owner_strategy_id, **kwargs):
        my_queue.put((name, symbol, timestamp, value))

    def build_payload(self):
        return {"definitions": [], "instances": [], "points": []}

    def flush_stream_snapshot(self):
        ...

    def set_stream_emitter(self, emitter):
        ...


run_backtest(..., indicator_recorder=QueueSink())
```

`IndicatorSink` 是"指标采集器"的公开扩展点（对标 Backtrader 的 Analyzer/Observer）：
- 内置的 `IndicatorRecorder` 天然满足该协议（回测默认用它，跑完 `build_payload` 累积导出）；
- 你可以传入自己的实现，把指标数据接进任意下游，无需触碰 AKQuant 内部。

### 实盘实时指标流

回测与实盘的指标流是**同构**的：`run_live` 同样接受 `on_event` 与 `indicator_recorder`，
产出与回测一致的 `indicator_point` / `indicator_snapshot` 事件，因此前端一套消费逻辑通吃。

```python
from akquant import run_live

run_live(
    strategy_cls=MyStrategy,
    instruments=instruments,
    broker="ctp",
    trading_mode="broker_live",
    on_event=on_event,   # 与 run_backtest 相同的事件回调
)
```

实盘是长跑进程，默认使用一个**只发不累积**的轻量流式 sink：它只发射 stream 事件、
不在内存里堆积历史点位，避免长时间运行导致内存无界增长。若只传 `on_event`（不传
`indicator_recorder`），`run_live` 会自动启用这个流式 sink。

### 实盘历史预热：历史段与实时段连续

图表通常既要画启动前的历史指标，又要接着画实时指标。不要用“先跑一段回测、再启动实盘”来拼：
两段之间可能漏 bar 或重复 bar，回测里的指标实例和实盘里的也不是同一个对象，EMA、MACD 这类
递归指标到了实盘要重新收敛，接缝处会跳变。改用 `run_live(preload_history=...)`，让同一个会话、
同一批指标实例先吃历史、再吃实时：

```python
from akquant import run_live

history = fetch_my_history()      # 截止到上一根已收盘 bar；框架不内置数据源

run_live(
    strategy_cls=MyStrategy,
    instruments=instruments,
    broker="klinedata",
    preload_history=history,      # DataFrame / {symbol: DataFrame} / list[Bar]，与 run_backtest(data=) 同形态
    on_event=on_event,
)
```

预热做什么：

- 灌入历史缓冲：首根实盘 bar 里 `get_history` 就有数据。未显式 `set_history_depth` 时，历史深度
  自动抬到预热深度（每标的最大根数），已设的更大值不会被下调；
- 喂 `subscribe_bars` 的窗口聚合器：预热期间闭合的窗口同样进入窗口历史；
- 计入 `warmup_period`：预热了 N 根就算已经攒了 N 根；
- 在 `on_start` **之后**逐根回放每个 `self.I()` 指标（含 `freq=` 窗口指标），所以 `on_start` 里
  声明的参数化指标同样吃得到。挂了指标脚本或多个 slot 时，回放统一排在主策略与全部 slot 的
  `on_start` 都跑完之后（先主策略、再各 slot），保证同名指标从第一个预热点起就带上最终的
  `series_key`。绘图点照常进指标流，只是带 `phase="preload"`；回放结束后同一批
  实例接着算实时 bar，点带 `phase="run"`。

预热不做什么：不触发 `on_bar`、不撮合、不动账户，零下单风险。在 `on_bar` 里手算、再用
`record_indicator` 上报的指标不会被回放——需要预热的指标请改用 `self.I()` 声明。

几条边界：

- 晚于会话启动时刻的行会被裁掉并告警（防止“看到未来”）；
- 实盘首根 bar 若不晚于该标的的预热末戳，会点名告警但**照常派发**（重复一根远比丢一根轻）。
  前端按同 `(owner_strategy_id, indicator_key, symbol, timestamp)` 用后到的点覆盖即可；
- 从存档热启动的策略跳过指标回放（状态已随存档恢复），历史缓冲仍会灌入；
- tick 驱动的会话灌入 bar 预热后会变成 bar/tick 双流，`on_timer` 等回调之外读历史需显式传
  `freq='bar'` 或 `freq='tick'`。

前端拿到的是一条连续序列，按 `phase` 决定是否区分样式即可。完整闭环见
`examples/73_lwc_live_indicators.py`：前 40 根作为预热、后 20 根实时到达，图上是一条不间断的线。

!!! warning "历史段一次下发，实时段才逐点"
    `phase="preload"` 的点是会话启动时一次性回放出来的批量结果，**不要**逐点 `series.update()` 推给前端：
    几千个点逐个推送、每个再等一次网络往返，图会花几分钟“从左往右画”。正确做法是收集 preload 点，
    按 `series_key` 分组后用 `series.setData()` 一次画完；之后 `phase="run"` 的点再逐个 `update()`。
    框架算 2 万根 × 3 个指标约 1～2 秒，慢通常出在接入方式。`examples/73_lwc_live_indicators.py`
    就是这样做的：preload 点进一个批，页面拿到后 `setData()`，实时点才走增量。

### 实时增量到 LWC 图表

前端如果用 lightweight-charts，不必自己解析 `to_indicator_message()` 的字段：
`akquant.lwc.to_lwc_update()` 把 point 消息转成 `series.update()` 可直接消费的增量
（`{owner_strategy_id, indicator_key, series_key, pane, series_type, color, confirmed, phase, point:{time, value}}`）。
按 `series_key` 找线：它与 `to_lwc_indicator_series()` 历史 series 的 `series_key` 同规则，
主策略与指标脚本同名时也不会合成一条。
`confirmed` 原样带出——将来未闭合 bar 的临时点与确认点同 `time`，LWC 的 `update()`
按 `time` 覆盖，前端零改动。`phase` 同样原样带出（`"preload"` / `"run"`，见上一节）。

```python
from akquant.lwc import load_lwc_js, to_lwc_update

def on_event(event):
    message = aq.to_indicator_message(event)
    update = to_lwc_update(message) if message else None
    if update is not None:
        push_to_page(update)           # 传输层由你决定：轮询 / WebSocket / SSE
```

`load_lwc_js()` 返回 vendored 的 LWC 源码文本，供自建页面内联（无 CDN）。完整闭环见
`examples/73_lwc_live_indicators.py`：`run_live(broker="replay")` + 标准库 `http.server`
轮询 + 首次出现时惰性建 series。HTTP/WS 传输刻意留在示例层，不进核心。


### 零依赖浏览器实时预览

如果你想先给业务方或前端同事一个“打开浏览器就能看到”的最小接入样板，而暂时不引入
`fastapi`、`uvicorn` 或 `websockets` 等依赖，可以直接参考
`examples/64_indicator_live_web.py`。

这个示例做了三件事：

- 用 `run_backtest(..., on_event=...)` 接收流式事件；
- 用 `aq.to_indicator_message(event)` 规整成前端友好消息；
- 用内置 `http.server` 暴露 `/state` JSON，并由浏览器轮询绘制 `close_echo` 折线。

现在这个示例同时支持两种轮询方式：

- 直接请求 `/state`，拿最近窗口内的完整快照；
- 请求 `/state?since_seq=123`，只拿 `seq > 123` 的增量消息，同时保留总量统计和最新游标。

当前返回结构也做了区分，便于前端减少分支歧义：

- 公共字段放在 `cursor`、`counts`、`latest_indicator_values`
- 全量模式把消息窗口放在 `window.point_messages` / `window.snapshot_messages`
- 增量模式把新增消息放在 `delta.point_messages` / `delta.snapshot_messages`

运行方式：

```bash
UV_INDEX_URL=https://pypi.org/simple uv run python examples/64_indicator_live_web.py --open
```

如果你只是想快速验证链路，也可以缩短保活时间：

```bash
UV_INDEX_URL=https://pypi.org/simple uv run python examples/64_indicator_live_web.py --keep-seconds 1
```

这个样板的定位不是完整前端产品，而是帮助你更快完成以下工作：

- 验证指标流是否已经成功出站；
- 让前端先对接稳定的消息结构和 `/state` JSON；
- 在不新增依赖栈的前提下演示实时指标预览效果。

### 当前输出结构

第一版实现会输出三类结构化结果：

- 指标定义：如 `display_name`、`pane`、`render_type`
- 指标实例：按 `strategy/symbol/indicator/meta` 归并后的实例信息
- 指标点位：按时间记录的数值序列

每个指标点位同时带有 `timestamp`（纳秒）与 `timestamp_ms`（毫秒）两个时间字段：前者供
Python 侧按纳秒解析，后者可被前端图表库直接使用，无需再做单位换算。指标 `meta` 采用
`ensure_ascii=False` 序列化，中文等非 ASCII 字符保持可读。

点位的 `symbol` 保证非空（见上文「关于 `symbol` 归属」），因此消费端可直接按
`symbol` + `indicator_key` 分组，不必处理"无归属"的情形。多标的回测下同名指标会
按标的拆成多条独立序列，切记按 `symbol` 过滤后再绘制，否则不同标的的同名指标会
串成一条线。

#### 指标身份与 series 键

指标定义的身份是 `(owner_strategy_id, indicator_key)`：主策略（owner 为 `_default`）
与指标脚本（owner 为其 `id`，默认取类名的 snake_case）可以使用同名键，各自保留自己的 `pane`、`color` 等定义。
`result.indicator_definitions` 的第一列是 `owner_strategy_id`，按 `(owner, key)` 排序；
点位本来就带 `owner_strategy_id`，消费端应按 `(owner_strategy_id, indicator_key)` 关联
定义与点位。

图上的 series 键只在冲突时加前缀：某个 `indicator_key` 只属于一个 owner 时，series 键
就是 `indicator_key`，现有图和前端不受影响；被多个 owner 使用时，series 键为
`"{owner}:{key}"`，例如 `_default:ema` 与 `sub:ema`。图例显示名在用户没写 label（展示名
等于 `indicator_key`）时随之变为 series 键；写了 label 时保持原样。LWC、d3Kline、plotly
三个出口统一经由 `akquant.indicator_identity.series_keys()` 取键，自建前端也应复用它。

candle 组在图上是一条 series，它的键由 `akquant.indicator_identity.chart_series_keys()`
计算：candle 组以 `(owner, group)`、普通线以 `(owner, indicator_key)` 进入**同一个**命名
空间。两个指标脚本都有 `ha` 组，或者一个脚本的 `ha` 组遇上另一个 owner 的普通线 `ha`，
双方都变成 `"{owner}:ha"`。图上出口（LWC、d3Kline、`to_raw_panes`、plotly）都用它；
实时流里分量的 `series_key` 仍按分量键（如 `ha.open`，冲突时 `ha_two:ha.open`）计算，
流消费者按 `(owner_strategy_id, group, symbol, timestamp)` 拼 K 线，不依赖组的键。
同一 owner 既有普通线 `ha`（例如直接 `record_indicator("ha")`）又有 candle 组 `ha` 时，
组保持 `ha`，普通线在图上改为 `ha:line`，并告警一次。

实时流（`on_event` 的 point / snapshot 消息、`to_lwc_update()`）同样带 `series_key`，
规则与上面相同，由同一个函数算出。流在发出每条消息时就要给出名字，而这时还没有完整的
定义表，所以录制器（回测的 `IndicatorRecorder`、实盘的 `StreamingIndicatorSink`）维护
一张会话级登记表：挂载时补登各策略与指标脚本已有的 `plot=True` 声明（`__init__` 里的
`self.I`），挂载后 `self.I(plot=True)` 的声明（`on_start` 里）即时登记，每次
`record_indicator` 也登记。因此在回测和实盘（含 `preload_history` 回放）中，`__init__` 与
`on_start` 里的声明都会在第一个点发出之前登记完毕，同名指标从第一个点起就是最终的
`series_key`。自定义 `IndicatorSink` 可实现可选方法
`declare(owner_strategy_id, indicator_key)` 接收这些登记，不实现也不影响 `record()`。

与历史接口有两处已知差异：

- 运行中途才有第二个 owner 用了已有的键（例如在 `on_bar` 里手写同名的
  `record_indicator`）时，该键的所有 owner 从这一刻起都改为 `"{owner}:{key}"`，此前已发出
  的点仍是不带前缀的名字，不追溯修改；每个键只打一次 warning。把同名指标用
  `self.I(plot=True)` 在 `__init__` / `on_start` 里声明，或改用不同的 `name`，可避免切换。
- 声明了 `plot=True`、但整场一个点都没产出的指标（例如始终未就绪）会参与实时流的冲突
  判断，却不在历史定义表里，可能出现实时侧加了前缀、历史侧没加的情况。

另外，point 的 `series_key` 在发出那一刻计算，snapshot item 的在本轮 flush 时计算：若冲突恰好
在同一个快照周期内出现，同一轮里先发的 point 与 snapshot item 的 `series_key` 可能不同，以
snapshot 为准。

这三层结构的目的，是让 AKQuant 负责“生产标准化指标数据”，而不是直接耦合某个具体前端图表库。

### 可选适配器：转成 d3Kline 的 `IndicatorOption`

如果你的前端是 d3Kline（金融界期魔方前端的 K 线渲染器），可以直接用
`akquant.chart.to_d3kline_options()` 把指标定义与点位转成 `IndicatorManager.create()`
能直接消费的结构，不必自己写映射：

```python
from akquant.chart import group_points_by_identity, to_d3kline_options, to_raw_panes

defs = result.indicator_definitions.to_dict(orient="records")
# (owner_strategy_id, indicator_key) -> 点位列表
points_by_identity = group_points_by_identity(
    result.indicator_df().to_dict(orient="records")
)
options = to_d3kline_options(defs, points_by_identity)  # 喂给前端 IndicatorManager.create()
panes = to_raw_panes(defs, points_by_identity)  # 保留 akquant 原生语义，供其它前端
```

它是**可选的消费端适配器**，与上文的流式桥接同一性质，不改变“生产者不耦合前端”的定位。
把它放进核心包的唯一理由是：回测服务与实盘服务两条链路都要产出这套结构，而前端用同一套
渲染逻辑消费——两边各写一份必然漂移，这里是单一事实来源。

几条刻意为之的转换规则（改动前请先理解原因）：

- **同一 pane 的多个指标打包成一个 option**，各指标是 `dataList` 里的一条 series。
  d3Kline 的 `IndicatorManager` 有 `MAX_SUB_INDICATORS = 3` 的上限且超限**静默丢弃**，
  按 pane 打包能让“5 个副图指标只占 2 个 pane”时只消耗 2 个名额。
- **主图（pane 0）不带 `style.height`**，避免挤压 K 线区；副图给默认 100。
- **无颜色时省略整个 `style` 字段**，而不是传 `{"color": null}`——前端的合并写法是
  `{color: 默认色, ...old, ...s.style}`，`null` 会覆盖掉默认色导致指标无色。
- `render_type` 的八个值降级到 d3Kline 仅有的 `line | bar`；发生降级时附带原值
  `render_type`，前端将来支持 area/scatter 时可无损升级。`candle` 组只输出 close 分量的
  一条 line，`name` 为组的 series 键。
- series 的 `name` 用 series 键（前端以 name 做合并身份，必须唯一）：同名键只属于一个
  owner 时就是 `indicator_key`，跨 owner 冲突时为 `"{owner}:{key}"`；中文展示名放在
  `label`。点位按 `(owner_strategy_id, indicator_key)` 查找，`to_raw_panes()` 的每个指标
  另带 `owner_strategy_id` 与 `series_key`。

输入用普通 `Mapping`（dict）而非特定模型，缺失字段按契约默认值处理（`pane=0`、
`render_type="line"`），`pane` 允许是字符串。

### 推荐边界

建议把职责切开：

- `AKQuant` 内部负责：
  - 指标计算
  - 指标记录
  - 指标查询
  - 指标导出
- 外部平台负责：
  - 存储服务
  - API 查询
  - WebSocket 推送
  - 前端图表页面

也就是说，AKQuant 更适合作为“指标生产者”，而不是企业前端平台本身。

### 推荐示例

- [60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py)
- [61_indicator_visualization_export_demo.py](https://github.com/akfamily/akquant/blob/main/examples/61_indicator_visualization_export_demo.py)
- [62_indicator_streaming_demo.py](https://github.com/akfamily/akquant/blob/main/examples/62_indicator_streaming_demo.py)
- [63_indicator_ws_bridge_demo.py](https://github.com/akfamily/akquant/blob/main/examples/63_indicator_ws_bridge_demo.py)
- [64_indicator_live_web.py](https://github.com/akfamily/akquant/blob/main/examples/64_indicator_live_web.py)

## 常见误区

- 误区 1：所有自定义指标都必须继承某个指标基类
  - 不是。预计算场景可以直接用 `PrecomputedIndicator(name, fn)`；增量场景传任何有 `update()` 的对象都行。
    `aq.Indicator` 是“只画图不交易”的指标脚本基类，与“写一个自定义指标”是两回事。
- 误区 2：多标的一定可以共用一个增量实例
  - 不行。传实例跨多标的会直接报错，正式策略请用 `factory=`。
- 误区 3：`warmup_bars=20` 会重复消费第一根正式 Bar
  - 不会。预热只使用正式开始前的历史数据。
- 误区 4：自定义指标自然支持热启动
  - 不一定。需要确认对象可 `pickle`，`factory` 也不能是 lambda。
- 误区 5：策略自定义指标和 `akquant.talib` 扩展是一回事
  - 不是，二者面向的层级不同。
- 误区 6：声明了指标就会自动画出来
  - 在 `aq.Strategy` 里不会，要写 `plot=True`（只写 `pane=` / `color=` 不够），与 Pine 的 `ta.*`
    不画、`plot()` 才画一致；在指标脚本 `aq.Indicator` 里默认画，`plot=False` 关闭。

## 选择建议

| 你的目标 | 建议方案 |
| :--- | :--- |
| 先快速验证一个想法 | `self.I(PrecomputedIndicator(name, fn))` |
| 单标的逐 Bar 更新 | `self.I(aq.EMA(20))` |
| 多标的正式策略 | `self.I(factory=...)` |
| 首根有效 Bar 就要有值 | `self.I(..., warmup_bars=N)` |
| 要在图上画出来 | 策略里写 `plot=True`，或改写成指标脚本 `aq.Indicator` |
| 判断金叉死叉 | `ind[0]` / `ind[1]` 序列回溯 |
| 需要断点续跑 | 指标状态可序列化，`factory` 用 `partial` 而非 lambda |
| 需要极致性能 | 先用内置的 100+ 个 Rust 指标类，再考虑自己写 Rust |

## 推荐阅读

- [策略开发手册](./strategy.md)
- [热启动指南](../advanced/warm_start.md)
- [AKQuant 指标全量说明](./rust_indicator_reference.md)
- [指标组合实战手册](./talib_indicator_playbook.md)
- [可运行示例：60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py)
- [可运行示例：72_declarative_indicators.py](https://github.com/akfamily/akquant/blob/main/examples/72_declarative_indicators.py)
