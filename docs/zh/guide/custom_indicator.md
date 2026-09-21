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
| 用 pandas 一次性计算整段历史 | `Indicator(name, fn)` | `self.I(Indicator(...), ...)` |
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
        # 声明一次 = 自动更新 + 自动绘图 + 可回溯
        self.ema_fast = self.I(aq.EMA(10), pane=0, color="#e91e63", label="EMA10")
        self.ema_slow = self.I(aq.EMA(30), pane=0, color="#3f51b5")
        self.rsi = self.I(aq.RSI(14), pane=1)

    def on_bar(self, bar: Bar) -> None:
        # [0] 是当前 bar，[1] 是上一根 —— 与 Pine 的 sma[0] / sma[1] 一致
        if self.ema_fast[1] is None or self.ema_slow[1] is None:
            return
        if self.ema_fast[1] <= self.ema_slow[1] and self.ema_fast[0] > self.ema_slow[0]:
            self.buy(bar.symbol, 100)
        # 无需再写 record_indicator —— 传了绘图参数就会自动上报
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
from akquant import Indicator, Strategy


class MyMomentum(Indicator):
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
                   outputs=("dif", "dea", "hist"))

# 读取：
self.macd[0]        # 整个元组 (dif, dea, hist)
self.macd.dif[0]    # 单个分量，同样支持 [1] 回溯
```

`outputs` 个数与指标实际分量数不符时会直接报错，不会静默错位。

## 路径二：预计算指标

当你的指标更适合一次性对完整 `DataFrame` 计算时，传一个 `Indicator(name, fn)`
实例即可，`self.I()` 会自动走向量化预计算路径。

### 最小示例

```python
from akquant import Indicator, Strategy


class PrecomputeMomentumStrategy(Strategy):
    def on_start(self):
        self.mom10 = self.I(
            Indicator("mom10", lambda df: df["close"] - df["close"].shift(10)),
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
- **多标的天然可用**：`Indicator` 自己按 symbol 缓存整段结果，不需要 `factory=`。

## 自动绘图

传了任一绘图参数（`pane` / `color` / `label` / `render_type` / `reference_lines` /
`scale_group`）或显式 `plot=True`，框架就会在每次更新后自动上报一个绘图点，
**无需在 `on_bar` 里手写 `record_indicator`**。

**默认不上报**——与 Pine 一致（`ta.sma()` 只计算，`plot()` 才画）。这不只是风格：
实盘的指标 sink 是每点一个事件，多标的 × 多指标默认全开会淹没前端链路。

未就绪的点（值为 `None`）不上报，对应 Pine 预热期的 `na`。

```python
# 只算不画
self.atr = self.I(aq.ATR(14), input_mode="hlc")

# 算且画
self.rsi = self.I(aq.RSI(14), pane=1, reference_lines=[
    {"value": 70, "label": "超买"},
    {"value": 30, "label": "超卖"},
])

# 没有其它绘图参数时，用 plot=True 显式开启
self.sma = self.I(aq.SMA(20), plot=True)
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
        self.sma5 = self.I(aq.SMA(20), freq="5min", intrabar=True, pane=0)

    def on_bar(self, bar):                 # 每根 1min bar
        v0 = self.sma5[0]                  # 有临时值时: 未闭合 5min 窗口上的试算值
        v1 = self.sma5[1]                  # 上一根已确认的 5min 值
        if not self.sma5.confirmed: ...    # 当前 [0] 是临时值
```

下标语义与 Pine 完全一致：有临时值时它占据 `[0]`，已确认序列整体后移一位。流事件里临时点带
`confirmed=false`，且**时间戳就是该窗口将来闭合的标签**，确认点随后以同一 `time` 到达——前端按
`(indicator_key, symbol, timestamp)` 覆盖即可（`akquant.lwc.to_lwc_update()` 已按此处理）。
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

## 可插拔 study：只画图不交易

TradingView 里 indicator 脚本与 strategy 脚本是两类东西，一张图上可以叠 N 个只画图的
indicator。AKQuant 的对应物是 `Study`：继承它、在 `on_start` 里用 `self.I()` 声明指标，
然后用 `studies=[...]` 挂到任意一次回测或实盘上。

```python
import akquant as aq
from akquant import Study, run_backtest


class RsiStudy(Study):
    def on_start(self):
        self.rsi = self.I(aq.RSI(14), pane=1)


class MacdStudy(Study):
    study_id = "macd_view"          # 省略时取类名 snake_case: "macd_study"

    def on_start(self):
        self.macd = self.I(aq.MACD(12, 26, 9), pane=2, outputs=("dif", "dea", "hist"))


result = run_backtest(strategy=MyStrategy, data=data, symbols=["600000"],
                      strategy_id="main", studies=[RsiStudy, MacdStudy])
result.indicator_df(owner="rsi_study")    # 只看这个 study 的点
result.viz.review(data)                   # study 的指标一样画到 LWC 图上
```

三条要点：

- **不新建管线**：每个 study 就是多策略拓扑里的一个 slot（`strategies_by_slot`），指标点带
  自己的 `owner_strategy_id`，窗口订阅跨 slot 合并，`run_live(studies=...)` 同一套语义。
- **不交易是硬保证**：`Study` 上所有交易 API（`buy` / `sell` / `place_*` / `order_target*` /
  `rebalance_*` / `cancel_*` …）一律抛 `StudyCannotTradeError`，在引擎回调里误写也会当场
  报错，不会静默成交。要下单请继承 `Strategy`。
- **id 不许撞**：`study_id` 与 `strategy_id` 或已有 slot key 重复时直接 `ValueError`；
  `studies=` 里塞进非 `Study` 的类会 `TypeError`——否则"不交易"的保证就没了。

可运行示例：[74_pluggable_studies.py](https://github.com/akfamily/akquant/blob/main/examples/74_pluggable_studies.py)。

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
    `render_type` 是一个**封闭枚举**，共 7 个值，消费者可据此穷举渲染分支：

    | 值 | 渲染 |
    | :--- | :--- |
    | `line` | 折线（默认） |
    | `area` | 折线 + 向零填充 |
    | `bar` | 垂直柱 |
    | `column` | `bar` 的别名（语义化：分类柱） |
    | `histogram` | `bar` 的别名（语义化：分布柱） |
    | `scatter` | 离散点标记 |
    | `signal` | 交易信号标记，渲染在主图 |

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
`render_type` 的 7 个值全部有对应：`line`/`area`/`bar`/`column`/`histogram`/`scatter`
各画各的，`signal` 走 K 线上的标记而不建独立序列。

```python
result.viz.review(market_data, filename="review.html")            # 带指标
result.viz.review(market_data, filename="review.html", include_indicators=False)
```

底层适配器是 `akquant.lwc.to_lwc_indicator_series()`，与 `akquant.chart.to_d3kline_options()`
是同一份指标契约面向两个前端的镜像实现；自建 LWC 页面可以直接复用。

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
- `panes`
- `render_types`
- `value_by_key`
- `items_by_key`
- `warmup_count`
- `has_warmup`

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
    该版本的旧消息时，回落到纳秒 `timestamp` 自行换算。

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

### 实时增量到 LWC 图表

前端如果用 lightweight-charts，不必自己解析 `to_indicator_message()` 的字段：
`akquant.lwc.to_lwc_update()` 把 point 消息转成 `series.update()` 可直接消费的增量
（`{indicator_key, pane, series_type, color, confirmed, point:{time, value}}`）。
`confirmed` 原样带出——将来未闭合 bar 的临时点与确认点同 `time`，LWC 的 `update()`
按 `time` 覆盖，前端零改动。

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

这三层结构的目的，是让 AKQuant 负责“生产标准化指标数据”，而不是直接耦合某个具体前端图表库。

### 可选适配器：转成 d3Kline 的 `IndicatorOption`

如果你的前端是 d3Kline（金融界期魔方前端的 K 线渲染器），可以直接用
`akquant.chart.to_d3kline_options()` 把指标定义与点位转成 `IndicatorManager.create()`
能直接消费的结构，不必自己写映射：

```python
from akquant.chart import to_d3kline_options, to_raw_panes

defs = result.indicator_definitions.to_dict(orient="records")
points_by_key = {
    key: group[["timestamp_ms", "value"]].to_dict(orient="records")
    for key, group in result.indicator_df().groupby("indicator_key")
}
options = to_d3kline_options(defs, points_by_key)  # 喂给前端 IndicatorManager.create()
panes = to_raw_panes(defs, points_by_key)  # 保留 akquant 原生语义，供其它前端
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
- `render_type` 的七个值降级到 d3Kline 仅有的 `line | bar`；发生降级时附带原值
  `render_type`，前端将来支持 area/scatter 时可无损升级。
- series 的 `name` 用 `indicator_key`（前端以 name 做合并身份，pane 内必须唯一），
  中文展示名放在 `label`。

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

- 误区 1：所有自定义指标都必须继承 `Indicator`
  - 不是。预计算场景可以直接用 `Indicator(name, fn)`；增量场景传任何有 `update()` 的对象都行。
- 误区 2：多标的一定可以共用一个增量实例
  - 不行。传实例跨多标的会直接报错，正式策略请用 `factory=`。
- 误区 3：`warmup_bars=20` 会重复消费第一根正式 Bar
  - 不会。预热只使用正式开始前的历史数据。
- 误区 4：自定义指标自然支持热启动
  - 不一定。需要确认对象可 `pickle`，`factory` 也不能是 lambda。
- 误区 5：策略自定义指标和 `akquant.talib` 扩展是一回事
  - 不是，二者面向的层级不同。
- 误区 6：声明了指标就会自动画出来
  - 不会。要传绘图参数或 `plot=True`，与 Pine 的 `ta.*` 不画、`plot()` 才画一致。

## 选择建议

| 你的目标 | 建议方案 |
| :--- | :--- |
| 先快速验证一个想法 | `self.I(Indicator(name, fn))` |
| 单标的逐 Bar 更新 | `self.I(aq.EMA(20))` |
| 多标的正式策略 | `self.I(factory=...)` |
| 首根有效 Bar 就要有值 | `self.I(..., warmup_bars=N)` |
| 要在图上画出来 | 加 `pane=` / `color=`，或 `plot=True` |
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
