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
| 给策略增加一个私有信号 | 自定义 `Indicator` / 自定义增量对象 | 在 `Strategy` 中注册 |
| 用 pandas 一次性计算整段历史 | `indicator_mode="precompute"` | `register_precomputed_indicator(...)` |
| 逐 Bar / 逐 Tick 维护状态 | `indicator_mode="incremental"` | `register_incremental_indicator(...)` |
| 给 `akquant.talib` 增加一个新函数名 | 修改 Python/Rust 兼容层源码 | 不属于运行时动态注册 |

如果你的目标只是“在策略里用一个自己的指标”，通常不需要修改 `akquant.talib`。

## 路径一：预计算指标

当你的指标更适合一次性对完整 `DataFrame` 计算时，优先使用 `precompute` 模式。

### 最小示例

```python
from akquant import Indicator, Strategy


class PrecomputeMomentumStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.indicator_mode = "precompute"
        self.mom10 = Indicator(
            "mom10",
            lambda df: df["close"] - df["close"].shift(10),
        )
        self.register_precomputed_indicator("mom10", self.mom10)

    def on_bar(self, bar):
        value = self.mom10.get_value(bar.symbol, bar.timestamp)
        if value == value and value > 0:
            self.buy(bar.symbol, 100)
```

### 何时适合

- 指标天然可向量化；
- 主要依赖 pandas `rolling` / `shift` / `ewm`；
- 更关心回测开发效率，而不是在线增量更新；
- 同一个 `symbol` 的整段历史可以提前准备好。

### 注意事项

- `Indicator(name, fn, **kwargs)` 的核心输入是一个返回 `pd.Series` 的函数；
- `get_value(symbol, timestamp)` 会从缓存序列中按时间取值；
- 指标会按 `symbol` 缓存结果，适合同一回测里重复访问。

## 路径二：增量指标

如果你希望指标在事件流中逐步更新，推荐使用 `incremental` 模式。

### 最小示例

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
    def __init__(self):
        super().__init__()
        self.indicator_mode = "incremental"

    def on_start(self):
        self.register_incremental_indicator(
            "mom10",
            indicator_factory=lambda: MyMomentum(period=10),
            source="close",
            symbols=["AAPL", "MSFT"],
            warmup_bars=10,
        )

    def on_bar(self, bar):
        value = self.mom10.value
        if value == value and value > 0:
            self.buy(bar.symbol, 100)
```

### 为什么推荐 `indicator_factory`

多标的策略里，增量指标通常都有内部状态。如果多个 `symbol` 共用同一个实例，状态很容易串线。

因此，推荐写法是：

```python
self.register_incremental_indicator(
    "mom10",
    indicator_factory=lambda: MyMomentum(period=10),
    source="close",
    symbols=["AAPL", "MSFT"],
)
```

而不是：

```python
self.mom10 = MyMomentum(period=10)
self.register_incremental_indicator("mom10", self.mom10, source="close")
```

后者更适合单标的或临时实验。

### `source` 代表什么

`source` 指定框架从行情对象里拿哪个字段喂给指标。最常见的是：

- `source="close"`
- `source="open"`
- `source="high"`
- `source="low"`
- `source="volume"`

如果你的指标需要多输入，建议优先参考策略手册中的增量接口约定，并确保你的 `update(...)` 参数顺序与框架喂入顺序一致。

## `warmup_bars` 怎么用

`warmup_bars` 用于在正式事件流开始前，先使用 `start_time` 之前的历史 Bar 预热指标。

适合以下场景：

- 你希望第一根有效 Bar 就拿到完整指标值；
- 指标依赖窗口历史，如 `period=20`；
- 你不想在 `on_bar` 里手工跳过前 N 根。

推荐示例可参考：

- [58_incremental_bootstrap_demo.py](https://github.com/akfamily/akquant/blob/main/examples/58_incremental_bootstrap_demo.py)
- [60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py)

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

运行结束后：

```python
result = ...

# 1) 在 Python 里直接读取
indicator_df = result.indicator_df(name="intrabar_spread", symbol="AAPL")

# 2) 在本地做一个轻量预览
fig = result.plot_indicators(
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

- `result.plot_indicators(...)`
- `from akquant.plot import plot_indicators`

这条内置路径的定位是“轻量 history preview”，特点是：

- 保持现有 `result.plot()` 继续只做账户 dashboard；
- 按 `pane` 自动拆分子图；
- 复用 `render_type`，第一版支持常见的 `line` / `bar`；
- 支持 `name`、`symbol`、`include_warmup` 过滤；
- 可直接输出为本地 HTML，方便和导出的 JSON 一起联调。

例如：

```python
fig = result.plot_indicators(
    name="intrabar_spread",
    symbol="AAPL",
    include_warmup=False,
    show=False,
    filename="indicator_preview.html",
    title="Indicator Preview",
)
```

如果你需要的是企业级多图联动、权限、持久化和实时订阅，这些仍建议放在外部平台实现；AKQuant 内部只提供最小预览和标准化数据输出。

### 报告中的可选指标区块

如果你希望把指标预览放进内置 HTML 报告，而不是单独输出一个图，也可以显式开启：

```python
result.report(
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

这三层结构的目的，是让 AKQuant 负责“生产标准化指标数据”，而不是直接耦合某个具体前端图表库。

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
  - 不是。预计算场景可以直接用 `Indicator(name, fn)`；增量场景也可以注册自定义对象，只要接口契合。
- 误区 2：多标的一定可以共用一个增量实例
  - 不建议。正式策略优先 `indicator_factory`。
- 误区 3：`warmup_bars=20` 会重复消费第一根正式 Bar
  - 不会。预热只使用正式开始前的历史数据。
- 误区 4：自定义指标自然支持热启动
  - 不一定。需要确认对象可 `pickle`。
- 误区 5：策略自定义指标和 `akquant.talib` 扩展是一回事
  - 不是，二者面向的层级不同。

## 选择建议

| 你的目标 | 建议方案 |
| :--- | :--- |
| 先快速验证一个想法 | `Indicator(name, fn)` + `precompute` |
| 单标的逐 Bar 更新 | `incremental` + 单实例 |
| 多标的正式策略 | `incremental` + `indicator_factory` |
| 首根有效 Bar 就要有值 | 增量模式 + `warmup_bars` |
| 需要断点续跑 | 确保指标状态可序列化 |
| 需要极致性能 | 再考虑 Rust 指标实现 |

## 推荐阅读

- [策略开发手册](./strategy.md)
- [热启动指南](../advanced/warm_start.md)
- [AKQuant 指标全量说明](./rust_indicator_reference.md)
- [指标组合实战手册](./talib_indicator_playbook.md)
- [可运行示例：60_custom_indicator_demo.py](https://github.com/akfamily/akquant/blob/main/examples/60_custom_indicator_demo.py)
