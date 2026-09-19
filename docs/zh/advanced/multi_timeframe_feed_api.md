# 多周期策略：三条路径

AKQuant 目前有三条独立的多周期路径，覆盖不同场景，互不替代：

1. **`subscribe_bars`（推荐）**：引擎原生，Rust 直接从基础 bar 聚合窗口，回测/实盘同一套代码。
2. **`BarGenerator`**：策略内的 Python 运行时工具，逐 bar 手工喂入聚合器。
3. **`feed.resample` / `feed.replay`**：离线数据编排，在喂给 `run_backtest` 之前就把多个频率的 feed 准备好。

## 怎么选

| 路径 | 适用 | 回测 | 实盘 | 高周期进历史与指标 | 日线时间戳口径 |
| --- | --- | --- | --- | --- | --- |
| `subscribe_bars`（推荐） | 策略内声明式订阅多周期 | 支持 | 支持 | 进 `get_history(freq=)` / `register_incremental_indicator(freq=)` | 当日最后一根基础 bar |
| `BarGenerator` | 策略内手工聚合，不想动引擎配置 | 支持 | 支持 | 不进历史/指标，只有聚合后回调收到的那一根 | 次日零点（pandas `resample` 口径） |
| `feed.resample` / `feed.replay` | 离线数据编排，喂给 `run_backtest` 之前就要多频率 feed | 仅回测 | 不支持 | 需要伪标的拼接，用户自己打时间戳 | 用户自定 |

## `subscribe_bars` 用法

在策略 `__init__` 里声明订阅（只能在 `__init__` 里调用，引擎启动后订阅表即冻结）：

```python
from akquant import Bar, Strategy


class FiveMinuteTrend(Strategy):
    """按 5 分钟窗口打印近 3 根收盘价, 用来展示 subscribe_bars 的实盘闭合时机."""

    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars(
            "5min", session_windows=[("09:30", "11:30"), ("13:00", "15:00")]
        )
        self.count = 0

    def on_start(self) -> None:
        self.set_history_depth(50)
        print(f"基础周期 self.freq = {self.freq!r}")

    def on_window_bar(self, bar: Bar) -> None:
        self.count += 1
        closes = self.get_history(3, bar.symbol, "close")  # 回调内自动定档到 5min
        print(
            f"[5min] {self.format_time(bar.timestamp)} "
            f"C={bar.close:.2f} 近3根={closes.round(2).tolist()}"
        )

    def on_stop(self) -> None:
        print(f"共收到 {self.count} 根 5 分钟窗口 bar")
```

完整可运行示例见
[examples/71_native_multi_timeframe_live.py](https://github.com/akfamily/akquant/blob/main/examples/71_native_multi_timeframe_live.py)（实盘/replay）与
[examples/14_multi_frequency.py](https://github.com/akfamily/akquant/blob/main/examples/14_multi_frequency.py)（回测）。

关键 API：

- `subscribe_bars(freq, callback=None, symbols=None, *, session_windows=None)`：声明订阅。`freq` 支持整数分钟 `"Nmin"`、整数小时 `"Nh"`、日线 `"1d"`。`symbols=None` 表示全部标的。
- `callback`：省略时闭合的窗口 bar 会回调策略的 `on_window_bar(bar)`；传入具体函数（如示例 14 的 `on_daily`）则改由该函数接收，`on_window_bar` 不再触发该条订阅。
- `on_window_bar(bar)`：默认窗口回调，`bar.freq` 是该 bar 所属的周期标签（如 `"5min"`、`"1d"`）。
- `current_window(symbol, freq)`：查看某标的当前正在形成、尚未闭合的窗口快照（只能在行情回调内调用），不会触发回调。
- `get_history(count, symbol, field, freq=)` / `get_history_multi(...)` / `register_incremental_indicator(..., freq=)`：`freq` 传已订阅的窗口周期标签，取的是该周期的历史序列；在窗口回调内可省略 `freq`（自动按当前回调定档）。

## 闭合规则：即时 vs 延迟

窗口何时闭合取决于**基础数据周期**（`self.freq`）是否已知：

- **基础周期已知**（回测传 `run_backtest(freq=...)`，或实盘/replay 行情网关声明 `metadata["freq"]`，见示例 71）：窗口在归属它的最后一根基础 bar 的 `on_bar` 之后**同一步**闭合并派发（零延迟）。
- **基础周期未知**（例如回测直接传一个纯 bar 的 `DataFrame`，没有走 `freq=` 或 `list[Tick]` 输入）：引擎无法判断"下一根基础 bar 是否已经跨入新窗口"，因此窗口要等**下一根落入新窗口的基础 bar 到达时**才闭合（晚一根），并在配置阶段打印一次性 `WARNING`。示例 14 就是这种情况：日线窗口在次日第一根 09:31 分钟 bar 到达后才闭合。`session_windows` 只负责正确切分交易时段边界，不能让窗口提前到即时闭合——即时闭合只看基础周期是否已知。

## 限制

- `freq` 仅支持整数分钟（`"Nmin"`）、整数小时（`"Nh"`）与日线（`"1d"`），不支持秒级、周线、月线；这些场景请改用 `feed.resample`。
- 窗口周期必须严格大于基础数据周期，否则 `subscribe_bars` 在引擎配置阶段报 `ValueError`。
- 同一周期、标的范围有重叠的多条订阅，`session_windows` 必须完全一致，否则报 `ValueError`（同周期+同 `session_windows`，一条全标的通用回调 + 一条特定标的专属回调是允许的，两个回调都会触发）。
- `extra` 字段不参与窗口聚合。
- 窗口 bar **不会**被撮合引擎用于成交判断——它只是信息，策略仍需在 `on_bar`（或其他基础行情回调）里正常下单。
- CTP 网关合成 bar 默认打的是区间起点戳；只有同时开启 `emit_ticks` 与 `emit_bars` 才会得到区间结束戳。若不同时开启，多周期聚合会因为时间戳口径不一致而错位一根，因此接入 CTP 做多周期时请同时打开 `emit_ticks` 与 `emit_bars`。
- 会话结束（回测结束、实盘/replay 会话终止）时，引擎会 flush 尚未闭合的尾部窗口并触发一次回调；此时下的单在回测里不会再有机会成交。若之后 `save_checkpoint()` 再从该快照 `run_from_checkpoint()` 续跑，尾部区间会重新形成一根同标签的窗口 bar（历史序列里会出现两根标签相同的 bar，这是已知且刻意保留的行为，不是 bug）。

## 从伪标的迁移

引擎原生多周期上线前，[examples/14_multi_frequency.py](https://github.com/akfamily/akquant/blob/main/examples/14_multi_frequency.py) 靠「伪标的」拼日线：把同一份分钟数据手工 `resample` 成日线，起一个 `000001.SZ_1D` 的假代码喂给引擎，策略里再按 `bar.symbol` 分流处理两条序列，还需要手工把时间戳往后打 15 小时来避免与分钟数据的时间戳冲突。现在改为：

```python
self.subscribe_bars(
    "1d",
    callback=self.on_daily,
    session_windows=[("09:30", "11:30"), ("13:00", "15:00")],
)
self.register_incremental_indicator(
    "daily_sma", aq.SMA(self.ma_window), source="close", freq="1d"
)
```

不再需要伪标的、不再需要手工 `resample`/打戳，日线窗口由 Rust 引擎直接从基础 bar 聚合，通过 `on_daily` 回调派发。完整前后对比见示例 14 的 git 历史与模块 docstring。

## 离线数据编排：`feed.resample` / `feed.replay`

`feed.resample`/`feed.replay` 是**离线**方案：在回测数据层面预先构造好多个
频率的 feed，再喂给 `run_backtest(data=...)`，本质是回测专属的多 feed 编排，
**不支持实盘**。适合"喂给引擎之前就要把多频率数据准备好"的场景。

### API

```python
resampled = feed.resample(
    freq="15min",
    agg={"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"},
    label="right",
    closed="right",
)

replayed = feed.replay(
    freq="1h",
    align="session",
    day_mode="trading",
    emit_partial=False,
    session_windows=[("09:30", "11:30"), ("13:00", "15:00")],
)
```

### 当前落地（最小可用）

- `BasePandasFeedAdapter` 已支持：
  - `resample(freq, agg=None, label="right", closed="right", emit_partial=True)`
  - `replay(freq, align="session", emit_partial=False, agg=None, label="right", closed="right")`
- 两者均返回可直接传入 `run_backtest(data=...)` 的适配器对象。
- 当前 `replay(align="session")` 已支持按交易日分区聚合，`emit_partial=False` 时按日丢弃尾部未闭合窗口。
- 可通过 `session_windows` 进一步按日内交易时段分区（例如午休前后分段），减少跨时段混合聚合。
- `align` 当前支持三种语义：
  - `session`：按交易日分区，可叠加 `session_windows`。
  - `day`：按日分区，不接收 `session_windows`，并支持 `day_mode`。
  - `global`：按全局时间轴聚合，不按交易日切段。
- `day_mode` 当前支持：
  - `trading`：按请求时区下的本地交易日切分。
  - `calendar`：按 UTC 自然日切分。

### 语义锁定

#### resample

- 输入：原始 `Bar/Tick` 流。
- 输出：按目标频率聚合后的 Bar 流。
- 默认聚合：
  - `open=first`
  - `high=max`
  - `low=min`
  - `close=last`
  - `volume=sum`
- 边界：默认 `label=right, closed=right`。

#### replay

- 输入：高频数据。
- 输出：按低频时钟重放的事件流。
- `align=session`：按交易时段边界对齐。
- `emit_partial=False`：未闭合窗口不发出。

### 事件对齐策略

- 时区统一：内部统一 UTC，展示层可转本地时区。
- 会话优先：跨日、午休等边界以 market session 为准。
- 缺口策略：
  - 价格列沿用“无成交不补值”。
  - volume 默认置 0。

### 一致性校验

- 与 pandas `resample` 在同参数下做结果比对。
- 对齐误差允许仅限浮点精度范围。
- 回测与实时流使用同一聚合实现。

## 运行时聚合（BarGenerator）

`BarGenerator` 是**运行时**方案：聚合逻辑内嵌在策略里，逐 bar 流式喂入，不
依赖任何离线数据编排。回测与实盘用的是**同一句** `update_bar` 调用——策略
代码本身不需要区分当前是在回测还是实盘运行。两者在相同参数下遵循同一套
时钟对齐语义（`label="right"`, `closed="right"`，与 pandas
`resample(label="right", closed="right")` 一致），因此同参数产出结果一致。

与 `subscribe_bars` 的关键差异：`BarGenerator` 聚合出的 bar **不会**进入
`get_history`/增量指标体系，只有传给回调函数的那一根可用；日线的时间戳口径
也不同（`BarGenerator` 用 pandas 的次日零点，`subscribe_bars` 用当日最后一根
基础 bar）。

最小用法：

```python
from akquant import BarGenerator, Strategy


class MyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        # 把 1 分钟 bar 聚合为 5 分钟 bar，窗口闭合时回调 on_5m
        self.bg = BarGenerator(self.on_5m, 5, "minute")

    def on_bar(self, bar):
        # 回测/实盘同一句调用
        self.bg.update_bar(bar)

    def on_5m(self, bar):
        # 收到聚合后的 5 分钟 bar，写自己的信号逻辑
        ...

    def on_stop(self):
        # 收尾时强制闭合尾部未满窗口，否则最后一根不完整窗口会被丢弃
        self.bg.flush()
```

- `window`/`interval` 组合出目标周期（如 `BarGenerator(cb, 15, "minute")`、
  `BarGenerator(cb, 1, "hour")`、`BarGenerator(cb, 1, "day")`）。
- `session_windows` 用于按交易时段分段聚合，避免跨午休等段拼接出脏 bar，例如
  `session_windows=[("09:30", "11:30"), ("13:00", "15:00")]`；不传则按纯时钟
  对齐（跨会话边界不做特殊处理）。
- `timezone` 指定时钟/session 对齐所在的市场时区（如 `"Asia/Shanghai"`）。
- `current(symbol)` 可查看某标的当前正在形成、尚未闭合的窗口快照，不会触发
  回调，适合盘中展示。
- 多标的相互独立聚合，互不影响。

完整可运行示例见
[examples/65_bar_generator.py](https://github.com/akfamily/akquant/blob/main/examples/65_bar_generator.py)。
