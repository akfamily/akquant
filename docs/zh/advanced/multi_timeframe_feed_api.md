# 多周期策略：三条路径

AKQuant 目前有三条独立的多周期路径，覆盖不同场景，互不替代：

1. **`subscribe_bars`（推荐）**：引擎原生，Rust 直接从基础 bar 聚合窗口，回测/实盘同一套代码。
2. **`BarGenerator`**：策略内的 Python 运行时工具，逐 bar 手工喂入聚合器。
3. **`feed.resample` / `feed.replay`**：离线数据编排，在喂给 `run_backtest` 之前就把多个频率的 feed 准备好。

## 怎么选

| 路径 | 适用 | 回测 | 实盘 | 高周期进历史与指标 | 日线时间戳口径 |
| --- | --- | --- | --- | --- | --- |
| `subscribe_bars`（推荐） | 策略内声明式订阅多周期 | 支持 | 支持 | 进 `get_history(freq=)` / `self.I(freq=)` | 交易日收盘时刻 |
| `BarGenerator` | 策略内手工聚合，不想动引擎配置 | 支持 | 支持 | 不进历史/指标，只有聚合后回调收到的那一根 | 交易日收盘时刻（与 `subscribe_bars` 同一聚合器） |
| `feed.resample` / `feed.replay` | 离线数据编排，喂给 `run_backtest` 之前就要多频率 feed | 仅回测 | 不支持 | 需要伪标的拼接，用户自己打时间戳 | 用户自定 |

## `subscribe_bars` 用法

在策略 `__init__` 里声明订阅（只能在 `__init__` 里调用，引擎启动后订阅表即冻结）：

```python
from akquant import Bar, Strategy


class FiveMinuteTrend(Strategy):
    """按 5 分钟窗口打印近 3 根收盘价, 用来展示 subscribe_bars 的实盘闭合时机."""

    def __init__(self) -> None:
        super().__init__()
        # 不传 session_windows：按品种时段表的交易块切分
        self.subscribe_bars("5min")
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

- `subscribe_bars(freq, callback=None, symbols=None, *, session_windows=None)`：声明订阅。`freq` 支持整数分钟 `"Nmin"`、整数小时 `"Nh"`、日线 `"1d"`。`symbols=None` 表示全部标的。不传 `session_windows`（推荐）时按品种时段表的交易块切分：开盘集合竞价并入上午块（A 股 09:15～11:30），收盘集合竞价与下午连续竞价同块（13:00～15:00）；日内窗口标签不超过块结束（A 股 1h 为 10:00、11:00、11:30、14:00、15:00）；日线窗口按交易日切分（夜盘归下一交易日），标签为交易日收盘时刻。`session_windows` 只在需要覆盖时段表时手写。
- `callback`：省略时闭合的窗口 bar 会回调策略的 `on_window_bar(bar)`；传入具体函数（如示例 14 的 `on_daily`）则改由该函数接收，`on_window_bar` 不再触发该条订阅。
- `on_window_bar(bar)`：默认窗口回调，`bar.freq` 是该 bar 所属的周期标签（如 `"5min"`、`"1d"`）。
- `current_window(symbol, freq)`：查看某标的当前正在形成、尚未闭合的窗口快照（只能在行情回调内调用），不会触发回调。
- `get_history(count, symbol, field, freq=)` / `get_history_multi(...)` / `self.I(..., freq=)`：`freq` 传已订阅的窗口周期标签，取的是该周期的历史序列；在窗口回调内可省略 `freq`（自动按当前回调定档）。

## 闭合规则：即时 vs 延迟

窗口何时闭合取决于**基础数据周期**（`self.freq`）是否已知：

- **基础周期已知**（回测传 `run_backtest(freq=...)`，或实盘/replay 行情网关声明 `metadata["freq"]`，见示例 71）：窗口在归属它的最后一根基础 bar 的 `on_bar` 之后**同一步**闭合并派发（零延迟）。
- **基础周期未知**（例如回测直接传一个纯 bar 的 `DataFrame`，没有传 `freq=`，也不是 `list[Tick]` 输入）：引擎无法判断"下一根基础 bar 是否已经跨入新窗口"，因此窗口要等**下一根落入新窗口的基础 bar 到达时**才闭合（晚一根），并在配置阶段打印一次性 `WARNING`。交易日的最后一批窗口（日线窗口、收盘所在块的日内窗口）不必等到次日：日终结算前闭合，并在 `on_after_trading` 之前派发。时段切分只负责正确的交易块边界，不能让窗口提前到即时闭合——即时闭合只看基础周期是否已知。示例 14 传了 `run_backtest(freq="1min")`，属于即时闭合。

## 限制

- `freq` 仅支持整数分钟（`"Nmin"`）、整数小时（`"Nh"`）与日线（`"1d"`），不支持秒级、周线、月线；这些场景请改用 `feed.resample`。
- 窗口周期必须严格大于基础数据周期，否则 `subscribe_bars` 在引擎配置阶段报 `ValueError`。
- 同一周期、标的范围有重叠的多条订阅，`session_windows` 必须完全一致，否则报 `ValueError`（同周期+同 `session_windows`，一条全标的通用回调 + 一条特定标的专属回调是允许的，两个回调都会触发）。
- `extra` 字段不参与窗口聚合。
- 窗口 bar **不会**被撮合引擎用于成交判断——它只是信息，策略仍需在 `on_bar`（或其他基础行情回调）里正常下单。
- 封闭的 bar 一律打区间终点戳 E 并带 `interval_start = S`，按右闭区间 `(S, E]` 分桶（回测 tick 聚合、CTP/klinedata 网关合成 bar 同一口径），多周期聚合不会因时间戳口径错位。
- replay 的 `gateway_options={"freq": ...}` 只声明基础周期（`self.freq` 与窗口即时闭合），不给回放的 bar 填 `interval_start`；需要区间起点时请在 `Bar(..., interval_start=...)` 里自带。
- 同一标的在同一时刻同时喂入两种周期的外部 bar（如 1min 与 5min）时，1min bar 上下的单可能被同刻的 5min bar 按更早的区间撮合。请只喂一种基础周期，高周期用 `subscribe_bars` 聚合。
- 会话结束（回测结束、实盘/replay 会话终止）时，引擎会 flush 尚未闭合的尾部窗口并触发一次回调；此时下的单在回测里不会再有机会成交。若之后 `save_checkpoint()` 再从该快照 `run_from_checkpoint()` 续跑，尾部区间会重新形成一根同标签的窗口 bar（历史序列里会出现两根标签相同的 bar，这是已知且刻意保留的行为，不是 bug）。

## 从伪标的迁移

引擎原生多周期上线前，[examples/14_multi_frequency.py](https://github.com/akfamily/akquant/blob/main/examples/14_multi_frequency.py) 靠「伪标的」拼日线：把同一份分钟数据手工 `resample` 成日线，起一个 `000001.SZ_1D` 的假代码喂给引擎，策略里再按 `bar.symbol` 分流处理两条序列，还需要手工把时间戳往后打 15 小时来避免与分钟数据的时间戳冲突。现在改为：

```python
self.subscribe_bars("1d", callback=self.on_daily)
self.daily_sma = self.I(
    aq.SMA(self.ma_window), name="daily_sma", source="close", freq="1d"
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
)
```

推荐不传 `session_windows`：按品种时段表的交易块切分，与 `subscribe_bars` 逐根一致；手写 `session_windows` 会走旧实现，不与引擎窗口对齐。

### 当前落地（最小可用）

- `BasePandasFeedAdapter` 已支持：
  - `resample(freq, agg=None, label="right", closed="right", emit_partial=True)`
  - `replay(freq, align="session", emit_partial=False, agg=None, label="right", closed="right", session_windows=None, sessions=None, trading_calendar=None, asset_type=None, base_interval_min=None)`
- 两者均返回可直接传入 `run_backtest(data=...)` 的适配器对象。
- `replay(align="session")`（不传 `session_windows`）与 `replay(align="day", day_mode="trading")` 用引擎 `subscribe_bars` 同一个窗口聚合器切分，对同一数据、同一时段表与 `subscribe_bars`、`BarGenerator` 产出相同的窗口：
  - 日内窗口（`Nmin`/`Nh`）按（交易日，交易块）切分，标签为本地时钟 ceil 到周期但不超过块结束（A 股 1h：10:00、11:00、11:30、14:00、15:00）。
  - 日线窗口（`1D`）按交易日切分（夜盘归下一交易日），标签为交易日收盘时刻。
  - `sessions`、`trading_calendar`、`asset_type` 与 `run_backtest` 同义（默认股票）；`base_interval_min` 是源数据周期（分钟），不传时按相邻时间戳间隔的众数推断。
  - OHLCV 取聚合器结果；`agg` 里的其他列（或改写了默认聚合方式的列）按聚合器给每行分配的窗口分组计算。
  - `emit_partial=False` 时，数据未到达标签的窗口若是该交易日最后一个窗口（或数据末尾），视为未闭合尾部丢弃。
  - 默认 `align="session"` 在中国时区（请求 `timezone` 缺省即 `Asia/Shanghai`）下按 A 股交易块切分，午休、收盘后的数据不会与交易时段混在一个窗口里。非中国市场或 24 小时数据须显式传 `FeedSlice(timezone=...)`，并按需传 `sessions=`、`asset_type=`（如期货夜盘 `sessions={"au": "CN_FUTURES_NIGHT_0230"}, asset_type="futures"`）。
  - 聚合前会剔除 open/high/low/close 任一为 NaN 的行，volume 的 NaN 按 0 计。
- 交易块截断只对默认 `label="right", closed="right"` 且 `freq` 为 `Nmin`/`Nh`/`1D` 时生效；其余情况沿用 pandas `resample` 的旧实现（按本地日期分区）。
- 可通过 `session_windows` 手写日内交易时段分区（例如午休前后分段）；终点早于起点表示跨午夜（如 `("21:00", "02:30")`）。手写时段走旧实现，不与引擎窗口逐根对齐。
- `align` 当前支持三种语义：
  - `session`：按交易日与交易块分区，或按手写 `session_windows` 分区。
  - `day`：按日分区，不接收 `session_windows`，并支持 `day_mode`。
  - `global`：按全局时间轴聚合，不按交易日切段。
- `day_mode` 当前支持：
  - `trading`：按品种时段表的交易日切分（夜盘归下一交易日）。
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
代码本身不需要区分当前是在回测还是实盘运行。

`BarGenerator` 底层就是引擎 `subscribe_bars` 用的同一个 Rust 窗口聚合器
（也可直接用 `akquant.WindowAggregator`），对同一数据、同一时段表产出相同的
窗口：日内窗口按（交易日，交易块）切分，标签为本地时钟 ceil 到周期但不超过
交易块结束（A 股 1h：10:00、11:00、11:30、14:00、15:00）；日线窗口按交易日
切分（夜盘归下一交易日），标签为交易日收盘时刻。窗口 bar 带 `freq` 与
`interval_start`。

与 `subscribe_bars` 的关键差异：`BarGenerator` 聚合出的 bar **不会**进入
`get_history`/增量指标体系，只有传给回调函数的那一根可用。

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
  `BarGenerator(cb, 1, "hour")`、`BarGenerator(cb, 1, "day")`）；`"day"` 只支持
  `window=1`。
- `timezone`（默认 `"Asia/Shanghai"`）、`sessions`、`trading_calendar`、
  `asset_type`（默认股票，也可传 `{symbol: asset_type}`，按完整代码匹配）与
  `run_backtest` 同义，决定品种的时段表与交易日。非中国市场（美股、加密货币等）
  必须显式传 `timezone`（如 `"America/New_York"`、`"UTC"`），否则会按 A 股时段表
  切分窗口；非中国时区默认全天连续。
- `base_interval_min` 是喂入 bar 的周期（分钟）。给定时 bar 区间终点到达窗口
  标签即闭合回调（如 11:30 那根 1 分钟 bar 到达就闭合 11:30 的窗口）；不给时
  等下一根落入新窗口的 bar 到达才闭合。
- `session_windows` 手写交易时段，覆盖时段表；一般不需要传，默认按品种时段表的交易块切分。例如
  `session_windows=[("09:30", "11:30"), ("13:00", "15:00")]`；终点早于起点表示
  跨午夜。
- `current(symbol)` 可查看某标的当前正在形成、尚未闭合的窗口快照，不会触发
  回调，适合盘中展示。
- 多标的相互独立聚合，互不影响。

完整可运行示例见
[examples/65_bar_generator.py](https://github.com/akfamily/akquant/blob/main/examples/65_bar_generator.py)。
