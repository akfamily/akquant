# RFC:TradingView 式指标处理(声明式指标层与四层路线)

> **状态**:提案(Proposed) · **日期**:2026-09-20 · **范围**:策略侧**指标声明/计算/上报**三件事的统一。P0 实施**声明式指标层**(`Strategy.I()` + series 回溯 + 自动绘图绑定),并**彻底收敛**旧的 `indicator_mode` 开关与两个 `register_*` 方法。**允许破坏性变更(直接删除旧入口,不留兼容垫片)**,与 [viz-namespace-and-lwc-review-rfc.md](viz-namespace-and-lwc-review-rfc.md)、[timer-api-rfc.md](timer-api-rfc.md)、[hooks-rfc.md](hooks-rfc.md) 的处理方式一致。
>
> 对标:**TradingView Pine Script**(`ta.*` 内建有状态指标 + `plot()` 声明式绘制 + `series[n]` 回溯 + `barstate` 实时/确认语义)、**backtesting.py**(`self.I()` 工厂方法)、**backtrader**(Lines 对象与 `[0]/[-1]` 索引)、**Freqtrade**(`populate_indicators` 向量化 + FreqUI 消费)。
>
> 与引擎**正交**:本 RFC 的 P0 只改 **Python 策略层 API** 与指标上报时机,不碰 Rust 撮合、不改回测数值。P2(实时未闭合 bar)会动 Rust 指标状态,届时另开 spec 评估 `__engine_rule_version__`。

---

## 0. 背景与动机

用户希望「像 TradingView 那样处理指标」。拆解后,诉求落在**四个层**上,且四层**都要**:

| 层 | 诉求 | Pine 对应物 |
|---|---|---|
| **L1 写法层** | 声明一次就自动更新、自动绘图,能回溯历史值 | `ta.sma(close,14)` + `plot()` + `sma[1]` |
| **L2 实时语义层** | 未收盘的 bar 上指标随 tick 实时刷新,收盘后确认为最终值 | `barstate.isrealtime` / `barstate.isconfirmed` |
| **L3 指标独立于策略** | 指标不写死在策略里,一个会话可挂 N 个只画图不交易的单元 | indicator 脚本 vs strategy 脚本 |
| **L4 渲染层** | 浏览器里就是 TradingView 那张图 | TradingView 图表 |

### 0.1 现状盘点:能力其实已经很强,但三套机制互不相通

**计算层有三种模式,且前两种被一个开关互斥**:

- **precompute**:`register_precomputed_indicator()` + `indicator.py::Indicator`,整段 DataFrame 向量算完缓存,`get_value(symbol, ts)` 做 asof 查询。**实盘用不了**——实盘没有完整 DataFrame。
- **incremental**:`register_incremental_indicator()`(`strategy.py`),有状态对象,框架在 `on_bar` / `on_tick` / `on_window_bar` **之前**自动 `update()`。已支持 per-symbol 实例化、`input_mode`(`source`/`hl`/`hlc`/`ohlc`/`close_volume`)、`freq=` 绑定 `subscribe_bars` 周期、`warmup_bars` 预热引导。
- **手工**:`on_bar` 里 `get_history(N)` 取 numpy 再 `aq.talib.EMA(arr,20)[-1]`,每根 bar 全量 O(N) 重算。

前两者由 `indicator_mode`(`StrategyRuntimeConfig` 的字段,默认 `"precompute"`)**互斥**切换。**Pine 里没有任何对应物**——它是历史包袱,还顺带制造了「注册方法与模式不匹配就报错」这一类纯仪式性错误。

**记录/绘图层与计算层完全脱钩**:`record_indicator()` 必须在每根 bar 手工调用,且 `pane` / `render_type` / `color` 等元数据**每次重传**。计算出一个值和把它画出来,在 AKQuant 里是两件毫不相干的事——而在 Pine 里 `plot(sma)` 就是一件事。

**无 series 回溯**:`IncrementalIndicatorBinding` 只暴露 `.value`(当前值)。要判断金叉这种最基本的形态,用户得自己维护 `deque`。

### 0.2 决定成本的关键事实:增量计算引擎早就有了

`src/indicators/*.rs` 已实现并导出 **103 个有状态增量指标类**到 Python 顶层:

```python
sma = aq.SMA(14)
sma.update(101.2)   # -> Option<f64>
sma.value           # 当前值
sma.is_ready        # 是否已满窗
```

覆盖 `EMA` / `RSI` / `MACD` / `BollingerBands` / `ATR` / `STOCH` / `SAR` / `KAMA` / `T3` / `ADX` / `OBV` / `WILLR` 等。**这就是 Pine `ta.*` 的等价物,且是 Rust 实现。**

所以本 RFC **不需要新建计算引擎**,缺的只是三层胶水:**门面**(一行声明)、**序列回溯**(`[n]`)、**绘图绑定**(声明即上报)。

---

## 1. 目标与非目标

**目标**

- 定义四层**共用**的指标声明契约,一次定死,避免 L2/L3/L4 各自开工时契约漂移。
- P0 交付 **L1 声明式指标层**:`Strategy.I()` 一行声明 = 自动增量更新 + 自动上报绘图 + `[0]/[1]` 回溯。
- P0 **彻底收敛**旧入口:删 `indicator_mode` 开关与两个 `register_*` 方法,`self.I()` 成为唯一声明入口。
- 为 L2 预留 `confirmed` 字段位,使 L2 落地时**不必**提升 `STREAM_SCHEMA_VERSION` 的 MAJOR。

**非目标**

- **P0 不做 L2/L3/L4**,本 RFC 对它们只定契约与阶段顺序,各自另开 spec。
- **不替换 `record_indicator()`**:它保留为底层逃生口,用于记录**非指标类**的自定义值(仓位、信号强度、风控中间量),`self.I()` 覆盖不了这些。
- **不动三出口字段契约**:`indicator_recording.py` / `live/_stream_sink.py` / `indicator_stream.py` 的 stream / DataFrame / export 三出口一致性已经建立(整数 pane、`timestamp_ms`、7 值 `render_type` 枚举),本轮只**往里加** `confirmed`。
- **不改 Rust 撮合**,P0 不动 `__engine_rule_version__`。

---

## 2. 四层共用契约

以下五项是 L1 定义、L2/L3/L4 单向消费的**公共契约**。改动它们需要修订本 RFC。

### 2.1 声明体 `IndicatorDeclaration`

一次声明携带**计算**与**绘图**两组信息:

| 组 | 字段 |
|---|---|
| 计算 | `indicator`(实例) / `factory`(工厂)、`source`、`input_mode`、`symbols`、`freq`、`warmup_bars`、`lookback` |
| 绘图 | `plot`、`pane`、`render_type`、`color`、`label`、`unit`、`precision`、`reference_lines`、`scale_group` |
| 多值 | `outputs`——多值指标的分量名元组 |

计算组直接沿用既有 `IncrementalIndicatorRegistration` 的语义(它已经过多周期、tick/bar 双流、warm start 三轮打磨),绘图组直接沿用 `record_indicator()` 的参数名,**不发明新词**。

### 2.2 series 语义

- `ind[0]` = **当前 bar** 的值,与 Pine 一致(Pine 里 `sma[0]` 等价于 `sma`)。
- `ind[1]` = 上一根,`ind[n]` = 前 n 根。
- `.value` 是 `[0]` 的别名,保留以兼容既有写法。
- **越界返回 `None`,不抛异常**——预热期天然越界,抛错会让每个策略都得先写 try。
- 回溯深度由 `lookback` 界定(默认 128),底层是 `deque(maxlen=lookback)`,**有界**:实盘长跑不泄漏。Pine 的 `max_bars_back` 默认 5000,但那是浏览器里的有限历史;实盘是无限流,必须有界。

### 2.3 确认状态 `confirmed`(L2 预留)

指标点事件的 payload **新增 `confirmed: bool`**:

- P0:恒为 `true`(所有指标都在 bar 闭合后计算)。
- L2:未闭合 bar 上的临时点发 `false`,bar 闭合时用同一 `timestamp` 再发一次 `true`。**前端约定:同 `(indicator_key, symbol, timestamp)` 的后来者覆盖先前者。**

**为什么 P0 就要加这个字段**:`STREAM_SCHEMA_VERSION` 是 MAJOR.MINOR 语义,加字段是 MINOR、改语义是 MAJOR。P0 加进去只是 `1.2 → 1.3`;等 L2 再加,前端已经按「无此字段 = 永远确认」写死了逻辑,那时就是破坏性的。

**P0 已落地**:回测 `IndicatorRecorder` 与实盘 `StreamingIndicatorSink` 的 point/snapshot 四个发射点都填 `confirmed`,`to_indicator_message()` 两个分支透出;payload 缺失该字段时默认 `True`(早期事件与第三方 `IndicatorSink` 都只在 bar 闭合后产点,当作已确认是正确语义)。`STREAM_SCHEMA_VERSION` 已升至 `1.3`。

### 2.4 study 挂载点(L3 预留)

`IndicatorDeclaration` 与驱动管线**不依赖 `Strategy` 的交易 API**。L3 的 study 是「只有指标声明、没有交易逻辑」的单元,复用同一个声明体与同一套 `_update_indicators` 驱动,挂在既有的多 slot 策略拓扑上,不新建并行管线。

**L3 已落地(2026-09-21)**:`akquant.Study(Strategy)` 把 `TRADING_API_NAMES` 里的 19 个交易方法全部换成抛 `StudyCannotTradeError`;`run_backtest(studies=[...])` / `run_live(studies=[...])` 经 `study.merge_studies_into_slots()` 并入 `strategies_by_slot`(key 取 `study_id`,省略为类名 snake_case;撞 key 抛 `ValueError`,非 `Study` 抛 `TypeError`);`indicator_df(owner=)` 按归属过滤。spike 先证明了 study 走 `strategies_by_slot` 在回测与实盘 replay 下**零改动即可用**,L3 的增量只是护栏与入口糖。

### 2.5 前端消费格式(L4 预留)

沿用三出口一致契约。多值指标按 `outputs` **展开成多个独立 `indicator_key`**:声明 `self.I(aq.MACD(12,26,9), outputs=("dif","dea","hist"), pane=2)` 产出 `macd.dif` / `macd.dea` / `macd.hist` 三条线。理由:前端的渲染单元是「一条线」,让它去解析元组等于把契约复杂度外推给每一个消费者。

---

## 3. P0 设计:`Strategy.I()`

### 3.1 签名

```python
def I(self, indicator, *, source="close", input_mode="source", symbols=None,
      freq=None, warmup_bars=0, lookback=128, name=None, outputs=None,
      plot=False, pane=None, render_type=None, color=None, label=None,
      unit=None, precision=None, reference_lines=None, scale_group=None,
      factory=None) -> IndicatorBinding
```

用法:

```python
class MaCross(Strategy):
    fast = IntParam(10, ge=2, le=200)
    slow = IntParam(30, ge=3, le=500)

    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars("5min")          # 窗口订阅仍然只能在 __init__

    def on_start(self) -> None:
        # 在 on_start 声明才拿得到 self.params
        self.ema_fast = self.I(aq.EMA(self.params.fast), pane=0, color="#e91e63",
                               label=f"EMA{self.params.fast}")
        self.ema_slow = self.I(aq.EMA(self.params.slow), pane=0, color="#3f51b5")
        self.rsi = self.I(aq.RSI(14), pane=1, reference_lines=[30, 70])
        self.macd = self.I(aq.MACD(12, 26, 9), pane=2, outputs=("dif", "dea", "hist"))
        self.ema5m = self.I(aq.EMA(20), freq="5min")   # 绑定到 5 分钟窗口, 不绘图

    def on_bar(self, bar: Bar) -> None:
        if self.ema_fast[1] <= self.ema_slow[1] and self.ema_fast[0] > self.ema_slow[0]:
            self.buy(symbol=bar.symbol, quantity=100)
        # 无需再写 record_indicator
```

### 3.2 分派规则

**删掉 `indicator_mode` 互斥开关的直接收益:两种模式可在同一策略共存。** `self.I()` 按传入对象分派:

- 对象有 `update()` → **增量路径**(Rust 的 103 个类、用户自写子类)。
- 对象是 `indicator.py::Indicator` 实例 → **预计算路径**,`_prepare_indicators` 在数据加载后向量算完。
- **实盘下走预计算路径要 fail-fast 报错**:实盘拿不到完整 DataFrame,静默返回 NaN 是最坏的失败方式——用户会以为策略在正常运行。

多标的:沿用既有语义,per-symbol 惰性实例化。传实例时按需 deepcopy,或显式传 `factory=` 控制构造。

### 3.3 声明位置:`__init__` 与 `on_start` 都支持,文档推荐 `on_start`

`self.params` 在 `__init__` **之后**才注入(既有 `examples/02_parameter_optimization.py` 正是在 `on_start` 里用 `self.params.fast_period` 声明指标)。因此:

- 参数化指标(`optimize.py` 网格搜索要扫的那些)**必须**在 `on_start` 声明;
- `__init__` 只适合字面量参数;
- `subscribe_bars` 仍**只能**在 `__init__`(引擎启动前要冻结订阅表),`self.I(freq=...)` 校验该周期已订阅,沿用既有报错文案指向 `subscribe_bars`。

### 3.4 自动上报的三条默认值

在既有 `_update_incremental_indicators` 内,指标 `update()` 之后**立即**上报——即 `on_bar` **之前**。这样用户在 `on_bar` 里手工 `record_indicator()` 的点与自动上报的点落在同一个 `flush_stream_snapshot()` 周期里,`timestamp` 一致,前端收到的是一个完整快照。

1. **默认不上报**(`plot=False`)。传了任一绘图参数(`pane`/`color`/`label`/`render_type`/`reference_lines`/`scale_group`)或显式 `plot=True` 才上报。
   *理由*:与 Pine 一致——`ta.sma()` 只计算,`plot()` 才画。且实盘 `StreamingIndicatorSink` 是**每点一个事件**,多标的 × 多指标默认全开会淹没前端链路。
2. **`is_ready=False` 时不上报**。Pine 预热期是 `na`,不画。避免前端收到一长串 NaN 还得自己过滤。
3. **`lookback` 默认 128**。足够绝大多数形态判断(交叉、背离、N 日新高),环形缓冲有界。

ready 判定统一为 `value is not None`,对象有 `is_ready` 属性时优先用它。**注意**:`MACD` / `BollingerBands` 这类复合指标**没有** `is_ready` getter(只有 `SMA`/`EMA`/`WMA` 等基础类有),不能假定它存在。

---

## 4. 破坏性变更清单

| 删除 | 替代 |
|---|---|
| `Strategy.indicator_mode` 属性 | 无需替代——`self.I()` 按对象类型自动分派 |
| `Strategy.register_precomputed_indicator()` | `self.I(Indicator(...))` |
| `Strategy.register_incremental_indicator()` | `self.I(aq.EMA(20), ...)` |
| `StrategyRuntimeConfig.indicator_mode` 字段 | 无。该配置项失去意义 |
| `BacktestConfig.indicator_mode` | 同上 |

**保留**:`record_indicator()`(非指标类自定义值的逃生口)、`Indicator` / `IndicatorSet`(`self.I()` 仍接受它们走预计算路径)。

### 4.1 外部消费者影响

`vendor/indicator_server_g/`(Pine 风格 DSL 的指标服务,**独立仓库、本仓 gitignore**)有 5 个策略文件使用 `indicator_mode` + `register_*` 旧 API:

```
examples/akquant_strategy_multi_indicators.py
examples/test_akquant_strategy.py
examples/test_strategy_indicator_same_file.py
indicators/boll_ema_macd_rsi_strategy.py
indicators/boll_ema_macd_rsi_record_strategy.py
```

**不在本仓修改**,但升级 AKQuant 版本时必须同步迁移,否则它会直接起不来。该项目面向用户的 DSL 用**字符串 pane**,与核心的整数 pane 分属不同层,迁移时不要顺手「统一」掉。

---

## 5. 阶段与依赖

依赖是**单向**的:L1 是地基,L2/L3/L4 都只消费 L1 的声明体,彼此之间无依赖。

```
        ┌─────────────────────────────┐
        │  L1 声明式指标层  (P0, 本轮) │
        └──────────────┬──────────────┘
           ┌───────────┼───────────┐
           ▼           ▼           ▼
      L2 实时语义   L3 study    L4 LWC 实时图
      (动 Rust)     (中小)       (中)
```

建议顺序:**L1 → L4 → L3 → L2**。L4 紧跟 L1 能最快形成「声明一行 → 浏览器里多一条线」的可见闭环;L2 最难且收益最依赖前端是否已能消费 `confirmed`,放最后。

### 5.1 L2 的核心难点(先行记录,避免开工时才发现)

未闭合 bar 上用 tick 驱动指标,**不能直接 `update()`**——那会把临时值写进增量状态,bar 真正闭合时状态已被污染,后续所有值全错。两条路:

- **`peek(value)`**:Rust 侧给每个增量类加「算出结果但不改状态」的方法。性能最好,但要为 103 个类逐个实现,且复合指标(MACD 内含三个 EMA)的 peek 要递归。
- **`clone() + update()`**:利用现有 `#[derive(Clone)]`,每个 tick 克隆一份状态算完丢弃。零新代码,但高频 tick 下每 tick 一次堆分配。

倾向 `clone + update` 起步(103 个类零改动),实测有瓶颈再对热点指标做 `peek`。**开工前须实测**克隆开销在真实 tick 频率下的量级。

**L2 Tier A 已落地(2026-09-21),且上面这段的前提被 spike 推翻**:Rust 侧确有 `#[derive(Clone)]`,但 pyo3 **不会**因此暴露 `__copy__`,Python 侧 `copy` / `deepcopy` / `pickle` 对 103 个类一律 TypeError——「零改动」不存在,两条路都要动 Rust。最终做法是**一个** Rust pyfunction `clone_indicator(obj)`(72 个类型的 `cast` 链由宏生成),Python 侧 `peek = clone → update → value`;用户自写 Python 指标走 `copy.deepcopy`。

落地范围刻意收在**窗口周期指标**(Tier A):`self.I(..., freq="5min", intrabar=True)` 后每根基础 bar 用 `ctx.current_window()` 的未闭合快照试算临时值,窗口闭合真 `update()`。这条路零新增 Rust 数据管道——未闭合快照的 `timestamp` 就是窗口将来闭合的标签(基础周期已知时),临时点与确认点天然同 time。`[0]` 语义对齐 Pine(临时值占 `[0]`,确认序列后移),新增 `binding.confirmed`。临时点只走流、不进 DataFrame / export。

**Tier B 也已落地(2026-09-21),且绕开了上面的两个前置问题**:① 不需要 Rust 暴露 `active_bars`——Python 侧从 tick 自己累一根 `PartialBar`(spike 证实双流回测里同一分钟的 tick 先到、闭合 bar 后到,bar 闭合即重置),H/L 也是真实的;临时点标签用聚合器的区间末打戳公式 `end_stamp_label = (ts // 间隔 + 1) * 间隔 − 1ns` 直接算,跨午休/跨日成立,并在 bar 真闭合时与 `bar.timestamp` 比对自校验,不符则该 symbol 停发临时点。② 不需要裁定破坏性语义——tick 从 `update` 改为 `peek` **只对显式 `intrabar=True` 的基础周期指标生效**,默认 False 时 tick 照旧 `update()`,纯 tick 策略零变化。**已知尾部现象**:回测数据末尾最后一笔 tick 开出的 bar 永不闭合(没有下一笔 tick 触发),它的临时点没有确认点;实盘下一笔 tick 就会闭合。

实施中挖出两个 bug:① **跳窗**:引擎顺序是 `on_bar` 先、`on_window_bar` 后,在闭合那根基础 bar 上 `current_window()` 已是下一窗口而上一窗口的确认值还没入账,照常试算会跳过一整根;修法是按快照标签跟踪,标签变了但确认未到就跳过这一帧。② **L1 遗留**:窗口指标确认点的时间戳取自 `current_bar`(基础 bar),`on_window_bar_event` 里 `_update_incremental_indicators` 跑在 `current_bar = 窗口 bar` 之前,正常闭合时两者恰好相等而掩盖,尾部 flush 时窗口标签(ceil)≠最后一根基础 bar 就露馅;修法是上报时显式传事件自身的 `timestamp`。

---

## 6. 验证

- 单元测试新增 `tests/test_indicator_declaration.py`,覆盖:`[n]` 回溯与越界返回 `None`、多值 `outputs` 拆分、`plot` 触发条件、预热期不上报、per-symbol 状态隔离、`freq=` 绑定窗口周期、实盘传预计算 `Indicator` 时 fail-fast、pickle / warm start 恢复(`is_restored` 分支语义必须保留)。
- **golden 回归**:确认删除 `indicator_mode` 没有改变任何回测数值。
- **examples 实跑 exit 0**(发布级要求,不是静态检查):受影响的 6 个示例 + 新增 `72_declarative_indicators.py`。
- 端到端手验:`62_indicator_streaming_demo.py` / `64_indicator_live_web.py` 改用 `self.I()` 后,浏览器里指标线照常出现,payload 带 `confirmed: true`。

---

## 7. 关联

- [viz-namespace-and-lwc-review-rfc.md](viz-namespace-and-lwc-review-rfc.md) —— L4 的 LWC 静态复盘已落地,实时推送是它的延伸
- [logging-rfc.md](logging-rfc.md)、[columnar-rfc.md](columnar-rfc.md) —— 同目录的其它长期方向
