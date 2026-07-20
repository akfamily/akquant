# RFC:列式原生数据层与引擎重构

> **状态**:草案(Draft) · **日期**:2026-07-18 · **范围**:AKQuant 核心(Rust 引擎 + Python 数据层),不涉及向后兼容承诺
>
> 本文是一次「不考虑兼容、以量化基础设施最佳实践为准」的整体重构设计,作为分期实施与评审的跟踪基线。对标实现:**nautilus_trader**(Rust+Python 事件驱动、回测/实盘同构、ParquetDataCatalog)、**ArcticDB**(版本化双时态存储)、**vectorbt**(列式向量化)、**Apache Arrow / DataFusion**(内存格式 + 流式计算)。

---

## 0. 背景与动机

当前数据链路为:

```
用户输入(pd.DataFrame / Dict / List[Bar] / DataFeed / DataFeedAdapter)
  → run_backtest 分派(isinstance)
  → prepare_dataframe()(pandas 时区归一)
  → df_to_arrays()(pandas → numpy)
  → feed.add_arrays(*arrays)   ← Rust 边界(收 numpy)
  → 引擎物化 Vec<Bar>(AoS,一行一个结构体)
```

三个根因问题:

1. **归一化逻辑重复**:`utils.load_bar_from_df`、`utils.df_to_arrays`、`utils.prepare_dataframe`、`feed_adapter.BasePandasFeedAdapter.normalize` 四处各写列名解析/时区/dtype,已出现细节漂移。
2. **介质割裂**:catalog 写用 pandas、factor 读用 polars、backtest 读用 pandas;同一份 parquet 三条路径三种引擎。
3. **AoS 内存布局**:引擎物化 `Vec<Bar>`,指标/因子跨 struct 跳读 → cache miss、无法 SIMD、数据集不能大于内存。

## 1. 目标与非目标

**目标**

- 全链路列式原生(SoA),ingest→撮合→指标→输出零拷贝,**永不物化 `Vec<Bar>`**。
- Catalog 成为唯一真相源:Arrow/Parquet、Hive 分区、mmap、流式、版本化。
- **Out-of-core**:验收目标 = 全 A 股分钟线数据集(> 内存)可跑通。
- 回测/实盘同构:策略代码零改切换 backtest ↔ live。
- 时点正确(point-in-time / 双时态):结构性杜绝复权与成分股的未来函数。
- 确定性重放:逐位可复现。

**非目标**

- 不承诺 API 向后兼容(可重命名/删除旧入口)。
- 不移除 pandas:pandas 退居**报表/研究/呈现边缘层**(plot、history_df、ml、analysis),核心路径不再依赖它。
- 引擎内部「行式 → 列式」是本 RFC 范围;更激进的「全 DataFusion 化查询引擎」不在本期。

## 2. 设计原则

| 原则 | 含义 |
|---|---|
| 列式原生 | 存储与批量计算 SoA;事件循环游标串行迭代 |
| 唯一真相源 | Catalog 是唯一入口,回测/实盘/研究都从它流式读 |
| 回测实盘同构 | 差异仅在注入的 DataEngine/ExecutionClient/Clock |
| 时点正确 | 复权因子、成分股、基本面按 sim 时点应用 |
| 确定性 | 稳定时序 tie-break + seeded RNG + catalog 版本绑定 |
| Out-of-core | mmap + 流式 RecordBatch,数据可大于内存 |

## 3. 规范数据模型(类型系统)

不再只有 OHLCV。每种数据是一个 Arrow schema,公共字段:`ts_event`(ns UTC)、`ts_init`(接收/生成时间)、`sequence`(同刻 tie-break)、`instrument_id`(dictionary 编码)。

```
Bar          : ts_event, ts_init, seq, iid, open, high, low, close, volume, open_interest?
QuoteTick    : ts_event, ts_init, seq, iid, bid, ask, bid_size, ask_size
TradeTick    : ts_event, ts_init, seq, iid, price, size, aggressor_side, trade_id
OrderBookΔ   : ts_event, ts_init, seq, iid, action, side, price, size, level
Instrument   : iid, asset_type, tick_size, lot_size, multiplier, margin_rate, expiry, ...
Adjustment   : iid, ex_date, knowledge_time, factor, cash_dividend, split_ratio      ← 双时态
Membership   : universe_id, iid, start, end, knowledge_time                          ← 防幸存者偏差
```

要点:

- **合约元数据 `Instrument` 独立成一等数据类型**——期货/期权的 tick_size/multiplier/margin/expiry 现散落在 `config.py`,应沉淀入 Catalog,供撮合、保证金、Greeks 共用。
- **`Adjustment` / `Membership` 为双时态表**,带 `knowledge_time`,是 §8 时点正确性的数据基础。

## 4. 存储层 · Catalog(唯一真相源)

对标 nautilus `ParquetDataCatalog` + ArcticDB 版本化:

```
{root}/
  data/{data_type}/{iid}/{year=YYYY}/{month=MM}/part-*.parquet   # Hive 分区, zstd
  instruments/instruments.parquet
  adjustments/{iid}.parquet
  _manifest/manifest.parquet     # iid × type × [t_start, t_end] → file, version
  _versions/{vN}/...             # append-only 不可变, 支持 "as-of catalog vN" 复现
```

- **读**:`memmap2` 映射 → 流式 `RecordBatchReader`;时间/标的过滤靠 parquet row-group 统计 + 谓词下推,不全量载入。
- **写**:canonical Arrow → 按 `(iid, ts)` 排序 → 分区落盘;文件 append-only 不可变。
- **版本/manifest**:回测绑定 catalog version → 数据可复现。
- 替换现状:`ParquetDataCatalog.write` 的 pandas 路径、`{root}/{symbol}.parquet` 扁平结构、factor/backtest 各读各的割裂。

## 5. 摄入漏斗(Python 薄适配,唯一归一化点)

```python
def normalize(source, schema=Bar) -> pa.RecordBatchReader:   # 流式, 非全量 Table
    match source:
        case pl.DataFrame() | pl.LazyFrame(): reader = source ... .to_arrow()
        case pa.Table() | pa.RecordBatchReader(): reader = source
        case pd.DataFrame(): reader = pa.Table.from_pandas(source)   # pandas 唯一入口
        case str() | Path(): reader = pl.scan_parquet(source) ...
        case list() if bars: reader = _bars_to_arrow(source)
    return _canonicalize(reader, schema)   # 列名/中文列/tz→UTC/dtype/symbol dict,一处实现
```

**删除清单(4 合 1)**:`utils.load_bar_from_df`、`utils.df_to_arrays`、`utils.prepare_dataframe`、`feed_adapter.BasePandasFeedAdapter.normalize` → 全部并入 `normalize`。中文列名(`日期/开盘/股票代码`)、naive→`Asia/Shanghai`→UTC 规则**只在此定义一次**。

## 6. Rust 核心 · 列式引擎(B 期主战场)

### 6.1 数据持有(SoA,Arrow 原生)

```rust
struct BarColumns {                 // 一个 instrument 的一段 chunk, 可 mmap
    ts_event: Int64Array,
    open: Float64Array, high: Float64Array,
    low: Float64Array,  close: Float64Array,
    volume: Float64Array,
}
```

引擎直接持有从 Catalog 流入的 `RecordBatch`,**不拆成 Bar 结构体**。Python↔Rust 走 **Arrow C Data Interface(PyCapsule)**,不锁 arrow-rs 与 pyarrow 版本。

### 6.2 合并时钟(替换 `feed.sort()` + Vec 迭代)

```rust
struct MergedClock {                // k 路归并
    cursors: Vec<Cursor>,           // 每 instrument 一个游标
    heap: BinaryHeap<(ts_event, sequence, iid, cursor_id)>,
}
// pop → 产出下一事件, 零分配
```

### 6.3 零分配 Bar 视图(`Copy`)

```rust
#[derive(Clone, Copy)]
struct BarRef<'a> { cols: &'a BarColumns, i: usize }
impl<'a> BarRef<'a> {
    fn close(&self) -> f64 { self.cols.close.value(self.i) }   // 直接读列 buffer
    // open / high / low / volume / ts_event 同理
}
```

- **撮合/策略循环**:串行游标推进 → 收益是 cache 局部性(顺序读同列)。
- **指标/因子/组合估值**:对列切片向量化 → SIMD + rayon 多核。
- **Python 边界**:`on_bar` 仍收一个 `Bar` pyclass,但它是「持索引懒读列」的薄视图,per-event 仅一次轻量构造;引擎内部始终列式。

**受影响的 Rust 模块**(现读 `Bar`,需改读 `BarRef`/列,实测 24 文件):
`execution/{stock,futures,simulated}.rs`、`data/{feed,aggregator,client}.rs`、`pipeline/stages/*.rs`、`history.rs`、`engine/core.rs`、`model/{types,market_data}.rs`、`event.rs` 等。

## 7. 回测/实盘同构

```rust
trait DataEngine     { fn next(&mut self) -> Option<Event>; }   // Backtest: Arrow重放 / Live: gateway推送
trait ExecutionClient { fn submit(&mut self, order); ... }      // Sim: 撮合器 / Live: 券商网关
trait Clock          { fn now(&self) -> ts; fn timers(); }      // TestClock(随事件) / LiveClock(墙钟)
```

- 策略只依赖 `Portfolio`/`OrderManager` facade,**backtest↔live 唯一差异是注入哪套三件套**。
- 现有 `gateway/`(ctp/miniqmt/ptrade + middleware/qmf 插件)接到 `ExecutionClient`/`LiveDataEngine`;`SimulatedExchange` 包住现有 `custom_matchers`。
- 收益:策略回测验证后**零改上实盘**——对含两个实盘插件的仓库价值最大。

## 8. 时点正确性 · 双时态(E 期,golden 会变更)

**当前问题**:全局 `qfq`(前复权)把未来拆分/分红烘焙进历史价 → 结构性未来函数;成分股用当前列表 → 幸存者偏差。

**改法**:

- 存**原始价 + `Adjustment` 因子序列**,读取时按 sim 当前时间应用(只用 `knowledge_time ≤ now` 的因子),回测走到哪复权到哪。
- `Membership` 双时态 → sim 时刻 T 只见 T 时点成分股。
- 基本面/信号带 `knowledge_time` → T 时刻只见 T 前已知数据。

**代价即收益**:回测数值相对现状**会变化**,但变化后才是正确的。E 期一次性修正并重锁 golden。

## 9. 确定性与复现

稳定排序 tie-break `(ts_event, sequence, iid)` + seeded RNG + catalog version 绑定 → 逐位可复现。`__engine_rule_version__` 随行为变更大版本跳变;golden 在 E 期一次性重生成后锁定。

## 10. Out-of-core 与并行

- **因子/指标**:rayon 跨标的、跨列并行 + SIMD。
- **参数寻优**(`optimize.py` 现为多进程):worker 共享 **mmap 的 catalog**,进程间零拷贝 → 内存占用大降。
- **组合估值**:持仓列向量化。

Out-of-core 对下游的反向约束:

| 下游 | 现状 | 改动 |
|---|---|---|
| `strategy_history` | 全历史在内存 | 有界窗口/流式切片(签名不变,语义收敛) |
| `plot/report` | 载入整条序列 | 大数据集降采样/分块渲染 |
| `optimize.py` | 每 worker 各载数据 | 共享 mmap catalog(收益) |
| `factor` | 已 polars lazy | 天然契合,基本不改 |

## 11. Python API 终态

```python
cat = Catalog(root)                                  # 唯一入口
cat.write(normalize(df_or_pl_or_parquet))            # 任意源 → 规范 Arrow
res = run_backtest(catalog=cat, instruments=[...],   # 从 catalog 流式读, out-of-core
                   strategy=MyStrategy, ...)
res = run_backtest(data=pl_df, strategy=...)         # 快速测试仍收内存 DF(漏斗→临时 Arrow)
```

- `data=` 保留内存 DF 便利路径(pl/pd/pa 平权);pandas 降为 `akquant[pandas]` 可选 extra。
- 策略 `on_bar/on_tick` 人体工学不变,`bar` 变为列式视图。

## 12. 影响面矩阵与「5 缝冻结」

实测爆炸半径:Rust 24 文件引用 `Bar`;Python 16 文件读 Bar 字段;25 模块 import pandas。破坏面通过冻结 5 条接口缝收敛:

| # | 缝(契约) | 依赖方 | 冻结策略 |
|---|---|---|---|
| 1 | **Bar 属性接口** `.open/.high/.low/.close/.volume/.timestamp/.symbol/.extra` | 16 个 Python 文件 + 用户策略 | 新 `BarRef` 保留同名 accessor,懒读列 → 下游零改 |
| 2 | **`DataFeed` 符号** | 两个插件 `from akquant import DataFeed` | 保留符号(内部换实现)或适配 |
| 3 | **`gateway.*`**(broker_models/trader_base/registry/protocols) | 两个插件**全部**依赖 | A–C 期不碰;D 期只做加法收敛 |
| 4 | **`strategy_history` 返回 `np.ndarray`/`pd.DataFrame`** | ml、rolling、用户 | Arrow→numpy 零拷贝,签名不变 |
| 5 | **`BacktestResult` 的 pandas 输出** | plot/analysis 全家 | 保持 pandas 表面,内部可 Arrow 计算 |

**插件仓库结论:基本免疫。** `akquant-middleware`/`akquant-qmf` 仅依赖 `gateway.*` + `DataFeed`,不碰 Bar 字段、归一化器、`run_backtest` 数据路径。A–C 期几乎无感;D 期被触及但为收益(接入统一 `ExecutionClient`);E 期若推 tick/quote 需扩 `broker_models`,是加法。

**pandas 定位澄清**:深嵌于报表/研究边缘(strategy_history.get_history_df、plot/*、ml/model、analysis/benchmark、talib/*、backtest/result),**不移除、退到边缘层**;列式化只接管 ingest→引擎→批量计算核心链路。

## 13. 删除 / 合并清单

| 动作 | 目标 |
|---|---|
| 合并 | `utils.load_bar_from_df` + `utils.df_to_arrays` + `utils.prepare_dataframe` + `feed_adapter.BasePandasFeedAdapter.normalize` → `normalize()` |
| 替换 | `feed.add_arrays(numpy…)` → `feed.add_arrow(RecordBatch)`(C Data Interface) |
| 重写 | `ParquetDataCatalog`(pandas/扁平)→ Arrow/Hive/mmap/版本化 |
| 重写 | `feed_adapter` 的 resample/replay(pandas)→ polars `group_by_dynamic` 或 Rust |
| 重构 | 引擎 AoS `Vec<Bar>` → SoA `BarColumns` + `MergedClock` + `BarRef` |
| 降级 | pandas → `akquant[pandas]` 可选 extra |

## 14. 分期路线图

| 期 | 交付 | 动 Rust | golden 策略 |
|---|---|---|---|
| **A** | 规范 Arrow schema + 单一 `normalize` 漏斗;删 4 归一化器;漏斗暂输出旧数组喂现引擎 | 否 | **不变** |
| **B** | 引擎 AoS→SoA + `MergedClock` + `BarRef`;execution/settlement/margin/statistics/history 改读列 | **重** | **不变**(纯重构,主战场) |
| **C** | Catalog 重写:Arrow/parquet + Hive + mmap + 流式 + manifest/version;**跑通 out-of-core 验收** | 中 | 不变 |
| **D** | DataEngine/ExecutionClient/Clock trait,backtest/live 同构;resample/replay 迁 polars/Rust | 重 | 不变 |
| **E** | Tick/Quote/Book 多类型;`Instrument` 独立;**双时态复权/universe** | 重 | **重生成**(§8 行为修正) |
| **F** | pandas 降 optional;API 收敛;文档/教材更新 | 否 | 不变 |

顺序关键:**B 保数值不变**(证明列式化只是重构),**E 才一次性改数值并重锁 golden**——两者分离,故障定位清晰。

## 15. A 期可实施细化(下一步落地起点)

> A 期原则:**不动 Rust、不改数值、golden 全绿**。只做「四合一归一化漏斗 + 过渡桥」,把重复逻辑收敛到一处,为 B 期列式化铺路。A 期漏斗**输出 `pa.Table`(全量物化即可)**,流式 `RecordBatchReader` 留到 C 期。

### 15.1 新增文件

**`python/akquant/schema.py`** —— 单一定义源(消除散落常量):

```python
DEFAULT_INPUT_TIMEZONE = "Asia/Shanghai"          # 现散落在 feed_adapter.py / data.py

COLUMN_ALIASES = {                                # 合并 df_to_arrays.targets + load_bar_from_df.required_map
    "timestamp": ["日期", "date", "datetime", "time", "timestamp"],
    "open":      ["开盘", "open"],
    "high":      ["最高", "high"],
    "low":       ["最低", "low"],
    "close":     ["收盘", "close"],
    "volume":    ["成交量", "volume", "vol"],
    "symbol":    ["股票代码", "symbol", "code", "ticker"],
}
REQUIRED_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")

# 规范 Arrow schema(§3);A 期仅用 OHLCV 子集,Tick/Quote 留待 E 期
BAR_SCHEMA = pa.schema([
    ("ts_event", pa.timestamp("ns", tz="UTC")),
    ("instrument_id", pa.dictionary(pa.int32(), pa.string())),
    ("open", pa.float64()), ("high", pa.float64()),
    ("low", pa.float64()), ("close", pa.float64()),
    ("volume", pa.float64()),
    # extra 数值列在 canonicalize 后作为 float64 追加列透传
])
```

**`python/akquant/normalize.py`**:

```python
def normalize(source, *, symbol=None, column_map=None) -> pa.Table:
    return _canonicalize(_to_arrow(source), symbol=symbol, column_map=column_map)

def _to_arrow(source) -> pa.Table:                # 仅负责"变成 Arrow",零业务逻辑
    # pl.DataFrame/LazyFrame → to_arrow; pa.Table → 原样;
    # pd.DataFrame → pa.Table.from_pandas(保留 index→列);
    # str|Path → pl.scan_parquet/csv; list[Bar] → _bars_to_arrow
    ...

def _canonicalize(tbl, *, symbol, column_map) -> pa.Table:
    # 列名解析(COLUMN_ALIASES + column_map 覆盖)、DatetimeIndex→ts、
    # naive→Asia/Shanghai→UTC、OHLCV→float64、symbol dict、extra 数值透传
    ...
```

### 15.2 过渡桥(对齐**真实** Rust 9 参签名,非 pyi 的 7 参)

Rust `DataFeed::add_arrays` 与 `from_arrays` 实际入参:
`(timestamps, opens, highs, lows, closes, volumes, symbol: Option<str>, symbols: Option<list[str]>, extra: Option<dict[str, ndarray]>)`。桥函数产出此 9 元组:

```python
def arrow_to_legacy_arrays(tbl) -> tuple:         # 替代 df_to_arrays 的返回
    ts   = tbl["ts_event"].to_numpy().astype("int64")     # 已 ns UTC,单一来源
    o,h,l,c,v = (tbl[k].to_numpy().astype("float64") for k in ("open","high","low","close","volume"))
    symbol, symbols = _resolve_symbol(tbl, single=...)    # 单标的→symbol_val;多标的→symbols
    extra = {name: tbl[name].to_numpy().astype("float64") for name in _extra_cols(tbl)} or None
    return (ts, o, h, l, c, v, symbol, symbols, extra)

def arrow_to_bars(tbl, symbol=None) -> list[Bar]: # 替代 load_bar_from_df
    return from_arrays(*arrow_to_legacy_arrays(tbl))
```

调用侧保持 `feed.add_arrays(*arrays)` / `from_arrays(...)` 不变。

### 15.3 删除 / 改调清单

| 目标 | 动作 |
|---|---|
| `utils.load_bar_from_df` | 删,调用点(`__init__`、`feed_adapter.to_bars`、`gateway/ctp/native.py`)改 `arrow_to_bars` |
| `utils.df_to_arrays` | 删,`backtest/engine.py` 6 处调用改 `arrow_to_legacy_arrays(normalize(...))` |
| `utils.prepare_dataframe` | 删,tz 归一并入 `_canonicalize`;engine 中 `prepare_dataframe(df)` 调用移除 |
| `feed_adapter.BasePandasFeedAdapter.normalize` | 改为 `return normalize(frame, symbol=symbol).to_pandas()`——**保留 pandas 输出**喂给 resample/replay(其 pandas 重采样留到 C/D 期),但列名/tz 逻辑不再自写 |
| `akquant.pyi` `add_arrays` | 修正为 9 参(现 7 参 stale,靠 `# type: ignore` 掩盖) |

### 15.4 行为并集清单(A 期最大风险:合并勿丢 quirk)

两个归一化器行为**不完全一致**,合并必须取并集,否则 golden 漂移:

1. naive → `Asia/Shanghai` → UTC(两者都有)
2. NaT → `Timestamp(0)` 填充(两者)
3. OHLCV `fillna(0.0).astype(float)`(两者)
4. **秒级时间戳自动修正**(`< 1e10` 则 ×1e9)—— **仅 `load_bar_from_df` 有**,合并后需决定是否对所有路径生效(建议保留并统一)
5. **`股票代码` 多标的检测**(`load_bar_from_df`)vs `symbol` 候选解析(`df_to_arrays`)—— 取并集
6. **extra 数值列透传** —— 仅 `df_to_arrays` 有,须保留
7. **`__index__` 时间戳**(DatetimeIndex 作时间源)—— 仅 `df_to_arrays` 有,须保留

### 15.5 `Bar` accessor 冻结清单(缝 1,B 期须逐项保留)

摘自 `akquant.pyi`,B 期把 `Bar` 变列式视图时须保持:
- 属性:`timestamp:int`、`symbol:str`、`extra:dict[str,float]`、`open/high/low/close/volume:float`、`timestamp_iso:str`
- 方法:`set_open/set_high/set_low/set_close/set_volume`、`__new__(timestamp,open,high,low,close,volume,symbol,extra?)`、`__repr__`
- ⚠️ **可变 setter 与零拷贝视图冲突**:`set_close` 等要写共享 Arrow buffer。B 期须定策略(Python 侧 `Bar` 保持 owned 副本,或 copy-on-write),此点在 B 期设计中单列。

### 15.6 golden 回归点

- 基线用例(`tests/golden/current/`):**`futures_margin`、`option_basic`、`stock_t1`** —— A 期后三者 `metrics.json` + `orders.parquet` 与 `baselines/` 逐位一致,`__engine_rule_version__` **不变**。
- 输入路径专项(新增单测,覆盖并集清单):中文列名、naive tz、`DatetimeIndex` 作时间源、多标的(symbol 列)、extra 列透传、秒级时间戳、`list[Bar]` 输入。
- 运行:`uv run pytest tests/golden/test_golden.py`;基线重生成(A 期**不应**触发):`uv run python tests/golden/runner.py --generate-baseline`。

## 16. 风险与对策

| 风险 | 对策 |
|---|---|
| B 期引擎重构面大(24 Rust 文件读 Bar) | 先冻结 `BarRef` 接口,模块逐个迁;每步 golden 回归 |
| Arrow/arrow-rs 版本耦合 | 一律走 **C Data Interface(PyCapsule)**,不共享 crate 版本 |
| Python per-event 对象开销 | Bar 视图懒读;高频路径提供批量回调 `on_bars(batch)` |
| 双时态需用户没有的数据(复权因子/knowledge_time) | 优雅降级:缺 `Adjustment` 回退「原始价 + 警告」,不硬失败 |
| out-of-core 与增量指标状态 | 指标状态随游标流转,不依赖全量在内存 |

## 17. 待决策项

1. **B/E 拆分**:已确认——B 保数值、E 改数值,分离实施。
2. **out-of-core 验收**:已确认——C 期须跑通「全 A 股分钟线 > 内存」stress 场景。
3. 待定:Rust 引擎内部是否在更远期进一步「全 DataFusion 查询化」(本 RFC 非目标)。
4. 待定:`on_bars(batch)` 批量回调是否作为 B 期一等 API,还是 D 期再引入。

---

## 18. A 期实施记录(2026-07-18)

**已落地**

- 新增 `python/akquant/schema.py`:`DEFAULT_INPUT_TIMEZONE`、`COLUMN_ALIASES`、`REQUIRED_FIELDS`、`OHLCV_FIELDS`、`BAR_SCHEMA` 单一定义源。
- 新增 `python/akquant/normalize.py`:`_timestamps_to_utc_ns`(共享 tz 助手)、`dataframe_to_arrays`/`dataframe_to_bars`(单一实现)、`to_frame`/`normalize`(多源漏斗)、`arrow_to_legacy_arrays`/`arrow_to_bars`(Arrow 过渡桥)、`coerce_to_pandas`。
- `utils.df_to_arrays` / `utils.load_bar_from_df` 改为对 `normalize` 的薄委托(去重, 签名不变)。
- `run_backtest` / `run_from_checkpoint` 两处分派入口加 `coerce_to_pandas`,`BacktestDataInput` 纳入 `pl.DataFrame`/`pl.LazyFrame`/`pa.Table`(issue #298)。
- 修正 `akquant.pyi` 的 `DataFeed.add_arrays` 为 9 参(补 `symbols`/`extra`)。
- 新增 `tests/test_normalize.py`(14 用例, 覆盖并集清单 + 多源输入)。

**验证**:golden 三用例逐位一致、`__engine_rule_version__` 仍为 `1.3.0`、golden 基线文件未改;`test_normalize`+`golden`+`test_engine` 共 232 passed;ruff check/format 全绿;docs 链接检查通过;polars/pandas/arrow 三路 `run_backtest` 末值一致(100200.0)。mypy 因 numpy 存根用 3.12 `type` 语句而在本地环境预先失败(与本次改动无关, 既有问题)。

**与原计划的差异(降风险)**:未删除 `prepare_dataframe` 及 engine 内 `df_to_arrays`/`prepare_dataframe` 调用点——它们与 `data_map_for_indicators`(指标预热)耦合, 贸然拆除有 golden 漂移风险。故 A 期保留三函数为薄委托(去重目标已达成), 调用点拆除 + `data_map_for_indicators` 解耦 + `feed_adapter.normalize`/resample 迁移下沉到 **A.2**。

**A 期发现的既有缺陷(留待 A.2 修)**:engine 的 `pd.DataFrame` 分支对"字符串日期列 + 无 symbol 列"形状会 `set_index` 后丢失时间戳, 报 `Missing columns: ['timestamp']`(pandas 输入亦复现, 非本次回归)。A.2 将 engine 直接改走 `normalize` 漏斗, 顺带修复此形状。

## 19. A.2 实施记录(2026-07-18)

**已落地(全量走漏斗)**

- `prepare_dataframe` 实现下沉为 `normalize.to_indicator_frame`(逐字移植, 行为不变);`utils.prepare_dataframe` 改薄委托。至此四处归一化逻辑全部收敛入 `normalize.py`。
- engine 两处数据分派改为**直接使用** `normalize` 的 `dataframe_to_arrays` / `to_indicator_frame`, 移除对 `utils.df_to_arrays` / `prepare_dataframe` 的间接依赖。
- **修复既有缺陷**:根因是 pandas 2.x 下 `set_index("date")` 得到 `str` dtype 索引, engine 旧判断 `index.dtype == "object"` 漏掉 → 索引未转 `DatetimeIndex` → 无 `timestamp`。改为稳健的 `not isinstance(index, pd.DatetimeIndex)` 判断(两处分支)。
- 新增回归测试:`run_backtest` 接受"字符串日期列"的 pandas/polars 输入且结果一致。

**验证**:golden 三用例逐位一致、`__engine_rule_version__` 仍 `1.3.0`;normalize+golden+engine 共 232 passed;字符串日期列 pandas/polars/arrow 三路 `run_backtest` 末值一致(100200.0);ruff + mypy(pre-commit 隔离环境)通过。

**A.2 后仍未做(留待后续)**:`feed_adapter` 的 resample/replay 仍是 pandas 实现(C/D 期迁 polars);Catalog 仍是旧结构(C 期重写);Rust 边界仍收 numpy(B 期列式化)。

## 20. B.1 实施记录(2026-07-18)

> **混合路线确认**:价格/OHLCV 用 **f64 列存**(输入本就是 f64,零精度损失);"钱"(结算/PnL/保证金)**保留 Decimal**。重构 Bar 时价格用 `Decimal::from_f64`,与旧 `from_arrays` **完全一致** → golden 逐位不变。全 f64 计算(SIMD)属改数值的大动作,归 E 期后。

**已落地**

- 新增 `src/data/columns.rs` 的 `BarColumns`(SoA:`ts:Vec<i64>` + `open/high/low/close/volume:Vec<f64>` + `symbols:Vec<String>` + `extra:HashMap<String,Vec<f64>>`):
  - `from_py_arrays`:与 `from_arrays` 相同的取数/校验/symbol 解析;
  - `reconstruct_bar(i)`:价格→`Decimal::from_f64`(含 NaN→0.0 告警),逐位一致;
  - `append`(多次 add_arrays 拼接)、`sort_by_timestamp`(稳定排序,等时间戳保插入序)。
- `batch::from_arrays` 重构为 `BarColumns::from_py_arrays(...).to_bars()`,输出不变。
- `SimulatedDataClient` 改为 **列式存 Bar(来自 `add_arrays`)+ `VecDeque<Event>` 存其余事件**(Tick / `List[Bar]` via add_bars);`peek/next` 按时间戳**流式归并**两者(等时列式 Bar 优先),`sort` 分别排序。
- `DataFeed.add_arrays` 直接构建 `BarColumns` 零拷贝存入(实时 `live_sender` 模式回退为重构 Bar 事件发送)。
- 无效数值告警源模块 `akquant.data.batch` → `akquant.data.columns`(消息/结构化上下文不变),同步更新 3 处测试断言。

**验证**:golden 三用例逐位一致、`engine_rule_version` 仍 `1.3.0`;**113 Rust 测试 + 233 Python 测试全过**;`cargo build`/`clippy` 在改动文件零警告。

**设计要点**:`Event` 枚举仍携 owned `Bar`,`next()` 仍按需重构 `Bar`(一次分配)——故 B.1 是**存储列式化 + 内存/地基**,尚未消灭消费端 AoS 物化。真正零物化的 `BarRef` 游标 + 撮合/结算读列 = **B.2**(需改 Event 枚举 + 全部消费者,巨大)。

**Rust 测试环境注**:`scripts/cargo-test.sh` 用 macOS 的 `DYLD_*`;Windows 下需把 base Python 目录(`sys.base_prefix`)加入 PATH 才能加载 pyo3 链接的 `pythonXX.dll`,否则测试二进制报 `STATUS_DLL_NOT_FOUND`。

## 21. B.2 架构结论 + B2′ 实施记录(2026-07-18)

### B.2(BarRef 零物化)判定为不可行 —— 已放弃

深入 Rust 后确认原 §6 的"`Event::Bar(BarRef)` 零物化"在当前架构下**不可行且低价值**:

- **自引用障碍**:`Engine` 同时持有 `state.feed`(列存)与 `current_event: Option<Event>`。让 `Event::Bar` 借用列存 = 同一结构体内自引用借用,安全 Rust 不允许。
- **owned 事件流**:`DataProcessor` 为多标的对齐会 `clone` 事件进 `current_symbol_events`(owned);feed 在 `Arc<Mutex<Box<dyn DataClient>>>` 后,索引方案则每次读字段要加锁。
- **低回报**:事件驱动撮合是逐 bar 串行,每事件物化一个 Bar 成本极小;cache/SIMD 红利在"扫整列"处,不在"一次一根"的撮合循环。

要硬做需 unsafe 自引用或重写整个 事件+pipeline+对齐 系统,风险极高、收益极低。故**放弃 B.2-as-BarRef**,把"列式性能"重定向到真正有价值处 → **B2′**。

### B2′(向量化列计算)—— 已落地

- 新增 `src/data/compute.rs`:`&[f64]` 切片上的向量化批量原语(零拷贝、编译器可自动向量化、可直接作用于 B.1 的 `BarColumns` 列,与增量指标互补):`sma/ema/wma/rolling_sum/rolling_min/rolling_max/rolling_std(ddof=1)/zscore/returns/log_returns/cumsum`。
- 11 个 numpy 零拷贝 Python 绑定 `vec_*`(`PyReadonlyArray1<f64>` 读入),注册进 `akquant` 模块 + `akquant.pyi` 手写存根(注:参数 `PyReadonlyArray1` 不满足 `PyStubType`,故这些函数不用 `#[gen_stub_pyfunction]`,存根手工维护)。
- 语义对齐 pandas:`vec_sma`/`vec_rolling_std`/`vec_returns` 等与 pandas 逐位一致(NaN 位置一致、std 为样本 ddof=1)。

**验证**:golden 不回归;**119 Rust(+6 compute)+ 28 Python(+11 vectorized)测试全过**;compute.rs clippy/ruff/mypy 干净。

**定位**:B2′ 与 polars 因子引擎**互补**——polars 用于研究期向量化;`vec_*` 是 Rust 原生、可零拷贝作用于引擎内列存(BarColumns),无 pandas/polars 往返,适合生产/嵌入路径与 Rust 侧因子。

## 22. C.2 实施记录(2026-07-18)—— out-of-core 核心机制

> **关键洞察**:out-of-core **契合现架构**(与 B.2 相反)——`DataClient` trait 本就是 pull-based(`peek_timestamp`/`next`),`CsvDataClient` 是现成流式模板,可复用 B.1 的时间戳归并。

### C.2a/b 已落地

- 新增 `src/data/parquet_stream.rs` 的 `ParquetStreamClient`(实现 `DataClient`):
  - 经 polars `scan_parquet(PlPath).slice(offset, chunk).collect()` **分块读取**,内存占用约 `chunk_size` 行,**与文件总量无关**(有界内存 → out-of-core)。
  - 逐块转 `Bar`(价格 `Decimal::from_f64`,与 `from_arrays` 一致);`next/peek` 从缓冲弹出,空则填充。
  - 只读流式源(`add()` 返回错误);假定 parquet 按 `timestamp` 升序,`sort()` 空实现。
  - 规范列:`timestamp`(i64 ns UTC)+ `open/high/low/close/volume`(f64)+ 可选 `symbol`(str)。
- `DataFeed.from_parquet(path, symbol=None, chunk_size=None)` 静态构造器(默认 chunk 65536)+ pyi 存根。

**验证**:2 个 Rust 单测(全量读取、时间戳升序、缓冲区 ≤ chunk)+ 2 个 Python 端到端(流式回测与等价内存回测**数值一致**、跨 chunk 边界不丢 bar);121 Rust 测试全过;golden 不回归;clippy/ruff/mypy 干净。

**用法**:`feed = DataFeed.from_parquet(path, "600000"); run_backtest(data=feed, ...)` → 有界内存回测。

### C.2c 已落地(2026-07-18)

- 新增 `normalize.write_canonical_parquet(source, path, symbol=None)`:任意源(pandas/polars/pyarrow/路径/`list[Bar]`)经漏斗规范化 → 写出 `timestamp`(i64 ns UTC)+ OHLCV(f64)+ `symbol`(str)、**按 timestamp 升序**、zstd 压缩的 parquet;导出为 `akquant.write_canonical_parquet` / `akquant.normalize`。
- **多标的 out-of-core 无需额外机制**:单个"按 ts 全局排序 + 带 symbol 列"的规范 parquet,经 `ParquetStreamClient`(读 symbol 列)流式产出即为跨标的按时序的 bar 流。已端到端验证(2 标的、小 chunk、`run_backtest(symbols=[A,B])`,各标的 bar 数正确)。
- 完整闭环:`write_canonical_parquet(源) → DataFeed.from_parquet(path) → 多标的有界内存回测`。249 Python 测试通过,golden 不回归。

### C.2d 已验收(2026-07-18)—— out-of-core 内存实测通过

新增 `scripts/stress_out_of_core.py`(分批生成大规模规范 parquet + 流式回测 + 峰值内存实测,Windows psapi 私有提交 / Unix getrusage,可选 `--compare` 对比内存路径)。**实测结果**:

| 行数 | 流式峰值内存 | 内存路径峰值 |
|---|---|---|
| 20,000 | 740 MiB | — |
| 100,000 | 768 MiB | — |
| 300,000 | 779 MiB | 983 MiB(+204 MiB) |

- **流式峰值近乎持平**:15× 行数(2万→30万)峰值仅 +5%(740→779 MiB)→ **内存有界、与数据总量无关**(峰值主要是 ~740 MiB 固定库开销,数据路径几乎不增)。
- 内存路径 30万行需 **+204 MiB**(全量载入数据的代价),且随数据持续增长直至 OOM。
- **流式耗时 ≈ 内存路径**(64s vs 66s),流式读几乎零额外开销。

**结论:数据路径 out-of-core 成立**。真正跑 >内存(数千万 bar)时内存路径 OOM、流式仍 ~持平。

### C.2 仍未做(留待后续,优先级低)

- **可选**:让 `run_backtest(catalog_path=...)` 自动走流式。**注意**:流式与 `data_map_for_indicators`(指标预计算 + 全量交易日元数据)本质冲突,仅适用**增量指标策略**;当前显式 `DataFeed.from_parquet` 路径已可用,自动接线价值有限。
- **引擎侧结果累积**:回测墙钟随 bar 数**超线性**(10万→30万:10s→64s,两路径皆然,非流式引入)——属引擎逐 bar 处理的既有性能问题,与 out-of-core 内存无关;真跑数千万 bar 需另行优化引擎吞吐(D 期后)。
- catalog 磁盘格式仍是 pandas 专有(`__index_level_0__`/us/naive);`write_canonical_parquet` 已提供规范格式独立写路径,是否迁移旧 catalog 待定。
- Windows 跑 Rust 测试需 `sys.base_prefix` 加 PATH(见 §20 注)。

---

*本 RFC 为设计基线,随分期实施更新;每期完成后回填「实际差异」与 golden 变更记录。*
