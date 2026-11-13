# RFC：列式数据层与引擎核心演进

> **状态**：部分落地（A、A.2、B.1、B2′、C.2 已完成；B.2 已放弃；C 其余、D、E、F 未开始） · **起草**：2026-07-18 · **重整**：2026-11-06 · **范围**：AKQuant 核心（Rust 引擎 + Python 数据层），不承诺向后兼容
>
> 本文是核心演进的跟踪基线：方向、分期、已落地的结论与遗留项。2026-11-06 重整时删去了已完成阶段的施工细节（A 期文件清单、过渡桥签名、逐期实施记录）和已放弃的 `BarRef` 设计，原文见 `512f74ed:docs/zh/meta/columnar-rfc.md`。
>
> 对标：**nautilus_trader**（Rust+Python 事件驱动、回测/实盘同构、ParquetDataCatalog）、**ArcticDB**（版本化双时态存储）、**vectorbt**（列式向量化）、**Apache Arrow / DataFusion**（内存格式与流式计算）。

---

## 0. 进度一览

| 期 | 内容 | 状态 | 结果 |
|---|---|---|---|
| **A / A.2** | 规范 schema + 单一归一化漏斗 | ✅ 已落地 | 四处归一化逻辑收敛到 `python/akquant/normalize.py`，常量收敛到 `schema.py`；golden 逐位不变 |
| **B.1** | 数据源存储列式化 | ✅ 已落地 | `src/data/columns.rs` 的 `BarColumns`（SoA，价格 f64 列）；`SimulatedDataClient` 列式存 bar，并与其余事件按时间戳流式归并 |
| **B.2** | `BarRef` 零物化游标 | ✗ 已放弃 | 理由见 §5.2 |
| **B2′** | 列上的向量化批量计算 | ✅ 已落地 | `src/data/compute.rs` + 11 个 `vec_*` 零拷贝绑定，语义与 pandas 逐位一致 |
| **C.2** | out-of-core 数据路径 | ✅ 已落地 | `ParquetStreamClient` 分块流式读取 + `write_canonical_parquet`；内存实测有界（§8） |
| **C** 其余 | Catalog 重写（Hive 分区、mmap、manifest、版本化） | 未开始 | |
| **D** | 回测/实盘同构 + 核心与 Python 解耦 | 进行中 | 解耦第 2 步（策略派发 trait 化）已完成，见 §6 |
| **E** | 多数据类型 + 双时态复权/成分股 | 未开始 | 唯一会改回测数值的一期 |
| **F** | pandas 降为可选依赖、API 收敛 | 未开始 | |

## 1. 背景与动机

起草时的数据链路是：用户输入（`pd.DataFrame` / `Dict` / `List[Bar]` / `DataFeed` / `DataFeedAdapter`）→ `run_backtest` 按类型分派 → pandas 时区归一 → pandas 转 numpy → `feed.add_arrays` 进 Rust → 引擎物化 `Vec<Bar>`。

三个根因问题：

1. **归一化逻辑重复**：列名解析、时区、dtype 在四处各写一份，已出现细节漂移。（A 期已解决）
2. **介质割裂**：catalog 写用 pandas、因子读用 polars、回测读用 pandas，同一份 parquet 三条路径。（C 期 Catalog 重写解决，未开始）
3. **行式内存布局**：数据集不能大于内存，批量计算无法向量化。（B.1、B2′、C.2 已解决数据路径部分）

## 2. 目标与非目标

**目标**

- 存储与批量计算列式原生（SoA）；数据可以大于内存（out-of-core）。
- Catalog 成为唯一真相源：Arrow/Parquet、Hive 分区、mmap、流式、版本化。
- 回测/实盘同构：策略代码不改即可在回测与实盘间切换。
- 时点正确（双时态）：结构性杜绝复权与成分股的未来函数。
- 确定性重放：逐位可复现。

**非目标**

- 不承诺 API 向后兼容（可以重命名或删除旧入口）。
- 不移除 pandas：pandas 退居报表、研究、呈现的边缘层，核心路径不再依赖它。
- 不追求事件循环零物化：撮合循环逐条处理事件，每个事件物化一个 `Bar` 的成本很小，B.2 已论证这条路不划算（§5.2）。
- 不做「全 DataFusion 查询化」的引擎。

## 3. 设计原则

| 原则 | 含义 |
|---|---|
| 列式原生 | 存储与批量计算用 SoA；事件循环用游标串行推进 |
| 唯一真相源 | Catalog 是唯一入口，回测、实盘、研究都从它流式读 |
| 回测实盘同构 | 两者的差异只在注入的数据源、执行端和时钟 |
| 核心不依赖 Python | 新增核心逻辑不出现 PyO3，PyO3 只放在各模块的 `python.rs` 绑定层（§6） |
| 时点正确 | 复权因子、成分股、基本面按模拟时点应用 |
| 确定性 | 稳定时序排序 + 固定随机种子 + 绑定 catalog 版本 |

## 4. 规范数据模型

每种数据是一个 Arrow schema，公共字段为 `ts_event`（ns，UTC）、`ts_init`（接收/生成时间）、`sequence`（同一时刻的排序依据）、`instrument_id`（字典编码）。

```text
Bar          : ts_event, ts_init, seq, iid, open, high, low, close, volume, open_interest?
QuoteTick    : ts_event, ts_init, seq, iid, bid, ask, bid_size, ask_size
TradeTick    : ts_event, ts_init, seq, iid, price, size, aggressor_side, trade_id
OrderBookΔ   : ts_event, ts_init, seq, iid, action, side, price, size, level
Instrument   : iid, asset_type, tick_size, lot_size, multiplier, margin_rate, expiry, ...
Adjustment   : iid, ex_date, knowledge_time, factor, cash_dividend, split_ratio      ← 双时态
Membership   : universe_id, iid, start, end, knowledge_time                          ← 防幸存者偏差
```

- 合约元数据 `Instrument` 应成为一等数据类型，沉淀进 Catalog，供撮合、保证金、Greeks 共用（现在散落在 `config.py`）。
- `Adjustment` / `Membership` 是双时态表，带 `knowledge_time`，是 §9 的数据基础。
- 现状：A 期只落地了 OHLCV 子集（`schema.py` 的 `BAR_SCHEMA`），其余类型属 E 期。

## 5. 引擎数据持有

### 5.1 已落地：存储列式、事件逐条物化

- **存储**：`BarColumns` 以 f64 列存价格与成交量（输入本就是 f64，没有精度损失）；涉及钱的计算（结算、盈亏、保证金）仍用 `Decimal`。从列重建 `Bar` 时价格走 `Decimal::from_f64`，与旧路径完全一致，因此 golden 逐位不变。全 f64 计算会改数值，归到 E 期之后。
- **归并**：`SimulatedDataClient` 列式存 `add_arrays` 进来的 bar，用队列存 tick 和 `List[Bar]`，`peek` / `next` 按时间戳流式归并两者（时间戳相同时列式 bar 优先）。
- **批量计算**：`compute.rs` 在 `&[f64]` 切片上提供 `sma/ema/wma/rolling_*/zscore/returns/log_returns/cumsum`，可直接作用于列存，与 polars 因子引擎互补（polars 用于研究，`vec_*` 用于引擎内和 Rust 侧因子）。

### 5.2 已放弃：`BarRef` 零物化游标（B.2）

原设计让 `Event::Bar` 携带指向列存的轻量视图，事件循环永不物化 `Bar`。深入实现后判定不可行且收益低：

- **自引用**：`Engine` 同时持有列存（`state.feed`）和 `current_event`，让事件借用列存就是同一结构体内的自引用借用，安全 Rust 不允许。
- **事件是 owned 的**：多标的对齐时 `DataProcessor` 会把事件克隆进当前时刻的事件集合；数据源在锁后面，按索引读字段每次都要加锁。
- **收益低**：撮合是逐 bar 串行，每个事件物化一个 `Bar` 的成本极小；缓存与 SIMD 的收益在「扫整列」处，已由 B2′ 拿到。

## 6. 回测/实盘同构与核心解耦（D 期）

### 6.1 目标形态

```rust
trait DataEngine      { fn next(&mut self) -> Option<Event>; }   // 回测：列存重放 / 实盘：网关推送
trait ExecutionClient { fn submit(&mut self, order); ... }       // 模拟撮合 / 券商网关
trait Clock           { fn now(&self) -> i64; fn timers(); }     // 随事件推进 / 墙钟
```

策略只依赖账户与订单的门面，回测与实盘的唯一差异是注入哪一套实现。现有 `gateway/`（ctp + middleware 插件）接到 `ExecutionClient` 与实盘数据源。

这个形态的前提是引擎核心不依赖 Python：上面三个 trait 和策略派发都要能在没有 Python 解释器的情况下运行和测试。

### 6.2 解耦路线：方向，不单独立项

2026-11-06 评估「能否直接用 Rust 写策略」时确定：把核心拆成纯 Rust crate 加绑定层是长期方向，但不作为一次性工程，而是每一步都单独交付收益：

1. **定规矩**：新增核心逻辑不出现 PyO3，PyO3 只放在各模块的 `python.rs`（`engine/`、`execution/`、`analysis/`、`calendar/` 已是这个结构）。
2. **策略派发 trait 化**（✅ 2026-11-06）：引擎只经 `src/engine/strategy_handler.rs` 的 `StrategyHandler` 调用策略，Python 策略是实现之一（`PyStrategy`）；`Engine::run_with_strategy` 可以直接运行 Rust 原生策略，`StrategyContext::submit_order` 是纯 Rust 下单入口；有原生策略跑完整会话的单测。
3. **按热点把策略语义下沉到 Rust**：Python 策略层约 1.1 万行（订单事件、OCO/括号单、`I()`、sizer、warmup、调度、本地止损……）。先下沉订单事件检查和框架钩子。订单事件检查已下沉（2026-11-08，`src/order_ledger.rs`）；下一块是 `on_bar_event` 的簿记（持仓 bar 计数、warmup 计数）。
4. **出现 Rust 使用方时**，再决定是否单独发布核心 crate、Rust 策略如何加载（Rust 没有稳定 ABI，跨 `.so` 传 trait 对象不安全）。

### 6.3 现状与下一步

- **`py` 令牌还去不掉**：派发 trait 仍带 `Python<'_>`，上下文仍是 `Py<StrategyContext>`。`py` 还承担三件事，各需单独抽象：live 数据源等待行情时释放 GIL（`feed.next_action(py)`）、流事件回调（`emit_stream_event`）、`StrategyContext` 是 pyclass（Python 策略要跨回调持有它）。
- **`StrategyContext` 的下单接口绑着 Python**：`buy` / `sell` 的参数是 Python 对象，Rust 策略目前只能自己构造 `Order` 再调 `submit_order`。
- **第 3 步的依据**（2026-11-06 实测，20 万根 1 分钟 bar、单标的）：空策略约 6.3 秒（约 32µs/bar）；cProfile 显示约一半时间在 Python 框架层（每根 bar 约 32 次 `getattr`），与用户 `on_bar` 无关。
- **耦合面**（2026-11-06 统计）：Rust 约 4.8 万行中约 2 万行不碰 PyO3；`execution`、`market`、`settlement`、`margin`、`statistics`、`calendar` 已基本独立，`engine`、`pipeline`、`context.rs` 是重度耦合区。

## 7. Catalog 重写（C 期）

对标 nautilus `ParquetDataCatalog` + ArcticDB 版本化：

```text
{root}/
  data/{data_type}/{iid}/{year=YYYY}/{month=MM}/part-*.parquet   # Hive 分区，zstd
  instruments/instruments.parquet
  adjustments/{iid}.parquet
  _manifest/manifest.parquet     # iid × type × [t_start, t_end] → file, version
  _versions/{vN}/...             # append-only 不可变，支持按 catalog 版本复现
```

- **读**：mmap 映射 → 流式读取；时间/标的过滤靠 parquet row-group 统计与谓词下推，不全量载入。
- **写**：规范 Arrow → 按 `(iid, ts)` 排序 → 分区落盘；文件不可变。
- **版本**：回测绑定 catalog 版本，数据可复现。
- **现状**：catalog 磁盘格式仍是 pandas 专有（`__index_level_0__`、us 精度、无时区），目录结构是扁平的 `{root}/{symbol}.parquet`；`write_canonical_parquet` 已提供规范格式的独立写路径。

## 8. Out-of-core 与并行

**C.2 实测**（`scripts/stress_out_of_core.py`，2026-07-18）：

| 行数 | 流式峰值内存 | 全量载入峰值 |
|---|---|---|
| 20,000 | 740 MiB | — |
| 100,000 | 768 MiB | — |
| 300,000 | 779 MiB | 983 MiB |

行数增加 15 倍，流式峰值只增加 5%（峰值主要是约 740 MiB 的固定库开销），耗时与全量载入相当。用法：`write_canonical_parquet(源) → DataFeed.from_parquet(path) → run_backtest(data=feed, ...)`；多标的只需一个按时间戳全局排序、带 `symbol` 列的规范 parquet。

限制与反向约束：

- 流式与指标预计算（`data_map_for_indicators` 需要全量数据与交易日元数据）本质冲突，只适用于增量指标策略；因此 `run_backtest` 没有自动接线流式路径，需要显式传 `DataFeed.from_parquet`。
- `strategy_history` 全历史在内存，要改为有界窗口；`plot` / `report` 对大数据集要降采样；`optimize.py` 的 worker 将来应共享 mmap 的 catalog。

并行方向：因子/指标用 rayon 跨标的、跨列并行 + SIMD；组合估值按持仓列向量化。

## 9. 时点正确性：双时态（E 期，golden 会变）

**问题**：全局前复权把未来的拆分、分红烘焙进历史价格，结构性引入未来函数；成分股用当前列表，引入幸存者偏差。

**改法**：

- 存原始价格加 `Adjustment` 因子序列，读取时按模拟当前时间只应用 `knowledge_time ≤ now` 的因子。
- `Membership` 双时态：模拟时刻 T 只能看到 T 时点的成分股。
- 基本面和信号带 `knowledge_time`：T 时刻只能看到 T 之前已知的数据。

回测数值相对现状会变化，变化后才是正确的。E 期一次性修正并重新锁定 golden；B 期保持数值不变、E 期才改数值，两者分离，便于定位问题。

## 10. 确定性与复现

稳定排序 `(ts_event, sequence, iid)` + 固定随机种子 + 绑定 catalog 版本 → 逐位可复现。`__engine_rule_version__` 随行为变更跳版本。

## 11. Python API 终态

```python
cat = Catalog(root)                                  # 唯一入口
cat.write(normalize(df_or_pl_or_parquet))            # 任意源 → 规范 Arrow
res = run_backtest(catalog=cat, instruments=[...],   # 从 catalog 流式读，out-of-core
                   strategy=MyStrategy, ...)
res = run_backtest(data=pl_df, strategy=...)         # 快速测试仍可传内存数据
```

- `data=` 保留内存数据的便利路径（polars / pandas / pyarrow 平权）；pandas 降为 `akquant[pandas]` 可选依赖。
- 策略 `on_bar` / `on_tick` 的用法不变。

## 12. 影响面与接口缝

后续各期通过冻结以下接口缝控制破坏面：

| # | 接口缝 | 依赖方 | 策略 |
|---|---|---|---|
| 1 | `Bar` 属性接口（`.open/.high/.low/.close/.volume/.timestamp/.symbol/.extra`） | Python 侧读 bar 字段的模块与用户策略 | 保持不变（`Bar` 仍是 owned 对象） |
| 2 | `DataFeed` 符号 | 插件 `from akquant import DataFeed` | 保留符号，内部可换实现 |
| 3 | `gateway.*`（broker_models / trader_base / registry / protocols） | middleware、klinedata 插件 | D 期只做加法 |
| 4 | `strategy_history` 返回 `np.ndarray` / `pd.DataFrame` | ml、rolling、用户 | 签名不变 |
| 5 | `BacktestResult` 的 pandas 输出 | plot / analysis | 保持 pandas 表面，内部可用 Arrow 计算 |

插件仓库基本不受影响：middleware 只依赖 `gateway.*` 与 `DataFeed`；D 期会接入统一的 `ExecutionClient`，属于加法；E 期如推 tick/quote 需扩展 `broker_models`，也是加法。

## 13. 遗留项与待决

**遗留项**

1. **标的列口径未统一**：`normalize.py` 里 `dataframe_to_arrays` 按别名表（`resolve_columns`）识别标的列，`dataframe_to_bars` 只认「股票代码」列，两者应取并集、统一走别名表。回测入口已先收敛（`resolve_columns` 识别后统一改名为 `symbol`），见 `tests/test_backtest_symbol_column_aliases.py`。
2. **`feed_adapter` 的 resample / replay 仍是 pandas 实现**：应迁到 polars `group_by_dynamic` 或 Rust（C/D 期）。
3. **Rust 边界仍收 numpy**：`feed.add_arrays(numpy…)` → `feed.add_arrow(RecordBatch)`（Arrow C Data Interface）未做。
4. **回测墙钟随 bar 数超线性增长**（C.2d 实测 10 万 → 30 万行：10 秒 → 64 秒，流式与全量两条路径都如此）：属引擎逐 bar 处理的吞吐问题，与 out-of-core 无关，要跑数千万根 bar 需另行优化。

**待决**

1. ~~`on_bars(batch)` 批量回调是否作为一等 API，以及在哪一期引入。~~ 已定（#441）：作为一等 API 引入，`Strategy.on_bars(batch: BarBatch)` 每个时间片回调一次，在 Python 框架层实现（收集 + 框架定时器），与列式分期无关；批次里仍是逐根物化的 `Bar` 对象，列式批次（零拷贝视图）留待解耦第 3 步。
2. 旧 catalog 是否迁移到规范格式。

## 14. 风险与对策

| 风险 | 对策 |
|---|---|
| Arrow / arrow-rs 版本耦合 | 一律走 Arrow C Data Interface（PyCapsule），不共享 crate 版本 |
| Python 每事件对象开销 | 解耦第 3 步下沉框架层语义；高频路径考虑批量回调 |
| 双时态需要用户没有的数据（复权因子、knowledge_time） | 优雅降级：缺 `Adjustment` 时回退为原始价并告警，不硬失败 |
| out-of-core 与增量指标状态 | 指标状态随游标流转，不依赖全量数据在内存 |
| 解耦后 Python 与 Rust 两套策略语义漂移 | 语义只在 Rust 实现一份，Python 层只做参数解析与转发；golden 锁行为 |
