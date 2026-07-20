# 设计草案:热启动配置继承 + 分阶段 result 合并(issue #282 二轮反馈)

> **状态**:第 1 部分(配置持久化)**已实现** · 第 2 部分(result 合并)**已实现**(`merge_results` / `MergedResult`,metrics 为核心子集) · 第 3 部分(API 重命名,破坏性)**已实现** · **日期**:2026-07-20 · **范围**:`python/akquant/checkpoint.py`、`python/akquant/backtest/engine.py`、`python/akquant/backtest/result.py` + 示例/文档 · **对应 issue**:[#282](https://github.com/akfamily/akquant/issues/282) 二轮反馈第 3、4 点 · **性质**:配置继承缺陷(已根治)+ 体验增强(result 合并)+ API 命名演进(热启动→checkpoint/resume)

---

## 0. 结论先行

issue #282 二轮反馈实跑核实结果:

| 点 | 内容 | 结论 |
|---|---|---|
| 1 | 二阶段重放一阶段 `on_before_trading` | **不可复现**。#329 已保证钩子去重;用户看到的是策略实例累加器随快照 pickle 保存的历史残留,非重新触发。建议文档提示在 `on_resume` 重置统计。 |
| 2 | 二阶段 `on_after_trading` 变每日首 bar 触发 | **不可复现**。precise / 默认模式冷热启动钩子序列结构一致。同上,累加器残留误导。 |
| 3 | 滑点/佣金未在热启动继承 | **滑点真 bug**,已在 `run_warm_start` 补 `set_slippage`/`set_volume_limit`(最小移植);佣金半通(走 `_resolve_stock_fee_rules`)。本 RFC 提出**根治**:快照持久化配置。 |
| 4 | 缺 result 合并函数 | **功能缺失**,本 RFC 提出 `merge_results` / `ResultMerger`。 |

本 RFC 只覆盖第 3 点的**根治方案**与第 4 点。第 3 点的最小热修已单独落地(不依赖本 RFC)。

## 1. 第 3 点根治:快照持久化回测配置

### 1.1 现状缺陷链(已对照源码核实)

- `save_snapshot`(`checkpoint.py:86-97`)只存 `engine_state`(Rust 二进制)+ `strategy` + `strategy_topology` + `snapshot_features`,**不含任何回测配置**(滑点、量比、fill_policy、市场模型、费率)。
- `EngineSnapshot`(`src/engine/python.rs:589-639`)序列化 portfolio/order_manager/instruments/risk_state 等,**不含** timers、slippage、fee rules、fill_policy——`load_state_bytes` 后这些全为引擎默认值。
- 因此 `run_warm_start` 必须把每一项配置**重新推导并 set 一遍**。团队已重配费率(`engine.py:5451-5528`)、fill_policy(`5539-5555`),却**漏了滑点/量比**(已由最小热修补上)。
- 根因:**"热启动重配所有配置"这件事没有单一数据源**,靠 `run_warm_start` 逐项手工补,漏一项就静默失效。滑点只是暴露出来的第一个;任何未来新增的引擎配置都可能重蹈覆辙。

### 1.2 目标与非目标

**目标**

- 让"分阶段回测继承完整回测配置"成为**默认且自动**的行为,用户只需关心真正变化的量:`data`(下阶段数据)、`symbols`(换月新代码)、`instruments`(资产属性)。
- 消除逐项手工重配的**遗漏面**:配置有单一权威来源(快照),`run_warm_start` 从中恢复。
- 显式传入的入参**优先级最高**,允许用户按阶段覆盖(如只改起止时间)。

**非目标**

- 不改 Rust `EngineSnapshot`(避免二进制格式版本迁移成本);配置存 Python 侧 pickle 层。
- 不强制用户改现有代码——旧快照(无配置字段)必须能继续 warm start(降级到当前逐项推导逻辑)。

### 1.3 提案:快照增加 `backtest_config` 字段

**A. `save_snapshot` 侧**

`save_snapshot(engine, strategy, filepath)` 目前签名不含 config。两个方案:

- **A1(推荐)**:新增可选参 `save_snapshot(engine, strategy, filepath, *, config=None, resolved_runtime=None)`。当 `run_backtest` 结束把**已解析的运行时配置**(归一化后的 slippage policy、volume_limit_pct、fill_policy、fee rules、market model、timezone、history_depth 等)挂到 `result` 上,用户 `save_snapshot(result.engine, result.strategy, path, config=result.resolved_config)` 即可。
- **A2**:`run_backtest` 在 `result` 上暴露 `result.resolved_config`(一个 dataclass/dict),`save_snapshot` 自动从 `engine`/`strategy` 反查——但引擎不留存原始 policy,反查不可靠。故取 A1。

快照结构扩展(向后兼容,新增键):

```python
snapshot = {
    "engine_state": ...,
    "strategy": ...,
    "strategy_topology": {...},
    "snapshot_features": {"history_buffer_snapshot": True},
    "backtest_config": {              # 新增,可选
        "slippage": {"type": "percent", "value": 0.0001},
        "volume_limit_pct": 0.25,
        "fill_policy": {"price_basis": "open", "temporal": "next_event", "bar_offset": 1},
        "commission_policy": {"type": "percent", "value": 0.0003},
        "stamp_tax_rate": 0.0,
        "transfer_fee_rate": 0.0,
        "min_commission": 5.0,
        "t_plus_one": False,
        "market_model": "simple" | "china",
        "timezone": "Asia/Shanghai",
        "history_depth": 50,
    },
    "version": _VERSION,
}
```

**B. `run_warm_start` 侧 —— 配置解析优先级**

统一成一条链,自上而下取第一个非 None:

```
显式入参(slippage=, config.strategy_config.slippage, ...)
  > 快照 backtest_config
  > 引擎/默认兜底
```

实现:在现有各配置解析点前,先从 `snapshot["backtest_config"]` 读默认值填充局部变量,再让现有"显式入参优先"逻辑覆盖。这样:

- 用户**什么都不传** → 完全继承一阶段配置(修复根因)。
- 用户**只改起止时间** → 其余继承,符合"注意力放在真正变化的量"。
- 用户**显式传 slippage** → 覆盖快照值(保留逐阶段调参能力)。

**C. 向后兼容**

`snapshot.get("backtest_config")` 为 None(旧快照)时,走当前逐项推导逻辑(已含滑点最小热修),行为不回退。

### 1.4 落地顺序(已完成)

1. ✅ `result.resolved_config` 暴露(`run_backtest` 收尾经 `_build_resolved_backtest_config` 填充 slippage/volume_limit/commission/stamp_tax/transfer_fee/min_commission/t_plus_one/timezone/history_depth/fill_policy)。
2. ✅ `save_snapshot(..., config=result|dict)` 经 `_extract_resolved_config` 写入快照 `backtest_config` 字段。
3. ✅ `warm_start` 把 `backtest_config` 挂到 `strategy._warm_start_backtest_config`;`run_warm_start` 建立 `显式入参 > config.strategy_config > 快照配置 > 默认` 优先级链(slippage/volume_limit/fee/fill_policy/t_plus_one)。
4. ✅ 回归测试:`test_run_warm_start_inherits_config_from_snapshot`(零显式入参继承)、`test_run_warm_start_explicit_arg_overrides_snapshot_config`(显式覆盖)、`test_run_warm_start_without_snapshot_config_is_backward_compatible`(旧快照降级)。前两者经 `git stash` 验证在未实现时失败、实现后通过。

**实现说明**:配置存 Python pickle 层(快照 dict 的 `backtest_config` 键),未改 Rust `EngineSnapshot`,旧快照 `snapshot.get("backtest_config")` 为 None 时降级到逐项推导,完全向后兼容。

## 2. 第 4 点:分阶段 result 合并

### 2.1 需求

完整回测 5 年 → 一条 5 年 equity_curve;按月分阶段 → 每段 result 只含当月曲线。用户需手动拼接。目标:官方 `merge_results` 自动处理曲线/交易/指标拼接,并防资产数量爆炸(清理已退市合约)。

### 2.2 可合并 / 需重算 / 需丢弃 三分

| 类别 | 字段 | 处理 |
|---|---|---|
| **可直接拼接**(时间序列) | `equity_curve`、`cash_curve`、`margin_curve`、`positions`(快照序列)、`orders`、`trades`、`executions`、`indicator_outputs` | 按时间戳排序去重拼接;相邻阶段边界时间戳去重(热启动首 bar 可能与上阶段末 bar 语义重叠) |
| **必须重算**(不可简单相加) | `metrics`(total_return、max_drawdown、sharpe、win_rate...) | 由合并后的 `equity_curve` + `trades` 重新计算;**不能**按阶段平均或相加。`initial_cash` 取第一阶段的,`end_market_value` 取最后阶段的 |
| **需丢弃/裁剪** | 已退市合约的 position snapshot、instrument 元数据 | 合并后清理配置了 `expiry_date` 且已过期的合约(用户明确要求防爆炸) |

### 2.3 API 设计

```python
def merge_results(
    *results: BacktestResult,
    drop_expired_instruments: bool = True,
    dedupe_boundary: bool = True,
) -> MergedResult:
    """按时间顺序合并多段分阶段回测结果。

    - 曲线/交易/订单/执行按时间戳拼接去重;
    - metrics 由合并后曲线+交易重算核心子集(非阶段简单相加);
    - drop_expired_instruments=True 时清理已退市合约快照,防资产爆炸;
    - dedupe_boundary=True 时去除相邻阶段边界重叠时间戳。
    """
```

**已落地决策(与草案的差异)**:

- **返回类型 = 新建 `MergedResult` 类**,而非 `BacktestResult`。原因:`BacktestResult` 包装 Rust `_raw` 对象(被 16 种方式访问),无法在 raw 层重建;`MergedResult` 用纯 pandas 帧鸭子类型复刻只读视图(`equity_curve`/`*_df`/`daily_returns`/`to_quantstats`/`metrics`/`metrics_df`),不侵入 `BacktestResult` 的 Rust 契约。代价:`plot()`/`report_quantstats()` 等依赖 Rust 对象的方法在合并结果上不可用。
- 配套链式便利:`merge_results(r1, r2)` 支持任意段数;单段 `merge_results(r)` 退化为恒等视图。

### 2.4 关键难点

- **metrics 重算依赖**:现在 metrics 在 Rust `generate_backtest_result`(`src/statistics/mod.rs:157`)里算。合并发生在 Python 侧,需要**纯 Python 的 metrics 计算路径**(或复用 `to_quantstats` + quantstats)。建议:合并 result 的 `metrics` 走 Python 重算(基于拼接 equity_curve/daily_returns),与 Rust 单段 metrics 语义对齐但独立实现,并加测试锁定两者在单段场景数值一致。
- **边界去重语义**:热启动首个 bar 的 equity 点可能与上阶段快照点时间戳相同,`upsert_timestamped_value`(Rust 侧已有 upsert 语义)提示"同戳取后者"。Python 合并沿用"同戳保留后一阶段值"。
- **positions 连续性**:分阶段的持仓快照本身连续(热启动恢复了 portfolio),拼接即可;去重同戳。

### 2.5 落地顺序(已完成)

1. ✅ `merge_results` + `MergedResult`(`python/akquant/backtest/merge.py`):曲线/orders/trades/executions/positions 拼接去重(纯 pandas)。
2. ✅ Python metrics **核心子集**重算(`_compute_core_metrics`)+ 与单段完整回测的 golden 一致性测试(`total_return_pct`/`end_market_value` `approx` 对齐)。
3. ✅ 退市合约清理(依据 instrument snapshot `expiry_date`)。
4. ✅ 时间重叠→`ValueError`;边界同戳去重(保留后段)。
5. ✅ 导出(`akquant.merge_results` / `akquant.MergedResult`)+ `.pyi` 存根。
6. ✅ 文档:热启动专题(zh+en)补"多阶段结果合并"小节 + `reference/api.md` 签名 + 测试 `tests/test_merge_results.py`(9 例)。

**metrics 范围决策**:全部 60 项指标由 Rust 引擎在完整回测中计算,无"从权益曲线算指标"的可复用 `#[pyfunction]` 入口。在 Python 全量复刻 60 项风险大(口径漂移)、收益低。故只重算能从合并曲线+交易明细**无歧义推导**的核心子集,口径对齐 `src/analysis/result.rs`(sharpe 用日收益算术均值×dpy、drawdown 用 cummax 峰值回撤、calmar=年化/最大回撤);其余字段访问时抛 `AttributeError` 并提示去单段 `BacktestResult` 读取。

## 3. API 命名演进:热启动 → checkpoint / resume(破坏性更新)

### 3.1 问题:一个名字扛了两种语义

调研主流框架后确认,AKQuant 的"热启动"存在**概念混淆**——它借用了 warm-up 的词,做的却是 checkpoint-resume 的事:

| 语义 | 做的事 | LEAN | NautilusTrader |
|---|---|---|---|
| **A. 指标预热(warm-up)** | 开跑前喂历史数据填满滚动窗口/指标,让策略起手即 `IsReady` | `SetWarmUp(period)` / `IsWarmingUp` | history bootstrap(catalog 喂数据) |
| **B. 状态续跑(checkpoint/resume)** | 保存引擎+策略完整状态(持仓/现金/订单/风控),从断点继续跑下一段 | 无内建(靠外部编排) | **State Persistence & Recovery**(Cache + Event Sourcing) |

`run_warm_start` 本质是 **B**,却叫了 **A** 的名,还顺带做了一点 A(`_bootstrap_incremental_indicators`)。一个函数扛两种语义,正是用户困惑与 #282 边角问题(滑点漏配、钩子累加器误判)的温床。

参考出处:[LEAN Warm Up Periods](https://www.quantconnect.com/docs/v2/writing-algorithms/historical-data/warm-up-periods)、[NautilusTrader State Persistence and Recovery](https://deepwiki.com/nautechsystems/nautilus_trader/7.3-state-persistence-and-recovery)、[NautilusTrader Event Sourcing](https://github.com/nautechsystems/nautilus_trader/blob/3eb18933/docs/concepts/event_sourcing.md)。

### 3.2 提案:直接重命名(不保留别名)

业界 B 语义的标准词是 **checkpoint / resume**。本 RFC 采**破坏性更新**,不保留 deprecated 别名(避免长期双名维护与文档歧义):

| 现名 | 新名 | 语义归位 |
|---|---|---|
| `run_warm_start()` | **`run_from_checkpoint()`** | B:从断点续跑 |
| `save_snapshot()` | **`save_checkpoint()`** | B:落盘断点(现实现就是 checkpoint,非只读 snapshot) |
| `checkpoint.warm_start()` | **`checkpoint.load_checkpoint()`** | B:恢复引擎+策略 |
| `warm_up` / `warmup_period` / `history_depth` / `infer_warmup_period` | **不变** | A:指标预热,语义本就正确,保持独立 |

命名优先级(B 场景):`resume` / `checkpoint restore` 优于 `warm start`——后者与 warm-up 天然混淆,新命名不再沿用。

### 3.3 迁移面(已勘定)

去除 `__pycache__` 后的真实引用面:

- **代码**:`python/akquant/__init__.py`(`__all__` 导出 4 处)、`backtest/engine.py`、`backtest/__init__.py` + `.pyi`、`checkpoint.py`、`backtest/result.py`;测试 `tests/test_strategy_extras.py`。
- **示例**(4 个):`examples/21_warm_start_demo.py`、`22_strategy_runtime_config_demo.py`、`56_functional_warm_start_demo.py`、`57_functional_multi_slot_warm_start_demo.py`——仅改内部 API 符号,**文件名保留**(避免级联改动 `tests/test_examples_regression.py` 的硬编码路径与 `done_*` 哨兵、README、doc 链接);按约定逐个实跑 `exit 0`。
- **文档**(zh+en):批量替换 API 符号;`warm_start.md` 专题页**文件名保留**(改名会级联 mkdocs nav 与全部内链),仅更新正文。`reference/api.md` 签名同步。

### 3.4 checkpoint 契约化(顺带根治"配置项漂移")

第 1 部分把配置塞进散装 dict 的 `backtest_config` 键,长期仍靠 `snapshot.get(...)` 逐项取,易漏。建议把 checkpoint 定义成**显式 dataclass**作为单一契约:

```python
@dataclass
class Checkpoint:
    engine_state: bytes
    strategy: Any
    strategy_topology: dict
    resolved_config: dict        # 第 1 部分的 backtest_config,契约化
    snapshot_features: dict
    version: str
```

- `save_checkpoint` / `load_checkpoint` 都引用此 dataclass,新增引擎配置只在一处加字段,结构上杜绝遗漏。
- **版本校验**:现有 `version` 字段仅记录、未校验。跨 AKQuant 版本恢复旧 checkpoint 时应显式告警/报错(schema 不兼容),而非静默行为漂移。
- **pickle 脆弱性**:`save_snapshot` 直接 pickle 策略实例,遇不可 pickle 对象即抛(如 `cannot pickle Order`,调试时已撞到)。短期不上 Nautilus 式事件溯源,但应在 `__getstate__` 更主动剔除运行时瞬态,并对残留累加器在文档提示 `on_resume` 重置(#282 第 1、2 点误判来源)。

### 3.5 落地顺序

1. Rust 无关,纯 Python 重命名 + `.pyi` 同步 + `__all__` 更新。
2. checkpoint dataclass 契约化 + 版本校验。
3. 示例/文档批量改名并实跑。
4. `__engine_rule_version__` 无需动(行为不变,仅 API 名变);发版说明标注**破坏性变更**。

## 4. 三部分关系

第 1 部分(配置持久化,**已实现**)、第 2 部分(result 合并)、第 3 部分(API 重命名)**互相正交**,可独立发版。建议顺序:第 1(已完成)→ 第 3(破坏性,趁早合并进下一个 major)→ 第 2(体验增强,随时跟进)。滑点最小热修已落地,不阻塞任何一部分。

## 5. 风险与开放问题

- **快照体积**:`backtest_config` 是小 dict,可忽略。
- **配置项漂移**:未来新增引擎配置需同步进 `backtest_config` 白名单——建议以 `resolved_config` dataclass 为单一定义点,`save_snapshot`/`run_warm_start` 都引用它,减少再次遗漏。
- **metrics 双实现**:Python 重算与 Rust 单段实现的一致性需 golden 级测试守护,否则合并 metrics 与单段 metrics 口径漂移。
- **开放问题**:`merge_results` 是否需要支持非连续时间段(跳空)?建议首版要求时间递增、允许 gap,不允许重叠段(重叠报错)。
