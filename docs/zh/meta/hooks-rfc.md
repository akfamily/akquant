# RFC:策略生命周期钩子体系重构

> **状态**:草案(Draft) · **日期**:2026-07-19 · **范围**:Python 策略层生命周期回调(`Strategy.on_*`);**允许破坏性变更(硬改名 / 移除,不保留兼容别名)**,与既往 `before_trading → on_before_trading` 的处理方式一致。
>
> 对标:**聚宽 JoinQuant**(`before_trading_start` / `after_trading_end` / `handle_data`)、**米筐 RQAlpha**(`before_trading` / `after_trading` / `handle_bar` / `open_auction`)、**Zipline**(`before_trading_start` / `handle_data`)、**backtrader**(`next` + `add_timer`,自行判跨日)、**nautilus_trader**(事件驱动 `on_*`)。

---

## 0. 背景与动机

issue #324 暴露出一类**"结束/边界型"钩子的成交时序缺陷**,顺藤摸出更系统的问题:部分钩子在事件驱动引擎里**没有独立的时钟刻度**,被"惰性补触发"到下一根 bar 的事件里,导致其中提交的 next-open 单晚一格成交;同时钩子命名与职责存在**冗余与"按意图命名"**的坏味道。

三个根因:

1. **结束型钩子无独立事件**:日终 / 会话终在时间线上处于两根 bar 之间的缝隙,惰性检测时引擎时钟已越过边界 → 订单 `created_at` 落到下一拍 → 撮合守卫(next-open 不得在提交拍成交,#307)把成交再推一格。
2. **`on_pre_open` 未兑现契约**:文档承诺"盘前信号、本次 open 成交",但定时器排在当日首根 bar 的**同一时刻**,订单撞上守卫 → 实际成交在**下一日 open**(官方示例 `52_pre_open_demo.py` 可复现)。
3. **命名/冗余**:`on_daily_rebalance` 与 `on_before_trading` **同阶段、同可见窗口**(功能重叠);`on_daily_rebalance*` 按"用户意图(调仓)"命名,而非按"触发时刻",违反生命周期钩子 when-not-what 原则。

## 1. 目标与非目标

**目标**

- 所有钩子的成交时序**自洽且符合文档契约**:next-open 单落在"提交拍之后的第一根 bar"。
- 结束/边界型钩子在**边界处的独立事件**里触发(与 `on_after_trading` / `on_session_end` 已落地的定时器方案一致)。
- 钩子命名**按时刻(when)**而非按意图(what);消除同一时刻的多名字冗余。
- 下单只发生在"决策型"钩子(`on_bar` / `on_tick` / `on_pre_open`);"结束型"钩子(`on_after_trading` / `on_session_end`)定位为**观测/收尾**。

**非目标**

- 不改 `on_bar` / `on_tick` / `on_timer` / `on_order` / `on_trade` / `on_reject` / `on_portfolio_update` / `on_start` / `on_stop` 等命名良好、职责清晰的回调。
- 不改引擎撮合核心(#307 守卫按设计保留)。

## 2. 设计原则

1. **When-not-what**:钩子名描述"何时触发",不描述"你打算做什么"。`on_bar` 而非 `on_signal`。
2. **一时刻一钩子**:同一时间线位置不应有两个语义重叠的回调。
3. **结束型 = 观测型**:日终/会话终用于统计、归档、清理;需要"收盘决策、次日开盘执行"时用 `on_bar`(next-open)或 `on_pre_open`。
4. **破坏性变更集中一次**:本轮硬改名,配套更新示例、文档、教材映射表与 CHANGELOG。

## 3. 现状审计(时序 + 命名)

| 钩子 | 类别 | 触发时机 | 时序 | 命名 | 处置 |
|---|---|---|---|---|---|
| `on_bar` / `on_tick` / `on_timer` | 决策 | 事件当拍 | ✓ | ✓ | 保留 |
| `on_before_trading` | 开始 | 当日首个常规事件 | ✓ | ✓(when) | 保留 |
| `on_pre_open` | 开盘 | 当日首根 bar timer | **✗ 落到次日 open** | ✓ | **修时序** |
| `on_after_trading` | 结束 | 日终 timer | ✓(已修 #324) | ✓ | 保留 |
| `on_session_start`(已移除) | 开始 | 曾在会话首个事件 | ✓ | 离群/近乎零用例 | **已移除**,改用 `ctx.session` |
| `on_session_end`(已移除) | 结束 | 曾用会话终 timer | — | 离群/近乎零用例 | **已移除**(连同会话终定时器机制回滚) |
| `on_daily_rebalance`(已移除) | 开始 | 曾与 `on_before_trading` 同阶段 | ✓ | **✗ 冗余+按意图** | **已移除,并入 `on_before_trading`** |
| `on_daily_rebalance_after_bar` → `on_cross_section` | 日中 | 当日首个完整切片后 timer | ✓ | 原名按意图 | **已改名为 `on_cross_section`** |

## 4. 变更提案

### 4.1 修复 `on_pre_open` 时序(bug)

- **根因**:`collect_pre_open_timer_entries` 把定时器排在 `start_ns`(当日首根 bar 时间戳),与首根 bar 同刻,订单 `created_at == 首根 bar ts` → 守卫拦截 → 次日成交。
- **修法**:把 pre_open 定时器排在**首根 bar 之前**(`start_ns - 1`,或市场集合竞价时刻),使 `created_at < 首根 bar ts`,当日 open 可成交。与 `on_after_trading` 用 `end_ns + 1`、`on_session_end` 用会话终定时器**同源对称**。
- **验收**:新增回归测试断言"`on_pre_open(T)` 下的默认 open 单成交在 **T 日 open**",覆盖首日与非首日、日线与日内。

### 4.2 移除 `on_daily_rebalance`,并入 `on_before_trading`

- 二者同阶段、同可见窗口("前一交易日信息可见");调仓逻辑本就应写在 `on_before_trading`。
- **破坏性**:实现 `on_daily_rebalance` 的策略需将逻辑迁入 `on_before_trading`。原先"先 `on_before_trading` 后 `on_daily_rebalance`"的两段顺序在同一回调内自然合并。
- 移除相关的 `_dispatch_daily_rebalance_if_needed` / `_framework_daily_rebalance_done_date` 分支。

### 4.3 `on_daily_rebalance_after_bar` → `on_cross_section`(已定名并落地)

- 其功能(当日首个跨标的完整 bar 切片后做同周期横截面处理)有真实用例,**保留**;改为按语义命名。
- **最终名**:`on_cross_section`。曾评估 `on_daily_first_slice` / `on_open_slice`,但前者的 `daily` 隐含"周/月频"钩子家族的伪需求(频率应在回调内用日历判断,不进钩子名),后者的 `open` 与 `on_pre_open` 撞词易混。`on_cross_section` 用量化标准术语「横截面」,无频率、无 open 歧义。
- 内部符号一并改名:payload `__framework_after_bar_rebalance__` → `__framework_cross_section__`(Python 与 Rust `src/**` 同步)、`collect_cross_section_timer_entries`、`_prime_framework_cross_section_timers`、`_framework_cross_section_*`、phase `cross_section`。

### 4.4 移除 `on_session_start` / `on_session_end`(已落地)

调研结论(见下"会话钩子调研"):这两个回调**近乎零真实用例**——全 workspace(含两个 plugin 仓)仅"全钩子演示"与框架行为测试引用,无任何真实策略使用;同类框架(聚宽/米筐/Zipline/backtrader/QuantConnect)均**不向策略暴露会话级 start/end 回调**;且 `on_session_end` 为正确时序还背了一整套"会话终定时器"机制。成本/收益严重失衡。

- **硬移除**两个回调,并**回滚 P1 的会话终定时器机制**(`dispatch_session_end_timer`、`_prime_framework_session_end_timers`、Rust `effective_session_at`、`_framework_session_end_*` 状态、`dispatch_time_hooks` 会话检测块)。
- **能力不丢**:session 概念在引擎内部(clock/`is_trading_time`/结算)仍承重;需要按会话(如期货日/夜盘)分支的策略,在 `on_bar` / `on_tick` 内读取 `self.ctx.session`(`TradingSession` 枚举)自行判断。
- `on_after_trading` 的日终定时器与本项无关,保留不受影响。

## 5. 破坏性变更清单(供 CHANGELOG)

1. **移除** `on_daily_rebalance`:迁移到 `on_before_trading`。
2. **改名** `on_daily_rebalance_after_bar` → `on_cross_section`。
3. **行为变更** `on_pre_open`:成交由"次日 open"修正为"本次 open"(修复 #324 家族);依赖旧(错误)行为的策略结果会变化。
4. **行为变更** `on_after_trading` / `on_session_end`:改为在边界处独立事件触发,其中的 next-open 单不再晚一格(#324)。
5. `__engine_rule_version__` 递增至 `1.3.1`,覆盖本轮全部 #324 家族修复。

## 6. 分期实施

- **P1(已完成)**:`on_after_trading`(日终定时器)、`on_session_end`(会话终定时器)时序修复 + 回归测试 + golden 重生成(#324)。
- **P2(已完成)**:`on_pre_open` 时序修复 + 回归测试(4.1)。
- **P3(已完成)**:`on_daily_rebalance` 移除、`on_daily_rebalance_after_bar` → `on_cross_section` 改名(4.2 / 4.3);同步代码、测试、示例、中英策略指南与 API 参考、教材第 5 章与示例映射表。
- **P4(已完成)**:会话钩子适用边界文档化(过渡)。
- **P5(已完成)**:移除 `on_session_start` / `on_session_end`,回滚会话终定时器机制,改文档为 `ctx.session` 范式(4.4)。

## 7. 验收 / 测试

- 每个"结束/边界型 + 开盘型"钩子均有一条**锚定成交拍**的回归测试(default 与 precise 两种边界分发模式)。
- `examples/` 中 `50_framework_hooks_demo.py` / `52_pre_open_demo.py` 更新为**断言**成交时刻,而非仅打印(防止契约再次静默漂移)。
- golden 基线保持指标不变(现有 golden 策略不使用这些钩子);仅 `engine_rule_version` 随行为变更递增。
