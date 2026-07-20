# 修复方案:issue #329 —— on_timer + 日频数据 deferred 幻影订单永久冻结

> **状态**:草案(Draft) · **日期**:2026-07-20 · **范围**:Rust 引擎核心(`src/execution/`、`src/pipeline/`、`src/risk/`、`src/settlement/`) · **对应 issue**:[#329](https://github.com/akfamily/akquant/issues/329) · **性质**:严重缺陷(资金可被永久冻结,静默失败)
>
> 与 [timer-api-rfc.md](timer-api-rfc.md) **正交**:本方案修引擎缺陷;那份收敛定时器注册 API 命名。API 重构**不能**替代本修复——重命名后的 `schedule_daily` 仍走同一条会死锁的用户 timer 路径。

---

## 0. 结论先行

这是一个**真实、严重**的核心引擎缺陷,报告者的四环根因分析已逐条对照源码核实,全部成立。触发组合(`on_timer` + 日线 + 满仓横截面轮动 + 组合内含停牌标的)常见且结构性命中,除拒单原因外无任何告警。**建议优先、独立修复,不等 API 重构。**

## 1. 缺陷链(已对照源码核实)

| 环 | 机制 | 源码位置 |
|---|---|---|
| 1. 同周期 deferral | 买单遇同批(`created_at` 相同)、跨标的、仍活跃的 reduce-first 单被推迟,保持 `New` 等 `finalize_timestamp` 统一处理;defer 判定**逐 bar 重估** | `simulated.rs:226-242`(判定)、`343-364`(注册进 `deferred_same_cycle_order_ids`) |
| 2. 用户 timer 清空 deferred 注册表 | Timer 事件走 `on_event` → `simulated.rs:704` **无条件** `prepare_slice_tracking(current_time)`,时间戳一变即清空 `deferred_same_cycle_order_ids`(`136-143`);而 `finalize_timestamp` 只在**下一根 Bar/Tick 时序前进**时触发(`data.rs:194-195`)——届时注册表已空,deferred 买单永远等不到 finalize。用户 timer 的 payload 是 `__daily__\|...`,**不命中** `data.rs:166-171` 的 `__framework_cross_section__\|` finalize 分支 | `simulated.rs:704`、`data.rs:164-180`、`execution.rs:44,62` |
| 3. 无法成交的卖单永不消亡 | 停牌标的当日无 bar,matcher 对 `volume<=0` 静默返回 `None`(`common.rs:180-182`),不成交/无回报/不拒单;默认 TIF=GTC 永不过期;日结过期仅 partition `TimeInForce::Day`(`settlement/manager.rs:166-168`);`reject_missing_symbol_orders` 要求 `created_at==last_timestamp` 且 `bar_offset==1`(`data.rs:75-80`),timer 单不命中 | `common.rs:180-182`、`settlement/manager.rs:166-168`、`data.rs:65-99` |
| 4. 幻影单压垮自由保证金 | 提交时 `CashMarginRule` → `check_affordability` 把所有 `status==New` 单投影进组合(`project_active_orders_into`,`risk/common.rs:365-406`);幻影买单花光投影现金 → `available=max(0,free_margin*safety)=0`(`470-473`)→ `required<=available` 对减仓卖单(`required` 仅佣金)同样失败(`475`)→ 无成交 → 单堆不散 → **永久冻结** | `risk/common.rs:365-406,426-506` |

**闭环**:停牌卖单永生 → 同批买单每根 bar 被重新 defer 且永不 finalize → 幻影 `New` 买单堆积 → 投影自由保证金归零 → 后续所有单(含能自救的卖单)被拒 → 无成交 → 单堆不消散。唯一出路是停牌标的复牌。

## 2. 目标与非目标

**目标**

- 恢复不变量:**任何时间戳前进之前,当前时间戳必先 finalize**(无论触发前进的是 Bar/Tick 还是任意 Timer)。
- 被钉住的 reduce 单(标的无 bar)不得跨 slice 无限期扣押同批买单。
- 无法成交的订单最终有终态(过期/撤销/拒单),不再以 `New` 永生。
- 真实减仓/平仓卖单不因无关在途买单占用的投影保证金被"保证金不足"拒掉。

**非目标**

- 不改同周期资金接力(sell 释放现金供 same-cycle buy 使用)的**正确**行为——只堵住其被 timer 打断后的病态分支。
- 不改 #307 next-open 撮合守卫。
- 不改定时器注册 API 命名(见 [timer-api-rfc.md](timer-api-rfc.md))。

## 3. 修复提案

报告者给了两个候选。**推荐候选 1(恢复 finalize 不变量)作为主修**:它修的是"不变量被破坏"这一根本问题,风险面清晰,golden 可锁行为;候选 2 改 defer 判定逻辑,对现有同周期成交/资金接力影响面更大,作为备选。

### 3.1【主修】恢复"时序前进前必先 finalize"不变量

**改法 A(引擎侧,首选)**:在 `data.rs` 的 `FeedAction::Timer` 分支(`164-180`),当 timer 时间戳 `> last_timestamp` 时,**对所有 timer**(不止 `__framework_cross_section__\|`)先调 `finalize_current_timestamp`,再派发 timer 事件。即把现有仅给框架横截面 timer 的待遇,推广到全部 timer。

**改法 B(执行器侧,与 A 二选一或叠加加固)**:让 `prepare_slice_tracking`(`simulated.rs:704`)**只在 Bar/Tick 事件上重锚**,忽略 Timer 事件——避免 timer 提前清空尚未 finalize 的 `deferred_same_cycle_order_ids`。

> 推荐先落 A(治本:先 finalize 再让 timer 推进时钟,deferred 单在被清空前已被处理);B 作为防御性加固,防止未来新增的 timer 路径重蹈覆辙。

**验收**:先让一笔买单 defer,再以更新时间戳触发 Timer `on_event`,断言该买单仍被 finalize/成交(当前实现失败,可先固化 bug 再修绿)。

### 3.2【卫生项 a】被钉住的无 bar reduce 单需有终态

`reject_missing_symbol_orders`(`data.rs:65-99`)当前按 `created_at == last_timestamp` 判定,timer 创建、标的无数据的订单永不命中。改为**按订单首个可撮合时间**判定(而非严格等于 `last_timestamp`),让停牌标的的挂单在其应撮合的 slice 结束时被拒/撤,而非永生。

**验收**:同批 reduce 单标的无 bar 时,断言买单**不跨 slice 重新 defer**——下一 slice 以成交或执行时拒单终结,而非永远 `New`。

### 3.3【卫生项 b】日结过期需同步通知执行器

日结过期(`settlement/manager.rs:164-175`)把 `TimeInForce::Day` 单标记 `Expired` 后,`data.rs:380-382` 仅将其 push 回 `orders`,**未调** `engine.execution_model.on_cancel`。执行器(`SimulatedExecutionClient.orders`)侧因此残留僵尸单,会继续参与 defer 判定与保证金投影。修复:过期路径同步调用 `on_cancel`,与 `reject_missing_symbol_orders`(`data.rs:94`)已有的做法对齐。

### 3.4【可选卫生项 c】提交时保证金检查豁免真实减仓单

提交时 `check_affordability` 对**真实减仓/平仓方向**(`margin_delta <= 0`,持仓充足)的订单,其资金需求仅为佣金,不应因无关在途买单占用的投影保证金被拒。可在提交闸门对这类订单豁免(执行时 `safety=0` 的检查作为真正的资金闸门兜底)。

> 此项**独立于死锁修复**:即便 3.1 已破锁,豁免也能让"减仓自救"更稳健。但它触及风控语义,建议单独评估;需断言无持仓的 `Auto` 卖单(实为开空)仍被正确拒掉。

## 4. 破坏性变更与版本

- 行为变更:timer 驱动策略的成交时序会因"timer 前先 finalize"而改变(此前 deferred 单晚成交或永不成交)。
- `__engine_rule_version__`:`1.3.1` → **`1.3.2`**。
- golden:现有 golden 策略不使用 `on_timer` 死锁路径,指标应不变;仅 `engine_rule_version` 随行为变更递增。若有基线漂移需逐项复核并记录。

## 5. 回归测试清单

1. **Rust `execution/simulated.rs`**:买单先 defer → 更新时间戳触发 Timer `on_event` → `finalize_timestamp`,断言该买单仍被 finalize/成交。
2. **Rust**:同批 reduce 单标的无 bar(停牌)时,断言买单不跨 slice 重新 defer,下一 slice 以成交或执行拒单终结。
3. **Rust `risk/common.rs`**:`active_orders` 放大额 `New` 买单 + 现金≈0,断言(修前)减仓卖单被 `Available:0` 拒;若采纳 3.4,断言真实减仓卖单放行、无持仓 `Auto` 卖单仍被拒。
4. **Python 端到端**:直接用 issue 所附 `repro_on_timer_deadlock.py`——断言停牌日后仍有新成交、无订单存活超过次一交易日、卖单不出现 `Available:0` 拒单。

## 6. 与报告者协作

报告者(@neoblackxt)已表示可提交含上述回归测试的 PR(候选 1 或 2)。**建议在 issue 回复中明确倾向候选 1(改法 A + 卫生项 a/b)**,据此提 PR;3.4 作为可选后续单独评估。

## 7. 临时 workaround(先给用户)

横截面日频调仓改用 `on_cross_section`(0.3.7 前名 `on_daily_rebalance_after_bar`):其框架 timer 走 `__framework_cross_section__\|` 路径,会正确触发 slice finalize,天然绕开本死锁。通用自定义时点的 `on_timer` 在本修复落地前,应避免用于满仓、组合含停牌标的的场景。
