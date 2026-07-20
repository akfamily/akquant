# RFC:定时器注册 API 命名收敛

> **状态**:已落地(Implemented,P1+P2+P3) · **日期**:2026-07-20 · **范围**:Python 策略层**定时器注册端**(`Strategy.schedule` / `Strategy.add_daily_timer`);回调端 `on_timer` **不动**。允许破坏性变更(硬改名,不留兼容别名),与 [hooks-rfc.md](hooks-rfc.md) 处理 `on_daily_rebalance_after_bar → on_cross_section` 的方式一致。
>
> 对标:**RQAlpha**(`scheduler.run_daily/_weekly/_monthly`)、**Zipline / QuantConnect**(`schedule_function(func, date_rule, time_rule)`)、**backtrader**(`add_timer(when, ...)` + `notify_timer`)。
>
> 与 [issue-329-fix-plan.md](issue-329-fix-plan.md) **正交**:本 RFC 只改注册端命名/分层,是体验优化;**不修**任何引擎缺陷。重命名后的方法仍走同一条用户 timer 路径,#329 的死锁必须由引擎修复独立解决。

---

## 0. 背景与动机

用户反馈:`on_timer` / `add_daily_timer` / `on_cross_section` / `on_daily_rebalance_after_bar` 这组接口看起来像"为同一个目的存在多套接口",不统一。审计后厘清:

- **`add_daily_timer`(或 `schedule`)↔ `on_timer` 是配对关系,不是冗余**:注册端(动词,主动调,通常在 `on_start`)埋 timer 事件,回调端(钩子,引擎回调)到点接住,靠 `payload` 分流。这是标准发布/订阅,与 backtrader `add_timer` + `notify_timer` 同源。
- **`on_cross_section` vs「`add_daily_timer` + `on_timer`」才是真正"能做同一件事"的两条路径**:前者是框架托管、成交时序对齐、绕开 #329 的横截面调仓专用入口;后者是通用自定义时点闹钟。[hooks-rfc.md](hooks-rfc.md) 已把二者职责分清,**本 RFC 不再触碰钩子端**。

真正的坏味道只在**注册端这一对方法**:

1. **动词分裂**:一个 `schedule`,一个 `add_..._timer`,同一族两个动词。
2. **周期标注不对称**:`add_daily_timer` 标了 `daily`,`schedule` 却没标 `once`。
3. **伪家族暗示**:有 `add_daily_timer` 却无 `weekly`/`monthly` 兄弟,`daily` 一词让用户误以为存在频率家族(RQAlpha 确实有)。

## 1. 目标与非目标

**目标**

- 注册端收敛为**同动词、成体系**的一族,消除动词分裂与周期标注不对称。
- 常见频率(每日/每周/每月)一行可达,不必在回调里手写 `_last_month` 之类的日历判定状态机。
- 保留"自由定义任意时点"的能力(通过底层原语 + 交易日历)。

**非目标**

- **不改回调端**:`on_timer(payload)` 命名良好、职责清晰,保留。
- **不新增回调钩子家族**:不做 `on_weekly` / `on_monthly`。频率进**注册端动词**(零引擎风险),绝不进**钩子端**(那会重蹈 #324/#329"钩子无独立时钟刻度"覆辙)。这是 hooks-rfc「频率不进钩子名」原则的正确适用边界:**它约束钩子端,不约束注册方法**。
- **不修 #329**:见 [issue-329-fix-plan.md](issue-329-fix-plan.md)。

## 2. 设计原则

1. **注册端 ≠ 钩子端**:注册方法用动词、可带周期(recurrence 是其核心语义,有信息量);钩子用 when-not-what、频率进回调。两套原则,别混用。
2. **分层:方法族托底常见,原语托底自由**。便捷方法覆盖 80% 常见频率;底层 `schedule` 原语 + 交易日历覆盖任意长尾节奏。
3. **一个频率方法只锁一个明确语义**。周/月频只认「首个交易日」一个定义,**不做 offset**(不追随 zipline `DateRules(days_offset=…)` 的参数化 DSL)。需要 month_end/offset/每两周等长尾节奏,回落到 `schedule` + 交易日历自行枚举。
4. **破坏性变更集中一次**,配套更新示例、中英文档、教材映射表、CHANGELOG。

## 3. 现状审计

| API | 角色 | 触发 | 命名问题 | 处置 |
|---|---|---|---|---|
| `schedule(trigger_time, payload)` | 注册·单次 | 任意单个时刻 | 未标 `once`,与 `add_daily_timer` 动词不一致 | **保留为自由原语**(语义已清晰,可选补 `_trading_days` 暴露) |
| `add_daily_timer(time_str, payload)` | 注册·每日 | 每交易日某时点 | 动词分裂 + `daily` 暗示伪家族 | **改名** `schedule_daily` |
| `on_timer(payload)` | 回调 | 事件当拍 | 无 | **保留不动** |

## 4. 变更提案

### 4.1 收敛为 `schedule` 方法族

```python
schedule(when, payload)              # 单次,任意时刻(保留)
schedule_daily(time_str, payload)    # 每日(add_daily_timer 改名)
schedule_weekly(time_str, payload)   # 每周首个交易日(新增,可选)
schedule_monthly(time_str, payload)  # 每月首个交易日(新增,可选)
```

- 同动词 `schedule_*`,成体系;`daily` 此刻是"相对单次的重复",语义自洽而非伪家族。
- `schedule_weekly/_monthly` **内部纯 Python**:在引擎已知的交易日序列上算出目标日期,逐个调 `schedule` 埋普通 timer 事件——**零引擎改动、零新增事件刻度**。全部回调仍汇进 `on_timer`。
- 每个非日频方法**只认「首个交易日」**:遇周初/月初停牌自动顺延到该周/月首个有交易的日子(直接在真实交易日集合上枚举,天然不踩停牌/节假日)。**不提供 offset 参数**。

### 4.2 暴露交易日历(自由层的关键)

`add_daily_timer` 内部已依赖 `strategy._trading_days`。将其以只读属性暴露(如 `self.trading_days`),让"非常规节奏"能优雅落在 `schedule` 上:

```python
def on_start(self):
    days = self.trading_days
    for i, d in enumerate(self._first_of_each_week(days)):   # 每两周首个交易日
        if i % 2 == 0:
            self.schedule(d.replace(hour=9, minute=30), "rebalance")
```

**可选** helper(本周/本月第 N 个 / 倒数第 N 个交易日):给了,长尾需求从"用户自己写循环"降为"调 helper 再 schedule";不给也能用,只是啰嗦。是否提供属产品取舍,不阻塞 4.1。

### 4.3 明确不做的方向(备选,记录否决理由)

- **方向:zipline / QuantConnect `DateRules`/`TimeRules` DSL**。最通用、能自然扩展 weekly/monthly + offset,但:① 引入一套规则对象类层级,与 akquant「payload → `on_timer` 极简配对」风格冲突;② `schedule_function` 本质是注册可调用对象,绕开 `on_timer`,稀释"所有 timer 汇进 on_timer"的干净不变量;③ 其核心卖点(频率做成调度器一等公民)与 hooks-rfc 已确立的「频率进回调判断」哲学在方向上对撞。**否决——过重且反哲学。**
- **方向:backtrader 单方法 `add_timer(when, repeat, ...)`**。用一个方法 + 参数覆盖 once/daily。但:① `when` 吃 `SESSION_START/END` token 会**撞 hooks-rfc P5**(刚硬移除 `on_session_end` 及会话终定时器,改用 `ctx.session`);② 单方法多参数 = stringly-typed,IDE 补全弱、错值到运行时才炸;③ 一旦约束成「只 once/daily、when 不吃 session」,`repeat` 参数退化到无信息量,实质等价于 4.1 的平行方法,却更难用。**否决——采纳其回调哲学(单一 `on_timer` 汇聚),不采纳其注册形态。**

## 5. 破坏性变更清单(供 CHANGELOG)

1. **改名** `add_daily_timer` → `schedule_daily`(硬改名,不留别名;PyPI 公开 API)。
2. **新增**(可选,本 RFC 可分期)`schedule_weekly` / `schedule_monthly`,语义锁「首个交易日」,不支持 offset。
3. **新增**(可选)`Strategy.trading_days` 只读属性 + 日历 helper。
4. 同步更新:`examples/` 定时器相关示例、中英策略指南与 API 参考、教材对应章与示例映射表。
5. `__engine_rule_version__`:**不变**——本 RFC 纯 Python 注册端命名,不改引擎行为、不改成交结果。

## 6. 分期实施

- **P1(已完成)**:`add_daily_timer` → `schedule_daily` 硬改名 + 文档/示例/教材同步。
- **P2(已完成)**:暴露 `trading_days` + `schedule_weekly` / `schedule_monthly`(锁首交易日语义)。
- **P3(已完成)**:日历 helper `nth_trading_day_of_month` / `nth_last_trading_day_of_month` / `nth_trading_day_of_week`。
- **(已完成)** `schedule_daily` / `schedule_monthly` 的 docstring 已补引导:**横截面调仓优先用 `on_cross_section`**(时序正确、绕开 #329),`schedule_*` + `on_timer` 用于通用自定义时点——作为 #329 修复之外的额外一道文档保险。

实现落点:`python/akquant/strategy_scheduler.py`(`schedule_daily`/`schedule_weekly`/`schedule_monthly`/`_nth_per_group` 等 impl)、`python/akquant/strategy.py`(公开方法 + `trading_days` 属性 + 三个日历 helper)。回归测试:`tests/test_api_rename_scheduler.py`(10 项)。

## 7. 验收 / 测试

- `schedule_daily` 改名后,原 `add_daily_timer` 用例全部迁移并通过;旧名调用应报错(无兼容别名)。
- `schedule_weekly/_monthly`:构造周初/月初停牌数据,断言触发落在该周/月**首个有交易的日子**,且回调进入 `on_timer` 携带正确 payload。
- `trading_days` 属性:断言与引擎内部交易日序列一致、只读。
- 示例:定时器相关 `examples/` 更新为可实跑(`exit 0`),ruff check/format 通过。
