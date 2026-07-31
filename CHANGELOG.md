# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `get_account()` 账户快照新增 `free_margin` 字段（= `equity - used_margin`），表示真正可用于新开仓的可用保证金，与下单因保证金不足被拒时日志里的 `Available` 口径一致；期货保证金账户下 `cash`（现金余额）通常大于 `free_margin`，股票现金账户下二者相等。同时修正了 `cash` 在文档与 docstring 中「可用资金」的误导性表述，明确其为「现金余额」。
- `BacktestConfig` 新增 `days_per_year`（年化天数因子，默认 252；数字货币 24/7 市场可设 365）与 `risk_free_rate`（年化无风险利率，默认 0.0）两个字段，用于参数化 Sharpe/Sortino/波动率等风险指标的年化口径。`risk_free_rate` 默认 0 不改变任何现有数值。

### Changed
- **策略参数声明改为内联字段（破坏性）**：策略参数的单一事实来源现在是类体内联字段——直接用 `IntParam` / `FloatParam` / `BoolParam` / `ChoiceParam` / `DateRangeParam` 赋值（如 `fast = IntParam(10, ge=2, le=200)`），经 `self.params.fast` 只读访问；`self.params` 在实例构造期即已校验就绪且 frozen，不支持运行期赋值。
    - **移除**：构造函数签名参数风格（`__init__(self, fast=10): self.fast=fast` 不再作为参数声明入口）、`PARAM_MODEL = XxxParams` 间接层（不再需要单独定义 `ParamModel` 子类）、适配层内部的 `_validate_with_signature` / `_build_signature_schema` 签名回退路径。`get_strategy_param_schema` / `validate_strategy_params` 现在只读取内联字段，不再回退到 `__init__` 签名推断。
    - **行为变更**：`start_time` / `symbols` / `end_time` 现在仅在策略显式声明为对应字段时才会被注入，不再隐式兜底；`strict_strategy_params=True`（默认）下，`param_grid` 或运行期 payload 中的未知键、越界取值（超出 `ge`/`le`、不在 `choices` 内）会直接报错，不再静默忽略；`strict_strategy_params=False` 时未知键会回退到字段默认值构造。
    - **迁移指引**：`def __init__(self, fast=10): self.fast = fast` 改为类体字段 `fast = IntParam(10)`；原先读取 `self.fast` 的地方改为 `self.fast` -> `self.params.fast`；派生初始化（如指标构造）从 `__init__` 移到 `on_start`。示例见 `examples/02_parameter_optimization.py`、`docs/zh/guide/strategy.md` §6.4「参数声明」。
- **策略钩子收敛（破坏性，硬改名/移除，不保留兼容别名）**：(0) **移除 `on_session_start` / `on_session_end`**——调研显示二者近乎零真实用例（无任何真实策略/plugin 使用），且同类框架均不向策略暴露会话级回调；session 概念在引擎内部仍承重，需要按会话（如期货日/夜盘）分支的策略请在 `on_bar` / `on_tick` 内读取 `self.ctx.session`（`TradingSession` 枚举）。连带回滚了为 `on_session_end` 正确时序而加的"会话终定时器"机制。(1) **移除 `on_daily_rebalance`**——它与 `on_before_trading` 同阶段、同"前一交易日信息可见"窗口，属于按用户意图（调仓）而非按触发时刻命名的冗余钩子；调仓逻辑请迁入 `on_before_trading`。(2) **`on_daily_rebalance_after_bar` 更名为 `on_cross_section`**——按语义（当日首个跨标的完整 bar 切片就绪后触发的横截面同周期调仓）命名，避免 `daily` 隐含周/月频钩子家族的伪需求（频率应在回调内用日历判断）、以及旧名过长；触发时机、可见窗口与成交语义不变。函数式入口、`LiveRunner` 与内部 timer payload（Python 与 Rust `src/**` 同步）一并更新。详见 `docs/zh/meta/hooks-rfc.md`。
- **`buy`/`sell`/`submit_order` 返回类型变更（破坏性）**：三者现统一返回 `OrderReceipt`（原为 `str` 订单号），回测与实盘（`broker_live`）两种模式返回类型一致；实盘 `submit_order` 此前会将多腿委托（如反手拆分的平仓+开仓、开平分离）收窄为单一 id 字符串，现已修复为返回携带全部腿 id 的完整 `OrderReceipt`。取单个订单 id 用 `receipt.primary`（首腿 broker_order_id，兼容旧用法），取全部腿 id 用 `receipt.order_ids`；`str(receipt)` 取 `group_id`（逻辑委托的客户端订单号），关联成交请用 `trade.group_id` 而非逐个 order_id 比对；新增 `cancel_group(group_id)` 用于一次性撤销一个逻辑委托的全部腿。
- **回测指标口径变更（破坏性）**：Sharpe / Sortino 比率的分子改为「日收益算术均值 × `days_per_year`」做年化，替代原先的 CAGR（复合年化），与 pyfolio/empyrical/quantstats 等主流实现一致，并与分母 `√days_per_year` 的年化口径匹配。升级后历史报告的 Sharpe/Sortino 数值会变化，不可直接与旧版逐值对比；UPI 与 Calmar 仍沿用 CAGR 口径。
- 策略交易日边界回调已硬切改名：`before_trading(trading_date, timestamp)` 更名为 `on_before_trading(trading_date, timestamp)`，`after_trading(trading_date, timestamp)` 更名为 `on_after_trading(trading_date, timestamp)`。
- 旧回调名不再保留兼容别名；升级到当前版本后，若策略仍实现 `before_trading` / `after_trading`，将不会再被框架触发，请同步迁移到新名称。
- 新增 `on_pre_open(event)` 框架回调，用于表达“盘前决策，本次 open 成交”；该回调默认下单语义会自动解析为 `price_basis=open, bar_offset=1, temporal=same_cycle`，不再要求用户自行用 `on_timer` 拼装开盘成交时序。
- 中英文策略指南与 API 参考现已补齐 `on_pre_open` 的语义说明、使用边界、相邻回调对比与示例入口。
- 教材第 5 章与教材目录页现已同步补充完整 `on_xxx` 回调地图、学习路径与相关示例入口，便于用户从教材直接理解各类回调的职责边界，并新增 `on_pre_open` 的推荐用法。
- 示例体系已补充 `examples/50_framework_hooks_demo.py` 与 `examples/51_class_tick_callbacks_demo.py`，分别覆盖框架边界钩子与类风格 `on_tick` 的最小可运行案例。
- 示例体系新增 `examples/52_pre_open_demo.py`，用于演示 `on_pre_open -> on_order/on_trade -> on_bar` 的触发顺序与当日 open 成交语义。
- **策略 API 批量改名与硬删（破坏性，不保留兼容别名）**：0.3.x 期间对 `Strategy` 公开面做了一轮收敛，以下旧名**已从 `Strategy` 移除**，旧调用会直接 `AttributeError`。此前这些变更未记入本文件，特此补齐——从 0.2.x / 0.3.早期升级请按下表逐项迁移。

    | 旧 API | 新 API | 迁移写法 |
    | --- | --- | --- |
    | `get_cash()` | `cash` 属性 | `self.get_cash()` → `self.cash` |
    | `get_portfolio_value()` | `equity` 属性 | `self.get_portfolio_value()` → `self.equity` |
    | `get_positions()` | `positions` 属性 | `self.get_positions()` → `self.positions` |
    | `hold_bar(symbol)` | `get_holding_bars(symbol)` | 仅改名，签名不变 |
    | `order_target_positions(...)` | `rebalance_positions(...)` | 仅改名，签名不变 |
    | `order_target_weights(...)` | `rebalance_weights(...)` | 仅改名，签名不变 |
    | `place_bracket_order(...)` | `place_bracket(...)` | 仅改名，签名不变 |
    | `create_oco_order_group(a, b)` | `place_oco(a, b)` | 仅改名；新版两个入参同时接受订单 id 与 `OrderReceipt` |
    | `register_indicator(name, ind)` | `register_precomputed_indicator(name, ind)` | 旧名本就是该方法的薄别名，去别名后请直接用全名 |
    | `stop_buy(symbol, trigger_price, quantity, price)` | `submit_order(...)` | `submit_order(symbol=..., side="Buy", quantity=..., trigger_price=..., price=...)`；`price=None` 即触发后转市价 |
    | `stop_sell(...)` | `submit_order(...)` | 同上，`side="Sell"` |
    | `buy_all(symbol)` | `order_target_percent(...)` | `order_target_percent(symbol=..., target_percent=1.0)` |

    注：`cash` / `equity` / `positions` 现为**只读属性**（`property`），不能再当方法调用——`self.cash()` 会抛 `TypeError`。执行后端（`strategy.execution`）层的同名方法未改动，自定义 broker 无需跟随调整。
- **回测成交策略从 dict 收敛为 `FillMode` 对象（破坏性）**：`fill_policy` 不再接受 dict，`make_fill_policy()` 已移除（保留为抛 `TypeError` 的报错壳，错误信息内嵌完整迁移映射）；引擎侧 `set_fill_policy(price_basis, bar_offset, temporal)` 重塑为 `set_fill_mode(mode: ExecutionMode, timer_timing: str)`，非法的 `(basis, offset)` 组合在枚举层已不可表达。`get_fill_policy()` 保留不变（checkpoint 反查依赖）。**迁移映射**：

    | 旧 dict | 新 `FillMode` |
    | --- | --- |
    | `{"price_basis": "open"}` | `NextOpen()` |
    | `{"price_basis": "close", "bar_offset": 0}` | `CurrentClose()` |
    | `{"price_basis": "close", "bar_offset": 0, "temporal": "next_event"}` | `CurrentClose(timer_fill_timing="deferred")` |
    | `{"price_basis": "close", "bar_offset": 1}` | `NextClose()` |
    | `{"price_basis": "ohlc4"}` | `NextAverage()` |
    | `{"price_basis": "hl2"}` | `NextHighLowMid()` |
- **`LiveRunner` 从公开 API 移除（破坏性）**：实盘入口统一为 `run_live(...)` 函数门面，与 `run_backtest` 对称。`from akquant import LiveRunner` 会 `ImportError`；原先 `LiveRunner(...).run(cash=..., duration=...)` 的两步用法改为单次 `run_live(..., cash=..., duration=...)` 调用，配置参数一一对应。
- **`add_daily_timer` 已移除，更名为 `schedule_daily`（破坏性，不保留兼容别名）**：定时器注册端统一为 `schedule` / `schedule_daily` / `schedule_weekly` / `schedule_monthly` 家族（回调端 `on_timer` 不变）。升级后旧调用会直接 `AttributeError`。**迁移**：`self.add_daily_timer("14:55:00", "rebalance")` 改为 `self.schedule_daily("14:55:00", "rebalance")`，参数与触发语义完全一致。注意 `schedule_weekly` / `schedule_monthly` 依赖回测交易日历（`_trading_days`），在**实盘**下会记录一条 warning 并被忽略；实盘的周期性任务请用 `schedule_daily` 配合回调内的日历判断。详见 `docs/zh/meta/timer-api-rfc.md`。
- **实盘 `subscribe()` 不再静默无效**：`Strategy.subscribe()` 只把标的写入 `_subscriptions`，而该列表**仅**被回测消费——实盘（`run_live`）的订阅集在会话启动时由 `instruments` 一次性交给行情网关，此后调用 `subscribe()` 不会触达任何网关。此前这一调用完全无声，用户以为订阅成功却收不到行情。现在实盘下调用 `subscribe()` 会记录一条 warning，指明应把该标的加入 `run_live(instruments=[...])`（每个标的只告警一次）。行为本身未变（仍会记录进 `_subscriptions`），回测语义完全不受影响。
- **`supports_short_sell` 在 `broker_live` 下开始真正生效**：该字段此前在实盘下单路径上没有任何消费者（仅服务 `order_target*` 的目标仓位计算），broker 声明了 `False` 也照样把开空单发给柜台，只能等柜台报错。现在 `side=Sell` + `position_effect=open` 会在下单前被本地拒绝。内置 CTP 声明 `True`，不受影响；`ptrade` / `miniqmt` 只声明支持 `auto`，其显式 `open` 早已被 `supported_position_effects` 拦截，行为不变。若自定义 broker 实际支持融券却被拦，请修正其 `get_capabilities()` 中的 `supports_short_sell` 声明。

### Fixed
- 修复 `broker_live` 下 `buy()` / `sell()` 缺省参数不解析的问题：`quantity=None` 原先直接透传到拆腿逻辑并以 `TypeError` 崩掉——等价于 `set_sizer()` 在实盘静默失效；`symbol=None` 原先会把 `None` 送进柜台请求。现按回测同口径解析：`symbol` 取当前 bar/tick，买入量走 `strategy.sizer`，卖出量全平持仓。另外解析后 `quantity <= 0` 时不再向柜台发 0 手单，改为返回空回执（与回测一致）。一处有意偏离：卖出全平在实盘取**可用**持仓而非总持仓，因为 A 股 T+1 下按总量报单会被柜台整单拒绝，取可用量退化为部分卖出。
- 修复 `broker_live` 下 `get_instrument()` / `get_instruments()` / `get_instrument_field()` 对**任何**标的都抛 `KeyError` 的问题：`run_live` 原先从不调用 `_set_instrument_snapshots`，策略侧快照字典恒为空。现由 `LiveRunner` 从传入的 `Instrument` 列表灌入。受实盘入口形态限制，`Instrument` 仅能回读 9 个字段，`option_type` / `strike_price` / `expiry_date` / `underlying_symbol` / `settlement_*` 在实盘快照中为 `None`（回测不受影响，仍由 `InstrumentConfig` 灌入完整字段）。
- 修复 `client_order_id` 跨会话撞号的问题：原格式 `{broker}-coid-{序号}` 的序号每次 `run_live` 从 0 起，重启后第一笔单又是 `...-coid-1`，与柜台里同名历史委托冲突——把 `client_order_id` 当幂等键的柜台会直接拒单（实测中间件返回 409 Conflict）。现格式为 `{broker}-{8 位会话标记}-{会话内序号}`，会话标记每次启动重新生成，跨重启与多进程并行均不撞号。
- 修复 `on_after_trading` 里下 next-open（`bar_offset=1`）单晚一格成交的问题（#324）。它原先在「下一根 bar」到来时才惰性补触发，引擎时钟已越过日界，其中提交的 next-open 单被撮合守卫推迟一格（盘后下单期望次日开盘成交，实际到第三日）。现改为按日终边界定时器在「当日收盘点」的独立事件里触发（早于下一根 bar），使其中提交的次日开盘单落在下一根 bar，而非晚一格。默认（非 precise）与 precise 模式下均已修正；`on_before_trading` 等开始型钩子本就与 `on_bar` 同拍触发，不受影响。`__engine_rule_version__` 相应升至 `1.3.1`。
- 修复 `on_pre_open` 未兑现「本次 open 成交」契约的问题（#324 家族）。其盘前定时器原先排在当日首根 bar 的同一时刻，订单 `created_at` 与该 bar 同拍、被 next-open 守卫拦截，导致实际成交在**下一日 open**（可由 `examples/52_pre_open_demo.py` 复现）。现将盘前定时器排在首根 bar 之前 1ns，使 `on_pre_open` 中下的默认开盘单落在**当日 open**，与文档承诺一致。
- 修复短回测区间下 Sharpe/Sortino 因「分子用 CAGR、分母用日波动 × √252」口径不一致而出现异常巨大值（如期货场景 Sharpe 高达数千万）的问题；改用日收益算术均值年化后数值回归合理量级。
- 修复 `on_timer` / `schedule_daily`（原 `add_daily_timer`）场景下，订单级 `fill_policy={"price_basis":"close","bar_offset":0,"temporal":"same_cycle"}` 未在当日 timer 事件内生效的问题；相关卖单现在会按当日 timer 时间与当日 close 成交，不再延后到下一交易日。
- 修复 framework 内部 `__framework_rebalance__` / `__framework_boundary__` timer 被误参与 same-cycle 撮合与终态统计的问题，避免出现 `+1ns` 的伪成交、partial fill 被过早补满，以及权益曲线尾部多出额外采样点。
- 修复 `on_pre_open` 首个交易日可能因延迟注册 timer 而不触发的问题；相关 framework timer 现会在回测事件循环开始前直接注入引擎，从而保证首日也能按预期开盘语义执行。

## [0.2.14] - 2026-04-21

### Added
- 新增 `on_expiry` / `on_expiry(ctx, event)` 回调与流式 `expiry` 事件；当引擎实际执行 `expiry_date` 驱动的到期结算或到期移除后，策略与 `run_backtest(..., on_event=...)` 均可收到通知。
- 新增 `examples/49_on_expiry_demo.py`，演示 `on_expiry` 回调、流式 `expiry` 事件以及结算后读取最新持仓状态。

### Changed
- 中英文 API 文档、策略指南、Quickstart、教材与示例总览已补充 `on_expiry` 的能力说明、使用边界与示例入口。

### Fixed
- 修复 `StrategyContext` / `ExpiryEvent` 类型声明错位问题，并同步完善 `BacktestStreamEvent`、函数式策略 `on_expiry` 与相关示例的类型签名，使 `mypy` / `pre-commit` 校验重新通过。

## [0.2.13] - 2026-04-20

### Fixed
- 修复日内调仓时机与行情数据跨符号排序问题，减少多标的同周期调仓时的执行顺序偏差。

## [0.2.12] - 2026-04-20

### Added
- 新增显式滑点策略写法，`run_backtest`、`StrategyConfig`、策略级 `strategy_slippage` 与订单级下单接口现支持 `{"type": "percent"|"fixed"|"ticks"|"zero", "value": ...}`。
- `short()` / `cover()` 现支持 `tag`、`fill_policy`、`slippage` 与 `commission`，与 `buy()` / `sell()` 的下单覆盖能力保持一致。

### Changed
- 期货教材示例与相关中文文档已切换为显式滑点 policy 写法，优先推荐 `percent` / `fixed` / `ticks`，并补充了成交时点与滑点语义的防踩坑说明。
- 内置 `broker_profile` 的滑点模板已迁移到显式 policy 表示，避免继续依赖裸数值语义。

### Deprecated
- 裸 `float` / `int` 形式的 `slippage` 仍保持兼容，但已进入弃用路径；当前会触发 `DeprecationWarning`，且对可疑的大滑点值给出明确提示。

### Fixed
- 为 `ticks` 滑点补充了 `tick_size` 解析与校验逻辑，避免多标的或缺少合约最小变动价位时静默使用错误滑点。

## [0.2.11] - 2026-04-16

### Fixed
- Fixed non-deterministic backtest metrics in multi-symbol runs when bars share the same timestamp.
- Ensured terminal equity/cash/margin snapshots are overwritten with the fully updated portfolio state before final metric calculation.

## [0.2.10] - 2026-04-15

### Added
- `run_backtest` now supports optional `on_event` callback and can emit stream events directly.
- Added `ChinaOptionsConfig` with prefix-level option fee configuration (`fee_by_symbol_prefix`).
- Added Engine API `set_options_fee_rules_by_prefix(symbol_prefix, commission_per_contract)`.
- Added readable time-string properties for `Trade.timestamp`, `Order.created_at`, and `Order.updated_at`.

### Changed
- `run_backtest_stream` is removed; stream scenarios should call `run_backtest(..., on_event=...)`.
- `run_backtest` always uses the unified stream core; runtime rollback flag `_engine_mode` is removed.
- Futures fee Engine API naming is standardized to `set_futures_fee_rules*`; legacy `set_future_fee_rules*` is removed.

## [0.2.9] - 2026-04-15

### Fixed
- Fixed benchmark return series index normalization and improved validation for report generation.

## [0.2.8] - 2026-04-14

### Added
- Added strategy start-time configuration support to the engine.

### Changed
- Improved futures margin risk handling.

## [0.2.7] - 2026-04-09

### Fixed
- Applied conditional open-price optimization according to the configured price-basis policy.

## [0.2.6] - 2026-04-09

### Added
- Added walk-forward model lifecycle management for the ML workflow.
- Added multi-symbol backtesting support and improved backtest window configuration.
- Added dictionary-based multi-symbol input support to the optimization workflow.

### Changed
- Improved rolling-training scheduling in parameter optimization.

## [0.2.5] - 2026-04-08

### Fixed
- Fixed missing `ctx.orders` in order-event callbacks.

## [0.2.4] - 2026-04-07

### Added
- Added same-bar cash reuse for sell-then-buy flows.

### Changed
- Distinguished automatic quantity adjustment from explicitly sized orders.

## [0.2.3] - 2026-04-06

### Fixed
- Fixed cross-category operator conflict detection in the factor-expression parser.

### Changed
- Cleaned up completed migration docs and outdated links.
- Refined examples by removing unused imports and obsolete configuration parameters.

## [0.2.2] - 2026-04-03

### Added
- Added `catalog_path` support for specifying the data directory in backtests.
- Added Top 8 rejected-order reason summaries in backtest output.

## [0.2.1] - 2026-04-02

### Added
- Added order-level execution overrides and strategy-level default execution settings.
- Added `NextClose` execution mode and unified the `symbols` parameter behavior.

### Changed
- Replaced `ExecutionMode` with `ExecutionPolicyCore`.
- Simplified the `price_basis` options under `fill_policy`.
- Updated the execution-semantics documentation and migration guidance.
