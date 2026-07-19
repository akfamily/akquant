# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `get_account()` 账户快照新增 `free_margin` 字段（= `equity - used_margin`），表示真正可用于新开仓的可用保证金，与下单因保证金不足被拒时日志里的 `Available` 口径一致；期货保证金账户下 `cash`（现金余额）通常大于 `free_margin`，股票现金账户下二者相等。同时修正了 `cash` 在文档与 docstring 中「可用资金」的误导性表述，明确其为「现金余额」。
- `BacktestConfig` 新增 `days_per_year`（年化天数因子，默认 252；数字货币 24/7 市场可设 365）与 `risk_free_rate`（年化无风险利率，默认 0.0）两个字段，用于参数化 Sharpe/Sortino/波动率等风险指标的年化口径。`risk_free_rate` 默认 0 不改变任何现有数值。

### Changed
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

### Fixed
- 修复 `on_after_trading` 里下 next-open（`bar_offset=1`）单晚一格成交的问题（#324）。它原先在「下一根 bar」到来时才惰性补触发，引擎时钟已越过日界，其中提交的 next-open 单被撮合守卫推迟一格（盘后下单期望次日开盘成交，实际到第三日）。现改为按日终边界定时器在「当日收盘点」的独立事件里触发（早于下一根 bar），使其中提交的次日开盘单落在下一根 bar，而非晚一格。默认（非 precise）与 precise 模式下均已修正；`on_before_trading` 等开始型钩子本就与 `on_bar` 同拍触发，不受影响。`__engine_rule_version__` 相应升至 `1.3.1`。
- 修复 `on_pre_open` 未兑现「本次 open 成交」契约的问题（#324 家族）。其盘前定时器原先排在当日首根 bar 的同一时刻，订单 `created_at` 与该 bar 同拍、被 next-open 守卫拦截，导致实际成交在**下一日 open**（可由 `examples/52_pre_open_demo.py` 复现）。现将盘前定时器排在首根 bar 之前 1ns，使 `on_pre_open` 中下的默认开盘单落在**当日 open**，与文档承诺一致。
- 修复短回测区间下 Sharpe/Sortino 因「分子用 CAGR、分母用日波动 × √252」口径不一致而出现异常巨大值（如期货场景 Sharpe 高达数千万）的问题；改用日收益算术均值年化后数值回归合理量级。
- 修复 `on_timer` / `add_daily_timer` 场景下，订单级 `fill_policy={"price_basis":"close","bar_offset":0,"temporal":"same_cycle"}` 未在当日 timer 事件内生效的问题；相关卖单现在会按当日 timer 时间与当日 close 成交，不再延后到下一交易日。
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
