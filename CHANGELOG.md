# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`on_bar` 与 `on_tick` 可同时触发（回测 + 实盘）**：引擎把 bar 与 tick 拆成两条并列的历史序列，此前二者共用一个缓冲区，tick 以退化 OHLC 写入会把历史 bar 挤出窗口——双流下 `get_history` 会静默返回混入 tick 价的错误序列。回测侧 `run_backtest(data=[Tick, ...], freq="1min")` 在照常投递 tick 的同时把它们聚合成 bar 投给 `on_bar`（`freq` 只在 `data` 为含 `Tick` 的列表时有意义，DataFrame 传 `freq` 会报错而非静默忽略）。实盘侧 klinedata 与 CTP 网关均新增 `emit_ticks` / `emit_bars`（`use_aggregator` 保留为兼容别名，按参数逐个回退——只显式传其一不会静默关掉另一路），双流下合成 bar 打区间结束戳以保证与 tick 混推时时间戳单调不倒退。
- **`get_history` / `get_history_map` / `get_history_multi` / `get_history_df` / `get_rolling_data` 新增 `freq` 参数**：取值 `'tick'` / `'bar'` / `None`。粒度做成参数而非新增 `get_tick_history` 方法，与 `run_backtest(freq=...)` 术语一致，并为将来的多周期（如 `'5min'`）留开取值域。**双流下省略 `freq` 会报错**，要求显式指定——不静默选一条序列；未识别的 `freq` 取值同样报错，不会兜底成 `'bar'`。纯 bar / 纯 tick 单流下行为不变（`None` 时照旧取该 symbol 唯一存在的那条序列）。
- **新增委托价 tick 对齐工具与股票侧校验开关**。配合下面「股票/基金委托价开始校验 tick 对齐」的行为变更，提供显式对齐手段与逃生开关：

    - `Strategy.round_to_tick(symbol, price, direction)`：按标的的 `tick_size` 对齐委托价，`direction` 取 `"down"`（买入侧保守）/ `"up"`（卖出侧保守）/ `"nearest"`（默认）。底层工具函数为 `akquant.utils.price.round_to_tick(price, tick, direction)`，不依赖策略上下文，可单测直接调用。
    - `ChinaStockConfig.enforce_tick_size`（缺省 `True`）：关掉股票/基金的 tick 校验。与 `ChinaFuturesConfig.enforce_tick_size` 对称。

    ```python
    # 策略内：买入向下取整，卖出向上取整，都不朝不利方向滑价
    price = self.round_to_tick(symbol, raw_price, "down")
    self.buy(symbol, 100, price=price)
    ```

    详见 [tick size 对齐指南](docs/zh/advanced/tick_size_alignment.md)。
- **`orders_df` / `executions_df` 现在导出开平语义**。#361 争的就是 `position_effect`，而这两张最常用的表原先都不导出它，使用者只能自行遍历 `Order` / `Trade` 对象才看得到。现在：

    - `executions_df` 新增 `position_effect`、`timestamp_iso`
    - `orders_df` 新增 `position_effect`、`reduce_only`、`created_at_iso`、`updated_at_iso`

    委托表比成交表更早暴露拆腿结果——`auto` 拆腿在下单时就定了，无需等成交。`reduce_only` 与 `position_effect` 成对：为真时不产生开仓腿，只看 effect 会看不懂为什么少了一条腿。ISO 列是 UTC 归一的稳定字符串，与已本地化的 `timestamp` / `created_at` 并存，便于跨系统对账与导出。

    导出词表与下单**入参同一套**（`auto` / `open` / `close` / `close_today` / `close_yesterday`），可直接用于筛选。这一点是刻意的：若沿用仓库既有的 `format!("{:?}").to_lowercase()` 惯例，`CloseToday` 会导出成 `closetoday`，而 API 接受的是 `close_today`，`df[df.position_effect == "close_today"]` 会**静默匹配不到**。`status` 那类只读列可以这么做，但 `position_effect` 是可回传值，入参与导出必须同词表，故新增 `PositionEffect::as_canonical_str()` 作为唯一词表源。

- `StrategyContext` 新增 `get_closable_position(symbol)` 与 `get_projected_position(symbol)`，分别是「还能平多少」与「仓位将落在哪」两个口径，供开平推断与目标仓位使用（详见 Fixed 中 #361 条目）。同时 `ExecutionBackend` 协议新增这两个方法；协议是公开且 `runtime_checkable` 的，故调用点按名取值、缺失则退回 `get_position()`，第三方自定义后端不会因此崩溃（代价是维持旧行为，需自行实现新方法才能获得修复）。`broker_live` 下这两个方法退回结算仓：柜台挂单快照 `UnifiedOrderSnapshot` 只有 `position_effect` 与 `filled_quantity`，没有 `side` 和 `quantity`，算不出剩余量也判不了方向。

- **实盘 `subscribe()` 现在真正下发到行情网关**。`MarketGateway` 协议一直定义着 `subscribe(symbols)`，各内置网关（`ctp` / `replay` / `miniqmt` / `ptrade`）也都实现了它，但 `LiveRunner` 从不调用——订阅链路只接了一半：`instruments` 进得去，`Strategy.subscribe()` 出不来。此前用户在 `on_start` 里 `subscribe()` 一个标的只会拿到一条 warning。现在 `run_live` 会给主策略与各 slot 策略装上转发器，`subscribe()` 调用即刻把订阅集下发到行情网关。

    下发的是 **`instruments` 与全部 slot 的 `_subscriptions` 的并集**：各网关的 `subscribe()` 是整集替换语义（`self.symbols = list(symbols)`），只下发增量会把其余标的静默退订。转发在 `subscribe()` 发生的当刻进行而非装配期——行情网关先于策略 `on_start` 启动，装配时还不知道策略要订什么。

    没有行情网关的 broker（`market_gateway=None`）不装转发器，`subscribe()` 退回记录 warning（措辞已更新为「该 broker 未提供行情网关」），因为那类 broker 确实无处可下发。回测语义完全不变。

- **行情源与交易源可分开指定**：`create_gateway_bundle` 与 `run_live` 新增 `market_broker` / `trader_broker` 两个可选参数。两种模式二选一——只传 `broker` 时由它同时提供两侧（原语义不变）；同时传 `market_broker` 与 `trader_broker` 时各供一侧，**`broker` 完全不参与构建**。只传其中一个会报错，要求把另一侧也写明：若让 `broker` 兼任缺失的那侧，它就一词双义了（读 `broker='qmf', market_broker='replay'` 必须先知道「qmf 只有交易通道」才能推出 `broker` 在此指交易源）。

    这补齐了两类「单边 broker」的组合——`GatewayBundle` 的 `market_gateway` / `trader_gateway` 本就是两个独立可选字段，`replay` 只有行情（不能下单），而某些券商/柜台插件只有交易通道（收不到任何行情、`current_tick` 恒为 `None`），此前二者无法拼接：

    ```python
    run_live(..., market_broker="replay", trader_broker="my_trade_only_broker")
    ```

    要点：`gateway_options` 同时传给两个 builder；两侧同名时只构建一次（builder 可能连柜台、起线程）；`metadata` 记录 `market_broker` / `trader_broker` 便于排障，且行情侧声明的会话级信息（如 `replay` 的 `bounded_event_total`，`LiveRunner` 据此在事件放完后结束会话）不会因分开指定而丢失；未注册的名字会报错并点名具体参数，而非静默缺失某一侧通道。副产品是「回放行情 + 真实柜台下单」的联调组合现在可行。

    另外，`create_gateway_bundle` 的 `broker` 参数由必填改为可选（分开指定时无需传）。所有既有调用点都以关键字传参，不受影响。

- `get_account()` 账户快照新增 `free_margin` 字段（= `equity - used_margin`），表示真正可用于新开仓的可用保证金，与下单因保证金不足被拒时日志里的 `Available` 口径一致；期货保证金账户下 `cash`（现金余额）通常大于 `free_margin`，股票现金账户下二者相等。同时修正了 `cash` 在文档与 docstring 中「可用资金」的误导性表述，明确其为「现金余额」。
- `BacktestConfig` 新增 `days_per_year`（年化天数因子，默认 252；数字货币 24/7 市场可设 365）与 `risk_free_rate`（年化无风险利率，默认 0.0）两个字段，用于参数化 Sharpe/Sortino/波动率等风险指标的年化口径。`risk_free_rate` 默认 0 不改变任何现有数值。
- **新增内置 broker `replay`：确定性回放行情源**。用于在没有真实柜台、也没有 `openctp-ctp` 等可选依赖的环境下跑通实盘数据通路（`DataFeed` → 引擎 → `on_bar` / `on_tick`），此前这条通路没有任何测试覆盖。行情数据经 `gateway_options` 传入，支持 `list[Bar]`、`list[Tick]` 与 `DataFrame`（DataFrame 走 `normalize.dataframe_to_bars`，多品种需提供 `股票代码` 列，否则退化为单标的）：

    ```python
    run_live(
        strategy_cls=MyStrategy,
        instruments=[...],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars},   # 或 {"bars": df} / {"ticks": ticks}
    )
    ```

    事件按时间戳升序推送（多品种全局交错——live feed 无法排序，推送顺序即引擎所见顺序），数据放完后会话自行结束，通常无需依赖 `duration`。**限制**：该 broker 只提供行情，`trader_gateway=None`，不模拟撮合成交，因此不能用于 `trading_mode="broker_live"`；并且**不覆盖 timer 语义**——回放数据带历史时间戳，而 live 引擎用墙钟判定 timer 到期，两条时间线错位，`on_timer` / `schedule_daily` 在回放会话中的行为不作保证。`build_replay_bundle` 会在构建时校验每条事件的时间戳为正数（非正时间戳会被引擎静默丢弃，导致声明的事件总数永远达不到、会话挂死），常见诱因是数据源日期列存在 `pd.to_datetime(errors="coerce")` 无法解析的值；若数据本身不受控，仍建议保留 `duration` 作为安全网。`examples/38_live_functional_strategy_demo.py` 已改用该 broker（并显式传入 `duration` 作为安全网），现可离线实跑。
- **`run_backtest` 支持 tick 输入**。`data=` 现接受 `list[Tick]` 与 `list[Bar | Tick]` 混合列表（此前只接受 `list[Bar]`，传 Tick 会在 Rust 层抛 `TypeError`）。纯 tick 回测下 `get_history()` 与**单值**增量指标（SMA/EMA/RSI 等）现在可用——tick 以退化 bar 写入历史（`open=high=low=close=price`），故 `get_history(sym, "close", n)` 返回成交价序列：

    ```python
    run_backtest(data=ticks, strategy=MyStrategy(), symbols=["600000"], ...)
    ```

    **限制**：tick 的 OHLC 恒等，因此需要真实最高/最低价的指标模式（`input_mode` 为 `"hl"` / `"hlc"` / `"ohlc"`，如 ATR、振幅类）在有 bar 来源（混合输入，或配合 `freq` 把 tick 聚合成 bar）时由 bar 驱动正常工作，tick 本身对它们静默跳过；只有当某个标的**全程只有 tick、从未有过任何 bar** 时，才会在会话结束时抛 `StrategyConfigurationError`（`ValueError` 子类）而非静默返回 0。`source` 单值模式（`open`/`high`/`low`/`close` 均映射为成交价，`volume` 为单笔量）与 `close_volume` 模式正常可用。

- **`run_backtest` 新增 `freq` 参数：把 tick 聚合成 bar**。传入后原始 tick 仍照常投递给 `on_tick`，同时额外合成 bar 投递给 `on_bar`，从而拿到完整 OHLC 语义与全部指标（含 ATR 等 H/L 类）：

    ```python
    run_backtest(data=ticks, freq="1min", strategy=MyStrategy(), symbols=["600000"], ...)
    ```

    `freq` 词汇与 `akquant.feed_adapter` 的 `resample(freq=...)` 一致，但聚合走 `BarAggregator`，**只支持整数分钟**（`"1min"` / `"5min"` / `"1h"`）；`"30s"` 之类会抛 `ValueError` 并指向 `feed_adapter.resample`，不会静默取整。**成交量口径**：回测中 `Tick.volume` 是单笔量，适配层显式声明单笔口径（`volume_is_cumulative=False`）并把每笔量原样交给聚合器，由聚合器直接求和，故合成 bar 的成交量等于区间内 tick 量之和、一笔不落。**时间戳口径**：合成 bar 打在**区间结束**（下一区间起点前 1 纳秒）而非区间起点——回测因果顺序要求：合成 bar 与源 tick 会被放进同一个 feed 再按时间戳排序，若打区间起点，bar 会排到形成它的 tick 之前，策略读到的是尚未发生的未来数据。末尾未满周期不产生 bar。`freq` 在数据不含 tick 时抛 `ValueError`（参数无意义）。
- **策略参数可迁移性改善（非破坏性）**：仍在用已废弃的构造函数签名写法（`def __init__(self, fast=10)`）声明参数的策略，现在在**类定义期**就会收到 `UserWarning` 点名参数与迁移路径，不必等到跑回测才发现参数没生效；`run_backtest` / `run_grid_search` 的 `Unknown strategy param(s)` 报错按成因分流——未声明任何内联字段时给出完整迁移写法，键名不匹配时列出该策略可用的字段清单。参数语义未变，内联字段仍是唯一入口。
- **新增 `ListParam`**：补上列表型参数的声明入口。此前 `run_backtest` 的 `symbols` 自动注入要求策略把 `symbols` 声明为参数字段，但 DSL 没有任何能声明 list 的函数，该机制实际不可达。
- **官方示例全部迁移**：`examples/` 下 17 个仍在演示构造函数签名写法的示例已改为内联字段声明（含教材配套 ch05/ch10/ch12）。
- **迁移提醒：`__init__` 搬到 `on_start` 会改变热启动（resume）下的执行次数**。「把派生初始化从 `__init__` 移到 `on_start`」是本次给出的推荐迁移写法，但两者生命周期不同——`__init__` 每个对象只跑一次，`on_start` 每次运行都会跑一次，**包括从快照恢复（resume）**（即 `self.is_restored` 为真时也会调用）。这带来两点实际影响：

    - 构造后立即可读的属性（如 `warmup_period`）迁移后不再能在 `__init__` 期读到，需等 `on_start` 执行完。
    - 更容易踩坑的是**累积状态**——计数器、缓存、已建仓记录等 per-object 状态若直接搬进 `on_start` 顶层，热启动时会被静默重新初始化，破坏跨阶段的连续性。这类初始化请收在 `if not self.is_restored:` 分支内；与运行历史无关的纯重建逻辑（如重新注册指标对象本身）才适合放在 `on_start` 顶层无条件执行。完整用法见 `examples/21_warm_start_demo.py`、`docs/zh/guide/strategy.md` §6.4。

### Changed
- **（破坏性）`freq='tick'` 时 `get_history` 系列的 `field` 限 `price`/`close`/`volume`**：tick 没有 `open`/`high`/`low`，此前请求这些字段会静默返回退化 OHLC（`price` 冒充 `high`），现改为显式抛 `ValueError`。`get_history_df` / `get_rolling_data` 固定取 OHLCV 五字段，故它们在 `freq='tick'` 下必然报错，报错文案会指向 `get_history(freq='tick', field='price')` 取成交价序列。仅影响显式传 `freq='tick'` 的调用；`freq='bar'` 与省略 `freq`（单流场景）不受影响。
- **股票/基金委托价开始校验 tick 对齐（行为变更，`__engine_rule_version__` 升至 1.4.0）**：此前只有期货侧校验最小变动价位，股票/基金的非对齐委托价（如 `tick_size=0.01` 却传 `2.8314`）在回测里会照常成交，到了实盘却被柜台风控拒单（`RISK_PRICE_TICK_INVALID`）——同一笔单回测通过、实盘失败。现在两侧口径统一：

    - **回测**：`StockMatcher` 对非对齐委托价 **reject**（与 `FuturesMatcher` 一致），不是静默取整。
    - **实盘**：`broker_live` 路径在报单前本地校验，报错文案直接给出可用的对齐价与 `round_to_tick` 用法，不必等柜台回一个难解的 400。
    - **作用范围**：仅 `stock` / `fund` 资产类型；标的未登记、拿不到 `tick_size` 或 `tick_size<=0` 时**跳过**校验（柜台自己也会校验，总比让没配 `instruments` 的用户完全下不了单好）。市价单（不传 `price`）不受影响。

    **不是**「悄悄把价格取整」——那会让回测拒单、实盘成交的分裂反向出现一次。需要自动对齐请显式调用 `Strategy.round_to_tick()`；要完全关掉校验用 `ChinaStockConfig(enforce_tick_size=False)`。golden 基线**零漂移**（既有基线用的都是对齐价），但规则版本已 bump，自建基线请重新生成。
- **标的未登记的报错文案给出成因与登记入口**：`get_instrument()` 原先只说 `Instrument config not found for symbol: X`，看不出成因，用户普遍误判为「broker 侧缺该标的配置」（实测反馈即如此），而真实原因几乎总是该标的没进 `run_live(instruments=[...])` / `BacktestConfig(instruments_config=[...])`。现在报错会附上**已登记标的清单**（超过 20 个截断）与登记入口。
- **trader-only broker 无行情时启动阶段给出告警**：`broker='middleware'` / `'qmf'` 这类只有交易通道的 broker 会让 `bundle.market_gateway is None`，于是 `on_bar` / `on_tick` 永不触发、`current_tick` 恒为 `None`。这是预期行为（`run_live` docstring 早已写明），但此前启动阶段完全静默，用户只看到「策略没反应」，无从判断是配置问题还是当时没行情。现在会记录一条 warning，并给出补救办法（`run_live(market_broker='<行情源>', trader_broker='<交易源>')` 的分离写法）。显式配置了 `market_broker` 时不告警。
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
- **tick 事件开始写入历史缓冲区、并推进增量指标（行为变更）**：此前 `Event::Tick` 只更新当前事件快照，既不进历史也不推进指标，且是**静默**的——不报错也不工作。现在两者都会发生。若既有策略在 tick 路径注册了增量指标并依赖它「不更新」，结果会变。`__engine_rule_version__` 已相应 bump。
- **回测数据列表改为逐元素类型校验（破坏性）**：`run_backtest(data=[...])` 此前只检查首元素类型，`[Bar, "garbage"]` 会漏到 Rust 层抛出难以定位的错误。现在在 Python 层逐元素校验，抛 `TypeError` 并指名位置索引与实际类型；空列表抛 `ValueError`。

### Fixed
- 修复同一根 bar 内「先平后开」反手时开平语义标错的问题（#361）。`buy()` / `sell()` 在默认 `position_effect="auto"` 下按**结算持仓**拆开平腿，而结算持仓不含同周期已提交未成交的平仓单——`close_position()` 紧跟 `buy()` 的写法里，第二笔看到的仓位仍是反向的，于是被判成平仓：反手两笔都标 `close`，本该是 `close + open`。做空方向不暴露此问题只是因为 `short()` / `cover()` 把 `position_effect` 硬编码为 `open` / `close`，从不走 auto 推断。

    不止是标签错。落账层（`order_manager`）只看 `side` 无条件加减，所以最终净仓与净值曲线一直是对的（这也是它长期未被发现的原因）；但风控与保证金层按 `position_effect` 投影，那条假 `close` 腿的仓位变化算作 0，**保证金需求为 0**——一笔实质开仓的单子零保证金过闸。同时 `is_reduce_first_order` 会把它排到开仓单之前参与 reduce-first 撮合排序。

    现改为按**可平仓**拆腿：结算持仓扣除在途平仓/减仓单占用（状态取 `New` / `Submitted` / `PartiallyFilled`，按 `quantity - filled_quantity` 折算，并对平仓量封顶）。刻意**不**投影在途开仓单——在途开仓单未必成交，不投影它使拆腿偏向判为开仓，即偏向多预留保证金（安全侧）。此口径与 vn.py `OffsetConverter`、RQAlpha `closable` 一致。`__engine_rule_version__` 相应升至 `1.3.7`。

- 修复 `order_target*` / `close_position()` 在同一根 bar 内重复下单造成超卖的问题（与 #361 同源）。它们同样按结算持仓算差额，先 `close_position()` 再 `order_target_percent()` 会在全平单之外再补一笔同向单——卖出量超过持仓。现改为按**投影持仓**（结算持仓叠加**全部**在途单的预期效果）算差额：目标仓位问的是「仓位最终落在哪」，开仓与平仓在途单都要计入，否则目标不收敛。同一根 bar 内重复调用 `close_position()` 现在是幂等的。

    与 `auto` 拆腿只扣在途平仓单的取舍不同，两者问的问题不同，故分为 `get_closable_position()` 与 `get_projected_position()` 两个口径，共用同一投影核心。

- 修复多策略下同周期跨策略挂单不可见的问题。`ctx.positions` 是账户级全局，但 `active_orders` 在 slot 循环**之前**只快照一次，slot 0 本周期提交的单要到循环体末尾才发给事件管理器，slot 1 因此看不到——两边口径不一致会让开平推断漏掉别的策略的减仓意图。现在 slot 之间累加已提交单（仅多策略时累加，单策略不多付克隆开销）。

- 修复 `examples/textbook/ch07_futures.py` 从不触发反手分支的问题：`warmup_period = 10` 少于 `get_history(count=ma_window + 1)` 要求的 11 根，返回数组首位为 `NaN`，示例又取 `closes[:-1][-10:]` 正好圈入该 `NaN`，均线恒为 `NaN`、比较恒为假、信号永久锁在初值。同时修正期末权益取错 metrics 键的问题：`end_portfolio_value` 从不存在，正确键是 `end_market_value`，原写法有 `if key in index else 0.0` 兜底，故静默打印 `0.00` 而不报错。

- 修复 `executions_df` 的 Python 兜底路径与 Rust 快路径 `side` 列取值不一致的问题：兜底路径用 `str(t.side).lower()` 产出 `orderside.sell`，而快路径产出 `sell`——同一列的值随走哪条路而变。

- 修复 `broker_live` 下 `buy()` / `sell()` 缺省参数不解析的问题：`quantity=None` 原先直接透传到拆腿逻辑并以 `TypeError` 崩掉——等价于 `set_sizer()` 在实盘静默失效；`symbol=None` 原先会把 `None` 送进柜台请求。现按回测同口径解析：`symbol` 取当前 bar/tick，买入量走 `strategy.sizer`，卖出量全平持仓。另外解析后 `quantity <= 0` 时不再向柜台发 0 手单，改为返回空回执（与回测一致）。一处有意偏离：卖出全平在实盘取**可用**持仓而非总持仓，因为 A 股 T+1 下按总量报单会被柜台整单拒绝，取可用量退化为部分卖出。
- 修复 `broker_live` 下 `get_instrument()` / `get_instruments()` / `get_instrument_field()` 对**任何**标的都抛 `KeyError` 的问题：`run_live` 原先从不调用 `_set_instrument_snapshots`，策略侧快照字典恒为空。现由 `LiveRunner` 从传入的 `Instrument` 列表灌入。受实盘入口形态限制，`Instrument` 仅能回读 9 个字段，`option_type` / `strike_price` / `expiry_date` / `underlying_symbol` / `settlement_*` 在实盘快照中为 `None`（回测不受影响，仍由 `InstrumentConfig` 灌入完整字段）。
- 修复 `client_order_id` 跨会话撞号的问题：原格式 `{broker}-coid-{序号}` 的序号每次 `run_live` 从 0 起，重启后第一笔单又是 `...-coid-1`，与柜台里同名历史委托冲突——把 `client_order_id` 当幂等键的柜台会直接拒单（实测中间件返回 409 Conflict）。现格式为 `{broker}-{8 位会话标记}-{会话内序号}`，会话标记每次启动重新生成，跨重启与多进程并行均不撞号。
- 修复 `on_after_trading` 里下 next-open（`bar_offset=1`）单晚一格成交的问题（#324）。它原先在「下一根 bar」到来时才惰性补触发，引擎时钟已越过日界，其中提交的 next-open 单被撮合守卫推迟一格（盘后下单期望次日开盘成交，实际到第三日）。现改为按日终边界定时器在「当日收盘点」的独立事件里触发（早于下一根 bar），使其中提交的次日开盘单落在下一根 bar，而非晚一格。默认（非 precise）与 precise 模式下均已修正；`on_before_trading` 等开始型钩子本就与 `on_bar` 同拍触发，不受影响。`__engine_rule_version__` 相应升至 `1.3.1`。
- 修复 `on_pre_open` 未兑现「本次 open 成交」契约的问题（#324 家族）。其盘前定时器原先排在当日首根 bar 的同一时刻，订单 `created_at` 与该 bar 同拍、被 next-open 守卫拦截，导致实际成交在**下一日 open**（可由 `examples/52_pre_open_demo.py` 复现）。现将盘前定时器排在首根 bar 之前 1ns，使 `on_pre_open` 中下的默认开盘单落在**当日 open**，与文档承诺一致。
- 修复短回测区间下 Sharpe/Sortino 因「分子用 CAGR、分母用日波动 × √252」口径不一致而出现异常巨大值（如期货场景 Sharpe 高达数千万）的问题；改用日收益算术均值年化后数值回归合理量级。
- 修复 `on_timer` / `schedule_daily`（原 `add_daily_timer`）场景下，订单级 `fill_policy={"price_basis":"close","bar_offset":0,"temporal":"same_cycle"}` 未在当日 timer 事件内生效的问题；相关卖单现在会按当日 timer 时间与当日 close 成交，不再延后到下一交易日。
- 修复 framework 内部 `__framework_rebalance__` / `__framework_boundary__` timer 被误参与 same-cycle 撮合与终态统计的问题，避免出现 `+1ns` 的伪成交、partial fill 被过早补满，以及权益曲线尾部多出额外采样点。
- 修复 `on_pre_open` 首个交易日可能因延迟注册 timer 而不触发的问题；相关 framework timer 现会在回测事件循环开始前直接注入引擎，从而保证首日也能按预期开盘语义执行。

- 修复函数式策略（`strategy=on_bar` 入口）无法复用「回调未重写就跳过」快路径，导致回测慢一个数量级的问题。判定函数 `_strategy_overrides_callback` 是**按类**比较的（`type(strategy)` 上的方法 vs 框架基类 `Strategy` 的默认实现），而内部包装器 `FunctionalStrategy` 在类体里**无条件**定义了全部回调转发方法（未提供对应函数时方法体只是 `if self._on_xxx_func is not None` 的空转）——于是 `on_before_trading` / `on_after_trading` / `on_portfolio_update` / `on_pre_open` / `on_cross_section` / `on_timer` 六个钩子一律被判为「已重写」，快路径全线失效。

    后果是引擎为**每根 Bar** 注册并分发整套框架钩子（pre-open / cross-section / time-hooks / portfolio-update），而下游无人消费：1000 根 Bar 的回测里空转分发 3000 次 timer 事件，占总耗时约 2/3。实测第 11 章 12 组合网格搜索：类风格约 1.0 秒，函数式约 11.8 秒（约 13 倍）。回测数值一直是对的（两版结果逐值一致），这也是它长期未被发现的原因。

    现给 `FunctionalStrategy` 加类标记 `_is_functional_wrapper`，判定函数在「该方法正是包装器自身那份」时改看用户是否真的提供了对应回调（`_on_<name>_func is not None`）；子类若二次重写该方法，`method` 不再是包装器那份，仍走原有的通用比较，因此继承 `FunctionalStrategy` 自定义钩子的写法不受影响。用类标记而非 `isinstance` 是为了避开 `strategy_framework_hooks` → `backtest.engine` 的循环导入。

    修复后同一网格搜索降至约 1.6 秒（1.62x），剩余差距是函数式每次回调多一跳 Python 转发的固有成本。**回测数值不变**（golden 基线通过、网格结果与类风格逐值一致），故 `__engine_rule_version__` 不变。

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
