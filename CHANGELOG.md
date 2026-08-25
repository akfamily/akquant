# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **成交事件流新增 `position_effect`（开平仓标志），前端可以区分开仓与平仓箭头**：`to_trade_message()` 产出的 `fill` 此前只有 `side`（buy/sell），拿不到开平语义 —— 前端只能画统一的买卖箭头，画不出「开仓 / 平仓」的区别（实测反馈是「开仓并没有画箭头」）。数据其实两侧都已就绪：回测侧 Rust `Trade` 结构本来就带 `position_effect`，实盘侧 `UnifiedTrade.position_effect` 也早有（middleware 已填真实值），缺的只是流出口没把它透出来。现在补上，取值是与下单入参**同一套**规范词：`auto` / `open` / `close` / `close_today` / `close_yesterday`。
  两侧词表统一小写。回测侧走 `PositionEffect::as_canonical_str()` 而**不是** `format!("{:?}").to_lowercase()`：后者会把 `CloseToday` 粘连成 `closetoday`，而入参与全仓其余口径都是 `close_today`，按规范词筛选会静默匹配不到（`src/model/types.rs` 里对这个写法有专门的警告注释）。实盘侧出口保留一次小写归一，兜住 middleware 字符串路径。老 broker 不填该字段时落 `auto`。`STREAM_SCHEMA_VERSION` 同步升到 `1.1`（新增可选字段，向后兼容加法），前端可据此协商识别新字段。
- **实盘会话收尾摘要新增两行事件丢弃计数**：`Dropped (foreign symbol)` 与 `Dropped (duplicate order)`，分别对应标的过滤与委托去重丢掉的事件数。这两项过滤本身按标的点名一次 WARNING 后就降级成 DEBUG，日志里很快看不见 —— 而万一标的归一化出错把**本会话自己的**回报也挡掉了，现象是「下单成功却收不到回调」这种极难排查的形态。计数挂在不会被日志级别关掉的地方，`foreign symbol` 数字异常大就是这个故障的直接信号。计数读取与格式化都做了异常隔离，不会因为一个可观测性字段而拖掉摘要主体（收益率、回撤、成交数）。
- **`UnifiedOrderStatus` 新增 `EXPIRED`，实盘终态口径与回测对齐（`__engine_rule_version__` 升至 1.5.2）**：回测侧 Rust `OrderStatus` 早有 `Expired`（日内单收盘未成、柜台把单判废），`strategy_order_events` 的终态集也一直含它；但实盘侧的 `UnifiedOrderStatus` 只有六个值，柜台报来的 `expired` 无处可落，只能被各 broker 的映射表兜底成 `SUBMITTED`（非终态）。后果不是"少显示一个状态"：该委托会一直出现在 `sync_open_orders()` 的结果里，`BrokerExecution.cancel_all_orders` 每轮恢复周期都对它撤一次，柜台回「委托状态错误不能撤单」，告警刷屏且永不收敛；`get_order()` 读到的也是一张永远活着的单。现在 `expired` 有了终态归属，`live/_payload_utils._TERMINAL_STATUSES`、`broker_event_adapter`（→ `OrderStatus.Expired`）、`create_default_mapper()` 的状态表、以及 ctp/miniqmt/ptrade 三个内置 broker 的 `_is_terminal_status()` 全部对齐同一口径。**自定义 broker 请检查自己的状态映射表**：柜台若有"过期/作废"这类状态，现在应映射到 `UnifiedOrderStatus.EXPIRED` 而不是继续兜底。不产生该状态的 broker 行为完全不变。
- **`docs/zh/advanced/adding_a_broker.md` 新增「委托状态映射：终态与非终态」一节**：列全两类状态的落点，并写明三条踩过的坑——未识别状态要兜底成非终态但**必须记 warning**（静默兜底会让柜台的新终态码值永远卡在挂单表里）；不要用 `status == Filled` 判成交（IOC / 最优五档即成剩撤类委托正常收尾就是 `Cancelled`，成交量在 `filled_quantity` 上）；不要复用 `create_default_mapper()` 的单字符码值（那是 CTP 口径，与恒生《数据词典》1203「委托状态」的数字码完全冲突，`0` 在两边分别是"全部成交"和"未报"）。
- **`run_live()` 新增 `strategy_runtime_config=` / `runtime_config_override=`，与 `run_backtest` 逐字对称（`__engine_rule_version__` 升至 1.5.1）**：`runtime_config` 的五个开关在实盘**本来就生效**——消费点直读 `strategy.runtime_config`，与运行模式无关，所以策略内部写 `self.runtime_config = StrategyRuntimeConfig(...)` 在实盘照常工作。缺的是**从入口统一下发**的能力：回测有 `run_backtest(strategy_runtime_config=...)`，实盘此前零参数，于是「同一份策略代码在两侧用不同口径」做不到——典型场景是回测用 `error_mode="raise"` 让问题尽早暴露、实盘用 `"continue"` 避免单个回调异常中断交易，此前只能改策略代码或在策略里按运行模式写分支。现在配置会下发到主策略**与每个槽位子策略**（对齐回测），冲突检测、`runtime_config_override` 取舍语义与告警去重都与回测共用同一份实现。下发时点刻意排在任何 `on_start` 之前：`runtime_config` 含 `indicator_mode`，它决定指标走增量还是预计算，而指标在 `on_start` 里注册。**不传该参数时行为完全不变**，策略自设值原样保留。与回测的一处差异：实盘入口没有 config 对象参数，因此没有回测那条从 `strategy_config.indicator_mode` 的兜底注入链路，实盘要改 `indicator_mode` 必须显式传 `strategy_runtime_config`。
- **`symbols` 里零数据的标的会发出告警**：白名单内的标的若全程没有任何行情事件（标的代码写错、数据源未覆盖、或所选时间范围内无交易），此前完全静默——回测照常跑完，结果里也看不出区别（`BacktestResult` 没有 `symbols`/`instruments` 属性，`trades_df` 为空时「没成交」与「没数据」无法区分）。现在能枚举输入集合的形态（`DataFrame`、`Dict[str, DataFrame]`、`List[Bar]`/`List[Tick]`）在运行前就逐标的告警；`DataFeed` 对象这种运行前无法枚举的形态由会话结束时的检查兜底；同一标的只报一次（运行前报过的不会在会话末重复报）。用 warning 不用 error：某标的在所选时间范围内确实无交易是合理场景。
- **`on_bar` 与 `on_tick` 可同时触发（回测 + 实盘）**：引擎把 bar 与 tick 拆成两条并列的历史序列，此前二者共用一个缓冲区，tick 以退化 OHLC 写入会把历史 bar 挤出窗口——双流下 `get_history` 会静默返回混入 tick 价的错误序列。回测侧 `run_backtest(data=[Tick, ...], freq="1min")` 在照常投递 tick 的同时把它们聚合成 bar 投给 `on_bar`（`freq` 只在 `data` 为含 `Tick` 的列表时有意义，DataFrame 传 `freq` 会报错而非静默忽略）。实盘侧 klinedata 与 CTP 网关均新增 `emit_ticks` / `emit_bars`（`use_aggregator` 保留为兼容别名，按参数逐个回退——只显式传其一不会静默关掉另一路），双流下合成 bar 打区间结束戳以保证与 tick 混推时时间戳单调不倒退。
- **`get_history` / `get_history_map` / `get_history_multi` / `get_history_df` / `get_rolling_data` 新增 `freq` 参数**：取值 `'tick'` / `'bar'` / `None`。粒度做成参数而非新增 `get_tick_history` 方法，与 `run_backtest(freq=...)` 术语一致，并为将来的多周期（如 `'5min'`）留开取值域。省略 `freq` 时按**当前所处的回调**自动定档（`on_bar` 里取 bar、`on_tick` 里取 tick，见下方「双流下 `on_bar` / `on_tick` 内省略 `freq` 不再报歧义错误」）；行情回调之外的双流场景仍要求显式指定——不静默选一条序列。未识别的 `freq` 取值同样报错，不会兜底成 `'bar'`。纯 bar / 纯 tick 单流下行为不变（`None` 时照旧取该 symbol 唯一存在的那条序列）。
- **新增策略只读属性 `self.freq`：策略终于能知道自己跑在什么周期上**。`run_backtest(freq=)` 此前只用于 tick→bar 聚合、用完即弃，klinedata 网关的 `period` 是纯网关内部参数，从未传给 `run_live` / `StrategyContext` / `Strategy`——策略**没有任何途径**读到数据周期，只能把周期写死在策略代码里或从外部参数重复传一遍。

    ```python
    def on_bar(self, bar):
        if self.freq == "1d":
            ...                      # 日线逻辑
        elif self.freq is None:
            ...                      # 周期未知：显式处理，不要假设
    ```

    取值**统一用回测侧口径**（`"1min"` / `"5min"` / `"1d"`）：klinedata 的 `"M1"` 由网关侧转换后注入，同一个策略在回测与实盘读到的是同一个值，策略代码因此不必按 broker 写兼容分支。回测侧由 `run_backtest(freq=)` 注入，实盘侧由行情网关经 `GatewayBundle.metadata["freq"]` 声明（`broker='replay'` 也支持 `gateway_options={"freq": "1min"}`）。

    **拿不到周期时为 `None`**，常见于：纯 bar 回测未传 `freq`、CTP 等只有逐笔源的网关、trader-only broker（无行情通道）、klinedata 的周线（回测侧 `freq` 只认整数分钟，周线没有对应写法，故声明 `None` 而不是编一个 `"10080min"`）。此时**刻意不从数据推断**——相邻 bar 的时间戳差会被停牌、跨日、午休误导，而一个错误的周期比未知的周期更危险（按周期折年化会静默错一个数量级）。属性只读，写入抛 `AttributeError` 并指向 `run_backtest(freq=)` / `gateway_options={"period": ...}`：数据粒度由数据源决定，允许策略写只会制造「我设了但没生效」的幻觉。
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
- **实盘恢复循环从单一 1 秒节奏拆成三层，不再每秒全量轮询柜台**：原先 heartbeat、`sync_open_orders()`、`sync_today_trades()`、`query_account()` 四个调用共用同一个 1 秒节拍，每轮无条件全跑 —— 挂单存在时就是每秒四次柜台请求，实测反馈是「后端在频繁调用接口」。现在按各自的必要频率分开：**heartbeat 每拍**（便宜，要快速发现断线）、**`query_account` 默认 5 秒**（策略要读 `get_account()`，不能太陈旧）、**两个全量 sync 默认 30 秒兜底**（恢复的本意是断线补齐，平时靠推送）。三档都可由 `gateway_options` 覆盖：`recovery_interval_sec` / `recovery_account_interval_sec` / `recovery_sync_interval_sec` —— 柜台的限流阈值是部署期才知道的外部约束，各家不同，硬编码不合适。越界只**钳制并告警**（约束是 tick ≤ account ≤ sync），非法值（负数、非数字、`nan`、`inf`）回退默认并告警，**都不抛异常**：让运维在启动时吃 `ValueError` 是最差的处理方式。
  两个配套行为：**断线重连成功后本轮立刻跑一次全量 sync**，不等 30 秒兜底 —— 这才是「断线补齐」真正该触发的时机；30 秒那档带 **±10% 抖动**，避免平台并发起多个任务时各自的全量 sync 对齐到同一秒打柜台。触发判断用单调时钟而非取模计数（循环 sleep 有漂移，取模会累积误差，也不受系统时间回拨影响）。**降频只针对重复轮询，没有给非交易时段加闸门** —— 盘后有合法事件（撤单确认、批量拒单、日终结算态变化、隔夜委托状态更新），粗暴挡掉会吞真实回报。
- **`UnifiedOrderSnapshot.timestamp_ns` 现在可以正常填写真实时间**：委托事件的去重键此前含 `timestamp_ns`，于是「插件**不要**给委托快照填真实时间」成了一条靠插件作者自觉遵守的**脆弱约定** —— 柜台的 `update_time` 每帧都变，填进去就等于让去重彻底失效（历史上「盘后还在推 `on_order`」正是这么来的）。现在键不再含该字段（跨轮去重改由状态指纹承担），这条约定随之解除，插件可以照常填真实时间，对订单审计有价值。
- **（破坏性）实盘柜台拒单不再从 `buy()`/`sell()`/`order_target_*` 抛异常**：改为返回空回执并触发
  `on_reject`（与回测口径一致）。原先靠 `try/except` 捕获柜台错误的策略需改用 `on_reject`。
- **（破坏性）`run_backtest(symbols=...)` / `run_from_checkpoint(symbols=...)` 从「注册哪些合约」改为「只跑哪些标的」（`__engine_rule_version__` 升至 1.5.0）**：此前不同数据输入形态对 `symbols` 的处理并不一致——`DataFrame`（含 `symbol` 列）、`Dict[str, DataFrame]`、`DataFeedAdapter` 三种形态本就按 `symbols` 过滤数据；但 `List[Bar]` / `List[Tick]` 与直接传入的 `DataFeed` 对象两种形态完全不过滤，数据里出现的任何标的都会正常触发 `on_bar` / `on_tick`，与传的 `symbols` 无关。这不只是「多算了几笔」——合约注册循环只遍历 `symbols`，于是这两种形态下数据里多出来的标的会带着**默认合约参数**参与撮合：期货乘数按 1 算（真实常为 10~300，持仓市值与权益错出几个数量级）、期货风控被整个跳过。极端情形是把全市场 `List[Bar]` 数据喂给策略而 `symbols` 只列了几个标的，其余标的照样建仓、照样计入统计。现在传了 `symbols` 就只有白名单内的标的进入引擎，五种输入形态行为统一；**不传 `symbols` 时行为完全不变**（沿用「数据即订阅」）。白名单为 `symbols` ∪ `config.instruments` ∪ `__init__` 阶段 `subscribe()` 已订阅的标的。**影响**：传了 `symbols` 的多标的回测结果会变（原先混进来的标的不再参与撮合与统计）。显式传空集合改为报错（那样会得到一个不放行任何标的的空回测，而不是静默退化为不过滤）。实盘不受影响——网关只推送已订阅的标的，「数据即订阅」在实盘天然成立。**迁移提示**：改动前 `symbol`/`symbols` 的签名默认值就是字面量 `"BENCHMARK"`（本仓库个别热启动示例也这么显式写过）。升级后显式传入的值一律按真实过滤条件处理，不再等价于「未传」，且不同数据形态下会向**相反方向**出错——`List[Bar]` / `DataFeed` 形态因数据里没有标的字面量等于 `"BENCHMARK"`，白名单谁都不放行，回测直接空跑（只有一条 WARNING）；`DataFrame` / `Dict[str, DataFrame]` 形态则反而完全不过滤。升级方式：把显式的 `symbols="BENCHMARK"` 直接**删掉（省略该参数）**。
- **（破坏性）传了 `symbols` 后，`on_start` 里 `subscribe()` 白名单外的标的会抛 `ValueError`**：时序上无法自动并入——`Engine::run` 内部先调 `on_start`，而数据加载与 `add_data` 发生在 `run()` 之前，前置过滤执行时 `_subscriptions` 尚为空。且「声明只跑 X，又订阅 Y」本身自相矛盾，报错比静默择一更清晰。迁移：把该标的加进 `symbols`，或去掉这次 `subscribe`。不传 `symbols` 时 `subscribe()` 不受任何约束；实盘的 `subscribe()` 是正常的动态订阅手段，同样不校验。
- **（破坏性）实盘网关 `Tick.volume` 语义改为单笔量，与回测口径对齐**：CTP、klinedata 两个行情网关此前把柜台/上游推的**当日累计成交量**原样塞进 `Tick.volume`，而回测侧 `Tick.volume` 一直是单笔量——同一策略读 `tick.volume`，回测和实盘拿到的量级能差出几个数量级，且是**静默**的，不报错也不告警。现在两个网关都按 symbol 记录上一次累计量并换算成单笔量（差分规则：`delta = 本次累计量 - 上一次累计量`；跨日重置/断线重连导致的负差分退回 `delta = 本次累计量`，因为此时它本身就约等于当日第一笔的单笔量）。**若你此前为了对齐语义已经自行加了「累计量转单笔量」的临时补丁，升级后必须移除，否则会被双重换算。** 分工未变：喂给内部 `BarAggregator`（`freq` 聚合 tick 为 bar 时用到）的仍是原始累计量（其 `volume_is_cumulative=True` 自己会处理），只有网关推给 `on_tick` / `add_tick` 的 `Tick.volume` 改成单笔量。**已知代价**：进程盘中启动时，某 symbol 收到的第一帧无法得知上一次累计量，换算规则记 `delta = 0`（而非误用累计量冒充单笔量），若策略里有「`volume == 0` 即跳过」式的防御逻辑，会连带跳过每个 symbol 的第一笔行情。
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
- **`get_history` / `get_history_multi` 对未登记标的不再静默返回全 NaN**：`ctx.history()` 在 Rust 侧 history 缓冲里完全找不到该 symbol 时返回 `None`，此前 Python 侧原样兜成 `np.full(count, np.nan)`，没有任何日志——策略拿到一串 NaN，却完全不知道是标的没订阅/登记错了还是别的原因（0821 平台测试反馈，回测、实盘两侧各报过一次）。现在这条分支(`arr is None`)会记一条 WARNING 并点名该 symbol，提示「通常意味着它没有被登记/订阅（检查 `instruments_config`/`symbols` 配置或标的代码是否写错），而不是数据源当天没数据」；同一 `(symbol, field)` 组合第二次起降为 DEBUG，避免策略每根 bar 都调用 `get_history` 时刷屏（去重集合挂在 strategy 实例上惰性建，与 `gateway/broker_event_bridge.py` 的 `_log_foreign_symbol` 同一套模式）。
  **刻意没有改的分支**：`len(arr) < count`（有数据但不够长）是预热不足的正常语义——函数契约本来就是左侧补 NaN，策略预热期每根 bar 都会触发这条路径，绝不能告警，否则会比原来的静默更糟（回测开头刷屏）。`get_history_df` / `get_rolling_data` / `get_history_map` 均转发自这两个入口，无需单独改。
- **broker recovery 的错误在默认 `compatible` 模式下不再完全静默**：`BrokerRecovery.handle_error` 此前第一行就是 `if recovery_mode != "strict": return`，而默认 `recovery_mode` 正是 `compatible`——心跳失败、重连失败、`sync_open_orders` / `sync_today_trades` / `query_account` 报错在默认配置下完全不打日志、不通知策略、不发 `recovery_error` observer 事件，柜台已经掉线而用户毫无感知（这也是「账户退出后任务仍显示运行中」这条反馈的一半根因）。现在把 strict 判断下移到既有的 `error_key` 去重之后，插入一条**两种模式都打**的 WARNING（带 `source` 与异常信息，如 `broker_recovery.heartbeat: ConnectionError: ...`），复用现成的去重键（同一错误连续发生只报一次，`run_cycle` 成功收尾会重置、恢复后再犯会再报）。**只补可见性，不改容错语义**：`compatible` 模式依旧不通知策略 `on_error`、不发 observer 事件，即依旧不中断交易；`strict` 模式原有的通知与事件行为不变。日志级别用 WARNING 而非 ERROR——单次 sync 失败下一轮会重试，够不上错误级别。
- **（实盘）修复账户存在挂单时策略每秒被重复推送同一个 `on_order`**：恢复循环每轮都会把柜台的全部未完成委托重新入队，而事件桥的去重集在每次 `drain_events` 后被清空 —— 那层去重只在「同一批」内生效，跨轮完全失效，配合每秒一轮的恢复节奏就成了每秒重推一次。表现是策略回调被同状态的挂单刷屏、订单审计日志每秒一条、接了 `on_broker_event` 的前端也跟着每秒收一次。现在新增**会话级委托状态指纹**（`status` + `filled_quantity` + `avg_fill_price` + `reject_reason`，**刻意不含时间戳** —— 含时间戳的键每次重推都会变、去重必然失效，这是这类缺陷最常见的写法），补上了成交侧早有、委托侧一直缺失的那处不对称。状态一变立刻放行，`New → PartiallyFilled → Filled` 这类真实推进一次都不会被吞；有界 FIFO 上限 50000，终态委托**不移除**（移除后下一轮重放会再推一次）。批内去重（`_event_keys`）保留不动，两层各管一段。
  去重键按 **`事件类型:委托号`** 建立独立命名空间，而不是裸委托号：ctp / miniqmt / ptrade 三个内置 broker 对同一次状态变化是**用同一个 payload 成对派发** `order` 与 `execution_report` 的（`ingest_order_event` 先 `order_callback` 再 `execution_callback`），而两者的四个指纹字段取值逐字相同 —— 共用命名空间会让 `on_execution_report` 被永久吞掉。同批内 `execution_report` 的键也补上了 `filled_quantity`：`UnifiedExecutionReport` 没有单独的「本次成交量」字段，它是区分连续两次部分成交的唯一信息，缺了它两条 `partially_filled` 会被误判成重复。
- **（实盘，破坏性）策略不再收到不属于本会话挂载标的的委托与成交回报**：柜台的 `sync_open_orders()` / `sync_today_trades()` 返回的是**全账户**数据，不限于本会话订阅的标的，此前这些回报会原样派发给策略 —— 策略因此看到别人的委托、`get_order()` 读到不属于自己的单。现在在事件桥入口按 `run_live(instruments=...)` 过滤（`order` / `execution_report` / `trade` 三类；`account` 无标的概念不过滤）。
  判据是两级的：**先认「这单是不是本会话自己报出来的」**（按 `broker_order_id`、再按 `client_order_id` 查会话内的委托映射），命中一律放行；查不到才按标的判。这一层是必需的——外部信号源经 `BrokerOrderSink` 直接下单时不经引擎的合约登记表，可以合法报出 `instruments` 之外的标的；它同时把「判据写错」的后果从「吞掉自己的回报」降级成「多派发一条」。标的比较在订阅集构建与回报比对**两侧都做后缀归一化**（只规整最后一段后缀，期货合约代码里有意义的小写如 `ag2612` 不受影响）：平台用 `000012.sz` 登记、柜台推 `000012.SZ` 是真实存在的写法差异，精确字符串比较会把自己的回报判成外来的并静默丢弃。所有边界一律**放行**（订阅集为空、载荷无 `symbol`、访问器抛异常），吞掉真实回报的代价远大于多派发一条；丢弃会按标的点名一次 WARNING、之后降 DEBUG，并计入会话丢弃计数。
  **仍然无法区分「同标的、不同任务」**：那需要柜台在推送帧里带任务或账户子标识，我们这侧没有可用判据。若同一账户下多个任务交易同一标的，彼此的回报仍会互相可见。
- **修复 `runtime_config` 读取侧默认值与配置类定义漂移**：`StrategyRuntimeConfig` 有五个字段，但读取侧兜底用的是一份**手写的四键字典**，漏了 `indicator_mode`——于是读取侧解析该字段会直接 `KeyError`（当前无调用点命中，故对用户不可见，但新增字段时同样的漏项会再次发生，且下一次可能就是可见的）。根因是这套东西散在三个文件：配置类在 `strategy.py`、写入侧 helper 在 `backtest/engine.py`（与回测毫无关系，只操作策略实例的属性）、读取侧默认值又在 `strategy_framework_hooks.py` 里手写第三份。现在配置类与写入侧统一收进新模块 `akquant/strategy_runtime_config.py`，读取侧默认值改为从 `dataclass` 的 `fields()` 派生，结构上不再可能漂移。`StrategyRuntimeConfig` 由 `akquant` 与 `akquant.strategy` 重新导出，`from akquant import StrategyRuntimeConfig` 等既有写法**不受影响**。
- 修复实盘会话（`run_live` / `LiveRunner.run`）结束时不触发策略 `on_stop` 的问题：三条停止路径（duration 到期、手动中断、异常中止）现在都会补发 `on_stop`，并在断开 broker 通道之前触发；同时补上实盘 slot 子策略此前从未触发的 `on_start`。
- **双流下 `on_bar` / `on_tick` 内省略 `freq` 不再报歧义错误**。tick 序列由引擎**无条件**写入历史缓冲区（`HistoryBuffer::update_tick`，与策略是否覆写 `on_tick` **无关**），因此只要订阅了 tick 流（实盘 `gateway_options={"emit_ticks": True}`、回测 `run_backtest(data=[Tick, ...], freq="1min")`），哪怕策略**只写了 `on_bar`**、从不读 tick，该标的也同时存在 bar 与 tick 两条序列。此前这种情形下 `get_history(...)` 不传 `freq` 会抛 `ValueError: symbol X 同时存在 bar 与 tick 两条历史序列`，实盘直接中止会话——用户被要求在一个「他不知道存在」的维度上做选择，而按单流写法写的策略代码本身毫无问题：

    ```python
    def on_bar(self, bar):                                  # 只有 on_bar
        close = self.get_history(20, bar.symbol, "close")    # 此前在双流下崩掉
    ```

    「双流下必须显式指定」这条规则对**同时挂 `on_bar` 与 `on_tick`** 的策略仍然必要（取哪条确实无从推断），但在 `on_bar` / `on_tick` 回调内部，调用点的意图并无歧义。现按**当前所处的回调**自动定档：`on_bar` 里省略 `freq` 等价于 `freq='bar'`，`on_tick` 里等价于 `freq='tick'`。**行情回调之外**（`on_timer`、`on_before_trading`、用户自建线程等）维持既有的歧义报错——那些位置推断不出该取哪条，不放宽成「全局默认取 bar」。五个历史入口（`get_history` / `get_history_multi` / `get_history_df` / `get_rolling_data` / `get_history_map`）全部覆盖。显式传入的 `freq` 优先，单流场景行为完全不变。字段合法性校验仍按**用户显式传入的** `freq` 判定，不按推断值——否则纯 tick 单流下既有的 `get_history(field='open')` 调用（一直是返回退化 OHLC）会突然开始报错。
- **并行参数优化在 worker 进程死亡时不再挂死，失败组合也不再静默丢失**。`run_grid_search(max_workers>1)`
  原先用 `multiprocessing.Pool.imap`，遇到 worker 进程级死亡（OOM、被 OOM-killer 杀、
  `os._exit`）会**永久挂起**——这是 CPython 的已知缺陷（[bpo-22393](https://bugs.python.org/issue22393)，
  十余年未修），跑几百个组合的网格时表现为「优化跑了几小时没有任何输出也不退出」。
  即便侥幸抛出异常，原代码 `except` 之后迭代器不再产出，**已提交但未取回的任务全部丢失**，
  用户既拿不到结果也不知道丢了多少。

    现改用 `concurrent.futures.ProcessPoolExecutor` + `as_completed` 逐 future 收结果：
    进程死亡会抛 `BrokenExecutor`（标准库刻意的快速失败设计，见
    [bpo-14148](https://bugs.python.org/issue14148)：worker 崩溃后队列与同步对象状态不可信，
    故整池置 broken 而不尝试恢复），该异常被捕获后落成带 `error` 字段的 `OptimizationResult`
    进入结果列表，**每个参数组合都有交代**。

    **注意池语义**：一个 worker 异常终止后整池进入 broken 状态，同批**尚未执行**的任务会一并
    失败（都落 error 列，不丢）。这不是「单任务失败其余照常跑完」——要做到那个需要池重建与
    任务重投。**刻意不做**：调研 QuantConnect LEAN / vnpy / backtrader / backtesting.py /
    PyAlgoTrade 后确认，量化框架**无一**做任务级重试，唯一商业级实现（LEAN）明确选择
    「记录失败、继续、不重试」；且优化场景下 worker 死亡的主因是内存不足，同参数重试大概率
    再次 OOM，只是把一次失败变成 N 次失败加 N 倍耗时。正确应对是调小 `max_workers` 或分批跑，
    而这需要的正是下面那条汇总信息。

    策略级异常（参数越界、数据不足、策略内部报错）的处理**未变**：一直由
    `_run_single_backtest` 内部兜住并落 `error` 列，从来不经过这条路径。
- **参数优化结束时汇总成功/失败数**。此前崩溃信息只逐条打 `logger.error`，几百个组合的网格里
  会被淹没，用户无从判断「崩了 3 个」还是「崩了 200 个」——而这两种情况的应对完全不同。
  现在收尾打一行 `优化完成: 成功 197, 失败 3 (worker 崩溃 3), 共 200 组, 耗时 412.5s`
  （`worker 崩溃 N` 仅在确有崩溃时出现；`db_path` 跳过的历史结果不计入，只反映本轮实跑）。
  失败占比 ≥50% 时追加一条 warning 指向排查方向——个别组合失败是正常的，过半失败几乎总意味着
  策略或数据本身有问题，而不是参数不好。口径对齐 LEAN 的 `_completedBacktest` /
  `_failedBacktest` 分列。单线程（`max_workers=1`）路径同样输出。
- 实盘报单被柜台回绝时不再抛穿策略回调，改为落成 `Rejected` 委托事件 + `order_reject`
  审计（`origin="broker"`）。
- 报单超时/连接断开时不再谎报拒单，改为 `on_error` + `order_submit_unknown` 审计——
  该情形下订单可能已在柜台，谎报拒单会诱导策略重复下单。
- 本地止损单在柜台返回「状态未知」（超时/断连）时不再自动重试。此前每次重试都会生成新的
  client_order_id，柜台无从去重，若报文其实已经到达柜台即造成真实的重复委托；现改为放弃该单并打
  CRITICAL + `on_error(..., "stop_trigger", ...)`。柜台明确拒单仍照旧重试（订单确定不存在）。
- `OrderReceipt` 新增只读字段 `failure`（`None` / `"rejected"` / `"unknown"`），使调用方能区分空回执
  是「无需交易」、「订单确定不存在」还是「状态不可知」。
- 多腿单（平今/平昨、反手）中途失败时保留已成功腿的回执，策略得以撤掉已发出的腿。
- 撤单失败不再抛异常并中断整轮 `cancel_all_orders`，改为逐单隔离 + `order_cancel_failed`
  审计；单笔失败不影响其余单撤销。
- **`instruments_config` 配了却在数据里找不到的标的会告警，不再静默丢弃**：合约快照与撮合层的合约表都按**数据里实际出现的** symbol 建，配置里 symbol 与数据对不上的条目会被完全静默丢弃，该标的回退到默认合约参数。最典型的撞法是数据用去后缀写法（`600487`）而配置写带后缀（`600487.SH`）：实测 `lot_size` 从配置的 100 变回默认 **1.0**（`tick_size` 恰好与股票默认值 0.01 相同，唯一露马脚的就是 `lot_size`），A 股下单数量随之不再整百，而用户以为配置已生效。现在运行前会点名这些标的、说明后果并列出数据里实际出现的标的供比对；冷启动（`run_backtest`）与热启动（`run_from_checkpoint`）共用同一检查。它与既有的三个集合（`filtered_out_symbols` 主动排除 / `symbols` 里有但数据没有 / adapter 泄漏）互不相交——这一条比的是「配置 vs 数据」，与 `symbols` 白名单无关，因此不受是否显式传 `symbols` 影响；`DataFeed` 对象输入无从枚举标的，跳过检查。
- **`on_order` 不再重复推送同一订单的同一状态**：`check_order_events` 每个 bar/tick 事件都会重扫在途与终态订单，而 `_emit_order_callback` 对 `on_order` 是**无条件调用**（`on_reject` 早有 `_framework_rejected_order_ids` 去重，`on_order` 没有等价物）。于是一张单只要还留在 `ctx.recent_rejected_orders` / `ctx.orders` 里，它每一拍都会再触发一次 `on_order`——表现为「回调全量推送」「同一张单跨交易日反复出现，看起来盘后还在推新回报」，写在 `on_order` 里的逻辑会被反复误触发。现在按**状态指纹**去重：键为「订单号 + 状态 + 已成交量 + 成交均价 + 拒单原因」，**刻意不含时间戳**——含 `updated_at` / `timestamp_ns` 的键每次重推都会变、去重完全失效；而只按订单号去重又会把 `New → PartiallyFilled → Filled` 这些真实状态推进整批吞掉。去重缓存有界（FIFO，上限 5 万条，模式与成交侧 `remember_trade_key` 一致），且**不随 checkpoint 落盘**：热启动后宁可把当前订单状态重推一轮，也不要因旧 key 残留而吞掉恢复后的第一次状态通报。
- **实盘会话因错误中止时，收尾摘要不再自称 "Manual Stop"**：策略回调抛异常时 `run_live` 会打 `CRITICAL` + traceback 并停止事件处理（设计如此，实盘不把异常继续上抛），但紧随其后的摘要标题此前硬编码为 `TRADING SUMMARY (Manual Stop)`——一次因故障中止的会话看起来像正常手动停止，而那条 `CRITICAL` 往往已被几十行日志淹没。现在该情形下标题为 `TRADING SUMMARY (ABORTED ON ERROR)`；正常结束与手动中断（含 `duration` 到期）的输出保持不变。
- **回测 DataFrame 的标的列识别改用项目统一的别名表，多标的不再被静默压成单标的**：`run_backtest(data=DataFrame)` 判断"是不是多标的数据"的判据此前是 `if "symbol" in df.columns`，**只认英文列名**。而 **AKShare 的标准输出列名是 `股票代码`**，项目自己的别名表 `schema.COLUMN_ALIASES` 也早已包含 `股票代码` / `symbol` / `code` / `ticker`（`normalize.resolve_columns` 与 `dataframe_to_arrays` 都在用），实盘侧的 `dataframe_to_bars` 更是**只认 `股票代码`**——两侧要求的列名正好相反。于是"从 AKShare 取多标的数据直接丢进 `run_backtest`"这个最自然的用法会被**静默**压成单标的：所有标的的 bar 混进一条时间序列（指标与撮合结果均不正确），`instruments_config` 里按真实标的配置的 `tick_size` / `lot_size` 整体失效，下单真实 symbol 只拿到 `Instrument not found`。

    现在判据改用 `resolve_columns(df)`，识别到的标的列统一重命名为 `symbol` 后走既有多标的路径；`symbol` 列的既有行为完全不变。另外，**未识别到标的列却检测到同一时间戳存在多行**时会打 `WARNING` 点出退化、给出可用的列名清单与替代输入（`Dict[str, DataFrame]` / `list[Bar]`）——同一时间戳多行是多标的被压平的可靠信号，真单标的数据每个时间戳只有一行，因此不会给单标的用法刷屏。三套标的列口径的彻底合并仍在 `docs/zh/meta/columnar-rfc.md` 挂账，此处只收敛回测入口这一条。golden 基线零漂移。
- **`order_target` / `order_target_value` / `order_target_percent` 的取整口径改为与撮合层一致**：下单侧取整此前只读**策略属性** `self.lot_size`（缺省 **1**），而撮合层校验读的是**标的登记值** `Instrument.lot_size`（`execution/common.rs` 对买单校验）。于是登记了 `lot_size=100` 的 A 股标的，`order_target_percent(0.2)` 仍按 1 股取整，算出的非整百数量必然被自己的风控拒掉——`Quantity 19743 is not a multiple of lot size 100`。**实盘尤其无解**：`run_live` 没有 `lot_size` 参数（实盘只接 `Instrument` 对象），除了在策略里手写 `self.lot_size = 100`，没有任何途径让取整逻辑知道登记值。

    现在两处取整（`_target_to_orders` 与 `calculate_max_buy_qty`）共用一个 `_resolve_lot_size()`，优先级为：

    1. 标的**未登记** `lot_size`（拿不到或 `<=0`）→ 用 `self.lot_size`，既有行为不变；
    2. `self.lot_size` 比登记值**更粗且是其整数倍**（如登记 100 而策略想按 200 下单）→ 尊重策略的显式意图（200 的倍数必然也是 100 的倍数，不会被拒）；
    3. 其余情况（含缺省的 1、以及与登记值不成倍数的值）→ 用登记值。

    `lot_size=1` 的标的（美股/加密）不会被凭空取整到整百。golden 基线零漂移（回测路径的 `self.lot_size` 由券商模板灌入，本就是 100）。
- **`Instrument` 构造期把交易所后缀归一化为大写，修复小写后缀导致的三处静默失败**：`Instrument("600028.sh", ...)` 这类写法此前被原样保留，而 `instruments` 是所有下游的**唯一源头**（实盘订阅集来自 `[inst.symbol for inst in instruments]`、标的属性快照字典、Rust 撮合层的合约登记都从它派生）。于是同一个小写会在三个下游各自以毫不相干的面貌炸开，使用者几乎不可能反推到同一个成因：

    - `get_instrument("600028.SH")` 抛 `KeyError`（快照字典的 key 是小写），报错文案会误导使用者以为标的没登记；
    - 实盘订阅集变成小写，而 broker 推送帧反解出的 symbol 恒为大写（如 middleware 的 `instrument_to_symbol`），按标的过滤时精确比较不命中 ⇒ **本任务自己的 order/trade 回报被静默丢弃**（只留一行 debug），表现为「下单成功却收不到回调」；
    - Rust 撮合层按小写登记合约，大写行情下单时查不到 ⇒ 该标的的**所有**订单被拒，`reject_reason` 为 `Instrument not found for 600028.SH`，且因拿不到 `tick_size` / `lot_size`，本地的价格对齐与整手校验一并失效。

    现在在构造期一次性归一化（与既有的 `symbol.trim()` 同属输入清洗），并在改写时打 `WARNING` 点出原值与新值，提示上游统一写法。**只规整最后一个 `.` 之后的后缀，证券代码段原样保留**——上期所/大商所的期货合约代码本身含有意义的小写（`ag2612`、`rb2601`）且柜台大小写敏感，对整个 symbol 取大写会把它改坏；无 `.` 的写法原样返回。已经使用大写后缀的用户零影响（不改写、不告警），golden 基线零漂移。
- **修复热启动省略 `symbols` 时会沿用上一段 checkpoint 里 pickle 残留的白名单的问题**：`Strategy._symbol_whitelist` 是普通实例属性，随策略对象整体 pickle 落盘。若阶段一 `run_backtest(..., symbols=[...])` 传了白名单，阶段二 `run_from_checkpoint(...)` 不再传 `symbols`，引擎层因未显式传入而正确「不过滤」，但策略层的 `_symbol_whitelist` 会通过 pickle 沿用阶段一的旧值——造成「引擎放行、策略层 `subscribe()` 仍按旧白名单拦截」的自相矛盾，且报错文案里的白名单是本次调用根本没传的值。现在改为无条件赋值：显式传了 `symbols` 就写入本次的白名单，没传就显式置 `None` 覆盖掉恢复出来的旧值。
- **修复 `symbols` 传字符串时白名单被按字符拆分的问题**：如 `symbols="600519"` 这种受支持的写法，下发给策略层用于 `subscribe()` 校验的白名单曾直接对未归一的原始参数取 `set(...)`，导致 `set("600519")` 按字符拆成 `{'6','0','0','5','1','9'}`，策略在 `on_start` 里 `subscribe("600519")` 自身标的反而被误判为「白名单外」并抛 `ValueError`。现在改用 `_resolve_effective_symbols` 已归一的标的列表构造白名单，字符串、元组、集合等写法均不受影响。
- **修复 `warmup_period` 在多标的下每个标的只预热约 `N / 标的数` 根的问题（行为变更，会改变多标的回测数值，`__engine_rule_version__` 升至 1.4.1）**：预热门槛此前用的是一个**跨标的的全局** bar 计数器，而历史缓冲区是按标的各自维护的。于是 `warmup_period = 60` 配 3 个标的时，每个标的只攒到 20 根就跨过门槛开始交易，`get_history(60, ...)` 里混入 `nan` 占位，指标用不足的数据计算——且**完全静默**，不报错也不告警。官方示例一律写 `self.warmup_period = self.params.long_window + 1`（指标窗口），而指标是按标的各自计算的，因此这些用法在多标的下全部失效。现在门槛按标的**独立**计数：标的 A 自己攒够 `warmup_period` 根就开始交易，不必等标的 B，`warmup_period` 无需乘以标的数量。**影响**：多标的策略（轮动、横截面、配对）的 `on_bar` 首次触发时间点后移，其后指标数值随之变化，既有回测结果与修复前不再可比——变化本身即修复生效的证据（此前是用不足的数据算出来的）。纯 tick 路径（`on_tick` 从来不受预热门槛约束）不受影响，golden 基线未变化；单标的策略的冷启动回测行为同样不受影响，但按标的计数需要新增 `_symbol_bar_counts` 字段随策略一起落盘——早于该字段存在的旧检查点（单标的与多标的都可能受影响）热启动恢复后，若不做兼容处理会把已经攒够的预热期重新走一遍，即使 Rust 历史缓冲区其实已经完整恢复。现在恢复旧检查点时改为查询实际恢复的历史深度而非从 0 重新计数：预热已攒满的直接放行，存档时真正处于预热期中途的会按剩余量继续预热，新格式检查点不受影响。ML 滚动训练仍按全局 bar 计数触发（`_rolling_step` 针对的是全局模型，语义未变）；但未配置 `validation_config` 时的取模触发在 per-symbol warmup 门槛导致部分 bar 事件被跳过后可能**永久错过**某次训练（而非仅仅延后），已在下方独立条目中修复为阈值语义。

    **收尾补充（会话结束告警）**：per-symbol 计数带来一个新的静默场景——若某标的的累计 bar 数在整个会话内始终不足 `warmup_period`（新上市标的历史太短、长期停牌、数据源对该标的覆盖不全，或 `warmup_period` 设得比该标的可用历史还长），它的 `on_bar` 会全程一次都不触发，策略对它没做任何决策，此前用户毫无提示，只会看到"这个标的怎么一直没信号"。放行逻辑本身没有问题（数据不足就不该拿它决策），问题只在于不可见。现在会话结束时（`_on_stop_internal`）会用 `WARNING`（不抛错，数据不足是常见且合理的情况）点名这类标的、它的累计 bar 数与 `warmup_period`，并给出可操作建议（调小 `warmup_period` 或为该标的补充历史数据）；已经触发过 `on_bar`（哪怕只有一次）的标的不受影响，避免噪音。

    **已知限制**：per-symbol 门槛的 `return` 只挡住了 `on_bar_event` 里的 `analyzer_manager.on_bar`，`on_trade` 路径（`strategy_order_events.py`）不受该门槛约束——预热期内产生的成交回报仍会正常到达 `analyzer_manager.on_trade`。这是修复前已存在的结构性不一致（此前预热期很短，不易暴露；per-symbol 化后某些标的的预热期可能等于其全部历史，不一致更容易被放大）。若自定义 `AnalyzerPlugin`（见 `docs/zh/advanced/analyzer_plugin_spec.md`）按"收到的 `on_bar` 次数"做分母或索引，在多标的 + 预热场景下会偏小，需要自行按标的维护计数，不能假设 bar 数与成交数同步递增。修复这一点需要改动 analyzer 的调度架构，超出本次范围，暂不改行为。
- **修复未配置 `validation_config` 时 ML 滚动训练触发用取模判断、遇到被跳过的 bar 事件会永久丢失一次训练的问题**：`should_trigger_training` 在没有 `validation_config` 的分支此前用 `_bar_count % _rolling_step == 0` 判断是否触发训练，这对"任何一次被跳过的 bar 事件"都不健壮——per-symbol warmup 门槛（见上一条目）会让某个标的在预热完成前的部分 bar 事件不经过训练判断就直接返回；若全局 `_bar_count` 恰好在被跳过的那次到达 `_rolling_step` 的倍数，下一次事件时模值已经错过 0，这次训练**永久丢失**、而非像 `validation_config` 分支那样只是顺延（该分支本就用 `>=` 阈值判断，不受影响）。极端情形下（`_rolling_step` 与预热跳过窗口重合）可导致 `model.fit` 全程一次都不会被调用，回测在没有任何训练过的模型下悄悄跑完全程，不报错也不告警。这不是本次改动引入的新缺陷——旧的全局计数器同样会在跳过窗口内丢失边界重合的触发，只是此前触发概率低很多，per-symbol warmup 生效后跳过窗口变长，缺陷更容易暴露，故一并修复。现在改为阈值语义：新增 `_last_train_bar_count` 状态（初值 0），触发条件为 `_bar_count - _last_train_bar_count >= _rolling_step`，消费触发时推进为当前 `_bar_count`——跳过的 bar 只会让触发顺延到下一根满足条件的 bar，不会丢；没有任何跳过时首次触发时机与取模完全一致。
- 修复 `freq='tick'` 下 `get_history` 系列 `field='price'` 报 `Invalid field` 的问题。`price` 是 tick 唯一的迁移出口（`open`/`high`/`low`/`close` 之外，tick 没有真正的开高低价），Python 侧白名单、报错文案与文档都已把它当作合法字段推荐给用户，但 Rust 侧字段解析只认 OHLCV，导致照文档改的用户反而撞上报错。现在 `price` 在实际取 tick 容器时等价于 `close`（tick 容器里 `close` 存的就是成交价）；bar 容器不受影响。
- 修复纯 tick 回测下 `ClosedTrade.mae` / `mfe` / `max_drawdown_pct` 静默恒为 0 的问题（回归修复）。`mae`/`mfe` 本身就是百分数（相对入场价的百分比），并没有对应的绝对值字段。tick/bar 历史序列拆分后，引擎内部计算这几个指标时仍只读 bar 容器，纯 tick 回测下该容器恒为空，取不到历史后按初值 0 写入成交记录，且不报错——若你此前跑过纯 tick 回测（`run_backtest(data=[Tick, ...])`，未配合 `freq` 聚合出 bar），报告里这几列为 0 很可能是本 bug 所致，并非策略本身盈亏波动为零，建议重新跑一次核对。混合输入（tick + bar 同时存在）与纯 bar 回测不受影响，golden 基线未变化。
- 修复 `run_live(broker="ctp", gateway_options={"emit_ticks": True, "emit_bars": True})` 被静默丢弃、`on_tick` 永不触发的问题。CTP 的 builder 链路（`_build_ctp` → `CTPMarketAdapter`）此前没有转发这两个参数，底层 `CTPMarketGateway` 始终拿到默认值（`emit_ticks=False`），且没有任何报错或告警提示配置未生效。现已补齐转发；逐参数回退、两者都为 `False` 时报错的语义仍由 `CTPMarketGateway` 统一实现，与 klinedata 网关保持同构。
- 修复双流（同一 symbol 同时存在 bar 与 tick 历史）下 ML 自动训练被歧义报错静默吞掉的问题：`on_train_signal` 此前调用不带 `freq` 的 `get_rolling_data()`，双流下撞上"两条历史序列同时存在"的 `ValueError`，被外层 `except Exception` 降级为一行 WARNING 后放过——训练一次都没跑，回测却照常出报告，指标全部来自未训练的模型。现在遇到这个特定的歧义错误会自动退回 `freq='bar'` 重试（ML 训练需要真实 OHLCV，tick 序列没有意义），且该分支单独用 ERROR 级别记录并点破"本次训练窗口已跳过"，与其它训练失败场景的 WARNING 区分开。单流场景行为不变。
- 缓解检查点跨版本恢复时含 tick 数据的策略被静默破坏的问题：tick/bar 历史序列拆分之前的存档里，bar 与 tick 共用同一个历史容器，用当前版本恢复这类旧存档会把整个旧容器当作 bar 序列装载，新写入的 tick 数据落入独立的 `tick_data` 容器——若原策略含 tick 数据，续跑后该 symbol 会同时拥有两条历史序列，取历史时可能报"两条历史序列同时存在"，或改传 `freq='tick'` 后只拿到续跑后的新 tick、历史窗口偏短。不改快照格式（Rust `HistoryBufferSnapshot::tick_data` 已用 `#[serde(default)]` 兼容旧存档，字段缺失与字段为空本身无法区分），改为在 `snapshot_features` 新增 `history_tick_split` 标记：`load_checkpoint` 遇到有 `history_buffer_snapshot` 但缺这个新标记的旧存档时，会发出 `RuntimeWarning` 并记录日志告警，提示重新用当前版本生成检查点。
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
