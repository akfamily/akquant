# API 参考

本 API 文档涵盖了 AKQuant 的核心类和方法。

快速跳转：

*   [run_live broker_live 执行语义](#live-broker-semantics)
*   [run_live 实盘历史预热](#live-preload-history)
*   [run_live 实盘时区、交易时段与收盘宽限](#live-timezone-sessions)
*   [SessionResolver 交易时段解析器](#akquant-session-resolver)

## 1. 高级入口 (High-Level API)

### `akquant.run_backtest`

最常用的回测入口函数，封装了引擎的初始化和配置过程。

```python
def run_backtest(
    data: Optional[BacktestDataInput] = None,
    freq: Optional[str] = None,
    strategy: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None] = None,
    strategy_source: Optional[Union[str, bytes, os.PathLike[str]]] = None,
    strategy_loader: Optional[str] = None,
    strategy_loader_options: Optional[Dict[str, Any]] = None,
    symbols: Optional[Union[str, List[str], Tuple[str, ...], set[str]]] = None,
    initial_cash: Optional[float] = None,
    commission_policy: Optional[CommissionPolicy] = None,
    commission_rate: Optional[float] = None,
    stamp_tax_rate: Optional[float] = None,
    transfer_fee_rate: Optional[float] = None,
    min_commission: Optional[float] = None,
    slippage: SlippageInput = None,
    volume_limit_pct: Optional[float] = None,
    timezone: Optional[str] = None,
    sessions: Optional[SessionsArg] = None,
    trading_calendar: Optional[Iterable[Any]] = None,
    t_plus_one: Optional[bool] = None,
    initialize: Optional[Callable[[Any], None]] = None,
    on_start: Optional[Callable[[Any], None]] = None,
    on_resume: Optional[Callable[[Any], None]] = None,
    on_train_signal: Optional[Callable[[Any], None]] = None,
    on_stop: Optional[Callable[[Any], None]] = None,
    on_tick: Optional[Callable[[Any, Any], None]] = None,
    on_order: Optional[Callable[[Any, Any], None]] = None,
    on_trade: Optional[Callable[[Any, Any], None]] = None,
    on_reject: Optional[Callable[[Any, Any], None]] = None,
    on_before_trading: Optional[Callable[[Any, Any, int], None]] = None,
    on_after_trading: Optional[Callable[[Any, Any, int], None]] = None,
    on_cross_section: Optional[Callable[[Any, Any, int], None]] = None,
    on_portfolio_update: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
    on_error: Optional[Callable[[Any, Exception, str, Any], None]] = None,
    on_expiry: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
    on_pre_open: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
    on_timer: Optional[Callable[[Any, str], None]] = None,
    context: Optional[Dict[str, Any]] = None,
    history_depth: Optional[int] = None,
    warmup_period: int = 0,
    lot_size: Union[int, Dict[str, int], None] = None,
    show_progress: Optional[bool] = None,
    start_time: Optional[Union[str, Any]] = None,
    end_time: Optional[Union[str, Any]] = None,
    catalog_path: Optional[str] = None,
    config: Optional[BacktestConfig] = None,
    custom_matchers: Optional[Dict[AssetType, Any]] = None,
    risk_config: Optional[Union[Dict[str, Any], RiskConfig]] = None,
    strategy_runtime_config: Optional[Union[StrategyRuntimeConfig, Dict[str, Any]]] = None,
    runtime_config_override: bool = True,
    strategy_id: Optional[str] = None,
    strategies_by_slot: Optional[Dict[str, Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]] = None,
    indicators: Optional[Sequence[Union[Type[Indicator], Indicator]]] = None,
    strategy_max_order_value: Optional[Dict[str, float]] = None,
    strategy_max_order_size: Optional[Dict[str, float]] = None,
    strategy_max_position_size: Optional[Dict[str, float]] = None,
    strategy_max_daily_loss: Optional[Dict[str, float]] = None,
    strategy_max_drawdown: Optional[Dict[str, float]] = None,
    strategy_reduce_only_after_risk: Optional[Dict[str, bool]] = None,
    strategy_risk_cooldown_bars: Optional[Dict[str, int]] = None,
    strategy_priority: Optional[Dict[str, int]] = None,
    strategy_risk_budget: Optional[Dict[str, float]] = None,
    strategy_fill_policy: Optional[Dict[str, FillMode]] = None,
    strategy_slippage: Optional[Dict[str, SlippageInput]] = None,
    strategy_commission: Optional[Dict[str, CommissionPolicy]] = None,
    portfolio_risk_budget: Optional[float] = None,
    risk_budget_mode: str = "order_notional",
    risk_budget_reset_daily: bool = False,
    analyzer_plugins: Optional[Sequence[AnalyzerPlugin]] = None,
    on_event: Optional[Callable[[BacktestStreamEvent], None]] = None,
    indicator_recorder: Optional[IndicatorSink] = None,
    broker_profile: Optional[str] = None,
    fill_policy: Optional[FillMode] = None,
    strict_strategy_params: bool = True,
    **kwargs: Any,
) -> BacktestResult
```

### `akquant.run_grid_search`

参数网格搜索入口，用于批量回测并按指标排序返回最优参数组合。

```python
def run_grid_search(
    strategy: Type[Strategy],
    param_grid: Mapping[str, Sequence[Any]],
    data: Any = None,
    max_workers: Optional[int] = None,
    sort_by: Union[str, List[str]] = "sharpe_ratio",
    ascending: Union[bool, List[bool]] = False,
    return_df: bool = True,
    warmup_calc: Optional[Any] = None,
    constraint: Optional[Any] = None,
    result_filter: Optional[Any] = None,
    timeout: Optional[float] = None,
    max_tasks_per_child: Optional[int] = None,
    db_path: Optional[str] = None,
    forward_worker_logs: bool = False,
    **kwargs: Any,
) -> Union[pd.DataFrame, List[OptimizationResult]]
```

**关键参数补充:**

*   `forward_worker_logs`: 并行优化时是否将子进程策略日志回传到主进程。
    *   `False`：吞吐优先，日志可能在主进程不可见。
    *   `True`：启用日志聚合，便于排障。
*   `strict_strategy_params`: 通过 `**kwargs` 传递给 `run_backtest`（默认在 `run_grid_search` 内为 `True`）。
    *   严格校验 `param_grid` 与策略构造参数匹配关系；
    *   参数不匹配时快速失败，避免静默回退。

### `akquant.run_walk_forward`

滚动优化入口。按窗口执行“样本内参数优化 + 样本外验证”，并拼接样本外资金曲线。

```python
def run_walk_forward(
    strategy: Type[Strategy],
    param_grid: Mapping[str, Sequence[Any]],
    data: pd.DataFrame,
    train_period: int,
    test_period: int,
    metric: Union[str, List[str]] = "sharpe_ratio",
    ascending: Union[bool, List[bool]] = False,
    initial_cash: float = 100_000.0,
    warmup_period: int = 0,
    warmup_calc: Optional[Any] = None,
    constraint: Optional[Any] = None,
    result_filter: Optional[Any] = None,
    compounding: bool = False,
    timeout: Optional[float] = None,
    max_tasks_per_child: Optional[int] = None,
    **kwargs: Any,
) -> pd.DataFrame
```

**关键参数补充:**

*   `**kwargs` 会透传到 `run_grid_search`（样本内优化阶段）与 `run_backtest`（样本外验证阶段）。
*   因此，`forward_worker_logs` 可用于控制样本内并行优化日志回传。
*   同时，`strict_strategy_params` 会在优化与回测阶段保持严格参数校验语义（默认严格）。

### `akquant.run_from_checkpoint`

从快照恢复并继续运行回测（支持多策略 slot 执行）。

```python
def run_from_checkpoint(
    checkpoint_path: str,
    data: Optional[BacktestDataInput] = None,
    show_progress: bool = True,
    symbols: Optional[Union[str, List[str], Tuple[str, ...], set[str]]] = None,
    commission_policy: Optional[CommissionPolicy] = None,
    strategy_runtime_config: Optional[Union[StrategyRuntimeConfig, Dict[str, Any]]] = None,
    runtime_config_override: bool = True,
    strategy_id: Optional[str] = None,
    strategies_by_slot: Optional[Dict[str, Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]] = None,
    strategy_max_order_value: Optional[Dict[str, float]] = None,
    strategy_max_order_size: Optional[Dict[str, float]] = None,
    strategy_max_position_size: Optional[Dict[str, float]] = None,
    strategy_max_daily_loss: Optional[Dict[str, float]] = None,
    strategy_max_drawdown: Optional[Dict[str, float]] = None,
    strategy_reduce_only_after_risk: Optional[Dict[str, bool]] = None,
    strategy_risk_cooldown_bars: Optional[Dict[str, int]] = None,
    strategy_priority: Optional[Dict[str, int]] = None,
    strategy_risk_budget: Optional[Dict[str, float]] = None,
    strategy_fill_policy: Optional[Dict[str, FillMode]] = None,
    strategy_slippage: Optional[Dict[str, SlippageInput]] = None,
    strategy_commission: Optional[Dict[str, CommissionPolicy]] = None,
    portfolio_risk_budget: Optional[float] = None,
    risk_budget_mode: str = "order_notional",
    risk_budget_reset_daily: bool = False,
    on_event: Optional[Callable[[BacktestStreamEvent], None]] = None,
    indicator_recorder: Optional[IndicatorSink] = None,
    config: Optional[BacktestConfig] = None,
    sessions: Optional[SessionsArg] = None,
    trading_calendar: Optional[Iterable[Any]] = None,
    **kwargs: Any,
) -> BacktestResult
```

`run_from_checkpoint` 使用与 `run_backtest` 相同的策略 slot、策略级风控与成交默认项；
对这些字段，优先级为：显式函数参数 > `config.strategy_config` > checkpoint 恢复值 / 默认值。

`sessions` / `trading_calendar` 与 `run_backtest` 同义。存档保存这两项，续跑不传就沿用存档里的配置；传入且与存档不一致时以传入为准并告警。

**通用行为说明（主要对应 `run_backtest`，`run_from_checkpoint` 共享其中的成交/风控/策略映射规则）:**

*   `data`: 回测数据。支持单个 DataFrame，`{symbol: DataFrame}` 字典，`List[Bar]`，`DataFeed`，或实现 `DataFeedAdapter.load(request)` 的对象。只有日期（本地 0 点）的数据改写到交易日收盘时刻只针对 DataFrame 与 Bar 列表输入；`DataFeed` 输入按原样使用，只有日期的数据请改用 DataFrame/Bar 列表，或把时间戳打在交易日收盘并传 `freq="1d"`。
*   `strategy`: 策略类、策略实例，或 `on_bar` 函数（函数式编程风格）。
*   `strategy_source` / `strategy_loader` / `strategy_loader_options`: 动态策略加载入口。`strategy=None` 时可直接从源码、路径或自定义加载器构造策略。
*   `initialize` / `on_start` / `on_resume` / `on_stop`: 函数式策略生命周期回调；其中 `on_resume(ctx)` 仅在 checkpoint 恢复后的热启动阶段触发，且先于 `on_start(ctx)`。
*   `on_tick` / `on_order` / `on_trade` / `on_reject` / `on_before_trading` / `on_after_trading` / `on_cross_section` / `on_portfolio_update` / `on_error` / `on_expiry` / `on_pre_open` / `on_timer` / `on_train_signal`: 函数式策略事件回调；其中 `on_expiry(ctx, event)` 在引擎实际执行到期结算后触发，`on_pre_open(ctx, event)` 在交易日首个时段开始前 1ns 触发（A 股 09:15 开盘集合竞价前），适合“盘前决策，本次 open 成交”，交易日钩子的完整时点见下文 [`akquant.Strategy`](#akquantstrategy)；`on_error(ctx, error, source, payload)` 会在其他用户回调抛出异常时触发；`on_train_signal(ctx)` 仅在 ML 滚动训练窗口触发。
*   `symbols`: 标的代码或代码列表。默认 `None`（**未显式传入**）：数据里出现的标的即视为要跑的标的（“数据即订阅”），行为与传入前完全一致。**显式传入**（哪怕传的是当前数据里已有的标的）后语义变为「只跑这些标的」：白名单外的标的会被前置过滤掉，不进入引擎、不参与撮合与统计（多标的输入下，回测结果可能因此变化）；显式传入空集合会报错，而不是退化为不过滤。白名单实际由 `symbols` ∪ `config.instruments` ∪（`__init__` 阶段）`self.subscribe()` 已订阅的标的三者合并而成；显式传了 `symbols` 后，`on_start` 里再 `subscribe()` 白名单外的标的会抛 `ValueError`（详见[策略指南](../guide/strategy.md)「策略生命周期」一节 `on_start` 的说明）。此校验仅作用于回测：实盘 `run_live` 从不下发白名单，`subscribe()` 不受约束。
    *   **`run_from_checkpoint` 的特例**：白名单合并的第三项不只是本次调用 `__init__` 阶段的 `subscribe()`——热启动恢复出来的策略实例带着**从存档 pickle 恢复的** `_subscriptions`（即上一段运行期间、含 `on_start` 里新增的全部订阅），这些也会并入白名单。也就是说，若阶段一的策略订阅过某个标的，阶段二哪怕不再显式 `subscribe()` 它，只要阶段二传了 `symbols`，该标的仍会留在合并后的白名单里——这是热启动「延续上一段状态」的整体语义的一部分，与「`__init__` 阶段订阅」并入 `run_backtest` 白名单的规则并非完全对称，请勿照搬 `run_backtest` 的白名单范围去推断 `run_from_checkpoint` 的实际过滤结果。
    *   **迁移提示（旧版 `symbols="BENCHMARK"` 写法）**：改动前，`symbol`/`symbols` 的签名默认值就是字面量 `"BENCHMARK"`（仓库内部分热启动示例也这么显式写过）。升级后，显式传入的值一律被当作真实过滤条件，不再等价于「未传」，且不同数据形态下会向**相反方向**出错：`List[Bar]` / `DataFeed` 形态因为数据里没有任何标的字面量等于 `"BENCHMARK"`，白名单谁都不放行，回测直接空跑（只有一条 WARNING，其余悄无声息）；`DataFrame` / `Dict[str, DataFrame]` 形态则因为 `"BENCHMARK"` 触发了既有前置过滤判据的短路，反而完全不过滤。升级方式很简单：把显式的 `symbols="BENCHMARK"` **删掉（省略该参数）**，不要迁移成其他字面量。
*   `initial_cash`: 初始资金。未显式传入时会回落到 `StrategyConfig.initial_cash`，其默认值为 `100000.0`。
*   `commission_policy`: 运行级默认佣金策略。支持三种模式：
    *   `{"type": "percent", "value": 0.0003}`: 按成交额比例收费。
    *   `{"type": "fixed", "value": 3.0}`: 每次成交固定收取 3 元。
    *   `{"type": "per_unit", "value": 0.01}`: 按成交数量线性收费，即 `fill_quantity * 0.01`。
    *   若显式提供，优先级高于 `commission_rate`；`commission_rate` 仍保留为兼容入口。
*   legacy 价格基准参数：已移除。
*   legacy 时序参数：已移除。
*   `fill_policy`: 运行级默认成交语义，接受一个 `FillMode` 对象（从 `akquant` 顶层导入）。五个命名模式：
    *   `NextOpen()`: 下一根 K 线开盘价成交（默认，无未来函数）。
    *   `NextClose()`: 下一根 K 线收盘价成交。
    *   `NextAverage()`: 下一根 K 线 OHLC4 均价成交。
    *   `NextHighLowMid()`: 下一根 K 线 HL2（高低中价）成交。
    *   `CurrentClose(timer_fill_timing="immediate"|"deferred")`: 当根收盘价成交；`timer_fill_timing` 仅影响 `on_timer` 触发的订单如何撮合（`immediate` 当期成交，`deferred` 顺延到下一根 bar）。
*   旧的 `fill_policy=dict`（`price_basis`/`bar_offset`/`temporal`）与 `make_fill_policy(...)` 已移除，传入 dict 会抛出 `TypeError`。请改用上述 `FillMode` 构造器。
*   `legacy_execution_policy_compat`（通过 `**kwargs`）: 已移除。
*   迁移建议：legacy 执行参数已不再接受，统一使用 `FillMode` 对象。
*   `strict_strategy_params`: 是否严格校验策略构造参数（默认 `True`）。
    *   当传入策略不接受的参数时会立即抛错；
    *   推荐保持默认值，避免参数错配被静默忽略导致回测结果偏差。
*   `freq`: 数据周期，默认 `None`。`data` 为含 `Tick` 的列表时是 tick 聚合周期（只支持整数分钟，如 `"1min"`），把 tick 合成 bar 并同时触发 `on_bar`；bar 数据（DataFrame、`List[Bar]`、catalog）也可传，用来声明 bar 周期并填 bar 的区间起点（`interval_start = 时间戳 − 周期`），写法为“正整数 + 单位”，单位 `min` / `h` / `d`（或 `day`），如 `"1min"`、`"5min"`、`"1h"`、`"1d"`。只有日期（本地 0 点）的日线不需要传，会自动改写到交易日收盘时刻；打在 15:00 等固定非 0 点时刻的日线要传 `freq="1d"`，否则每根 bar 按该时刻所在时段撮合（15:00 按“时间戳前 1ns”即 14:59:59.999… 判为收盘集合竞价），订单排到下一根 bar，并告警一次。策略的 `self.freq` 即该值。
*   `sessions`: 代码前缀 → 时段模板名或自定义时段表（`"default"` 为全局默认），默认 `None`。不传时中国时区按资产类型取默认（股票/基金 `CN_STOCK`、期货 `CN_FUTURES_DAY`、期权 `CN_OPTION`，其他资产 `ALWAYS_OPEN`），其他时区全天连续（`ALWAYS_OPEN`）。可选模板见 `aq.SessionResolver.templates()`；自定义时段表写法见下文 `BacktestConfig.china_futures` 一节的时段说明。
*   `trading_calendar`: 交易日列表（`date` / 字符串 / `Timestamp`），默认 `None`。不传时中国市场按工作日推算（节假日也被当作交易日，节假日前后的夜盘与日终结算会算错），并告警一次；`ALWAYS_OPEN` 每天都算，不告警。数据时刻超出所传日历覆盖范围时报 `ValueError`。
*   `t_plus_one`: 是否启用 T+1 交易规则。不传时取 `broker_profile` 模板的设置（三个内置 A 股模板都开），没有模板时为 False。如果启用，将强制使用中国市场模型。
*   `slippage`: 全局滑点 (默认 `None`，即不加滑点)。必须写成策略字典，例如 `{"type": "percent", "value": 0.0001}` 代表 1bp (0.01%) 的百分比滑点，或 `{"type": "fixed", "value": 0.2}` 代表固定价差。不再接受裸数字（含 `0`），传入会抛 `TypeError`。要关掉滑点（包括覆盖 `broker_profile` 模板里的滑点）传 `{"type": "zero"}`；不传或传 `None` 表示不覆盖，设了 `broker_profile` 时沿用模板的滑点，否则不加滑点。
*   `volume_limit_pct`: 成交量限制比例，默认 `None`（不限制，等同 0）。设为 0.25 时单根 bar 最多成交该 bar 成交量的 25%，截断量按手数向下取整，不足一手则本根不成交、订单继续挂着。订单剩余量不超过 `成交量 × 比例` 时仍一次成交（可能是零股，比如卖出最后剩下的 50 股）；IOC/FOK 订单若取整后的截断量为 0，当根撤销。**bar 的成交量必须与下单数量同单位**：AKShare 的 `stock_zh_a_hist` 等接口成交量单位是“手”，需先乘以 100 再开启量限。量限只作用于 bar 撮合，tick 驱动的撮合不受量限约束。数量可为小数的品种（加密货币、外汇）需设置 `lot_size`（默认 1），否则截断后的数量会向下取整为整数。
*   `warmup_period`: 策略预热期。指定需要预加载的历史数据长度（Bar 数量），用于计算指标。
*   `start_time` / `end_time`: 回测开始/结束时间。若传入 naive 时间字符串或 `Timestamp`，会按当前 `timezone` 解释，再转换为 UTC 参与过滤。
*   `catalog_path`: 当 `data` 未显式传入时，可从该目录按 `ParquetDataCatalog` 规则加载数据。
*   `config`: `BacktestConfig` 配置对象，用于集中管理配置。
*   `lot_size`: 最小交易单位。如果是 `int`，应用于所有标的；如果是字典，按标的匹配。
*   `custom_matchers`: 自定义撮合器字典。
*   `risk_config`: 风控配置。支持字典 (e.g., `{"max_position_pct": 0.1}`) 或 `RiskConfig` 对象。如果同时提供了 `config.strategy_config.risk`，此参数将覆盖其中的同名字段。
*   `strategy_runtime_config` / `runtime_config_override`: 运行时行为注入与冲突处理开关，支持 `StrategyRuntimeConfig` 或 `dict`。
*   `strategies_by_slot`: 可选多策略映射。键为 slot id，值为策略类/实例/函数式 on_bar 回调；用于启用 slot 迭代执行。
*   `indicators`: 可选指标脚本列表（`aq.Indicator` 的子类或实例），每个脚本并入 `strategies_by_slot` 成为独立 slot，key 取类定义行的 `id`（省略为类名 snake_case）。与 `strategy_id` 或已有 slot key 重复时报 `ValueError`，混入非 `aq.Indicator` 时报 `TypeError`。`run_live(indicators=...)` 语义相同。见下文 [`akquant.Indicator`](#akquant-indicator)。
*   `strategy_fill_policy`: 可选策略级默认成交策略映射（`strategy_id -> FillMode`）。
    下单时优先级：订单级 `fill_mode` > `strategy_fill_policy[strategy_id]` > 运行级 `fill_policy`。
*   `strategy_slippage`: 可选策略级默认滑点映射（`strategy_id -> slippage`）。
    下单时优先级：订单级 `slippage` > `strategy_slippage[strategy_id]` > 运行级 `slippage`。
*   `strategy_commission`: 可选策略级默认佣金映射（`strategy_id -> commission`）。
    下单时优先级：订单级 `commission` > `strategy_commission[strategy_id]` > 运行级佣金模型。
*   `commission` / `strategy_commission` 中的 `CommissionPolicy` 与运行级 `commission_policy` 共享同一结构：

```python
{"type": "percent" | "fixed" | "per_unit", "value": non_negative_number}
```

    *   `percent`: 按成交额比例收费。
    *   `fixed`: 每次成交固定金额，不随成交数量变化。
    *   `per_unit`: 按成交数量收费，适合“每股/每手/每份”线性收费场景。
*   配置分层（推荐心智模型）：
    1) 订单级（`buy/sell/submit_order` 传参）；
    2) 策略映射级（`strategy_*`，按 `strategy_id/slot`）；
    3) 运行级（`run_backtest` 参数）；
    4) 市场默认（market model 内建默认规则）。
*   T+1 范围说明：当前 `t_plus_one` 仍是运行级/市场级开关，不支持按 `strategy_id` 分层配置。
*   `analyzer_plugins`: 可选 Analyzer 插件列表。插件接收 `on_start/on_bar/on_trade/on_finish` 生命周期回调，结果汇总到 `result.analyzer_outputs`。
*   `on_event`: 可选事件回调。不传时内部使用 no-op 回调并保持阻塞返回语义；传入时可实时消费事件。
*   `broker_profile`: 可选 broker 参数模板，用于快速注入费率/滑点/最小手数/T+1 等默认值，显式参数优先（滑点同理：要覆盖模板里的滑点须显式传 `slippage={"type": "zero"}` 或其他策略字典，不传或传 `None` 时模板滑点生效）。内置模板：`cn_stock_miniqmt`、`cn_stock_t1_low_fee`、`cn_stock_sim_high_slippage`。

**fill_policy 推荐示例（主路径）：**

```python
import akquant as aq
from akquant import NextClose, CurrentClose

# 下一根 K 线收盘价成交
result = aq.run_backtest(
    data=data,
    strategy=MyStrategy,
    symbols="000001",
    fill_policy=NextClose(),
)

# 当根收盘价 + timer 订单顺延到下一根 bar
result = aq.run_backtest(
    data=data,
    strategy=MyStrategy,
    symbols="000001",
    fill_policy=CurrentClose(timer_fill_timing="deferred"),
)
```

**执行语义速查（五个命名模式）：**

| 场景 | `FillMode` |
| :--- | :--- |
| next-open 风格成交（默认） | `NextOpen()` |
| 当根收盘价成交 | `CurrentClose()` |
| 下一根收盘价成交 | `NextClose()` |
| 下一根 OHLC 均价成交 | `NextAverage()` |
| 下一根 HL2 成交 | `NextHighLowMid()` |

说明：
* 只有 `CurrentClose` 支持 `timer_fill_timing` 参数；其余模式的 `on_timer` 订单均在下一根 bar 成交。
* `timer_fill_timing="immediate"`（默认）：timer 触发即在当根收盘价成交；`"deferred"`：timer 不构成成交点，顺延到下一根 bar。它只影响 `on_timer` 订单，对普通 `on_bar` 订单无影响。

**DataFeedAdapter 用法（多时间框）:**

```python
import akquant as aq

base = aq.CSVFeedAdapter(path_template="/data/{symbol}.csv")

feed_15m = base.resample(freq="15min", emit_partial=False)
feed_replay = base.replay(
    freq="1h",
    align="session",            # session | day | global
    day_mode="trading",         # 仅 align='day' 时生效: trading | calendar
    emit_partial=False,
    session_windows=[("09:30", "11:30"), ("13:00", "15:00")],  # 仅 align='session'
)

result = aq.run_backtest(
    data=feed_replay,
    strategy=MyStrategy,
    symbols="000001",
    show_progress=False,
)
```

*   `align="session"`: 按交易日分区，可叠加 `session_windows`。
*   `align="day"`: 按日分区，不接收 `session_windows`；`day_mode` 支持 `trading/calendar`。
*   `align="global"`: 按全局时间轴聚合，不按交易日切段。
*   参数建议：统一使用 `symbols`。`run_backtest`/`run_from_checkpoint` 已不再接受 `symbol` 参数。

**兼容与迁移说明:**

*   推荐逐步将实时 UI / 日志 / 告警接入迁移到 `run_backtest(..., on_event=...)`。
*   流式场景统一使用 `run_backtest(..., on_event=...)`。
*   legacy 执行语义兼容开关已移除。
*   legacy 执行参数与 `legacy_execution_policy_compat` 不再接受。
*   公开执行配置全量统一使用 `FillMode` 对象（`fill_policy=NextOpen()` 等）。
*   在 PyCharm 中若未开启终端仿真，原生进度条可能不可见；可开启 `Emulate terminal in output console` 或改用 `on_event` 的 `progress` 事件输出文本进度。
*   阶段 5 后不再提供运行时参数级回滚开关；如需回滚请使用版本级回滚策略。

**阶段 5 迁移 FAQ:**

*   `run_backtest` 是否改名？不改名，调用方式保持不变。
*   `run_backtest` 是否仍可不传 `on_event`？可以，不传时仍返回同样的结果对象语义。
*   PyCharm 看不到进度条怎么办？先确认 `show_progress=True`，并在 Run 配置中开启 `Emulate terminal in output console`；若仍不可见，使用 `on_event` 消费 `progress` 事件打印文本进度。
*   线上出现问题如何回退？使用版本级回滚，不再支持 `_engine_mode` 参数级回切。
*   还可以继续用 `symbol` 吗？不可以。请统一迁移到 `symbols`。

### `akquant.merge_results`

```python
def merge_results(
    *results: BacktestResult,
    drop_expired_instruments: bool = True,
    dedupe_boundary: bool = True,
) -> MergedResult
```

把 `run_from_checkpoint` 分阶段续跑产生的多段 `BacktestResult` 按时间顺序合并成
一个 `MergedResult`，提供与 `BacktestResult` 一致的只读视图（`equity_curve` /
`cash_curve` / `margin_curve` / `orders_df` / `trades_df` / `executions_df` /
`positions_df` / `daily_returns` / `to_quantstats`）。

**行为:**

*   曲线与订单/交易/执行/持仓按时间戳拼接；`dedupe_boundary=True` 时去除相邻段
    重叠的边界时间戳（同戳保留后一段，对齐引擎 upsert 语义）。
*   各段必须**时间递增、互不重叠**（允许 gap）；重叠段抛 `ValueError`。
*   `drop_expired_instruments=True` 时，依据各段策略 instrument snapshot 的
    `expiry_date` 清理已过期合约的持仓行，避免长区间资产爆炸。

**metrics 为核心子集:** `MergedResult.metrics` / `metrics_df` 仅重算能从合并权益
曲线 + 交易明细无歧义推导的指标(`total_return_pct` / `max_drawdown` /
`sharpe_ratio` / `sortino_ratio` / `calmar_ratio` / `volatility` /
`annualized_return` / `win_rate` / `profit_factor` / `end_market_value` 等，
口径对齐单段回测)。其余依赖引擎内部态的字段不提供,访问时抛 `AttributeError`;
如需完整 60 项指标,请在单段完整回测的 `BacktestResult` 上读取。

### 流式参数与事件 (`run_backtest`)

**关键参数:**

*   `on_event`: 流式事件回调函数（可选），参数为 `BacktestStreamEvent`；不传时内部使用 no-op 回调。
*   `stream_progress_interval`: `progress` 事件采样间隔（正整数）。
*   `stream_equity_interval`: `equity` 事件采样间隔（正整数）。
*   `stream_batch_size`: 事件批量刷新阈值（正整数）。
*   `stream_max_buffer`: 缓冲区上限（正整数）。
*   `stream_error_mode`: 回调异常处理策略。
    *   `"continue"`: 回调报错后继续回测，并在结束事件中回传统计信息。
    *   `"fail_fast"`: 回调首次报错后立即终止，并抛出异常。
*   `stream_mode`: 流式模式。
    *   `"observability"`: 观测模式，允许采样与非关键事件背压丢弃。
    *   `"audit"`: 审计模式，禁用采样并采用阻塞背压（不丢弃非关键事件）。
*   `strategy_id`（通过 `**kwargs` 透传）: 为交易相关事件与结果打上策略归属，默认 `_default`。

**事件结构 (`BacktestStreamEvent`):**

*   `run_id`: 本次流式回测 ID。
*   `seq`: 事件序号（单调递增）。
*   `ts`: 事件时间戳（纳秒）。
*   `event_type`: 事件类型。
*   `symbol`: 关联标的（部分事件为空）。
*   `level`: 事件级别（如 `info`、`warn`、`error`）。
*   `payload`: 事件内容字典（字符串键值）。

**常见 `event_type`:**

*   生命周期: `started`, `finished`
*   采样更新: `progress`, `equity`
*   交易相关: `order`, `trade`, `risk`, `expiry`
*   运行异常: `error`
*   行情流: `tick`

**交易事件 payload 常用字段 (`order`/`trade`/`risk`/`expiry`):**

*   `owner_strategy_id`: 策略归属 ID（默认 `_default`）。
*   `order_id`: 订单 ID（`order`/`trade`/`risk`）。
*   `symbol`: 标的代码（`order`/`risk`）。
*   `status`: 订单状态（`order`）。
*   `filled_qty`: 订单已成交量（`order`）。
*   `trade_id`: 成交 ID（`trade`）。
*   `price`: 成交价格（`trade`）。
*   `quantity`: 成交数量（`trade`）。
*   `reason`: 风控拒绝原因（`risk`）。
*   `expiry_date`: 到期日（`expiry`，`YYYYMMDD`）。
*   `quantity_before`: 到期前持仓数量（`expiry`）。
*   `quantity_closed`: 本次因到期关闭的数量（`expiry`）。
*   `cash_flow`: 到期结算现金流（`expiry`）。
*   `settlement_type`: 到期结算模式（`expiry`，如 `cash`、`settlement_price`、`force_close`）。
*   `settlement_price`: 实际采用的结算价（`expiry`，存在时提供）。
*   流式 `expiry` 事件不含 `fee` 字段；期权行权结算费只出现在 `on_expiry` 回调的事件字典里（`fee`）。

**`finished.payload` 常用字段:**

*   `status`: `completed` 或 `failed`
*   `processed_events`: 已处理事件数
*   `total_trades`: 总成交笔数
*   `callback_error_count`: 回调报错次数
*   `dropped_event_count`: 背压丢弃事件总数
*   `dropped_event_count_by_type`: 按事件类型聚合的丢弃计数（`event=count` 逗号拼接）
*   `stream_mode`: 当前流式模式（`observability` 或 `audit`）
*   `sampling_enabled`: 是否启用采样（`true`/`false`）
*   `backpressure_policy`: 背压策略（`drop_non_critical` 或 `block`）
*   `last_callback_error`: 最近一次回调报错信息（存在时提供）
*   `reason`: 失败原因（存在时提供）

### `akquant.BacktestConfig`

用于集中配置回测参数的数据类。

```python
@dataclass
class BacktestConfig:
    strategy_config: StrategyConfig
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    instruments: Optional[List[str]] = None
    instruments_config: Optional[Union[List[InstrumentConfig], Dict[str, InstrumentConfig]]] = None
    china_futures: Optional[ChinaFuturesConfig] = None
    china_options: Optional[ChinaOptionsConfig] = None
    benchmark: Optional[str] = None
    timezone: str = "Asia/Shanghai"
    show_progress: bool = True
    history_depth: int = 0

    # Analysis & Bootstrap
    bootstrap_samples: int = 1000
    bootstrap_sample_size: Optional[int] = None
    analysis_config: Optional[Dict[str, Any]] = None

```

### `akquant.StrategyConfig`

策略层面的配置，包含资金、费率和风控。

```python
@dataclass
class StrategyConfig:
    initial_cash: float = 100000.0
    commission_rate: float = 0.0
    commission_policy: Optional[Dict[str, Any]] = None
    stamp_tax_rate: float = 0.0
    transfer_fee_rate: float = 0.0
    min_commission: float = 0.0
    enable_fractional_shares: bool = False
    round_fill_price: bool = True
    slippage: Optional[Dict[str, Any]] = None
    volume_limit_pct: float = 0.0
    max_long_positions: Optional[int] = None
    max_short_positions: Optional[int] = None
    exit_on_last_bar: bool = True
    risk: Optional[RiskConfig] = None
    strategy_id: Optional[str] = None
    strategies_by_slot: Optional[Dict[str, Any]] = None
    strategy_source: Optional[str] = None
    strategy_loader: Optional[str] = None
    strategy_loader_options: Optional[Dict[str, Any]] = None
    strategy_max_order_value: Optional[Dict[str, float]] = None
    strategy_max_order_size: Optional[Dict[str, float]] = None
    strategy_max_position_size: Optional[Dict[str, float]] = None
    strategy_max_daily_loss: Optional[Dict[str, float]] = None
    strategy_max_drawdown: Optional[Dict[str, float]] = None
    strategy_reduce_only_after_risk: Optional[Dict[str, bool]] = None
    strategy_risk_cooldown_bars: Optional[Dict[str, int]] = None
    strategy_priority: Optional[Dict[str, int]] = None
    strategy_risk_budget: Optional[Dict[str, float]] = None
    strategy_fill_policy: Optional[Dict[str, Dict[str, Any]]] = None
    strategy_slippage: Optional[Dict[str, Dict[str, Any]]] = None
    strategy_commission: Optional[Dict[str, Dict[str, Any]]] = None
    portfolio_risk_budget: Optional[float] = None
```

### `akquant.InstrumentConfig`

用于配置单个标的属性的数据类。

```python
@dataclass
class InstrumentConfig:
    symbol: str
    asset_type: Union[
        Literal["STOCK", "FUTURES", "FUND", "OPTION"],
        Literal["ETF_OPTION", "CONVERTIBLE_BOND"],  # 品种预设, 展开为 OPTION / FUND
        InstrumentAssetTypeEnum
    ] = InstrumentAssetTypeEnum.STOCK
    multiplier: float = None   # 合约乘数; 缺省按预设或 1.0 填充
    margin_ratio: float = None # 保证金率 (0.1 表示 10% 保证金); 缺省按预设或 1.0 填充
    tick_size: Optional[float] = None  # 最小变动价位; 缺省按预设或资产类型填充
    lot_size: Optional[int] = None

    # 费率与执行 (资产专用, 覆盖全局; ChinaMarket 下期权按张计费不受影响)
    commission_rate: Optional[float] = None
    min_commission: Optional[float] = None
    stamp_tax_rate: Optional[float] = None
    transfer_fee_rate: Optional[float] = None
    slippage: Optional[Dict[str, Any]] = None

    # 期权相关
    option_type: Optional[
        Union[Literal["CALL", "PUT"], InstrumentOptionTypeEnum]
    ] = None
    strike_price: Optional[float] = None
    expiry_date: Optional[Union[int, date, datetime]] = None
    underlying_symbol: Optional[str] = None
    option_margin_model: Optional[InstrumentOptionMarginModelEnum] = None
    implied_volatility: Optional[float] = None
    reference_volatility: Optional[float] = None
    settlement_type: Optional[
        Union[
            Literal["cash", "settlement_price", "force_close"],
            InstrumentSettlementTypeEnum
        ]
    ] = None
    settlement_price: Optional[float] = None  # 期权: 到期日标的结算价
    sellable_after_days: Optional[int] = None
    static_attrs: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    # 以下由 __post_init__ 填写, 不可传入
    product_preset: Optional[str]      # 展开前的品种预设名, 未用预设时为 None
    defaulted_fields: FrozenSet[str]   # 由缺省值填充的字段名
```

`commission_rate` / `min_commission` / `stamp_tax_rate` / `transfer_fee_rate` 在 `run_backtest` 与 `run_from_checkpoint` 中按标的覆盖市场费率（ChinaMarket 下股票/基金四项全部生效、期货只认 `commission_rate`；SimpleMarket 下四项对所有资产生效）。ChinaMarket 下期权按张计费不受这些字段影响；但显式 `use_china_market=False` 走 SimpleMarket 时，覆盖同样作用于期权。`slippage` 字段已接入撮合，优先级：订单级 `slippage=` > 品种级 `InstrumentConfig.slippage` > 策略级 `strategy_slippage` > 全局 `StrategyConfig.slippage`；`run_from_checkpoint` 续跑同样按此优先级生效。`run_from_checkpoint` 续跑段现在会补齐期权市场规则（`china_options` 配置下发、行权结算费）与按品种费用 / 滑点覆盖；已知限制：若第二阶段调用完全不传 `instruments`，第一阶段仅由数据键重新登记的期权会被判定为 STOCK；期货市场规则（`ChinaFuturesConfig`、期货前缀费率规则）续跑时仍不会重新下发（交易时段 `sessions` / `trading_calendar` 随存档保存，续跑不传时沿用）；识别出第一阶段的期权可能使续跑切换到 ChinaMarket。

常用枚举（均可在 `akquant` 顶层直接访问）：

- `InstrumentAssetTypeEnum`: `STOCK` / `FUTURES` / `FUND` / `OPTION`
- `InstrumentOptionMarginModelEnum`: `RATIO` / `CHINA_SINGLE_LEG` / `US_BROKER_SINGLE_LEG` / `US_BROKER_SINGLE_LEG_VOL_ADJUSTED`
- `InstrumentOptionTypeEnum`: `CALL` / `PUT`
- `InstrumentSettlementTypeEnum`: `CASH` / `SETTLEMENT_PRICE` / `FORCE_CLOSE`

示例：

```python
conf = akquant.InstrumentConfig(
    symbol="IF2506",
    asset_type=akquant.InstrumentAssetTypeEnum.FUTURES,
    settlement_type=akquant.InstrumentSettlementTypeEnum.CASH,
)
```

### `akquant.InstrumentSnapshot`

策略侧可访问的标的静态属性快照对象（由引擎注入，通常通过 `Strategy.get_instrument*` 读取）。

```python
@dataclass(frozen=True)
class InstrumentSnapshot:
    symbol: str
    asset_type: Literal["STOCK", "FUTURES", "FUND", "OPTION"]
    multiplier: float
    margin_ratio: float
    tick_size: float
    lot_size: float
    option_margin_model: Optional[Literal["RATIO", "CHINA_SINGLE_LEG", "US_BROKER_SINGLE_LEG", "US_BROKER_SINGLE_LEG_VOL_ADJUSTED"]] = None
    option_type: Optional[Literal["CALL", "PUT"]] = None
    strike_price: Optional[float] = None
    expiry_date: Optional[int] = None  # YYYYMMDD
    underlying_symbol: Optional[str] = None
    implied_volatility: Optional[float] = None
    reference_volatility: Optional[float] = None
    settlement_type: Optional[Literal["CASH", "SETTLEMENT_PRICE", "FORCE_CLOSE"]] = None
    settlement_price: Optional[float] = None
    static_attrs: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
```

要点：

*   `expiry_date` 使用 `int(YYYYMMDD)` 语义。
*   快照在 `on_start` 即可访问。
*   建议在策略中通过 `get_instrument` / `get_instrument_config` / `get_instrument_field` 访问。
*   **回测与实盘的字段覆盖不同**：回测快照由 `InstrumentConfig` 灌入，字段齐全；实盘（`run_live`）只接 `Instrument` 对象，它仅能回读 `symbol` / `asset_type` / `multiplier` / `margin_ratio` / `tick_size` / `lot_size` / `option_margin_model` / `implied_volatility` / `reference_volatility`，因此 `option_type` / `strike_price` / `expiry_date` / `underlying_symbol` / `settlement_type` / `settlement_price` / `static_attrs` 在实盘快照里为 `None`（或空 dict）。期权策略若依赖这些字段，需自行通过策略参数或 `context` 传入。

### 配置系统详解 (Configuration System)

AKQuant 提供了灵活的配置系统，允许用户通过多种方式设置回测参数。

#### 1. 配置层级 (Hierarchy)

配置对象采用树状结构组织，`BacktestConfig` 是顶层入口：

```text
BacktestConfig (回测场景)
├── StrategyConfig (策略与账户)
│   ├── initial_cash (初始资金)
│   ├── commission_policy / commission_rate (默认佣金)
│   ├── slippage (默认滑点)
│   └── RiskConfig (风控规则)
│       ├── safety_margin (安全垫)
│       └── max_position_pct (持仓限制)
└── InstrumentConfig (资产属性)
    ├── multiplier (合约乘数)
    └── commission_rate (资产专用佣金，覆盖 StrategyConfig)
```

中国期货扩展配置位于 `BacktestConfig.china_futures`，用于管理前缀级规则：

- `instrument_templates_by_symbol_prefix`: 品种模板（乘数/保证金/tick/手数/费率）
- `fee_by_symbol_prefix`: 品种费率覆盖
- `validation_by_symbol_prefix`: 品种撮合校验开关覆盖
- 交易时段已不在 `ChinaFuturesConfig` 配置：`enforce_sessions` / `session_profile` / `sessions` 与 `ChinaFuturesSessionConfig`、`ChinaOptionsConfig.sessions`、`ChinaOptionsSessionConfig` 均已删除（传入会报带迁移方法的错误），改用 `run_backtest(sessions=...)`，如 `sessions={"rb": "CN_FUTURES_NIGHT_23"}`；不受时段约束用 `sessions={"default": "ALWAYS_OPEN"}`
  - `sessions` 的值可以是模板名，也可以是自定义时段表（段序列 `(start, end, session[, anchor])`，如 `[("21:00", "02:30", "continuous"), ("09:00", "15:00", "continuous")]`）。规则：段外即闭市，不写 `closed`；未写第 4 项时 `start >= "18:00"` 的段锚定前一交易日（夜盘），可用 `"prev"` / `"day"` 显式指定；`end <= start` 表示跨午夜，只允许夜盘段；时刻须为严格两位的 `HH:MM`；同一锚定内及夜盘展开后与当日段之间不得重叠；键按代码前缀匹配（更长前缀优先，如 `shfe.au`），去空格、转小写后不得重复；自定义表未配 `trading_calendar` 时缺省日历为工作日。详见[第 7 章「AKQuant 中国期货配置速览」](../textbook/07_futures.md)
  - 期权未被 `sessions` 前缀匹配时，若其 `underlying_symbol` 是已登记的期货，跟随该期货的时段表；否则用 `CN_OPTION`

配置对象采用“构造即校验”：

- `symbol_prefix` 为空会直接报错
- 模板数值范围非法（如 `multiplier <= 0`）会直接报错
- 同一列表内前缀重复会报错并标注冲突项索引

#### 2. 参数优先级 (Priority)

`run_backtest` 函数的参数解析遵循以下优先级（由高到低）：

1.  **显式参数 (Explicit Arguments)**:
    *   直接传递给 `run_backtest` 的参数优先级最高。
    *   例如：`run_backtest(start_time="2022-01-01")` 会覆盖 `config.start_time`。
2.  **配置对象 (Config Objects)**:
    *   如果显式参数为 `None`，则从 `config` (`BacktestConfig`) 中读取。
    *   多策略字段可集中配置在 `config.strategy_config`（如 `strategy_id`、
        `strategies_by_slot`、`strategy_max_*`、`strategy_priority`、
        `strategy_risk_budget`、`portfolio_risk_budget`）。
3.  **默认值 (Defaults)**:
    *   如果上述两者都未提供，则使用系统默认值。

中国期货扩展（`BacktestConfig.china_futures`）推荐使用以下优先级口径：

| 配置项 | 高优先级 | 中优先级 | 默认值 |
|---|---|---|---|
| 合约参数（乘数/保证金/tick/手数） | `InstrumentConfig` 显式字段 | `instrument_templates_by_symbol_prefix` | `run_backtest` 默认参数 |
| 品种费率 | `fee_by_symbol_prefix` | 模板 `commission_rate` | `StrategyConfig.commission_policy` 或 `StrategyConfig.commission_rate` |
| 品种校验开关 | `validation_by_symbol_prefix` | 模板 `enforce_tick_size / enforce_lot_size` | 全局 `ChinaFuturesConfig.enforce_*` |
| 交易时段 | `run_backtest(sessions=...)` 按品种前缀指定 | 资产类型默认时段表 | `CN_STOCK` / `CN_FUTURES_DAY` / `CN_OPTION` / `ALWAYS_OPEN` |
| 市场路由 | `use_china_futures_market=False` 或混合资产回落 | `use_china_futures_market=True` 且纯期货 | `use_simple_market` |

中国股票扩展（`BacktestConfig.china_stock`）的委托校验开关：

| 配置项 | 字段 | 默认值 | 说明 |
|---|---|---|---|
| 委托价 tick 对齐校验 | `ChinaStockConfig.enforce_tick_size` | `True` | 关掉后股票/基金的非对齐委托价不再被撮合拒单。与 `ChinaFuturesConfig.enforce_tick_size` 对称 |

```python
from akquant import BacktestConfig, ChinaStockConfig, StrategyConfig

config = BacktestConfig(
    strategy_config=StrategyConfig(initial_cash=100000.0),
    china_stock=ChinaStockConfig(enforce_tick_size=False),  # 关闭 tick 校验
)
```

`enforce_tick_size` 缺省开启，非对齐委托价（如 `tick_size=0.01` 却传 `2.8314`）在回测撮合与实盘报单前都会被**拒单**而非静默取整——目的是让回测与实盘口径一致。需要自动对齐请用 `Strategy.round_to_tick()`，详见 [tick size 对齐指南](../advanced/tick_size_alignment.md)。

口径说明：

*   同级规则冲突时，以显式规则覆盖模板规则。
*   撮合校验路径按更具体前缀优先（更长匹配优先）。

中国期权扩展配置位于 `BacktestConfig.china_options`（`ChinaOptionsConfig`），用于管理中国期权费率。缺省值取沪深 ETF 期权现行口径，单位均为元/张：

| 字段 | 缺省值 | 说明 |
|---|---|---|
| `commission_per_contract` | `5.0` | 券商佣金，双向收取，卖出开仓照收（原 `fee_per_contract`，已改名） |
| `exchange_fee_per_contract` | `1.3` | 交易所交易经手费，双向收取 |
| `clearing_fee_per_contract` | `0.3` | 中国结算交易结算费，双向收取 |
| `exercise_fee_per_contract` | `0.6` | 行权结算费，到期时只向实值被行权的多头收取 |
| `sell_open_exempt` | `True` | 卖出开仓（含备兑开仓）免收经手费与结算费；交易所恢复收费时设为 `False` |
| `fee_by_symbol_prefix` | `None` | `ChinaOptionsFeeConfig` 列表，按品种前缀覆盖上面五项（字段同名同缺省，`commission_per_contract` 必填） |
| `use_china_market` | `True` | 是否使用 ChinaMarket；只有显式设为 `False` 才走 SimpleMarket（此时期权按成交额百分比计费，行权结算费为 0） |

以上缺省值只适用于沪深 ETF 期权；商品期货期权 / 股指期权（如 RB、MO、IO 等）费率与减免规则不同，请用 `fee_by_symbol_prefix` 显式设置费率与 `sell_open_exempt`。

只要回测里有期权合约，**默认就使用 ChinaMarket**（不再需要配置 `china_options` 或 `t_plus_one=True`）。副作用：同一回测里的 ETF 等基金改按 ChinaMarket 的 FUND 费率计费；同一回测里的期货沿用用户设置的百分比佣金率，非百分比佣金策略无法应用到 ChinaMarket 期货（回退为期货缺省费率并打一条 warning）。

卖单按"成交前持仓"拆成两段计费：先平掉多头的部分照收全部费用，超出多头的部分视为卖出开仓，`sell_open_exempt=True` 时只收佣金。`position_effect="auto"` 下，一笔穿越零持仓的卖单会被拆成一笔平仓单和一笔开仓单。行权结算费计入到期事件 `ExpiryEvent.fee` 并从到期现金流中扣除（`cash_flow` 为毛额，账户实收 `cash_flow - fee`），**计入** `trade_metrics.total_commission`。

到期平仓会作为一笔合成成交记入 `trades_df`（`order_tag="__expiry__"`、`order_id="expiry:<symbol>:<trading_date>"`），成交价期权取每单位内在价值、期货/股票取结算价，手续费为行权结算费；胜率等交易统计随之包含到期平仓，但该行的 `mae`/`mfe` 恒为 `0`（未回放持仓期内的价格历史）。合成成交**不会**出现在 `orders_df` / `executions` 中——它不是一笔订单。保证金账户期货到期时，`ExpiryEvent.cash_flow` 为平仓盈亏（与实际入账一致）；现金账户、股票、期权仍按到期现金流毛额计（`cash_flow - fee` 入账）。

中国期权扩展推荐使用以下优先级口径：

| 配置项 | 高优先级 | 中优先级 | 默认值 |
|---|---|---|---|
| 期权费率（按张） | `fee_by_symbol_prefix` | `ChinaOptionsConfig` 全局字段 | 上表缺省值 |
| 市场路由 | `use_china_market=False` 时 SimpleMarket | — | 有期权合约即 ChinaMarket |

品种预设：`InstrumentConfig.asset_type` 除四个基础类型外，还可以写品种预设名，在 `__post_init__` 中展开，显式传入的字段一律优先；展开后原始预设名保存在 `InstrumentConfig.product_preset`，由缺省值填充的字段名记在 `defaulted_fields`。

| 字段 | `ETF_OPTION` | `CONVERTIBLE_BOND` |
|---|---|---|
| 底层 asset_type | OPTION | FUND |
| multiplier | 10000 | 1（价格按每张计） |
| tick_size | 0.0001 | 0.001 |
| lot_size | 1 | 10 |
| sellable_after_days | 0 | 0 |
| option_margin_model | CHINA_SINGLE_LEG | — |
| 费用 | 上述全局期权规则 | 全局 FUND 规则；可用 `InstrumentConfig.commission_rate` 等字段按品种覆盖 |

```python
from akquant import InstrumentConfig

option = InstrumentConfig(
    symbol="10007000.SH",
    asset_type="ETF_OPTION",
    option_type="CALL",
    strike_price=3.0,
    expiry_date=20260325,
    underlying_symbol="510050.SH",
)
cb = InstrumentConfig(symbol="113050.SH", asset_type="CONVERTIBLE_BOND")
```

`multiplier`、`margin_ratio`、`tick_size` 的缺省值为 `None`，在 `__post_init__` 中按"预设 → 1.0（`tick_size` 为按资产类型的缺省值）"填充。

期货 vs 期权配置能力对照：

| 能力维度 | 中国期货（`china_futures`） | 中国期权（`china_options`） |
|---|---|---|
| 路由开关 | `use_china_futures_market` | `use_china_market` |
| 全局费率 | `StrategyConfig.commission_policy` / `StrategyConfig.commission_rate` 或模板费率 | `commission_per_contract` 等四项按张费用 |
| 前缀费率覆盖 | `fee_by_symbol_prefix` | `fee_by_symbol_prefix` |
| 合约参数模板 | 支持（乘数/保证金/tick/手数） | 不支持 |
| 撮合校验开关 | 支持（tick/手数，含前缀覆盖） | 不支持 |
| 交易时段 | `run_backtest(sessions=...)` 按前缀指定 | `run_backtest(sessions=...)` 按前缀指定；未匹配时跟随已登记的期货标的，否则 `CN_OPTION` |
| 前缀匹配策略 | 更长前缀优先 | 更长前缀优先 |

股票配置推荐使用以下优先级口径：

| 配置项 | 高优先级 | 中优先级 | 默认值 |
|---|---|---|---|
| 股票费率（佣金/印花税/过户费/最低佣金） | `InstrumentConfig` 单标的费率字段 | `StrategyConfig` 全局费率字段（含 `commission_policy` / `commission_rate`） | `run_backtest` 内置默认值 |
| 交易单位（`lot_size`） | `InstrumentConfig.lot_size`（显式设置） | `run_backtest(lot_size=...)` 全局设置 | `1` |
| 市场制度（T+1） | `run_backtest(t_plus_one=...)` 显式参数 | `broker_profile` 模板（内置 A 股模板为 `True`） | `False` |
| 市场模型 | `use_china_market()` | `use_simple_market()` | 引擎默认市场配置 |

股票侧说明：

*   当前股票没有按代码前缀的模板层（不像期货的 `china_futures` 前缀模板）。
*   生产场景建议优先用 `InstrumentConfig` 精确配置重点股票，再用 `StrategyConfig` 作为全局兜底。

#### 3. 风控配置合并 (Risk Config Merging)

`risk_config` 参数的处理逻辑比较特殊，旨在支持“基准配置 + 快速覆盖”的模式：

*   **基准**: 首先加载 `config.strategy_config.risk`（如果存在）。
*   **覆盖**: 如果提供了 `risk_config` 参数（字典或对象），它将覆盖基准配置中的同名字段。
    *   这允许你在不修改 Config 对象的情况下，通过 `run_backtest(..., risk_config={"max_position_pct": 0.5})` 快速调整风控参数进行测试。

#### 4. 策略运行时配置注入 (Strategy Runtime Config Injection)

`run_backtest` 与 `run_from_checkpoint` 支持 `strategy_runtime_config` 参数：

*   支持 `StrategyRuntimeConfig` 对象或 `dict`。
*   用于在不修改策略类代码的前提下注入运行时行为开关。
*   示例：`run_backtest(..., strategy_runtime_config={"error_mode": "continue"})`。
*   校验行为：未知字段或非法值会快速失败，并给出字段级错误信息。
*   冲突处理：`runtime_config_override=True` 时应用外部配置；`False` 时保留策略侧配置。
*   上述冲突规则在 `run_backtest` 与 `run_from_checkpoint` 中保持一致。
*   对同一策略实例、同一冲突内容，告警日志会自动去重。
*   优先级规则：显式传入的 `strategy_runtime_config` 参数高于转发配置映射中的同名配置。
*   故障速查入口：参考 [Runtime Config 指南](../advanced/runtime_config.md)。

```python
from akquant import StrategyRuntimeConfig, run_backtest

result = run_backtest(
    data=data,
    strategy=MyStrategy,
    strategy_runtime_config=StrategyRuntimeConfig(
        error_mode="continue",
        portfolio_update_eps=1.0,
    ),
)
```

#### 5. 最佳实践 (Best Practices)

*   **简单脚本**: 直接使用 `run_backtest` 的扁平参数（如 `initial_cash`, `start_time`）。
*   **生产/复杂策略**: 构建完整的 `BacktestConfig` 对象，以便于版本管理和复用。
*   **页面化参数输入**: 在策略类中内联声明参数字段（`IntParam` 等，例如 `fast_period = IntParam(10, ge=2, le=200)`，运行时通过 `self.params.fast_period` 访问），并使用 `get_strategy_param_schema` / `validate_strategy_params` 完成前后端参数联动与校验。
*   **参数调优**: 使用 `run_grid_search` 时，通常通过修改 Config 对象或传入 override 参数来实现。

### 日志配置 API (Logging)

AKQuant 作为库使用时默认保持静默；未显式配置前，`akquant` 根 logger 仅挂载 `NullHandler`。

#### `akquant.LogConfig`

高级日志配置对象，供 `configure_logging(...)` 使用。

核心字段：

*   `level`: 全局回退等级。
*   `console`: 是否启用控制台 handler。
*   `console_level` / `file_level`: handler 级别覆盖。
*   `console_format` / `file_format`: 文本 formatter 覆盖。
*   `console_show_context` / `file_show_context`: 文本模式下是否附带结构化上下文。
*   `console_json` / `file_json`: 是否对对应 handler 启用 JSON line 输出。
*   `filename`: 文件日志路径。
*   `file_mode`: 文件模式，默认 `a`。
*   `file_max_bytes` / `file_backup_count`: 启用按大小轮转时的阈值与保留份数。
*   `profile`: 预设 profile，支持 `research`、`optimize`、`live`。
*   `reset_handlers`: 是否重置 AKQuant 自己管理的 handler。
*   `propagate`: 是否向上游 logger 传播。
*   `mask_sensitive`: 是否对敏感字段脱敏（默认 `True`）。密钥类（`password`/`token`/`api_key` 等）全掩码、账户类（`user_id`/`account` 等）保留尾 4 位；在 handler 层兜底，任何调用点忘记脱敏也不会泄漏。
*   `order_audit_file`: 实盘订单审计的独立 JSON 文件路径。设置后，`broker_live` 下每一笔订单的提交/回报/成交/撤单/拒单会额外以 JSON line 写入该文件（`akquant.audit.order` 命名空间），用于事后对账与复盘。
*   `order_audit_level`: 审计文件级别，默认 `INFO`。
*   `order_audit_max_bytes` / `order_audit_backup_count`: 审计文件按大小轮转的阈值与保留份数（默认保留 5 份）。
*   `language`: **控制台**审计消息语言，`"en"`（默认）/`"zh"`。仅影响控制台的订单审计行渲染；文件与 JSON 恒为英文 canonical，结构化字段（`event`/`side`/`price` 等）任何语言下不变，因此 grep/告警/对账不会因语言分裂。

#### `akquant.configure_logging`

```python
def configure_logging(config: LogConfig) -> logging.Logger
```

使用结构化配置初始化或重配 `akquant` 日志系统。

推荐示例：

```python
import akquant

akquant.configure_logging(
    akquant.LogConfig(
        profile="live",
        level="INFO",
        console=True,
        console_json=False,
        filename="logs/live.log",
        file_level="DEBUG",
        file_json=True,
        file_max_bytes=10_000_000,
        file_backup_count=5,
    )
)
```

行为说明：

*   `profile` 只填充未显式指定的字段，显式参数优先级更高。
*   `profile="optimize"` 默认文本格式会带 `processName`，便于区分 worker。
*   `profile="live"` 适合打开结构化上下文或 JSON 输出。
*   Rust 侧运行路径中的 `akquant.*` warning 也会桥接进入 Python `logging`，并尽量恢复为统一的结构化字段。
*   例如执行链路中的保证金不足拒单、收盘过期、取消未知订单、同一切片 `same-cycle` 延后等 warning，会携带 `phase="execution"`，并在可用时附带 `symbol`、`order_id`、`strategy_id`、`slot`、`event_time_iso`。

#### `akquant.register_logger`

```python
def register_logger(
    filename: Optional[str] = None,
    console: bool = True,
    level: str = "INFO",
) -> None
```

兼容快捷接口，适合快速打开日志，不暴露高级字段。内部会转成 `configure_logging(LogConfig(...))`。

#### `akquant.get_logger`

```python
def get_logger(name: Optional[str] = None) -> logging.Logger
```

获取 `akquant` 命名空间下的 logger：

*   `get_logger()` -> `akquant`
*   `get_logger("strategy")` -> `akquant.strategy`
*   `get_logger("gateway.live")` -> `akquant.gateway.live`

#### `akquant.set_log_level`

```python
def set_log_level(level: Union[str, int]) -> None
```

修改当前 `akquant` 根 logger 的 level。

#### 使用边界

*   `self.log(...)` 面向人类阅读的策略调试日志。
*   `run_backtest(..., on_event=...)` 面向机器消费的统一事件流，更适合实时 UI、告警、审计落盘。
*   在 `on_order` / `on_trade` / `on_reject` 中使用 `self.log(...)` 时，日志会自动携带 `order_id` / `client_order_id` 等结构化字段。
*   Rust 执行层与数据层产生的 warning 不需要用户手动接管；只要已经配置了 `akquant` logger handler，它们就会进入同一套文本或 JSON 输出链路。

## 2. 策略开发 (Strategy)

### `akquant.Strategy`

策略基类。用户应继承此类并重写回调方法。

**回调方法:**

*   `on_start()`: 策略启动时触发。用于订阅 (`subscribe`) 和注册指标。
*   `on_bar(bar: Bar)`: K 线闭合时触发。
*   `on_tick(tick: Tick)`: Tick 到达时触发。
*   `on_order(order: Order)`: 订单状态更新时触发（如成交、取消、拒绝）。
*   `on_trade(trade: Trade)`: 订单成交时触发。
*   `on_reject(order: Order)`: 订单首次进入 `Rejected` 时触发一次。实盘（`broker_live`）下，柜台明确回绝的报单也经此回调回吐，`buy()`/`sell()`/
    `order_target_*` **不抛异常**；若因超时/断连导致订单状态不可知，则改走
    `on_error(error, "order_submit", request)`，不会伪造拒单。
*   `on_expiry(event: Dict[str, Any])`: 到期结算回调。仅当引擎实际执行 `expiry_date` 驱动的到期结算/移除后触发；回调时账户状态已更新。示例见：`examples/49_on_expiry_demo.py`。
*   `on_before_trading(trading_date, timestamp)`: 交易日最早的连续竞价开始后 1ns 触发，每个交易日至多一次（回测只为有行情的交易日触发，实盘按交易日历逐日触发；A 股 09:30:00 之后；有夜盘的期货交易日从前一交易日夜盘开始，周一交易日在周五 21:00 触发；交易日没有连续竞价段时不触发）。该回调按“前一交易日/前一时点信息可见”的语义工作。
*   `on_pre_open(event: Dict[str, Any])`: 交易日首个时段开始前 1ns 触发，每个交易日一次（A 股 09:15 前，即 09:14:59.999…；有夜盘的期货为前一交易日 20:55 夜盘集合竞价前），不早于上一交易日的日终结算；回测与实盘一致。同一交易日它一定先于 `on_before_trading`。适合“盘前决策，本次 open 成交”；默认下单语义等价于 `NextOpen()`，这类单参加开盘集合竞价统一撮合（撮合在连续竞价开始时进行，成交价为竞价价，成交时间记在首个连续竞价事件上）。`event["expected_open_at"]` 是本次触发时刻，不是开盘时刻。示例见：`examples/52_pre_open_demo.py`。
*   `on_cross_section(trading_date, timestamp)`: 横截面同周期调仓钩子。在框架看到当日首个“跨标的完整 bar 切片”后触发，每个交易日最多一次；与 `on_before_trading` 不同，它可以看到当日历史和当前账户快照，适合收盘价同周期调仓。调仓频率（日/周/月）在回调内用日历判断。
*   `on_after_trading(trading_date, timestamp)`: 日终结算完成后触发，每个交易日一次。此时 T+1 已解冻、当日 Day 单已过期，其中下的单属于下一交易日。回测在收盘时刻；实盘在收盘 + `day_close_delay`（见 [run_live 实盘时区与交易时段](#live-timezone-sessions)）。回测数据在盘中结束、当日未结算时不触发，停止时也不补发。
*   `on_portfolio_update(snapshot)`: 账户快照变化时触发。
*   `on_error(error, source, payload=None)`: 用户回调抛异常时触发。是否在触发 `on_error`
    后继续向上抛出，取决于 `error_mode`/`re_raise_on_error`（默认继续抛出）——这条规则适用于
    `on_bar`/`on_tick` 等经框架调度路径产生的普通回调异常，回测与实盘（`broker_live`）下行为一致。
    但 `broker_live` 下由**柜台通信失败**触发的 `on_error`——包括下单状态未知（`source="order_submit"`）、
    本地止损单重试失败（`"stop_trigger"`）、撤单失败（`"order_cancel"`/`"order_cancel_all"`），
    以及**同步派发的 `on_order`/`on_reject`/`on_trade` 回调自身抛出的异常**（source 为 `"on_order"`/
    `"on_reject"`/`"on_trade"`）——一律只调用 `on_error` 后吞掉，**不受 `re_raise_on_error` 控制，
    也不会继续向上抛出**（实现分别位于 `python/akquant/gateway/order_submitter.py::_dispatch_reject_order`、
    `python/akquant/gateway/broker_execution.py::_notify_error`/`_notify_stop_error`/`_handle_cancel_failure`、
    `python/akquant/live/_runner.py::_safe_strategy_callback`）。
*   `on_timer(payload: str)`: 定时器触发。
*   `on_stop()`: 策略停止时触发。
*   `on_train_signal(context)`: 滚动训练信号触发 (ML 模式)。

`on_pre_open` 推荐写法：

```python
def on_pre_open(self, event: Dict[str, Any]) -> None:
    signal = self.compute_pre_open_signal()
    if signal > 0:
        self.buy("000001", quantity=100)
```

说明：若这里不显式传 `fill_mode`，框架会默认按 `NextOpen()`（当日 open）语义处理订单。

**属性与快捷访问:**

*   `self.symbol`: 当前正在处理的标的代码。
*   `self.close`, `self.open`, `self.high`, `self.low`, `self.volume`: 当前 Bar/Tick 的价格和成交量。
*   `self.position`: 当前标的持仓辅助对象 (`Position`)，包含 `size` 和 `available` 属性。
*   `self.now`: 当前回测时间 (`pd.Timestamp`)。
*   `self.runtime_config`: 运行时行为配置对象 (`StrategyRuntimeConfig`)。
*   `self.trading_day -> Optional[datetime.date]`: 当前时刻所属交易日。盘中属本交易日，夜盘属下一交易日（周五 21:30 属下周一）；收盘后至下一交易日开盘前属下一交易日；实盘收盘宽限期内到达的行情仍归当日；尚无上下文（如 `on_start` 中）时为 `None`。
*   `self.session_of(symbol=None) -> TradingSession`: 品种此刻的交易时段，与撮合同口径；不传 `symbol` 即当前事件的时段 `ctx.session`。只能在运行期间调用。
*   `self.is_market_open(symbol=None, include_auction=False) -> bool`: 品种此刻是否处于连续竞价；`include_auction=True` 时 `OpeningAuction` / `AuctionHold` / `ClosingAuction` 也算开市。
*   `self.portfolio_update_eps`: 账户快照更新阈值，低于该变化量不触发 `on_portfolio_update`（默认 `0.0`）。
*   `self.error_mode`: 错误处理模式，`"raise"` 或 `"continue"`（默认 `"raise"`）。
*   `self.re_raise_on_error`: 用户回调异常后是否继续抛出（默认 `True`）；仅对经框架调度的普通
    回调异常生效（见上文 `on_error` 的范围说明）。`broker_live` 下由柜台通信失败触发的
    `on_error`（`order_submit`/`stop_trigger`/`order_cancel`/`order_cancel_all` 等）不受此项
    控制，恒不重抛。
*   `self.ctx`: 策略上下文 (`StrategyContext`)，提供底层 API 访问。与交易时段、订单终态相关的只读字段：
    *   `ctx.session`: 当前事件的交易时段（`TradingSession`，7 个值：`Closed`、`OpeningAuction`、`AuctionHold`、`Continuous`、`Break`、`ClosingAuction`、`PostClose`）。
    *   `ctx.trading_day`: 当前时刻所属交易日（同 `self.trading_day`）。
    *   `ctx.recent_rejected_orders`: 本步被拒的订单（框架据此补发 `on_reject`）。
    *   `ctx.recent_expired_orders`: 最近一次日终结算置为 `Expired` 的 Day 单。
    *   `ctx.recent_finalized_orders`: 本步未成交就离开在途账本的订单（撤单、过期、拒单）的终态快照。

**交易方法:**

*   `buy(symbol=None, quantity=None, price=None, trigger_price=None, ...)`: 买入（开多/平空）。
    *   如果不指定 `price`，则为市价单。
    *   如果指定 `price`，则为限价单。
    *   如果指定 `trigger_price`，则为止损/止盈单 (Stop Market)。
    *   不传 `symbol` 时取当前 bar/tick 的标的；在无行情上下文的回调（如 `on_start`）中必须显式传入。
    *   不传 `quantity` 时按 `self.sizer` 计算下单量（默认 `FixedSize(100)`，可用 `set_sizer()` 替换），算出的数量按标的最小交易单位（`lot_size`）向下取整，不足一手则不下单；`lot_size` 为 1 时不取整（小数数量照常下单）。显式给的 `quantity` 不取整。
*   `sell(symbol=None, quantity=None, price=None, trigger_price=None, ...)`: 卖出（平多/开空）。参数同上，但**不传 `quantity` 时不走 sizer，而是全平当前持仓**：回测取总持仓，`broker_live` 取可用持仓（A 股 T+1 下当日买入部分不可卖，按总量报单会被柜台整单拒绝）。
*   解析后下单量 `<= 0` 时不报单，返回空回执（`len(receipt) == 0`、`receipt.primary == ""`）。
*   `receipt.failure`（只读，`None` / `"rejected"` / `"unknown"`）：区分空回执的三种成因。
    *   `None`：无需交易（如解析后下单量 `<= 0`），不代表任何失败。
    *   `"rejected"`：柜台明确拒单，订单**确定不存在**，重试是安全的。
    *   `"unknown"`：提交遇到超时/断连等异常，订单状态**不可知**（报文可能已到柜台）——
        消费方**不得**假设订单不存在而重发，否则若报文其实已被柜台接受，重发就是一笔
        真实的重复委托；应改为等待下一轮 `sync_open_orders` 对账浮出真实状态。
*   `submit_order(..., order_type="StopTrail", trail_offset=..., trail_reference_price=None)`: 提交跟踪止损单。`trail_offset` 必须大于 0。
*   `submit_order(..., order_type="StopTrailLimit", price=..., trail_offset=..., trail_reference_price=None)`: 提交跟踪止损限价单。`price` 与 `trail_offset` 必填。
*   `submit_order(..., broker_options={...})`: 可选 broker 扩展参数透传（回测阶段仅记录在订单对象 `order.broker_options` 上，便于联调与审计）。
*   `place_trailing_stop(symbol, quantity, trail_offset, side="Sell", trail_reference_price=None, ...) -> str`: 跟踪止损助手，触发后按市价执行。
*   `place_trailing_stop_limit(symbol, quantity, price, trail_offset, side="Sell", trail_reference_price=None, ...) -> str`: 跟踪止损限价助手，触发后按限价执行。
*   `rebalance_weights(target_weights, price_map=None, liquidate_unmentioned=False, allow_leverage=False, rebalance_tolerance=0.0, ...)`: 按多标的目标权重调仓。
    *   `target_weights` 形如 `{symbol: weight}`，默认要求权重和不超过 `1.0`。
    *   `liquidate_unmentioned=True` 时，会将未出现在目标字典中的现有持仓目标设为 `0`。
    *   执行顺序为先卖后买，减少现金约束导致的调仓失败。
    *   `rebalance_tolerance` 按组合市值比例跳过小偏差，降低无效换手。
*   `cancel_order(order_id: str)`: 撤销指定订单。
*   `cancel_all_orders(symbol)`: 取消指定标的的所有挂单。如果不指定 `symbol`，则取消所有挂单。
*   `place_oco(first_order_id, second_order_id, group_id=None) -> str`: 创建 OCO 订单组。组内任一订单成交后，另一订单会被自动撤单。
*   `place_bracket(symbol, quantity, entry_price=None, stop_trigger_price=None, take_profit_price=None, ...) -> str`: 创建 Bracket 订单。先提交进场单，进场成交后自动提交止损/止盈；当止损与止盈同时存在时会自动绑定 OCO。

**数据与工具:**

*   `get_history(count, symbol, field="close", freq=None) -> np.ndarray`: 获取历史数据数组（返回滚动缓冲的安全快照拷贝，非零拷贝）。
*   `get_history_multi(count, symbol, fields=("open","high","low","close","volume"), freq=None) -> Dict[str, np.ndarray]`: 单次跨界批量取回多字段，语义等价于逐字段 `get_history`，`get_history_df` 内部即基于它。
*   `get_history_map(count, symbols, field="close", freq=None) -> Dict[str, np.ndarray]`: 批量获取多个标的历史数据。
*   `rebalance_to_topn(scores, top_n, weight_mode="equal", ...) -> List[str]`: 根据打分选取 TopN 并执行调仓，支持等权或按分数归一化。
*   `get_history_df(count, symbol, freq=None) -> pd.DataFrame`: 获取历史数据 DataFrame (OHLCV)。
*   `freq` 参数（`get_history` / `get_history_map` / `get_history_multi` / `get_history_df` / `get_rolling_data` 均支持）：取值 `'tick'` / `'bar'` / `None`。
    *   `None`（默认）：该 symbol 只有 bar 序列时取 bar，只有 tick 序列时取 tick（单流下行为不变）。**双流下（该 symbol 同时存在 bar 与 tick 两条序列）按当前所处的回调自动定档**：`on_bar` 里等价于 `freq='bar'`，`on_tick` 里等价于 `freq='tick'`。注意 tick 序列由引擎无条件写入，**与策略是否覆写 `on_tick` 无关**——只要订阅了 tick 流，只写 `on_bar` 的策略也处于双流之中，靠的就是这条推断。**行情回调之外**（`on_timer`、`on_before_trading`、用户自建线程等）无从推断意图，双流下仍报 `ValueError`，要求显式传 `freq='bar'` 或 `freq='tick'`——不会静默选一条。未识别的取值同样报错，不会兜底成 `'bar'`。
    *   `freq='tick'` 时 `field` 只支持 `price`/`close`/`volume`：tick 没有 open/high/low，传这些字段会抛 `ValueError`（此前会静默返回退化 OHLC，`price` 冒充 `high`，破坏性变更）。`get_history_df` / `get_rolling_data` 固定取 OHLCV 五字段，因此在 `freq='tick'` 下必然报错，请改用 `get_history(freq='tick', field='price')`。
    *   回测中让 `on_bar` 与 `on_tick` 同时触发：`run_backtest(data=[Tick, ...], freq="1min")`（含 `Tick` 的列表下 `freq` 是聚合周期；bar 数据也可传 `freq` 声明周期，含 `"1d"`，只用来填 bar 区间起点，不产生 tick）。实盘中的等价开关是 `gateway_options={"emit_ticks": True, "emit_bars": True}`（klinedata、CTP 网关均支持，`use_aggregator` 保留为兼容别名）：两个网关对 `emit_ticks`/`emit_bars` 本身的处理规则一致——按参数逐个回退（只显式传其一不会静默关掉另一路），若回退后 `emit_ticks` 与 `emit_bars` 都为 `False` 则报错而非静默不推送任何数据。`broker="ctp"` 走的是 `run_live(..., gateway_options=...)` → builder 转发这条链路，此前该链路会把这两个键静默丢弃（`on_tick` 永不触发且无任何报错），现已修复为原样转发到底层 `CTPMarketGateway`。**klinedata 有一个前置条件 CTP 没有**：klinedata 多一个 `drive` 参数，缺省为 `drive="bar"`，而 `drive="bar"` 只订阅 K 线频道、逐笔帧根本不会到达。因此 `drive="bar"` 配**显式** `emit_ticks=True` 会直接**报错**并给出可照抄的双流配方（此前是静默压成 `emit_ticks=False`，表现为「`on_tick` 永不触发且不知道为什么」）——要让 `on_tick` 真正触发，必须显式传 `"drive": "tick"`，仅传 `emit_ticks=True` 不够。经 `use_aggregator` 别名回退推导出的值仍按 `drive` 处理以保持兼容。CTP 没有 `drive` 这一层，`emit_ticks=True` 会直接生效。
*   `get_position(symbol) -> float`: 获取当前持仓量。返回值仍为数量，不返回对象。
*   `get_available_position(symbol) -> float`: 获取可用持仓量。
*   `positions -> Dict[str, float]`: 获取所有标的持仓（只读属性）。
*   `self.position.entry_price -> float`: 通过 `Position` helper 获取当前标的持仓均价。
*   `self.position.avg_price -> float`: `entry_price` 的别名。
*   `ctx.get_position_entry_price(symbol) -> float`: 获取指定标的当前持仓均价。
*   `ctx.get_position_entry_prices() -> Dict[str, float]`: 获取所有标的当前持仓均价。
*   `get_holding_bars(symbol) -> int`: 获取当前持仓持有的 Bar 数量。
*   `cash -> float`: 获取当前可用资金（只读属性）。
*   `freq -> Optional[str]`: 数据周期（只读属性），取值为回测侧口径 `"1min"` / `"5min"` / `"1d"` 等。
    *   回测侧由 `run_backtest(freq=)` 注入（tick 聚合周期，或 bar 数据声明的周期）；实盘侧由行情网关声明（klinedata 把自己的 `period="M1"` 转换成 `"1min"`，`broker="replay"` 支持 `gateway_options={"freq": "1min"}`）。两侧统一到回测口径，故同一个策略在回测与实盘读到同一个值，无需按 broker 写兼容分支。
    *   **拿不到周期时为 `None`**：纯 bar 回测未传 `freq`、CTP 等只有逐笔源的网关、trader-only broker（无行情通道）、klinedata 周线（回测侧 `freq` 只认分钟、小时与日，周线无对应写法）。框架**不从数据推断**周期——相邻 bar 的时间戳差会被停牌、跨日、午休误导，错误的周期比未知的周期更危险。请显式处理 `None`。
    *   只读，写入抛 `AttributeError`：数据粒度由数据源决定，改这个属性不会真的改变粒度。
*   `get_account() -> Dict[str, float]`: 获取账户详情快照。常见字段包括 `cash`、`equity`、`market_value`、`notional_value`、`frozen_cash`、`margin`、`used_margin`、`free_margin`、`unrealized_pnl`、`borrowed_cash`、`short_market_value`、`maintenance_ratio`、`account_mode`、`accrued_interest`、`daily_interest`。
    *   现金账户 / 现货账户下，`market_value` 通常表示持仓市值。
    *   期货保证金账户下，`equity` 表示账户权益，`used_margin` 表示已占用保证金，`notional_value` 表示期货名义敞口，`unrealized_pnl` 表示浮动盈亏；期货持仓不会像股票那样把全额名义本金直接计入 `cash` 扣减，也不会把名义敞口直接映射为 `market_value`。
    *   `cash` 是现金余额，`free_margin`（= `equity - used_margin`）才是可用于新开仓的资金，与下单被拒时日志里的 `Available` 口径一致。期货保证金账户下开仓不从 `cash` 扣减保证金，因此 `cash` 通常大于 `free_margin`；股票现金账户下二者相等。
    *   在策略回调内，如果你只想读取“当前账户总权益”，优先使用 `equity`；其口径与 `get_account()["equity"]` 对齐。
*   `get_order(order_id) -> Order`: 获取指定订单详情。
*   `get_open_orders(symbol) -> List[Order]`: 获取当前未完成订单列表。
*   `get_trades() -> List[ClosedTrade]`: 获取所有已平仓交易记录。
*   `subscribe(instrument_id: str)`: 订阅行情。
*   `log(msg: str, level: int)`: 输出带时间戳的日志。
*   `schedule(trigger_time, payload)`: 注册单次定时任务。
*   `schedule_daily(time_str, payload)`: 注册每日定时任务(交易日历上的每个交易日触发，回测与实盘一致)。
*   `schedule_weekly(time_str, payload)`: 每周首个交易日触发(按交易日历，节假日自动顺延；回测与实盘一致)。
*   `schedule_monthly(time_str, payload)`: 每月首个交易日触发(按交易日历，节假日自动顺延；回测与实盘一致)。
*   `trading_days -> List[pd.Timestamp]`: 只读交易日序列，配合 `schedule` 自定义节奏。
*   `nth_trading_day_of_month(n)` / `nth_last_trading_day_of_month(n)` / `nth_trading_day_of_week(n)`: 交易日历辅助，返回每月/周第 n 个(或倒数第 n 个)交易日。
*   `to_local_time(timestamp) -> pd.Timestamp`: 将 UTC 时间戳转换为本地时间。
*   `format_time(timestamp, fmt) -> str`: 格式化时间戳。

**标的静态属性 API（推荐）:**

*   `get_instrument(symbol) -> InstrumentSnapshot`: 获取单个标的静态属性快照。
*   `get_instruments(symbols=None) -> Dict[str, InstrumentSnapshot]`: 获取多个标的静态属性快照字典；`symbols=None` 时返回全部。
*   `get_instrument_field(symbol, field) -> Any`: 获取单个标的字段值。
*   `get_instrument_config(symbol, fields=None) -> Union[Any, Dict[str, Any], InstrumentSnapshot]`: 兼容接口；支持整对象、单字段或多字段读取。
*   `round_to_tick(symbol, price, direction="nearest") -> float`: 按该标的的 `tick_size` 对齐委托价。`direction` 取 `"down"`（买入侧保守）/ `"up"`（卖出侧保守）/ `"nearest"`。股票/基金的非对齐委托价会被撮合与实盘报单前校验**拒单**（见 [tick size 对齐指南](../advanced/tick_size_alignment.md)），需要自动对齐时显式调用本方法。

说明：

*   这些接口在 `on_start` 即可使用（启动阶段已注入快照）。
*   推荐优先使用这些接口读取静态属性，而不是依赖 `bar.extra`。

**机器学习支持:**

*   `set_rolling_window(train_window, step)`: 设置滚动训练窗口。
*   `get_rolling_data(length, symbol, freq=None)`: 获取滚动训练数据 (X, y)。`freq` 语义与限制同上一节的 `get_history` 系列（底层基于 `get_history_df`）。
*   `prepare_features(df, mode)`: (需重写) 特征工程与标签生成。

**类定义行参数:**

*   `class MyStrategy(aq.Strategy, title="趋势", id="trend")`：`title` 为显示名（省略为类名），`id` 为机器标识。写了 `id` 时它就是 `strategy_id`，与 `run_backtest/run_live(strategy_id=...)` 不一致报 `ValueError`；不写 `id` 时 `strategy_id` 仍默认 `"_default"`。策略不接受 `overlay=`（报 `TypeError`）。
*   `self.I(...)` 在策略里**默认不画图**，要画写 `plot=True`；`pane` / `color` / `label` 等绘图参数不再隐含 `plot=True`。

### `akquant.Indicator` {: #akquant-indicator }

指标脚本基类，对标 Pine v5 的 `indicator()`：只声明指标、不交易，经 `run_backtest/run_live(indicators=[...])` 挂载。

```python
class DualMA(aq.Indicator, title="双均线", overlay=True, id="dual_ma"):
    def on_start(self):
        self.fast = self.I(aq.EMA(10), name="ema_fast")
```

*   `title`: 显示名，只用于展示，省略为类名。
*   `overlay`: `True` 画主图，`False`（默认）画附图。`self.I` 默认画图：`overlay=True` 时默认 `pane=0`，否则默认 `pane=1`（pane 编号只在本脚本内有效）；`plot=False` 关闭。
*   `id`: 机器标识（即指标点的 `owner_strategy_id`），省略为类名 snake_case。
*   `replaces_price`: `True` 表示本脚本替代价格 K 线（如平均 K 线），只是给前端的提示，隐藏原 K 线由前端决定（内置复盘页另有 `result.viz.review(hide_price=)` 与页面按钮）；必须同时写 `overlay=True`（否则 `ValueError`），值须为 `bool`（否则 `TypeError`），可被子类继承，默认 `False`。`X.resolve_replaces_price()` 返回生效值。`aq.Strategy` 上写报 `TypeError`。
*   全部交易 API（`buy` / `sell` / `submit_order` / `place_*` / `order_target*` / `rebalance_*` / `close_position` / `cancel_*`）抛 `akquant.IndicatorCannotTradeError`。
*   详见[自定义指标指南](../guide/custom_indicator.md)。

### `akquant.PrecomputedIndicator`

向量化预计算指标：`PrecomputedIndicator(name, fn)`，`fn(df) -> pd.Series` 对单个标的的完整 `DataFrame` 一次性计算，经 `self.I(PrecomputedIndicator(...))` 声明。只用于回测；不支持含 Tick 的输入、`freq=` 与 `warmup_bars=`。

### `akquant.HeikinAshi`

内置 Rust 增量指标：平均 K 线（Heikin-Ashi）。

```python
ha = aq.HeikinAshi()
ha.update(open, high, low, close)  # -> (ha_open, ha_high, ha_low, ha_close)，第一根即有值
```

*   公式：`ha_close = (O+H+L+C)/4`；`ha_open` 第一根取 `(O+C)/2`，之后取前一根 `(ha_open+ha_close)/2`；`ha_high = max(H, ha_open, ha_close)`；`ha_low = min(L, ha_open, ha_close)`。
*   另有 `value` 属性与批量方法 `update_many_ohlc`；支持 `clone_indicator` 与 `intrabar=True`。没有 `akquant.talib` 批量函数。
*   画成 K 线：`self.I(aq.HeikinAshi(), name="ha", input_mode="ohlc", outputs=("open", "high", "low", "close"), render_type="candle")`。四个分量（`ha.open` 等）的定义带 `group="ha"` 与 `role`；LWC 与 plotly 画成一条 K 线，d3Kline 降级为 close 一条线。见[自定义指标指南](../guide/custom_indicator.md)。

### `akquant.inspect_strategy`

```python
def inspect_strategy(source: Union[str, os.PathLike[str]]) -> InspectResult
```

静态检查策略/指标脚本文件或源码字符串：只用 `ast` 解析，不导入、不执行用户代码，只接受明文（UTF-8，可带 BOM）。返回 `InspectResult`（`classes`、`error`，`to_dict()` 可序列化为 JSON），每个类给出 `kind`（`"indicator"` / `"strategy"` / `"unknown"`）、`title`、`id`、`overlay`、`replaces_price`（策略为 `None`，指标默认 `False`）、`draws_indicators` 与每条 `self.I` 的 `name` / `pane` / `plot` / `outputs` / `label`；静态确定不了的值为 `None` 并记入 `dynamic`。同时识别 `akquant` 与 `qmfquant` 的导入。以 `.py` 结尾但不存在的路径报 `FileNotFoundError`，入参类型错报 `TypeError`，其余问题写进 `error` / `dynamic`。详见[自定义指标指南](../guide/custom_indicator.md)「静态检查」一节。

### `akquant.Bar`

K 线数据对象。

*   `timestamp`: Unix 时间戳 (纳秒)，为 bar 的**区间终点**（右闭区间 `(interval_start, timestamp]`）。只有日期的日线改写到交易日收盘时刻。
*   `interval_start`: 区间起点 (UTC 纳秒，不含)，`None` 表示未知。tick 合成的 bar、声明了 `freq` 的 bar 数据、只有日期的日线、引擎聚合的窗口 bar 都带它；未知时引擎按“时间戳前 1ns”判时段。构造时可传 `Bar(..., interval_start=...)`。
*   `open`, `high`, `low`, `close`, `volume`: OHLCV 数据。
*   `symbol`: 标的代码。
*   `extra`: 扩展数据字典 (`Dict[str, float]`)。
*   `freq`: 周期标签，基础 bar 为 `None`，引擎聚合出的窗口 bar 带 `"5min"` 等。
*   `timestamp_iso`: UTC ISO 8601 时间字符串。

### `akquant.Tick`

Tick 数据对象。

*   `timestamp`: Unix 时间戳 (纳秒)。
*   `price`: 最新价。
*   `volume`: 成交量。**单笔量**（与回测语义一致），不是累计量。
    *   实盘网关（CTP、klinedata）推给 `on_tick` / `add_tick` 的 `Tick.volume` 是这一笔的成交量；柜台/上游行情原始推的是**当日累计成交量**，网关内部按 symbol 做差分换算后才对外暴露单笔量。
    *   与之相对，网关内部用于 `freq` 聚合（tick 合成 bar）的 `BarAggregator` 吃的是**原始累计量**（构造时声明 `volume_is_cumulative=True`，由聚合器自己差分求和）——这是刻意的分工，不要把它当成 bug "修" 成单笔量，否则聚合量会被腰斩。
    *   **已知代价**：进程盘中启动时，某 symbol 收到的第一帧行情因为没有上一次累计量可比对，换算不出真实单笔量，此时 `Tick.volume` 记为 `0`（表示"未知"，而非误用累计量冒充单笔量）。若策略里有「`volume == 0` 就跳过」之类的防御逻辑，会连带跳过每个 symbol 的第一笔行情。
*   `symbol`: 标的代码。

### `akquant.run_live`（broker_live 执行语义） {: #live-broker-semantics }

实盘 broker 路由可通过 `gateway_options` 传入网关特定参数：

```python
from akquant import run_live

run_live(
    strategy_cls=on_bar,
    instruments=instruments,
    broker="ctp",
    trading_mode="broker_live",
    gateway_options={"execution_semantics_mode": "strict"},
)
```

`gateway_options.execution_semantics_mode`：

| 取值 | 默认值 | 行为 | 推荐场景 |
| :--- | :--- | :--- | :--- |
| `strict` | 是 | `Cancelled` / `Rejected` / `Filled` 等终态由订单回报 (`OnRtnOrder`) 最终确认。错误回报会先缓存拒单原因，再在后续订单回报中补齐。 | 生产实盘 |
| `compatible` | 否 | 在部分错误/撤单路径允许本地立即推进终态，以兼容历史行为。 | 迁移过渡 |

严格模式注意事项：

*   撤单请求发送成功不等于 `Cancelled`，需等待 `OnRtnOrder(Cancelled)`。
*   收到错误回报不等于 `Rejected`，最终状态以订单回报为准。

### `akquant.run_live(preload_history=...)`（实盘历史预热） {: #live-preload-history }

```python
run_live(..., preload_history=history)
```

*   `preload_history`：`pd.DataFrame` / `dict[str, pd.DataFrame]` / `list[Bar]` / `None`（默认）。形态与列约定同 `run_backtest(data=)`。框架不内置数据源，数据由调用方提供，应截止到上一根已收盘 bar。
*   **会做**：灌入历史缓冲（策略的 `_history_depth` 自动抬到预热深度，不下调已设的更大值）；喂 `subscribe_bars` 窗口聚合器；计入 per-symbol `warmup_period`；在主策略与全部 slot（含指标脚本）的 `on_start` 都跑完之后逐根回放 `self.I()` 指标（含 `freq=` 窗口指标），绘图点以 `phase="preload"` 进入指标流。
*   **不会做**：不触发 `on_bar`、不撮合、不动账户。
*   **校验**：缺必需列、或一个标的都匹配不上 `instruments` 时抛 `ValueError`；部分标的不匹配、晚于会话启动的行、空数据均告警后跳过；实盘首根 bar 不晚于预热末戳时告警但照常派发。
*   **热启动**：从存档恢复的策略跳过指标回放，历史缓冲仍会灌入。
*   `preload_bars`：`int` / `None`（默认）。行情 broker 在 `GatewayBundle.history_provider` 上提供了历史拉取函数（如 `klinedata`）且未传 `preload_history` 时生效，拉到的历史走与 `preload_history` 相同的路径。`None` 自动：每标的取 `min(10000, max(3 × 推断 warmup, 500, warmup_period, 历史深度))` 根（各项对主策略与全部 slot 取最大值），tick 驱动的行情源不开启；`0` 关闭；`N > 0` 取 N 根（截到 10000，tick 驱动也生效）；负数或非整数抛 `ValueError`。显式 `preload_history` 优先，此时不调用行情源。拉取抛异常告警后照常启动，返回空记 INFO。
*   `GatewayBundle.history_provider`：`Callable[[Sequence[str], int], list[Bar]] | None`（默认 `None`）。broker 插件可选实现：给各标的拉最近 count 根已收盘 bar，按 `(timestamp, symbol)` 升序返回，周期由行情源自身决定；建议逐标的拉取，单标的失败只跳过它并告警。
*   详见[自定义指标指南](../guide/custom_indicator.md)「实盘历史预热」一节。

### `akquant.run_live(timezone=..., sessions=...)`（实盘时区与交易时段） {: #live-timezone-sessions }

```python
run_live(
    ...,
    timezone="Asia/Shanghai",
    sessions={"rb": "CN_FUTURES_NIGHT_23"},
    trading_calendar=trading_days,  # 交易所交易日列表
    day_close_delay=60.0,
)
```

*   `timezone`：IANA 时区名，默认 `"Asia/Shanghai"`，传 `None` 等同默认值，非法时区名抛 `ValueError`。与 `run_backtest(timezone=)` 同义：引擎按它划分交易日、计算日终结算时刻与 Day 单过期、选默认时段表（非中国时区全天连续）；`preload_history` / replay 的 DataFrame 里不带时区的时间按它本地化；策略的 `self.timezone` 同步为该值。
*   **须与交易所时区一致**：接中国市场（如 `klinedata` 行情、`middleware` 交易）时保持默认，不要改成本机或服务器所在时区，否则交易日、日终结算与 Day 单过期会按该时区划分，与交易所实际不符。
*   `sessions` / `trading_calendar`：与 `run_backtest` 同义（模板名或自定义时段表），写法见上文 `BacktestConfig.china_futures` 一节的时段说明与[第 7 章](../textbook/07_futures.md)。
*   **按时段撮合只在 paper/replay**：模拟撮合下，时段为闭市、休市或集合竞价时订单排队，不按连续竞价成交。`broker_live` 下订单直接报到柜台、不按时段排队，时段表只决定 `ctx.session`、`is_market_open()`、交易日划分与日终结算、交易日钩子的时刻。期货不传 `sessions` 时按 `CN_FUTURES_DAY` 日盘时段，夜盘行情的时段判为闭市（paper 下订单排到日盘才撮合），有夜盘的品种必须传 `sessions=`（如 `{"rb": "CN_FUTURES_NIGHT_23"}`）。
*   **长期运行的实盘必须传 `trading_calendar=`**：不传时按工作日推算，节假日也会开日与结算（启动时告警一次）；行情源报告交易日（如 CTP 的 `TradingDay`）时，只能在所涉交易日开日前修正缺省日历。live_server 0.12.0 起，klinedata 会话自动从行情服务取沪深交易日历传入（行情服务以 jwt 模式部署时需要 `X-QMF-Session`，live_server 不带，取不到日历，退回工作日推算并告警）。
*   `day_close_delay`：`float`，默认 `60.0`（秒），取值 0～86400，越界或非有限值抛 `ValueError`，非数字抛 `TypeError`。实盘日终结算（Day 单过期、T+1 解冻、收盘快照）与 `on_after_trading` 在“会话收盘 + `day_close_delay`”触发，收盘快照仍记在收盘时刻；按交易日历登记，当日没有任何行情也会结算。回测不受影响（收盘时刻结算）。
*   **收盘宽限**：实盘（含 replay）在收盘后 `day_close_delay` 秒内、且早于下一交易日首个时段开始时到达的行情仍归当日（宽限不越过下一交易日开盘，因此 `ALWAYS_OPEN` 这类收盘即次日开盘的时段表实际没有宽限，除非显式日历有空档；`day_close_delay=0` 时也没有宽限）——`ctx.trading_day` 为当日，时段为 `Closed`（自定义表有盘后段时为 `PostClose`），不撮合、不产生权益点，也不提前开下一交易日；宽限期内下的 Day 单仍属下一交易日。结算时刻不变，结算价为收盘时刻的最新价（只认时间戳不晚于收盘的原始行情）。网关由 tick 合成 bar 时，收盘后 `day_close_delay − 5` 秒内的 tick 并入收盘那根 bar，该 bar 在这段钳制宽限结束后 2 秒补发（`day_close_delay` 不足 5 秒时不并入）。`Engine.trading_day_at` 是纯日历查询，不考虑宽限。
    *   已知限制：宽限期内存档后续跑，当日结算价按存档时的最新价；某品种收盘前没有 tick、只有宽限期 tick 时，结算价等于宽限价；结算之后才到的收盘 bar 仍派发给 `on_bar`；双流下同品种的宽限 tick 先于收盘 bar 推出，时间戳会倒退一次；多张时段表时宽限按会话级收盘（取最晚）判，早收盘品种在其自身收盘后、会话收盘前的行情不属宽限，照常计入当日与结算价，并报一次“闭市时段”告警。
*   `akquant.live.StreamingIndicatorSink(on_event, *, run_id="live")`：实盘用的流式指标录制器，每个指标点即时发给 `on_event`、不在内存累积。`run_live` 传了 `on_event` 而未传 `indicator_recorder` 时默认使用它；需要自行组合时可从 `akquant.live` 导入。

## 3. 核心引擎 (Core)

### `akquant.Engine`

回测引擎的主入口 (通常通过 `run_backtest` 隐式使用)。

**配置方法:**

*   `set_timezone_name(timezone: str)`: 设置 IANA 时区名称，例如 `Asia/Shanghai`、`UTC`、`US/Eastern`。推荐优先使用此方法，以正确处理 DST 和历史时区规则。
*   `set_timezone(offset: int)`: 设置固定时区偏移秒数。仅作为兼容接口保留，不包含 DST / 历史时区规则。
*   `use_simulated_execution()` / `use_realtime_execution()`: 设置执行环境。
*   `set_fill_mode(mode: ExecutionMode, timer_timing: str)`: 设置运行级默认执行模式。`mode` 取 `ExecutionMode` 枚举（`NextOpen`/`NextClose`/`NextAverage`/`NextHighLowMid`/`CurrentClose`），`timer_timing` 取 `"same_cycle"`/`"next_event"`（仅对 `CurrentClose` 有意义）。日常使用推荐通过 `run_backtest(..., fill_policy=NextOpen())` 传入 `FillMode` 对象，由框架翻译到该底层方法。
*   `get_fill_policy()`: 获取当前执行模式对应的核心三元组 `(price_basis, bar_offset, temporal)`（内部表示）。
*   `set_history_depth(depth)`: 设置历史数据缓存长度。
*   `configure_sessions(sessions=None, trading_calendar=None)`: 配置交易时段表与交易日历，写法同 `run_backtest(sessions=, trading_calendar=)`。低层 `Engine`（含 `use_simple_market()`）同样按时段表撮合，不再隐含全天可交易：不调用时中国时区按资产类型取默认时段表；不受时段约束请传 `sessions={"default": "ALWAYS_OPEN"}`。
*   `session_config`: 当前生效的 `(sessions, trading_calendar)`，随存档保存。
*   `trading_day_at(timestamp_ns) -> datetime.date`: 时刻所属交易日（收盘之后属下一交易日）。纯日历查询，不考虑实盘收盘宽限；只读，`run()` 前后都可调用；时刻超出 `trading_calendar` 覆盖范围时报 `ValueError`。
*   `current_trading_day` / `last_settled_trading_day`: 当前交易日 / 已完成日终结算的最后交易日，尚无时为 `None`。

**市场与费率配置:**

*   `use_simple_market()`: 启用简单市场（按比例佣金兼容入口）。
*   `use_simple_market_policy(type, value)`: 启用简单市场并显式设置佣金模式。
*   `use_china_market()`: 启用中国市场 (股票)。
*   `use_china_futures_market()`: 启用中国期货市场。
*   `set_stock_fee_rules(commission, stamp_tax, transfer_fee, min_commission)`: 设置股票费率。
*   `set_stock_fee_policy(type, value, stamp_tax, transfer_fee, min_commission)`: 设置股票佣金模式与费率。
*   `set_futures_fee_rules(commission_rate)`: 设置期货费率。
*   `set_futures_fee_rules_by_prefix(symbol_prefix, commission_rate)`: 设置期货品种前缀费率。
*   `set_futures_validation_options(enforce_tick_size, enforce_lot_size)`: 设置期货撮合前校验开关。
*   `set_futures_validation_options_by_prefix(symbol_prefix, enforce_tick_size, enforce_lot_size)`: 设置期货品种前缀校验开关。
*   `set_fund_fee_rules(...)`: 设置基金费率。
*   `set_option_fee_rules(commission_per_contract, exchange_fee_per_contract, clearing_fee_per_contract, exercise_fee_per_contract, sell_open_exempt)`: 设置期权费率（元/张）。
*   `set_options_fee_rules_by_prefix(symbol_prefix, commission_per_contract, exchange_fee_per_contract, clearing_fee_per_contract, exercise_fee_per_contract, sell_open_exempt)`: 设置期权品种前缀费率。
*   `set_instrument_fee_override(symbol, commission_rate=None, min_commission=None, stamp_tax_rate=None, transfer_fee_rate=None)`: 设置单个标的的费用覆盖，未传的项沿用市场配置。
*   `set_slippage(type, value)`: 设置滑点 (Fixed 或 Percent)。
*   `set_volume_limit(limit)`: 设置成交量限制 (如 0.1 表示不超过 Bar 成交量的 10%)。

命名约定说明：

*   期货费率接口统一使用复数命名 `set_futures_fee_rules*`。
*   旧单数命名 `set_future_fee_rules*` 已移除，不再对外暴露。

### `akquant.DataFeed`

`DataFeed` 是引擎内部的事件数据源封装，适合在你希望显式控制“数据如何进入引擎”时直接使用。

**构造与工厂方法:**

*   `DataFeed()`: 创建一个空的历史数据源。
*   `DataFeed.from_csv(path, symbol)`: 直接从 CSV 文件创建数据源；适合由 Rust 侧按行读取并驱动事件流。
*   `DataFeed.create_live()`: 创建实时数据源，适合供 gateway / 行情推送场景写入事件。

**写入方法:**

*   `add_bar(bar)`: 向数据源追加单个 `Bar`。
*   `add_bars(bars)`: 向数据源批量追加 `Bar` 列表。
*   `add_tick(tick)`: 向实时数据源追加单个 `Tick`。
*   `add_arrays(timestamps, opens, highs, lows, closes, volumes, symbol)`: 通过数组快速批量构建 `Bar` 并注入数据源；可选 `interval_starts=` 传区间起点列。
*   `add_trading_day_hint(timestamp_ns, trading_day, source)`: 推送行情源报告的交易日（仅实盘数据源，回测数据源上调用抛 `RuntimeError`），与行情同序进入引擎。传了 `trading_calendar=` 时不一致只告警；未传时在所涉交易日开日前按它修正缺省（工作日）日历。replay 回放可用 `gateway_options={"trading_day_hints": [(ts_ns, date), ...]}` 模拟。
*   `sort()`: 对当前历史数据源按事件时间排序。

**使用边界:**

*   若你只是做普通回测，优先使用 `run_backtest(data=...)`，直接传 `DataFrame`、`List[Bar]` 或 `DataFeedAdapter` 即可。
*   若你需要复用同一数据源对象、显式切换历史/实时模式，或直接接入 `Engine.add_data(feed)`，则使用 `DataFeed` 更合适。
*   `from_csv(...)` / `add_arrays(...)` 中如果遇到非法浮点值，Rust 侧会记录 warning，并通过 AKQuant 的 Python `logging` 体系输出，例如 `akquant.data.client`、`akquant.data.batch`。

### `akquant.SessionResolver` {: #akquant-session-resolver }

交易时段解析器，与引擎同一套规则：把时间点或 bar 区间解析为 `(交易日, 时段)`。适合在策略外做时段、交易日计算（如自建数据预处理）。

```python
SessionResolver(
    sessions: Optional[Mapping[str, Union[str, Sequence[Sequence[str]]]]] = None,
    trading_calendar: Optional[List[datetime.date]] = None,
    timezone: str = "Asia/Shanghai",
)
```

*   `sessions` / `trading_calendar`：同 `run_backtest`；未匹配前缀时按资产类型取默认（股票/基金 `CN_STOCK`、期货 `CN_FUTURES_DAY`、期权 `CN_OPTION`、其他 `ALWAYS_OPEN`）。
*   `resolve_instant(symbol, timestamp, asset_type=None) -> (date, TradingSession)`：解析时间点（tick）。
*   `resolve_bar(symbol, timestamp, interval_start=None, asset_type=None) -> (date, TradingSession)`：解析 bar 区间 `(interval_start, timestamp]`；`interval_start` 为 `None` 时按时间戳前 1ns。
*   `day_bounds(symbol, trading_day, asset_type=None) -> (start_ns, end_ns)`：交易日窗口（第一个时段开始、最后一个时段结束）。
*   `trading_blocks(symbol, trading_day, asset_type=None)`：交易块 `[(起点, 终点), ...]`（首尾相接的非休市、非闭市时段合并成的连续区间，如 A 股为 09:15～11:30 与 13:00～15:00）；`block_at(symbol, timestamp, asset_type=None)` 返回时刻所在交易块。
*   `set_active([(symbol, asset_type), ...])`：登记在用标的，会话级收盘取它们时段表收盘的最大值；`set_option_underlyings({期权: 标的})`：标的为已登记期货的期权跟随期货时段表。
*   `session_day_end(trading_day)`、`session_trading_day(timestamp)`、`is_session_day(day)`、`next_session_day(day)`：会话级收盘时刻、时刻所属交易日（夜盘属下一交易日）、是否交易日、下一交易日。
*   `templates()`（静态方法）：内置模板名；`uses_fallback_calendar`：是否在用缺省日历（未传 `trading_calendar`）；`uses_all_days_fallback`：未传日历且在用时段表全为每天交易的表（如 `ALWAYS_OPEN`），此时周末也算交易日。
*   时刻一律为 UTC 纳秒。

```python
import pandas as pd

import akquant as aq

resolver = aq.SessionResolver(sessions={"rb": "CN_FUTURES_NIGHT_23"})
ts = pd.Timestamp("2024-01-05 21:30", tz="Asia/Shanghai").value  # 周五夜盘
day, session = resolver.resolve_instant("rb2405", ts, aq.AssetType.Futures)
print(day, session)  # 2024-01-08 TradingSession.Continuous（属下周一）
```

### `akquant.WindowAggregator` {: #akquant-window-aggregator }

单周期窗口聚合器：把基础 bar 流聚合成 `freq` 周期的窗口 bar，规则与引擎 `subscribe_bars` 相同（日内窗口按交易日与交易块切分、标签不超过块结束；日线窗口按交易日切分、标签为交易日收盘）。适合在引擎之外复用同一套聚合口径。

```python
WindowAggregator(
    freq: str,
    session_windows: Optional[List[Tuple[str, str]]] = None,
    base_interval_min: Optional[int] = None,
    resolver: Optional[SessionResolver] = None,
    timezone: Optional[str] = None,
    asset_types: Optional[Dict[str, AssetType]] = None,
)
```

*   `freq`：窗口周期，`"Nmin"` / `"Nh"` / `"1d"`。
*   `session_windows`：手写本地交易时段，给定则覆盖时段表；一般不需要传。
*   `base_interval_min`：基础 bar 周期（分钟），给定时按 bar 区间即时闭合。
*   `resolver` / `timezone`：不传 `resolver` 时按 `timezone`（缺省 `"Asia/Shanghai"`）新建默认解析器；两者时区不一致时报错。
*   `asset_types`：品种 → 资产类型；也可用 `set_asset_type(symbol, asset_type)` 逐个设置。
*   `update(bar) -> List[Bar]`：喂入一根基础 bar，返回本次闭合的窗口 bar；`flush()` 闭合全部在形成窗口；`current(symbol)` 查看在形成窗口。
*   `update_arrays(...)`：批量喂入单品种基础 bar 列，返回 `(窗口 bar 列表, 行→窗口下标)`；`flush=True` 时会闭合本聚合器里所有品种的在形成窗口，与 `update()` 混用时请对本品种单独建聚合器。
*   bar 落在品种时段表全部交易块之外时（如期货误用股票时段表），每个品种告警一次。

### `akquant.gateway` 自定义 Broker 注册

可通过注册表机制按名称接入自定义 broker，而无需修改内置工厂分支。

**注册表 API:**

*   `register_broker(name, builder)`: 注册 broker 构建函数。
*   `unregister_broker(name)`: 取消注册 broker。
*   `get_broker_builder(name)`: 查询 broker 构建函数。
*   `list_registered_brokers()`: 获取当前已注册 broker 列表。

**Builder 签名:**

```python
def builder(
    feed: DataFeed,
    symbols: Sequence[str],
    use_aggregator: bool,
    **kwargs: Any,
) -> GatewayBundle:
    ...
```

**示例:**

```python
from akquant import DataFeed
from akquant.gateway import create_gateway_bundle, register_broker

register_broker("demo", demo_builder)
bundle = create_gateway_bundle(
    broker="demo",
    feed=DataFeed(),
    symbols=["000001.SZ"],
)
```

### 行情源与交易源分开指定 {: #mixed-market-trader-broker }

`GatewayBundle` 的 `market_gateway` 与 `trader_gateway` 是两个独立可选字段，因此
一个 broker 可以只提供其中一侧：`replay` 只有行情（`trader_gateway=None`，不能
下单），而某些券商/柜台插件只有交易通道（`market_gateway=None`，收不到行情）。

`create_gateway_bundle` 与 `run_live` 支持把两侧分开指定：

两种模式，二选一：

*   **单 broker**：只传 `broker`，由它同时提供行情与交易两侧（原语义，不变）。
*   **分开指定**：同时传 `market_broker` 与 `trader_broker`，各供一侧；此时
    `broker` **完全不参与构建**。

```python
run_live(
    strategy_cls=MyStrategy,
    instruments=instruments,
    market_broker="replay",   # 行情源
    trader_broker="demo",     # 交易源
    trading_mode="paper",
    gateway_options={"bars": bars},
)
```

**只传其中一个会报错**，要求把另一侧也写明。这是刻意的设计：如果让 `broker` 去
兼任缺失的那一侧，它就一词双义了——读 `broker='middleware', market_broker='replay'` 时，
你必须先知道「middleware 只有交易通道」才能推断出 `broker` 在这里指交易源，而参数名本身
没有表达这件事。两侧都写明则无需这层推断。

要点：

*   `gateway_options` 会**同时**传给两个 builder，两侧所需参数放在同一个 dict 里即可。
*   两侧同名时只构建一次（builder 可能连柜台、起线程，构建两次有副作用）。
*   `metadata` 会记录 `market_broker` / `trader_broker` 便于排障；行情侧声明的
    会话级信息（如 `replay` 的 `bounded_event_total`）不会因分开指定而丢失。
*   未注册的名字会报错并点名具体参数（`market_broker must be one of: ...`），
    而不是静默缺失某一侧通道。

## 4. 交易对象 (Trading Objects)

### `akquant.Order`

*   `id`: 订单 ID。
*   `symbol`: 标的代码。
*   `side`: `OrderSide.Buy` / `OrderSide.Sell`。
*   `order_type`: `OrderType.Market` / `OrderType.Limit` / `StopMarket` 等。
*   `status`: `OrderStatus.New` / `Filled` / `Cancelled` 等。
*   `quantity` / `filled_quantity`: 委托/成交数量。
*   `price`: 委托价格。
*   `average_filled_price`: 成交均价。
*   `trigger_price`: 触发价格。
*   `time_in_force`: 有效期 (`GTC`, `IOC`, `FOK`, `Day`)。
*   `created_at` / `updated_at`: 时间戳。
*   `tag`: 标签。
*   `reject_reason`: 拒绝原因。

### `akquant.Trade`

单次成交记录（一个订单可能对应多次成交）。

*   `id`: 成交 ID。
*   `order_id`: 对应订单 ID。
*   `symbol`: 标的代码。
*   `side`: 方向。
*   `quantity`: 成交数量。
*   `price`: 成交价格。
*   `commission`: 手续费。
*   `timestamp`: 成交时间。

### `akquant.ClosedTrade`

已平仓交易记录（开仓+平仓的完整周期）。

*   `entry_time` / `exit_time`: 开/平仓时间。
*   `entry_price` / `exit_price`: 开/平仓价格。
*   `quantity`: 数量。
*   `pnl`: 盈亏金额。
*   `return_pct`: 收益率。
*   `duration`: 持仓时间。
*   `mae` / `mfe`: 最大不利/有利变动。

## 5. 投资组合与风控 (Portfolio & Risk)

### `akquant.RiskConfig`

风控配置。

```python
@dataclass
class RiskConfig:
    active: bool = True
    check_cash: bool = True
    safety_margin: float = 0.0001
    max_order_size: Optional[float] = None
    max_order_value: Optional[float] = None
    max_position_size: Optional[float] = None
    restricted_list: Optional[List[str]] = None
    max_position_pct: Optional[float] = None
    sector_concentration: Optional[Union[float, tuple]] = None
    max_account_drawdown: Optional[float] = None
    max_daily_loss: Optional[float] = None
    stop_loss_threshold: Optional[float] = None
    account_mode: str = "cash"
    enable_short_sell: bool = False
    initial_margin_ratio: float = 1.0
    maintenance_margin_ratio: float = 0.3
    financing_rate_annual: float = 0.08
    borrow_rate_annual: float = 0.10
    allow_force_liquidation: bool = True
    liquidation_priority: str = "short_first"
```

账户级字段说明：

*   `max_account_drawdown`: 最大回撤阈值（0~1 小数）。以历史权益峰值为基准，当前权益回撤超过阈值后，新的下单请求会被拒绝。
*   `max_daily_loss`: 单日亏损阈值（0~1 小数）。以当日首次风控检查时的权益为基准，当日亏损超过阈值后，新的下单请求会被拒绝。
*   `stop_loss_threshold`: 账户净值止损阈值（0~1 小数）。当当前权益低于“规则首次生效时权益 × 阈值”后，新的下单请求会被拒绝。

资金校验字段说明：

*   `check_cash`: 是否启用资金/保证金校验，默认 `True`。它**同时**控制提交时与成交时两道闸门（股票、期货、期权一律适用）。设为 `False` 即允许账户透支（现金/保证金可以为负），以便回测跑到底；它只影响下单时的校验，**不会**关闭两融账户的日终强制平仓（那由 `allow_force_liquidation` 独立控制）。
*   校验的量是 `max(保证金占用变化量 + 费用, 0)`，其中变化量是**带符号**的：
    *   **开仓/加仓**变化量为正，需要现金预付。
    *   **减仓/平仓**会释放保证金，变化量为负，先抵掉自己的费用——因此**卖单不需要预先持有现金**，与 A 股"卖出费用从成交款净额扣除"的规则一致。满仓后只剩几十元现金时，全量卖单照常成交。
    *   仅当卖出所得连费用都盖不住时（如持 100 股 0.01 元的股票），该值仍为正并照常验资；放过它等于让现金透支，而透支只有 `check_cash=False` 才允许。
*   `safety_margin`: 提交时闸门的安全垫（默认 `0.0001`），按 `(1 - safety_margin)` 折减可用资金，用于吸收"提交时用预估价、成交时用真实价"的偏差。成交时闸门不留安全垫。
*   提交时闸门用**参考价**估算：限价单用限价，市价单用最新价。`fill_policy=NextOpen()` 时最新价就是上一根的收盘价，而真实成交价是下一根的开盘价，所以跳空高开的日子里，按开盘价算好的满仓市价买单可能在提交时就被拒（拒单原因 `Insufficient margin`，其中 `Required = 数量 × 参考价 + 佣金`，`Available = 可用资金 × (1 − safety_margin)`）。成交时闸门再按真实成交价校验一次，不留安全垫。规避办法：用限价单，限价按你能接受的最高价给；或 `buy(symbol, quantity=None)` 交给仓位管理器按可用资金计算数量，资金不足时引擎会自动缩到买得起的整手数（提交时与成交时都会缩；显式给 `quantity` 的买单不缩量）；或者按 `参考价 × (1 + 预留比例)` 自己计算数量。

这些拒单原因会体现在 `orders_df.reject_reason` 字段中。成交时闸门产生的拒单，原因里会带 `at execution` 标记。订单此前已部分成交（如开了 `volume_limit_pct` 跨多根 bar 成交）时，成交时被拒的只是剩余部分：状态为 `cancelled`，`filled_quantity` 与 `avg_price` 保留已成交部分，原因仍写在 `reject_reason`。

## 6. 结果分析 (Analysis)

### `akquant.BacktestResult`

回测结果对象。

**属性:**

*   `metrics_df`: 绩效指标表格 (Sharpe, Drawdown 等)。其中交易相关主字段包括 `closed_trade_count`、`execution_count`、`open_position_count`。
*   `trades_df`: 所有平仓交易记录表格。
*   `orders_df`: 所有委托记录表格。含 `position_effect`（开平语义）、`reduce_only`、`created_at_iso` / `updated_at_iso`（UTC ISO 串）。
*   `executions_df`: 所有成交流水表格（优先使用 Rust IPC/dict 快速导出）。含 `position_effect` 与 `timestamp_iso`。

!!! tip "开平语义（`position_effect`）"
    取值为 `auto` / `open` / `close` / `close_today` / `close_yesterday`，与下单
    入参**同一套词表**，可直接用于筛选（如 `df[df.position_effect == "close_today"]`）。

    `buy()` / `sell()` 在默认的 `position_effect="auto"` 下会自动拆开平腿：反手
    时先出 `close` 腿再出 `open` 腿。这两列就是查看拆腿结果的地方——委托表在
    下单时即可见，成交表在成交后可见。
*   `positions_df`: 每日持仓详情。
*   `equity_curve`: 权益曲线 (List[Tuple[timestamp, value]])。
*   `cash_curve`: 现金曲线 (List[Tuple[timestamp, value]])。
*   `margin_curve`: 保证金曲线 (List[Tuple[timestamp, value]])。
*   `equity_curve_daily`: 日频权益曲线（按日末值聚合）。
*   `cash_curve_daily`: 日频现金曲线（按日末值聚合）。
*   `margin_curve_daily`: 日频保证金曲线（按日末值聚合）。
*   `trades`: `ClosedTrade` 对象列表。
*   `executions`: `Trade` 对象列表 (所有成交流水)。
*   `snapshots`: 每日 `PositionSnapshot` 列表。

**分析方法:**

*   `exposure_df(freq="D")`: 组合暴露分解（净暴露、总暴露、杠杆）。
*   `attribution_df(by="symbol", use_net=True, top_n=None)`: 按 symbol/tag 做归因汇总。
*   `capacity_df(freq="D")`: 容量代理指标（订单数、成交率、换手）。
*   `benchmark_analysis(benchmark=None, curve_freq="raw")`: 返回结构化 benchmark analysis，可直接供前端/API 使用。
*   `export_benchmark_analysis(path, benchmark=None, format="json", curve_freq="raw")`: 将 benchmark analysis 导出为 JSON 或 parquet 产物。
*   `top_reject_reason_types(top_n=10)`: 按拒单类型聚合拒单统计，并附带一条示例明细。
*   `orders_by_strategy()`: 按 `owner_strategy_id` 聚合订单统计。
*   `executions_by_strategy()`: 按 `owner_strategy_id` 聚合成交流水统计。
*   `get_event_stats()`: 返回流式事件统计摘要（如 `processed_events`、`dropped_event_count`、`callback_error_count`、`backpressure_policy`、`stream_mode`）。
*   `report(..., curve_freq="D" | "raw")`: 生成 HTML 报告时，默认使用日频末值曲线，也可切回原始频率。

```python
orders_by_strategy = result.orders_by_strategy()
executions_by_strategy = result.executions_by_strategy()
benchmark_analysis = result.benchmark_analysis(
    benchmark=benchmark_returns,
    curve_freq="D",
)

# benchmark_analysis 常用字段:
# - schema_version, available, reason
# - benchmark.label
# - summary.total_excess / annual_excess / tracking_error (小数，需 ×100 展示为 %)
# - summary.information_ratio / beta (无量纲，不乘)
# - summary.alpha (小数，需 ×100 展示为 %)
# - series[*].date / strategy_return / benchmark_return / excess_return (小数)
# - series[*].strategy_cum_return / benchmark_cum_return / excess_cum_return (小数)

# 常用字段示例
# orders_by_strategy:
# - owner_strategy_id, order_count, filled_order_count,
#   ordered_quantity, filled_quantity, ordered_value, filled_value,
#   fill_rate_qty, fill_rate_value
#
# executions_by_strategy:
# - owner_strategy_id, execution_count, total_quantity,
#   total_notional, total_commission, avg_fill_price

event_stats = result.get_event_stats()
# 常见字段:
# - processed_events, dropped_event_count, callback_error_count,
#   backpressure_policy, stream_mode, reason
```

---

## 7. 数据输入与向量化 (Data I/O & Vectorized Compute)

### 7.1 `run_backtest` 的数据输入类型

`run_backtest(data=...)` 接受多种输入,内部统一归一化后进入引擎:

*   `pandas.DataFrame` / `Dict[str, pandas.DataFrame]`
*   `polars.DataFrame` / `polars.LazyFrame` / `pyarrow.Table`(一等输入,内部零成本转 pandas 路径)
*   `List[Bar]`
*   `DataFeed`(含流式 `DataFeed.from_parquet`,见 7.3)/ `DataFeedAdapter`

### 7.2 `akquant.write_canonical_parquet`

```python
def write_canonical_parquet(source, path, *, symbol=None) -> Path: ...
```

将任意来源(pandas / polars / pyarrow / 路径 / `List[Bar]`)规范化并写出**可流式(out-of-core)读取**的 Parquet:列 `timestamp`(int64 纳秒 UTC)+ `open/high/low/close/volume`(float64)+ `symbol`(str),按 `timestamp` 升序、zstd 压缩。产物可由 `DataFeed.from_parquet` 有界内存流式读取。

### 7.3 `akquant.DataFeed.from_parquet`

```python
@staticmethod
def from_parquet(path, symbol=None, chunk_size=None) -> DataFeed: ...
```

从规范 Parquet 创建**有界内存(out-of-core)流式**数据源:数据按 `chunk_size` 行(默认 65536)分块从磁盘读取,回测峰值内存与数据总量无关。要求 Parquet 按 `timestamp` 升序;含 `symbol` 列即支持多标的。详见「数据准备与加载指南 · 2.6」。

### 7.4 向量化列计算 `akquant.vec_*`

在 `numpy` 数组上零拷贝、向量化的批量列计算原语(与逐点增量指标互补,适合整列一次性求值):

| 函数 | 说明 |
| :--- | :--- |
| `vec_sma(values, period)` | 简单移动平均 |
| `vec_ema(values, period)` | 指数移动平均 |
| `vec_wma(values, period)` | 加权移动平均 |
| `vec_rolling_sum/min/max(values, period)` | 滚动求和 / 最小 / 最大 |
| `vec_rolling_std(values, period)` | 滚动样本标准差(ddof=1) |
| `vec_zscore(values, period)` | 滚动 z-score |
| `vec_returns(values)` / `vec_log_returns(values)` | 简单 / 对数收益率 |
| `vec_cumsum(values)` | 累积求和 |

语义与 pandas 对齐(NaN 位置一致、`rolling_std` 为样本标准差)。

```python
import numpy as np
import akquant as aq

close = np.array([10.0, 11.0, 12.0, 11.0, 13.0])
ma = aq.vec_sma(close, 3)     # 前 period-1 个为 NaN
z = aq.vec_zscore(close, 3)
```
