# API 参考

本 API 文档涵盖了 AKQuant 的核心类和方法。

快速跳转：

*   [run_live broker_live 执行语义](#live-broker-semantics)

## 1. 高级入口 (High-Level API)

### `akquant.run_backtest`

最常用的回测入口函数，封装了引擎的初始化和配置过程。

```python
def run_backtest(
    data: Optional[BacktestDataInput] = None,
    strategy: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None] = None,
    strategy_source: Optional[Union[str, bytes, os.PathLike[str]]] = None,
    strategy_loader: Optional[str] = None,
    strategy_loader_options: Optional[Dict[str, Any]] = None,
    symbols: Union[str, List[str], Tuple[str, ...], set[str]] = "BENCHMARK",
    initial_cash: Optional[float] = None,
    commission_policy: Optional[CommissionPolicy] = None,
    commission_rate: Optional[float] = None,
    stamp_tax_rate: Optional[float] = None,
    transfer_fee_rate: Optional[float] = None,
    min_commission: Optional[float] = None,
    slippage: SlippageInput = None,
    volume_limit_pct: Optional[float] = None,
    timezone: Optional[str] = None,
    t_plus_one: bool = False,
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
    symbols: Union[str, List[str], Tuple[str, ...], set[str]] = "BENCHMARK",
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
    config: Optional[BacktestConfig] = None,
    **kwargs: Any,
) -> BacktestResult
```

`run_from_checkpoint` 使用与 `run_backtest` 相同的策略 slot、策略级风控与成交默认项；
对这些字段，优先级为：显式函数参数 > `config.strategy_config` > checkpoint 恢复值 / 默认值。

**通用行为说明（主要对应 `run_backtest`，`run_from_checkpoint` 共享其中的成交/风控/策略映射规则）:**

*   `data`: 回测数据。支持单个 DataFrame，`{symbol: DataFrame}` 字典，`List[Bar]`，`DataFeed`，或实现 `DataFeedAdapter.load(request)` 的对象。
*   `strategy`: 策略类、策略实例，或 `on_bar` 函数（函数式编程风格）。
*   `strategy_source` / `strategy_loader` / `strategy_loader_options`: 动态策略加载入口。`strategy=None` 时可直接从源码、路径或自定义加载器构造策略。
*   `initialize` / `on_start` / `on_resume` / `on_stop`: 函数式策略生命周期回调；其中 `on_resume(ctx)` 仅在 checkpoint 恢复后的热启动阶段触发，且先于 `on_start(ctx)`。
*   `on_tick` / `on_order` / `on_trade` / `on_reject` / `on_before_trading` / `on_after_trading` / `on_cross_section` / `on_portfolio_update` / `on_error` / `on_expiry` / `on_pre_open` / `on_timer` / `on_train_signal`: 函数式策略事件回调；其中 `on_expiry(ctx, event)` 在引擎实际执行到期结算后触发，`on_pre_open(ctx, event)` 在每个交易日首个常规行情事件前触发，适合“盘前决策，本次 open 成交”；`on_error(ctx, error, source, payload)` 会在其他用户回调抛出异常时触发；`on_train_signal(ctx)` 仅在 ML 滚动训练窗口触发。
*   `symbols`: 标的代码或代码列表。
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
*   `t_plus_one`: 是否启用 T+1 交易规则 (默认 False)。如果启用，将强制使用中国市场模型。
*   `slippage`: 全局滑点 (默认 0.0)。例如 0.0001 代表 1bp (0.01%) 的滑点，采用百分比模型。
*   `volume_limit_pct`: 成交量限制比例 (默认 0.25)。限制单笔成交不超过该 Bar 总成交量的百分比。
*   `warmup_period`: 策略预热期。指定需要预加载的历史数据长度（Bar 数量），用于计算指标。
*   `start_time` / `end_time`: 回测开始/结束时间。若传入 naive 时间字符串或 `Timestamp`，会按当前 `timezone` 解释，再转换为 UTC 参与过滤。
*   `catalog_path`: 当 `data` 未显式传入时，可从该目录按 `ParquetDataCatalog` 规则加载数据。
*   `config`: `BacktestConfig` 配置对象，用于集中管理配置。
*   `lot_size`: 最小交易单位。如果是 `int`，应用于所有标的；如果是字典，按标的匹配。
*   `custom_matchers`: 自定义撮合器字典。
*   `risk_config`: 风控配置。支持字典 (e.g., `{"max_position_pct": 0.1}`) 或 `RiskConfig` 对象。如果同时提供了 `config.strategy_config.risk`，此参数将覆盖其中的同名字段。
*   `strategy_runtime_config` / `runtime_config_override`: 运行时行为注入与冲突处理开关，支持 `StrategyRuntimeConfig` 或 `dict`。
*   `strategies_by_slot`: 可选多策略映射。键为 slot id，值为策略类/实例/函数式 on_bar 回调；用于启用 slot 迭代执行。
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
*   `broker_profile`: 可选 broker 参数模板，用于快速注入费率/滑点/最小手数等默认值。内置模板：`cn_stock_miniqmt`、`cn_stock_t1_low_fee`、`cn_stock_sim_high_slippage`。

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
    slippage: Union[float, Dict[str, Any], None] = 0.0
    volume_limit_pct: float = 0.25
    max_long_positions: Optional[int] = None
    max_short_positions: Optional[int] = None
    exit_on_last_bar: bool = True
    indicator_mode: str = "precompute"
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
        InstrumentAssetTypeEnum
    ] = InstrumentAssetTypeEnum.STOCK
    multiplier: float = 1.0    # 合约乘数
    margin_ratio: float = 1.0  # 保证金率 (0.1 表示 10% 保证金)
    tick_size: float = 0.01    # 最小变动价位
    lot_size: Optional[int] = None

    # 费率与执行 (资产专用)
    commission_rate: Optional[float] = None
    min_commission: Optional[float] = None
    stamp_tax_rate: Optional[float] = None
    transfer_fee_rate: Optional[float] = None
    slippage: Optional[Union[float, Dict[str, Any]]] = None

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
    settlement_price: Optional[float] = None
    static_attrs: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
```

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
- `enforce_sessions`: 是否严格按交易时段控制成交
- `session_profile`: 中国期货会话模板（`CN_FUTURES_DAY`=`CN_FUTURES_COMMODITY_DAY` / `CN_FUTURES_CFFEX_STOCK_INDEX_DAY` / `CN_FUTURES_CFFEX_BOND_DAY` / `CN_FUTURES_NIGHT_23` / `CN_FUTURES_NIGHT_01` / `CN_FUTURES_NIGHT_0230`）

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
| 交易时段 | `china_futures.sessions` 显式配置 | `session_profile` 模板 | ChinaMarket 默认会话 |
| 市场路由 | `use_china_futures_market=False` 或混合资产回落 | `use_china_futures_market=True` 且纯期货 | `use_simple_market` |

口径说明：

*   同级规则冲突时，以显式规则覆盖模板规则。
*   撮合校验路径按更具体前缀优先（更长匹配优先）。

中国期权扩展配置位于 `BacktestConfig.china_options`，用于管理中国期权费率：

- `fee_per_contract`: 全局每张合约手续费
- `fee_by_symbol_prefix`: 按品种前缀覆盖每张合约手续费
- `use_china_market`: 是否切换到 ChinaMarket
- `sessions`: 可选时段覆盖（不与期货会话配置冲突时生效）

中国期权扩展推荐使用以下优先级口径：

| 配置项 | 高优先级 | 中优先级 | 默认值 |
|---|---|---|---|
| 期权费率（按张） | `fee_by_symbol_prefix` | `fee_per_contract` | `set_option_fee_rules` 默认配置 |
| 市场路由 | `use_china_market=True` | 混合资产时自动 ChinaMarket | `use_simple_market` |

期货 vs 期权配置能力对照：

| 能力维度 | 中国期货（`china_futures`） | 中国期权（`china_options`） |
|---|---|---|
| 路由开关 | `use_china_futures_market` | `use_china_market` |
| 全局费率 | `StrategyConfig.commission_policy` / `StrategyConfig.commission_rate` 或模板费率 | `fee_per_contract` |
| 前缀费率覆盖 | `fee_by_symbol_prefix` | `fee_by_symbol_prefix` |
| 合约参数模板 | 支持（乘数/保证金/tick/手数） | 不支持 |
| 撮合校验开关 | 支持（tick/手数，含前缀覆盖） | 不支持 |
| 会话覆盖 | 支持（`sessions`） | 支持（`sessions`） |
| 前缀匹配策略 | 更长前缀优先 | 更长前缀优先 |

股票配置推荐使用以下优先级口径：

| 配置项 | 高优先级 | 中优先级 | 默认值 |
|---|---|---|---|
| 股票费率（佣金/印花税/过户费/最低佣金） | `InstrumentConfig` 单标的费率字段 | `StrategyConfig` 全局费率字段（含 `commission_policy` / `commission_rate`） | `run_backtest` 内置默认值 |
| 交易单位（`lot_size`） | `InstrumentConfig.lot_size`（显式设置） | `run_backtest(lot_size=...)` 全局设置 | `1` |
| 市场制度（T+1） | `run_backtest(t_plus_one=...)` 显式参数 | `Engine.set_t_plus_one(...)` 引擎设置 | `False` |
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
*   `on_reject(order: Order)`: 订单首次进入 `Rejected` 时触发一次。
*   `on_expiry(event: Dict[str, Any])`: 到期结算回调。仅当引擎实际执行 `expiry_date` 驱动的到期结算/移除后触发；回调时账户状态已更新。示例见：`examples/49_on_expiry_demo.py`。
*   `on_before_trading(trading_date, timestamp)`: 每个本地交易日首次进入常规交易会话时触发一次；默认回测路径下该会话通常表现为 `Continuous`。该回调按“前一交易日/前一时点信息可见”的语义工作。
*   `on_pre_open(event: Dict[str, Any])`: 每个交易日首个常规行情事件前触发一次。适合“盘前决策，本次 open 成交”；默认下单语义等价于 `NextOpen()`（下一根 open 成交）。示例见：`examples/52_pre_open_demo.py`。
*   `on_cross_section(trading_date, timestamp)`: 横截面同周期调仓钩子。在框架看到当日首个“跨标的完整 bar 切片”后触发，每个交易日最多一次；与 `on_before_trading` 不同，它可以看到当日历史和当前账户快照，适合收盘价同周期调仓。调仓频率（日/周/月）在回调内用日历判断。
*   `on_after_trading(trading_date, timestamp)`: 离开常规交易会话时触发；若先跨日则在下一事件补发。
*   `on_portfolio_update(snapshot)`: 账户快照变化时触发。
*   `on_error(error, source, payload=None)`: 用户回调抛异常时触发，默认触发后继续抛出。
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
*   `self.enable_precise_day_boundary_hooks`: 是否启用边界定时器精确交易日钩子（默认 `False`）。该开关只影响日边界 hooks 的触发精度，不改变 `on_before_trading` 中 `get_history()`、`get_account()`、`equity` 等接口的可见数据窗口。
*   `self.portfolio_update_eps`: 账户快照更新阈值，低于该变化量不触发 `on_portfolio_update`（默认 `0.0`）。
*   `self.error_mode`: 错误处理模式，`"raise"` 或 `"continue"`（默认 `"raise"`）。
*   `self.re_raise_on_error`: 用户回调异常后是否继续抛出（默认 `True`）。
*   `self.ctx`: 策略上下文 (`StrategyContext`)，提供底层 API 访问。

**交易方法:**

*   `buy(symbol=None, quantity=None, price=None, trigger_price=None, ...)`: 买入（开多/平空）。
    *   如果不指定 `price`，则为市价单。
    *   如果指定 `price`，则为限价单。
    *   如果指定 `trigger_price`，则为止损/止盈单 (Stop Market)。
    *   不传 `symbol` 时取当前 bar/tick 的标的；在无行情上下文的回调（如 `on_start`）中必须显式传入。
    *   不传 `quantity` 时按 `self.sizer` 计算下单量（默认 `FixedSize(100)`，可用 `set_sizer()` 替换）。
*   `sell(symbol=None, quantity=None, price=None, trigger_price=None, ...)`: 卖出（平多/开空）。参数同上，但**不传 `quantity` 时不走 sizer，而是全平当前持仓**：回测取总持仓，`broker_live` 取可用持仓（A 股 T+1 下当日买入部分不可卖，按总量报单会被柜台整单拒绝）。
*   解析后下单量 `<= 0` 时不报单，返回空回执（`len(receipt) == 0`、`receipt.primary == ""`）。
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

*   `get_history(count, symbol, field="close") -> np.ndarray`: 获取历史数据数组（返回滚动缓冲的安全快照拷贝，非零拷贝）。
*   `get_history_multi(count, symbol, fields=("open","high","low","close","volume")) -> Dict[str, np.ndarray]`: 单次跨界批量取回多字段，语义等价于逐字段 `get_history`，`get_history_df` 内部即基于它。
*   `get_history_map(count, symbols, field="close") -> Dict[str, np.ndarray]`: 批量获取多个标的历史数据。
*   `rebalance_to_topn(scores, top_n, weight_mode="equal", ...) -> List[str]`: 根据打分选取 TopN 并执行调仓，支持等权或按分数归一化。
*   `get_history_df(count, symbol) -> pd.DataFrame`: 获取历史数据 DataFrame (OHLCV)。
*   `get_position(symbol) -> float`: 获取当前持仓量。返回值仍为数量，不返回对象。
*   `get_available_position(symbol) -> float`: 获取可用持仓量。
*   `positions -> Dict[str, float]`: 获取所有标的持仓（只读属性）。
*   `self.position.entry_price -> float`: 通过 `Position` helper 获取当前标的持仓均价。
*   `self.position.avg_price -> float`: `entry_price` 的别名。
*   `ctx.get_position_entry_price(symbol) -> float`: 获取指定标的当前持仓均价。
*   `ctx.get_position_entry_prices() -> Dict[str, float]`: 获取所有标的当前持仓均价。
*   `get_holding_bars(symbol) -> int`: 获取当前持仓持有的 Bar 数量。
*   `cash -> float`: 获取当前可用资金（只读属性）。
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
*   `schedule_daily(time_str, payload)`: 注册每日定时任务(每个交易日触发)。
*   `schedule_weekly(time_str, payload)`: 每周首个交易日触发(节假日/停牌自动顺延)。
*   `schedule_monthly(time_str, payload)`: 每月首个交易日触发(节假日/停牌自动顺延)。
*   `trading_days -> List[pd.Timestamp]`: 只读交易日序列，配合 `schedule` 自定义节奏。
*   `nth_trading_day_of_month(n)` / `nth_last_trading_day_of_month(n)` / `nth_trading_day_of_week(n)`: 交易日历辅助，返回每月/周第 n 个(或倒数第 n 个)交易日。
*   `to_local_time(timestamp) -> pd.Timestamp`: 将 UTC 时间戳转换为本地时间。
*   `format_time(timestamp, fmt) -> str`: 格式化时间戳。

**标的静态属性 API（推荐）:**

*   `get_instrument(symbol) -> InstrumentSnapshot`: 获取单个标的静态属性快照。
*   `get_instruments(symbols=None) -> Dict[str, InstrumentSnapshot]`: 获取多个标的静态属性快照字典；`symbols=None` 时返回全部。
*   `get_instrument_field(symbol, field) -> Any`: 获取单个标的字段值。
*   `get_instrument_config(symbol, fields=None) -> Union[Any, Dict[str, Any], InstrumentSnapshot]`: 兼容接口；支持整对象、单字段或多字段读取。

说明：

*   这些接口在 `on_start` 即可使用（启动阶段已注入快照）。
*   推荐优先使用这些接口读取静态属性，而不是依赖 `bar.extra`。

**机器学习支持:**

*   `set_rolling_window(train_window, step)`: 设置滚动训练窗口。
*   `get_rolling_data(length, symbol)`: 获取滚动训练数据 (X, y)。
*   `prepare_features(df, mode)`: (需重写) 特征工程与标签生成。

### `akquant.Bar`

K 线数据对象。

*   `timestamp`: Unix 时间戳 (纳秒)。
*   `open`, `high`, `low`, `close`, `volume`: OHLCV 数据。
*   `symbol`: 标的代码。
*   `extra`: 扩展数据字典 (`Dict[str, float]`)。
*   `timestamp_iso`: UTC ISO 8601 时间字符串。

### `akquant.Tick`

Tick 数据对象。

*   `timestamp`: Unix 时间戳 (纳秒)。
*   `price`: 最新价。
*   `volume`: 成交量。
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
*   `set_option_fee_rules(...)`: 设置期权费率。
*   `set_slippage(type, value)`: 设置滑点 (Fixed 或 Percent)。
*   `set_volume_limit(limit)`: 设置成交量限制 (如 0.1 表示不超过 Bar 成交量的 10%)。
*   `set_market_sessions(sessions)`: 设置交易时段。

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
*   `add_arrays(timestamps, opens, highs, lows, closes, volumes, symbol)`: 通过数组快速批量构建 `Bar` 并注入数据源。
*   `sort()`: 对当前历史数据源按事件时间排序。

**使用边界:**

*   若你只是做普通回测，优先使用 `run_backtest(data=...)`，直接传 `DataFrame`、`List[Bar]` 或 `DataFeedAdapter` 即可。
*   若你需要复用同一数据源对象、显式切换历史/实时模式，或直接接入 `Engine.add_data(feed)`，则使用 `DataFeed` 更合适。
*   `from_csv(...)` / `add_arrays(...)` 中如果遇到非法浮点值，Rust 侧会记录 warning，并通过 AKQuant 的 Python `logging` 体系输出，例如 `akquant.data.client`、`akquant.data.batch`。

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
兼任缺失的那一侧，它就一词双义了——读 `broker='qmf', market_broker='replay'` 时，
你必须先知道「qmf 只有交易通道」才能推断出 `broker` 在这里指交易源，而参数名本身
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

这些拒单原因会体现在 `orders_df.reject_reason` 字段中。

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
# - summary.total_excess / annual_excess / tracking_error
# - summary.information_ratio / beta / alpha
# - series[*].date / strategy_return / benchmark_return / excess_return
# - series[*].strategy_cum_return / benchmark_cum_return / excess_cum_return

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
