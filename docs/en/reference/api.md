# API Reference

This API documentation covers the core classes and methods of AKQuant.

Quick links:

*   [LiveRunner broker-live semantics](#live-broker-semantics)

## 1. High-Level API

### `akquant.run_backtest`

The most commonly used backtest entry function, encapsulating the initialization and configuration process of the engine.

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
    on_session_start: Optional[Callable[[Any, Any, int], None]] = None,
    on_session_end: Optional[Callable[[Any, Any, int], None]] = None,
    on_before_trading: Optional[Callable[[Any, Any, int], None]] = None,
    on_after_trading: Optional[Callable[[Any, Any, int], None]] = None,
    on_daily_rebalance: Optional[Callable[[Any, Any, int], None]] = None,
    on_daily_rebalance_after_bar: Optional[Callable[[Any, Any, int], None]] = None,
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
    strategy_fill_policy: Optional[Dict[str, FillPolicy]] = None,
    strategy_slippage: Optional[Dict[str, SlippageInput]] = None,
    strategy_commission: Optional[Dict[str, CommissionPolicy]] = None,
    portfolio_risk_budget: Optional[float] = None,
    risk_budget_mode: str = "order_notional",
    risk_budget_reset_daily: bool = False,
    analyzer_plugins: Optional[Sequence[AnalyzerPlugin]] = None,
    on_event: Optional[Callable[[BacktestStreamEvent], None]] = None,
    broker_profile: Optional[str] = None,
    fill_policy: Optional[FillPolicy] = None,
    strict_strategy_params: bool = True,
    **kwargs: Any,
) -> BacktestResult
```

**Key Parameters:**

*   `data`: Backtest data. Supports a single DataFrame, a `{symbol: DataFrame}` dictionary, `List[Bar]`, `DataFeed`, or any object implementing `DataFeedAdapter.load(request)`.
*   `strategy`: Strategy class or instance. Also supports passing an `on_bar` function (functional style).
*   `strategy_source` / `strategy_loader` / `strategy_loader_options`: Dynamic strategy loading entry points. When `strategy=None`, the framework can build the strategy from source, a path, or a custom loader.
*   `initialize` / `on_start` / `on_resume` / `on_stop`: Functional-strategy lifecycle callbacks for initialization, start, resume, and stop stages.
*   `on_tick` / `on_order` / `on_trade` / `on_reject` / `on_session_start` / `on_session_end` / `on_before_trading` / `on_after_trading` / `on_daily_rebalance` / `on_daily_rebalance_after_bar` / `on_portfolio_update` / `on_error` / `on_expiry` / `on_pre_open` / `on_timer` / `on_train_signal`: Functional event callbacks. `on_expiry(ctx, event)` fires only after the engine actually executes expiry settlement/removal.
*   `symbols`: Preferred parameter. Symbol or list of symbols.
*   `initial_cash`: Initial cash. If omitted, it falls back to `StrategyConfig.initial_cash`, whose default is `100000.0`.
*   `commission_policy`: Run-level default commission policy. Supported modes:
    *   `{"type": "percent", "value": 0.0003}`: commission as a percentage of turnover.
    *   `{"type": "fixed", "value": 3.0}`: a fixed amount charged on each fill.
    *   `{"type": "per_unit", "value": 0.01}`: charged linearly by filled quantity, i.e. `fill_quantity * 0.01`.
    *   When explicitly provided, it takes precedence over `commission_rate`; `commission_rate` remains as a backward-compatible shorthand for percent mode.
*   Legacy price-basis parameter: Removed.
*   Legacy timer-temporal parameter: Removed.
*   `fill_policy`: Unified fill semantics.
    *   `price_basis`: `open`, `close`, `ohlc4` (OHLC average), or `hl2` (high-low midpoint).
    *   `bar_offset`: `0` or `1`. Required for full three-axis semantics.
    *   Reserved (not implemented yet): `mid_quote`, `vwap_window`, `twap_window` (currently raises `NotImplementedError`).
    *   `temporal`: `same_cycle` or `next_event`.
*   `legacy_execution_policy_compat` (via `**kwargs`): Removed.
*   Migration hint: legacy execution parameters are no longer accepted; use `fill_policy`.
*   `strict_strategy_params`: Whether to strictly validate strategy constructor parameters (default `True`).
    *   Raises immediately if unsupported constructor parameters are provided.
    *   Recommended to keep enabled to avoid silent parameter mismatch and distorted backtest results.
*   `t_plus_one`: Enable T+1 trading rule (Default False). If enabled, it forces usage of China Market Model.
*   `slippage`: Global slippage (Default 0.0). E.g., 0.0001 means 1bp (0.01%) slippage, using percent model.
*   `volume_limit_pct`: Volume limit percentage (Default 0.25). Limits single trade to not exceed this percentage of the bar's total volume.
*   `warmup_period`: Strategy warmup period. Specifies the length of historical data (number of Bars) to preload for indicator calculation.
*   `start_time` / `end_time`: Backtest start/end time. Naive strings or `Timestamp` values are interpreted in the current `timezone` before being converted to UTC for filtering.
*   `catalog_path`: When `data` is omitted, load data from this directory using `ParquetDataCatalog` rules.
*   `config`: `BacktestConfig` object for centralized configuration.
*   `risk_config`: Risk configuration. Supports dict (e.g., `{"max_position_pct": 0.1}`) or `RiskConfig` object. Overrides fields in `config.strategy_config.risk` if both are provided.
*   `strategy_runtime_config` / `runtime_config_override`: Runtime behavior injection and conflict-resolution controls. Accepts `StrategyRuntimeConfig` or `dict`.
*   `strategy_id`: Primary strategy ownership id. Default `_default`.
*   `strategies_by_slot`: Optional multi-strategy mapping. Keys are slot ids and values are strategy class/instance/functional callback used by slot-iterative execution.
*   `strategy_max_order_size` / `strategy_max_order_value` / `strategy_max_position_size`: Optional strategy-level risk maps keyed by strategy id.
*   `strategy_max_daily_loss` / `strategy_max_drawdown`: Optional strategy-level stop maps keyed by strategy id.
*   `strategy_reduce_only_after_risk` / `strategy_risk_cooldown_bars`: Optional post-risk behavior maps keyed by strategy id.
*   `strategy_priority` / `strategy_risk_budget` / `portfolio_risk_budget`: Optional scheduling/budget controls.
*   `strategy_fill_policy`: Optional strategy-level default fill policy map keyed by strategy id.
    Resolution order at submit time: order-level `fill_policy` > `strategy_fill_policy[strategy_id]` > run-level `fill_policy`.
*   `strategy_slippage`: Optional strategy-level default slippage map keyed by strategy id.
    Resolution order at submit time: order-level `slippage` > `strategy_slippage[strategy_id]` > run-level `slippage`.
*   `strategy_commission`: Optional strategy-level default commission map keyed by strategy id.
    Resolution order at submit time: order-level `commission` > `strategy_commission[strategy_id]` > run-level commission model.
*   `commission` / `strategy_commission` use the same `CommissionPolicy` payload as run-level `commission_policy`:

```python
{"type": "percent" | "fixed" | "per_unit", "value": non_negative_number}
```

    *   `percent`: percentage of turnover.
    *   `fixed`: fixed amount per fill.
    *   `per_unit`: linear by filled quantity, suitable for per-share / per-lot / per-unit fee models.
*   Configuration layers (recommended mental model):
    1) order-level (`buy/sell/submit_order` args);
    2) strategy-map level (`strategy_*`, keyed by `strategy_id/slot`);
    3) run-level (`run_backtest` args);
    4) market defaults (built-in market-model defaults).
*   T+1 scope note: `t_plus_one` is currently a run/market-level switch, not a per-`strategy_id` layered setting.
*   `risk_budget_mode` / `risk_budget_reset_daily`: Risk budget accounting mode and reset policy.
*   `analyzer_plugins`: Optional analyzer plugin list. Plugins receive `on_start/on_bar/on_trade/on_finish` callbacks and final outputs are stored in `result.analyzer_outputs`.
*   `on_event`: Optional stream callback. When omitted, an internal no-op callback keeps legacy blocking return semantics; when provided, runtime events are emitted.
*   `broker_profile`: Optional broker template preset for quick defaults (fees/slippage/lot size). Built-ins: `cn_stock_miniqmt`, `cn_stock_t1_low_fee`, `cn_stock_sim_high_slippage`.

**Recommended fill_policy examples (primary path):**

```python
# Next bar close fill
result = aq.run_backtest(
    data=data,
    strategy=MyStrategy,
    symbols="000001",
    fill_policy={"price_basis": "close", "bar_offset": 1, "temporal": "same_cycle"},
)

# Current-close price with next-event temporal matching
result = aq.run_backtest(
    data=data,
    strategy=MyStrategy,
    symbols="000001",
    fill_policy={"price_basis": "close", "bar_offset": 0, "temporal": "next_event"},
)
```

**Execution semantics quick map (three-axis only):**

| Scenario | `fill_policy` |
| :--- | :--- |
| Next-open style fill | `{"price_basis":"open","bar_offset":1,"temporal":"same_cycle"}` |
| Current-close style fill | `{"price_basis":"close","bar_offset":0,"temporal":"same_cycle"}` |
| Next-bar close fill | `{"price_basis":"close","bar_offset":1,"temporal":"same_cycle"}` |
| Next-bar OHLC average fill | `{"price_basis":"ohlc4","bar_offset":1,"temporal":"same_cycle"}` |
| Next-bar HL2 fill | `{"price_basis":"hl2","bar_offset":1,"temporal":"same_cycle"}` |

Notes:
* For `open/ohlc4/hl2`, `bar_offset` is fixed to `1` (`0` is not supported).
* For `close + bar_offset=1` (next-bar close), the primary time-shift semantics come from `bar_offset`; keep `temporal="same_cycle"` to avoid confusion.
* `temporal` differences are mainly meaningful for `close + bar_offset=0` (current-close) scenarios.

### `akquant.run_grid_search`

Grid-search entry for batch backtesting and metric-based parameter ranking.

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

**Key parameter notes:**

*   `forward_worker_logs`: Whether to forward worker-process strategy logs to the main process during parallel optimization.
    *   `False`: throughput-first; worker logs may be invisible in main-process output.
    *   `True`: enables log aggregation for debugging.
*   `strict_strategy_params`: Passed via `**kwargs` into `run_backtest` (defaulted to `True` inside `run_grid_search`).
    *   Enforces strict match between `param_grid` keys and strategy constructor parameters.
    *   Fails fast on mismatch to avoid silent fallback.

### `akquant.run_walk_forward`

Walk-forward entry. Executes rolling "in-sample optimization + out-of-sample validation" and concatenates OOS equity curves.

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

**Key parameter notes:**

*   `**kwargs` are forwarded to both `run_grid_search` (in-sample optimization) and `run_backtest` (out-of-sample validation).
*   Therefore, `forward_worker_logs` controls worker-log forwarding during in-sample parallel optimization.
*   `strict_strategy_params` stays effective across optimization and validation phases (strict by default).

### `akquant.run_warm_start`

Resume a backtest from snapshot state and continue execution.

```python
def run_warm_start(
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
    strategy_fill_policy: Optional[Dict[str, FillPolicy]] = None,
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

`run_warm_start` uses the same strategy-slot, strategy-level risk, and strategy-level execution defaults as `run_backtest`.
For these fields, priority is:

1. explicit function arguments
2. `config.strategy_config`
3. restored/default values

**DataFeedAdapter Usage (Multi-Timeframe):**

```python
import akquant as aq

base = aq.CSVFeedAdapter(path_template="/data/{symbol}.csv")

feed_15m = base.resample(freq="15min", emit_partial=False)
feed_replay = base.replay(
    freq="1h",
    align="session",            # session | day | global
    day_mode="trading",         # effective only when align='day': trading | calendar
    emit_partial=False,
    session_windows=[("09:30", "11:30"), ("13:00", "15:00")],  # session only
)

result = aq.run_backtest(
    data=feed_replay,
    strategy=MyStrategy,
    symbols="000001",
    show_progress=False,
)
```

*   `align="session"`: Partition by trading day, optionally with `session_windows`.
*   `align="day"`: Partition by day without `session_windows`; `day_mode` supports `trading/calendar`.
*   `align="global"`: Aggregate on the full timeline without day partitioning.
*   Parameter recommendation: always use `symbols`. `run_backtest`/`run_warm_start` no longer accept `symbol`.
*   Migration status: `symbol` has been removed from `run_backtest`/`run_warm_start`; migrate all calls to `symbols`.

**Compatibility & Migration Notes:**

*   Prefer migrating realtime UI/logging/alerting to `run_backtest(..., on_event=...)`.
*   Stream use cases are unified under `run_backtest(..., on_event=...)`.
*   Legacy execution policy compatibility gate has been removed.
*   Legacy execution parameters and `legacy_execution_policy_compat` are no longer accepted.
*   Use `fill_policy` for all public execution configuration.
*   Since Phase 5, runtime rollback flags are removed; use release-level rollback when needed.

**Phase-5 Migration FAQ:**

*   Is `run_backtest` renamed? No, the public entry name stays unchanged.
*   Can `run_backtest` still be called without `on_event`? Yes, and result-return semantics stay the same.
*   How do we roll back in production? Use release-level rollback; `_engine_mode` runtime fallback is removed.
*   Can we still use `symbol`? No. Migrate to `symbols`.

### Stream Parameters & Events (`run_backtest`)

**Key Parameters:**

*   `on_event`: Optional stream callback receiving `BacktestStreamEvent`; if omitted, an internal no-op callback is used.
*   `stream_progress_interval`: Sampling interval for `progress` events (positive int).
*   `stream_equity_interval`: Sampling interval for `equity` events (positive int).
*   `stream_batch_size`: Flush threshold for buffered events (positive int).
*   `stream_max_buffer`: Maximum buffered events (positive int).
*   `stream_error_mode`: Callback exception handling policy.
    *   `"continue"`: Continue backtest on callback errors and report summary in
        final `finished` event.
    *   `"fail_fast"`: Stop immediately and raise once callback throws.
*   `stream_mode`: Stream mode.
    *   `"observability"`: observability-oriented mode with sampling and non-critical dropping under backpressure.
    *   `"audit"`: audit-oriented mode with sampling disabled and blocking backpressure for non-critical events.
*   `strategy_id` (forwarded via `**kwargs`): Tags trading events and results with strategy ownership. Default is `_default`.

**Event Schema (`BacktestStreamEvent`):**

*   `run_id`: Stream run id.
*   `seq`: Monotonic event sequence.
*   `ts`: Event timestamp in nanoseconds.
*   `event_type`: Event type.
*   `symbol`: Related symbol (nullable for some events).
*   `level`: Event level (e.g., `info`, `warn`, `error`).
*   `payload`: Event payload as string key-value map.

**Common `event_type` values:**

*   Lifecycle: `started`, `finished`
*   Sampling: `progress`, `equity`
*   Trading: `order`, `trade`, `risk`, `expiry`
*   Runtime exceptions: `error`
*   Market data: `tick`

**Common trading payload fields (`order`/`trade`/`risk`/`expiry`):**

*   `owner_strategy_id`: Strategy ownership id (default `_default`).
*   `order_id`: Order id (`order`/`trade`/`risk`).
*   `symbol`: Symbol (`order`/`risk`).
*   `status`: Order status (`order`).
*   `filled_qty`: Filled quantity (`order`).
*   `trade_id`: Trade id (`trade`).
*   `price`: Fill price (`trade`).
*   `quantity`: Fill quantity (`trade`).
*   `reason`: Risk rejection reason (`risk`).
*   `expiry_date`: Expiry date in `YYYYMMDD` form (`expiry`).
*   `quantity_before`: Position quantity before expiry settlement (`expiry`).
*   `quantity_closed`: Quantity closed by expiry settlement (`expiry`).
*   `cash_flow`: Cash flow generated by expiry settlement (`expiry`).
*   `settlement_type`: Expiry settlement mode such as `cash`, `settlement_price`, or `force_close` (`expiry`).
*   `settlement_price`: Effective settlement price when available (`expiry`).

**Common `finished.payload` fields:**

*   `status`: `completed` or `failed`
*   `processed_events`: Number of processed events
*   `total_trades`: Number of trades
*   `callback_error_count`: Total callback errors
*   `dropped_event_count`: Total number of events dropped under backpressure
*   `dropped_event_count_by_type`: Dropped count grouped by event type (`event=count` comma-separated)
*   `stream_mode`: Effective stream mode (`observability` or `audit`)
*   `sampling_enabled`: Whether sampling is enabled (`true`/`false`)
*   `backpressure_policy`: Backpressure policy (`drop_non_critical` or `block`)
*   `last_callback_error`: Latest callback error message (when present)
*   `reason`: Failure reason (when present)

### `akquant.BacktestConfig`

Data class for centralized backtest configuration.

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

Configuration at the strategy level, including capital, fees, and risk.

```python
@dataclass
class StrategyConfig:
    initial_cash: float = 100000.0
    commission_rate: float = 0.0
    commission_policy: Optional[Dict[str, Any]] = None
    stamp_tax_rate: float = 0.0
    transfer_fee_rate: float = 0.0
    min_commission: float = 0.0

    # Execution
    enable_fractional_shares: bool = False
    round_fill_price: bool = True
    slippage: Union[float, Dict[str, Any], None] = 0.0
    volume_limit_pct: float = 0.25
    exit_on_last_bar: bool = True
    indicator_mode: str = "precompute"

    # Position Sizing
    max_long_positions: Optional[int] = None
    max_short_positions: Optional[int] = None

    risk: Optional[RiskConfig] = None

    # Multi-strategy topology & strategy-level controls
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

A data class used to configure the properties of a single instrument.

```python
@dataclass
class InstrumentConfig:
    symbol: str
    asset_type: Union[
        Literal["STOCK", "FUTURES", "FUND", "OPTION"],
        InstrumentAssetTypeEnum
    ] = InstrumentAssetTypeEnum.STOCK
    multiplier: float = 1.0    # Contract multiplier
    margin_ratio: float = 1.0  # Margin ratio (0.1 means 10% margin)
    tick_size: float = 0.01    # Minimum price variation
    lot_size: Optional[int] = None

    # Costs & Execution (Asset Specific)
    commission_rate: Optional[float] = None
    min_commission: Optional[float] = None
    stamp_tax_rate: Optional[float] = None
    transfer_fee_rate: Optional[float] = None
    slippage: Optional[Union[float, Dict[str, Any]]] = None

    # Option specific
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

Common enums (available directly from top-level `akquant`):

- `InstrumentAssetTypeEnum`: `STOCK` / `FUTURES` / `FUND` / `OPTION`
- `InstrumentOptionMarginModelEnum`: `RATIO` / `CHINA_SINGLE_LEG` / `US_BROKER_SINGLE_LEG` / `US_BROKER_SINGLE_LEG_VOL_ADJUSTED`
- `InstrumentOptionTypeEnum`: `CALL` / `PUT`
- `InstrumentSettlementTypeEnum`: `CASH` / `SETTLEMENT_PRICE` / `FORCE_CLOSE`

Example:

```python
conf = akquant.InstrumentConfig(
    symbol="IF2506",
    asset_type=akquant.InstrumentAssetTypeEnum.FUTURES,
    settlement_type=akquant.InstrumentSettlementTypeEnum.CASH,
)
```

### `akquant.InstrumentSnapshot`

Static instrument metadata snapshot available to strategies (injected by engine; usually accessed via `Strategy.get_instrument*` APIs).

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

Notes:

*   `expiry_date` uses `int(YYYYMMDD)` semantics.
*   Snapshot data is available in `on_start`.
*   Prefer `get_instrument` / `get_instrument_config` / `get_instrument_field` in strategy code.

### Configuration System Explained

AKQuant provides a flexible configuration system that allows users to set backtest parameters in multiple ways.

#### 1. Hierarchy

Configuration objects are organized in a tree structure, with `BacktestConfig` as the top-level entry point:

```text
BacktestConfig (Simulation Scenario)
├── StrategyConfig (Strategy & Account)
│   ├── initial_cash
│   ├── commission_policy / commission_rate (Default commission)
│   ├── slippage (Default)
│   └── RiskConfig (Risk Rules)
│       ├── safety_margin
│       └── max_position_pct
└── InstrumentConfig (Asset Properties)
    ├── multiplier
    └── commission_rate (Asset-specific override)
```

#### 2. Priority

Parameter resolution in `run_backtest` follows this priority order (highest to lowest):

1.  **Explicit Arguments**:
    *   Parameters passed directly to `run_backtest` have the highest priority.
    *   Example: `run_backtest(start_time="2022-01-01")` overrides `config.start_time`.
2.  **Configuration Objects**:
    *   If explicit arguments are `None`, values are read from `config` (`BacktestConfig`).
    *   Multi-strategy fields can be centralized in `config.strategy_config`
        (`strategy_id`, `strategies_by_slot`, `strategy_max_*`, `strategy_priority`,
        `strategy_risk_budget`, `portfolio_risk_budget`).
3.  **Defaults**:
    *   If neither provides a value, system defaults are used.

#### 3. Risk Config Merging

The `risk_config` parameter has special handling logic designed to support a "Baseline + Override" pattern:

*   **Baseline**: First loads `config.strategy_config.risk` (if it exists).
*   **Override**: If `risk_config` parameter (dict or object) is provided, it overrides fields in the baseline configuration.
    *   This allows you to quickly adjust risk parameters for testing without modifying the main Config object, e.g., `run_backtest(..., risk_config={"max_position_pct": 0.5})`.

#### 4. Strategy Runtime Config Injection

`run_backtest` and `run_warm_start` support `strategy_runtime_config`:

*   Accepted formats: `StrategyRuntimeConfig` or `dict`.
*   Purpose: Inject runtime behavior switches without modifying strategy class code.
*   Example: `run_backtest(..., strategy_runtime_config={"error_mode": "continue"})`.
*   Validation: Unknown keys and invalid values fail fast with field-level errors.
*   Conflict handling: `runtime_config_override=True` applies external config; `False` keeps strategy-side config.
*   The same conflict rules apply consistently to both `run_backtest` and `run_warm_start`.
*   Conflict warnings are deduplicated per strategy instance for identical conflict payloads.
*   Priority rule: explicit `strategy_runtime_config` parameter has higher priority than forwarded config maps.
*   Troubleshooting quick lookup: see [Runtime Config Guide](../advanced/runtime_config.md).

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

#### 5. Best Practices

*   **Simple Scripts**: Use flat parameters of `run_backtest` directly (e.g., `initial_cash`, `start_time`).
*   **Production/Complex Strategies**: Build a complete `BacktestConfig` object for version control and reuse.
*   **UI-Driven Parameter Input**: Declare `PARAM_MODEL` in strategy classes (`akquant.ParamModel`) and use `get_strategy_param_schema` / `validate_strategy_params` for frontend-backend parameter consistency.
*   **Parameter Tuning**: When using `run_grid_search`, modify the Config object or pass override parameters as needed.

### Logging API

AKQuant stays quiet by default when imported as a library; until explicitly configured, the `akquant` root logger only carries a `NullHandler`.

#### `akquant.LogConfig`

Advanced logging config object used by `configure_logging(...)`.

Core fields:

*   `level`: global fallback level.
*   `console`: whether to enable the console handler.
*   `console_level` / `file_level`: per-handler level overrides.
*   `console_format` / `file_format`: text formatter overrides.
*   `console_show_context` / `file_show_context`: whether human-readable output should append structured context.
*   `console_json` / `file_json`: whether the corresponding handler should emit JSON lines.
*   `filename`: target log file path.
*   `file_mode`: file open mode, default `a`.
*   `file_max_bytes` / `file_backup_count`: size-based rotation threshold and retention count.
*   `profile`: preset profile, one of `research`, `optimize`, or `live`.
*   `reset_handlers`: whether to reset AKQuant-managed handlers.
*   `propagate`: whether records should propagate to parent loggers.

#### `akquant.configure_logging`

```python
def configure_logging(config: LogConfig) -> logging.Logger
```

Initializes or reconfigures the `akquant` logging system through a structured config.

Recommended example:

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

Behavior notes:

*   `profile` only fills unspecified fields; explicit config values always win.
*   `profile="optimize"` uses a process-aware default text format so worker output is easier to distinguish.
*   `profile="live"` is the natural place to enable structured context and/or JSON output.
*   Rust-side runtime warnings under `akquant.*` are also bridged into Python `logging` and restored into the same structured field model whenever possible.
*   For example, execution-path warnings such as insufficient-margin rejects, session-close expiry, unknown cancel requests, or same-slice `same-cycle` deferrals carry `phase="execution"` and may also include `symbol`, `order_id`, `strategy_id`, `slot`, and `event_time_iso`.

#### `akquant.register_logger`

```python
def register_logger(
    filename: Optional[str] = None,
    console: bool = True,
    level: str = "INFO",
) -> None
```

Compatibility helper for quickly enabling logging without exposing advanced fields. Internally this maps to `configure_logging(LogConfig(...))`.

#### `akquant.get_logger`

```python
def get_logger(name: Optional[str] = None) -> logging.Logger
```

Fetches a logger under the `akquant` namespace:

*   `get_logger()` -> `akquant`
*   `get_logger("strategy")` -> `akquant.strategy`
*   `get_logger("gateway.live")` -> `akquant.gateway.live`

#### `akquant.set_log_level`

```python
def set_log_level(level: Union[str, int]) -> None
```

Updates the current `akquant` root logger level.

#### Boundary Guidance

*   `self.log(...)` is the primary human-readable strategy logging path.
*   `run_backtest(..., on_event=...)` is the machine-consumable event stream and is better suited for realtime UI, alerting, and audit sinks.
*   Inside `on_order` / `on_trade` / `on_reject`, `self.log(...)` automatically carries structured fields such as `order_id` and `client_order_id`.
*   Rust execution/data warnings do not require manual user wiring; once an `akquant` logger handler is configured, they flow through the same text or JSON logging pipeline.

### `akquant.RiskConfig`

Configuration for risk management.

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

    # Account Level Risk
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

## 2. Strategy Development (Strategy)

### `akquant.Strategy`

Strategy base class. Users should inherit from this class and override callback methods.

**Callback Methods:**

*   `on_start()`: Triggered when the strategy starts. Used for subscription (`subscribe`) and indicator registration.
*   `on_bar(bar: Bar)`: Triggered when a Bar closes.
*   `on_tick(tick: Tick)`: Triggered when a Tick arrives.
*   `on_order(order)`: Triggered when order state changes.
*   `on_trade(trade)`: Triggered when trade report arrives.
*   `on_reject(order)`: Triggered once when an order becomes `Rejected`.
*   `on_expiry(event: Dict[str, Any])`: Triggered after an `expiry_date` driven settlement/removal is actually executed. Portfolio state is already updated when the callback runs. See `examples/49_on_expiry_demo.py` for a runnable example.
*   `on_session_start(session, timestamp)`: Triggered on session transition start.
*   `on_session_end(session, timestamp)`: Triggered on session transition end.
*   `on_before_trading(trading_date, timestamp)`: Triggered once when the regular trading session starts each local day; on the default backtest path this session is usually exposed as `Continuous`. This callback follows a "previous trading day / previous snapshot only" visibility model.
*   `on_pre_open(event: Dict[str, Any])`: Triggered once before the first regular event of each trading day. Use it for "pre-open decision, current open fill" workflows; default order semantics resolve to `price_basis=open, bar_offset=1, temporal=same_cycle`. See `examples/52_pre_open_demo.py`.
*   `on_daily_rebalance(trading_date, timestamp)`: Daily rebalance hook, triggered at most once per trading day and in the same phase as `on_before_trading`. It also exposes only previous-trading-day / previous-snapshot information.
*   `on_daily_rebalance_after_bar(trading_date, timestamp)`: Daily rebalance hook that runs after the first complete cross-symbol bar slice of the trading day. Unlike `on_daily_rebalance`, it can see the current day's bar history and current account snapshot, and is intended for same-cycle close-style rebalances.
*   `on_after_trading(trading_date, timestamp)`: Triggered when leaving the regular trading session, or replayed on next event after day rollover.
*   `on_portfolio_update(snapshot)`: Triggered when cash/equity/position snapshot changes.
*   `on_error(error, source, payload=None)`: Triggered when user callback raises, then exception is re-raised by default.
*   `on_timer(payload: str)`: Triggered by timer.
*   `on_stop()`: Triggered when the strategy stops.
*   `on_train_signal(context)`: Triggered by rolling training signal (ML mode).

Recommended `on_pre_open` pattern:

```python
def on_pre_open(self, event: Dict[str, Any]) -> None:
    signal = self.compute_pre_open_signal()
    if signal > 0:
        self.buy("000001", quantity=100)
```

Note: if you do not pass an explicit `fill_policy` here, the framework defaults to current-open order semantics.

**Properties & Shortcuts:**

*   `self.symbol`: The symbol currently being processed.
*   `self.close`, `self.open`, `self.high`, `self.low`, `self.volume`: Current Bar/Tick price and volume.
*   `self.position`: Position object for current symbol, with `size` and `available` properties.
*   `self.now`: Current backtest time (`pd.Timestamp`).
*   `self.runtime_config`: Runtime behavior config object (`StrategyRuntimeConfig`).
*   `self.enable_precise_day_boundary_hooks`: Enable boundary timer based precise day hooks (default `False`). This switch changes trigger precision only; it does not change the visibility window of `get_history()`, `get_account()`, or `equity` inside `on_before_trading` / `on_daily_rebalance`.
*   `self.portfolio_update_eps`: Snapshot threshold; changes below it skip `on_portfolio_update` (default `0.0`).
*   `self.error_mode`: Error handling mode, `"raise"` or `"continue"` (default `"raise"`).
*   `self.re_raise_on_error`: Whether to re-raise user callback exception after `on_error` (default `True`).

**Trading Methods:**

*   `buy(symbol, quantity, price=None, trigger_price=None, ...)`: Buy (open long / close short).
    *   Market order if `price` is not specified.
    *   Limit order if `price` is specified.
    *   Stop order (Stop Market) if `trigger_price` is specified.
*   `sell(symbol, quantity, price=None, trigger_price=None, ...)`: Sell (close long / open short). Same parameters as above.
*   `short(symbol, quantity, price=None, ...)`: Short sell.
*   `cover(symbol, quantity, price=None, ...)`: Buy to cover.
*   `submit_order(..., order_type="StopTrail", trail_offset=..., trail_reference_price=None)`: Submit a trailing stop order. `trail_offset` must be greater than 0.
*   `submit_order(..., order_type="StopTrailLimit", price=..., trail_offset=..., trail_reference_price=None)`: Submit a trailing stop-limit order. `price` and `trail_offset` are required.
*   `submit_order(..., broker_options={...})`: Optional broker extension fields passthrough (backtest currently records them on `order.broker_options` for debugging/audit).
*   `place_trailing_stop(symbol, quantity, trail_offset, side="Sell", trail_reference_price=None, ...) -> str`: Helper for trailing stop orders, promoted to market order when triggered.
*   `place_trailing_stop_limit(symbol, quantity, price, trail_offset, side="Sell", trail_reference_price=None, ...) -> str`: Helper for trailing stop-limit orders, promoted to limit order when triggered.
*   `order_target_value(target_value, symbol, price=None)`: Adjust position to target value.
*   `order_target_percent(target_percent, symbol, price=None)`: Adjust position to target account percentage.
*   `rebalance_weights(target_weights, price_map=None, liquidate_unmentioned=False, allow_leverage=False, rebalance_tolerance=0.0, ...)`: Rebalance a multi-asset portfolio by target weights.
    *   `target_weights` is `{symbol: weight}` and by default requires total weight `<= 1.0`.
    *   `liquidate_unmentioned=True` sets all existing non-mentioned positions to target `0`.
    *   Orders are submitted in sell-first then buy-second order to reduce cash-lock conflicts.
    *   `rebalance_tolerance` skips tiny drifts by portfolio-value ratio to reduce churn.
*   `close_position(symbol)`: Close position for a specific instrument.
*   `cancel_order(order_id: str)`: Cancel a specific order.
*   `cancel_all_orders(symbol)`: Cancel all pending orders for a specific instrument. If `symbol` is omitted, cancels all orders.
*   `place_oco(first_order_id, second_order_id, group_id=None) -> str`: Create an OCO order group. Once one order is filled, the peer order is canceled automatically.
*   `place_bracket(symbol, quantity, entry_price=None, stop_trigger_price=None, take_profit_price=None, ...) -> str`: Create a bracket order. The entry order is submitted first, then stop-loss/take-profit exits are submitted after entry fill; if both exits exist, they are linked as OCO automatically.

**Data & Utilities:**

*   `get_history(count, symbol, field="close") -> np.ndarray`: Get history data array (a safe snapshot copy of the rolling buffer, not zero-copy). Supports `open/high/low/close/volume` and any numeric extra fields (e.g., `adj_close`, `adj_factor`).
*   `get_history_multi(count, symbol, fields=("open","high","low","close","volume")) -> Dict[str, np.ndarray]`: Fetch multiple fields in a single FFI crossing; identical in behavior to per-field `get_history`, and used internally by `get_history_df`.
*   `get_history_df(count, symbol) -> pd.DataFrame`: Get history data DataFrame (OHLCV).
*   `get_position(symbol) -> float`: Get current position size. This still returns a numeric quantity, not an object.
*   `get_available_position(symbol) -> float`: Get available position size.
*   `positions -> Dict[str, float]`: Get all positions by symbol (read-only property).
*   `self.position.entry_price -> float`: Get the current symbol's average entry price via the `Position` helper.
*   `self.position.avg_price -> float`: Alias of `entry_price`.
*   `ctx.get_position_entry_price(symbol) -> float`: Get the current average entry price for one symbol.
*   `ctx.get_position_entry_prices() -> Dict[str, float]`: Get current average entry prices for all symbols.
*   `cash -> float`: Get current available cash (read-only property).
*   `get_account() -> Dict[str, float]`: Get an account snapshot. Common fields include `cash`, `equity`, `market_value`, `notional_value`, `frozen_cash`, `margin`, `used_margin`, `free_margin`, `unrealized_pnl`, `borrowed_cash`, `short_market_value`, `maintenance_ratio`, `account_mode`, `accrued_interest`, and `daily_interest`.
    *   In cash / spot-style accounts, `market_value` usually represents marked position value.
    *   In futures margin accounts, `equity` is account equity, `used_margin` is margin in use, `notional_value` is futures notional exposure, and `unrealized_pnl` is marked floating PnL. Futures trades do not deduct full notional from `cash` the way spot buys do, and notional exposure is not mirrored into `market_value` as if it were spot inventory.
    *   `cash` is the cash balance; `free_margin` (= `equity - used_margin`) is the amount actually available to open new positions and matches the `Available` value shown in the rejection log when an order is rejected. In futures margin accounts, opening a position does not deduct margin from `cash`, so `cash` is usually larger than `free_margin`; in stock cash accounts the two are equal.
    *   Inside strategy callbacks, prefer `equity` when you only need current total equity; it is aligned with `get_account()["equity"]`.
*   `get_order(order_id) -> Order`: Get details of a specific order.
*   `get_open_orders(symbol) -> List[Order]`: Get list of open orders.
*   `subscribe(instrument_id: str)`: Subscribe to market data. Must be called explicitly for multi-asset backtesting or live trading to receive `on_tick`/`on_bar` callbacks.
*   `log(msg: str, level: int)`: Log with timestamp.
*   `schedule(trigger_time, payload)`: Register a one-time timer task.
*   `add_daily_timer(time_str, payload)`: Register a daily timer task.

**Instrument Metadata APIs (Recommended):**

*   `get_instrument(symbol) -> InstrumentSnapshot`: Return static metadata snapshot for one symbol.
*   `get_instruments(symbols=None) -> Dict[str, InstrumentSnapshot]`: Return snapshot dict for multiple symbols; returns all when `symbols=None`.
*   `get_instrument_field(symbol, field) -> Any`: Return one metadata field value.
*   `get_instrument_config(symbol, fields=None) -> Union[Any, Dict[str, Any], InstrumentSnapshot]`: Compatibility API for full object, single field, or multi-field access.

Notes:

*   These APIs are available in `on_start` (snapshots are injected before start callbacks).
*   Prefer these APIs for static metadata access instead of relying on `bar.extra`.

**Machine Learning Support:**

*   `set_rolling_window(train_window, step)`: Set rolling training window.
*   `get_rolling_data(length, symbol)`: Get rolling training data (X, y).
*   `prepare_features(df, mode)`: (Override required) Feature engineering and label generation.

### `akquant.Bar`

Bar data object.

*   `timestamp`: Unix timestamp (nanoseconds).
*   `open`, `high`, `low`, `close`, `volume`: OHLCV data.
*   `symbol`: Instrument symbol.

### `akquant.live.LiveRunner` (broker live semantics) {: #live-broker-semantics }

For live broker routing, `LiveRunner` accepts broker-specific options through `gateway_options`.

```python
runner = LiveRunner(
    strategy_cls=on_bar,
    instruments=instruments,
    broker="ctp",
    trading_mode="broker_live",
    gateway_options={"execution_semantics_mode": "strict"},
)
```

`gateway_options.execution_semantics_mode`:

| Value | Default | Behavior | Recommended |
| :--- | :--- | :--- | :--- |
| `strict` | Yes | Terminal states (`Cancelled` / `Rejected` / `Filled`) are finalized by broker order callbacks (`OnRtnOrder`). Error callbacks cache reject reasons and merge them into subsequent order callbacks. | Production live trading |
| `compatible` | No | Allows immediate local terminal-state transitions for selected error/cancel paths to keep legacy behavior. | Migration / temporary compatibility |

Strict-mode notes:

*   Cancel request sent does not imply `Cancelled`; wait for `OnRtnOrder(Cancelled)`.
*   Error callback received does not imply `Rejected`; final status is confirmed by order callback.

## 3. Core Engine

### `akquant.Engine`

The main entry point for the backtesting engine (usually used implicitly via `run_backtest`).

**Configuration Methods:**

*   `set_timezone_name(timezone: str)`: Set an IANA timezone name such as `Asia/Shanghai`, `UTC`, or `US/Eastern`. This is the recommended API because it preserves DST and historical timezone rules.
*   `set_timezone(offset: int)`: Set a fixed timezone offset in seconds. Kept only as a compatibility fallback and does not preserve DST or historical timezone rules.
*   `use_simulated_execution()` / `use_realtime_execution()`: Set execution environment.
*   `set_fill_policy(price_basis, bar_offset, temporal)`: Set unified three-axis execution policy (recommended).
*   `get_fill_policy()`: Get current three-axis execution policy.
*   `set_history_depth(depth)`: Set history data cache length.

**Market & Fee Configuration:**

*   `use_simple_market()`: Enable simple market (legacy percent-commission shorthand).
*   `use_simple_market_policy(type, value)`: Enable simple market with an explicit commission mode.
*   `use_china_market()`: Enable China market.
*   `set_stock_fee_rules(commission, stamp_tax, transfer_fee, min_commission)`: Set fee rules.
*   `set_stock_fee_policy(type, value, stamp_tax, transfer_fee, min_commission)`: Set stock commission mode and fee rules.

### `akquant.DataFeed`

`DataFeed` is the engine-facing event source abstraction. Use it when you want explicit control over how data enters the engine.

**Constructors & factories:**

*   `DataFeed()`: Create an empty historical feed.
*   `DataFeed.from_csv(path, symbol)`: Create a feed directly from a CSV file; useful when you want Rust-side row iteration to drive the event stream.
*   `DataFeed.create_live()`: Create a live feed suitable for gateway / market data push scenarios.

**Input methods:**

*   `add_bar(bar)`: Append one `Bar` to the feed.
*   `add_bars(bars)`: Append a batch of `Bar` objects.
*   `add_tick(tick)`: Append one `Tick` to a live feed.
*   `add_arrays(timestamps, opens, highs, lows, closes, volumes, symbol)`: Build bars from arrays and inject them into the feed efficiently.
*   `sort()`: Sort the current historical feed by event time.

**When to use it:**

*   For normal backtests, prefer `run_backtest(data=...)` with a `DataFrame`, `List[Bar]`, or `DataFeedAdapter`.
*   Use `DataFeed` when you want to reuse the same feed object, switch explicitly between historical and live modes, or wire data into `Engine.add_data(feed)` yourself.
*   If `from_csv(...)` or `add_arrays(...)` encounters invalid floating-point values, Rust emits warnings that flow into AKQuant's Python `logging` pipeline, such as `akquant.data.client` and `akquant.data.batch`.

### `akquant.gateway` Custom Broker Registry

You can plug in a custom broker by name through the registry without editing built-in factory branches.

**Registry APIs:**

*   `register_broker(name, builder)`: Register a broker builder.
*   `unregister_broker(name)`: Unregister a broker.
*   `get_broker_builder(name)`: Resolve a broker builder.
*   `list_registered_brokers()`: List currently registered brokers.

**Builder signature:**

```python
def builder(
    feed: DataFeed,
    symbols: Sequence[str],
    use_aggregator: bool,
    **kwargs: Any,
) -> GatewayBundle:
    ...
```

**Example:**

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

## 4. Trading Objects

### `akquant.Order`

*   `id`: Order ID.
*   `symbol`: Instrument symbol.
*   `side`: `OrderSide.Buy` / `OrderSide.Sell`.
*   `order_type`: `OrderType.Market` / `OrderType.Limit` etc.
*   `status`: `OrderStatus.New` / `Filled` / `Cancelled` etc.
*   `quantity` / `filled_quantity`: Order / Filled quantity.
*   `average_filled_price`: Average filled price.

### `akquant.Instrument`

Contract definition.

```python
Instrument(
    symbol="AAPL",
    asset_type=AssetType.Stock,
    multiplier=1.0,
    margin_ratio=1.0,
    tick_size=0.01,
    option_type=None,
    strike_price=None,
    expiry_date=None,
    lot_size=1,
    underlying_symbol=None,
    settlement_type=None
)
```

## 5. Portfolio & Risk

### `akquant.RiskConfig`

Risk configuration.

```python
@dataclass
class RiskConfig:
    active: bool = True
    safety_margin: float = 0.0001
    max_order_size: Optional[float] = None
    max_order_value: Optional[float] = None
    max_position_size: Optional[float] = None
    restricted_list: Optional[List[str]] = None
    max_position_pct: Optional[float] = None
    sector_concentration: Optional[Union[float, tuple]] = None

    # Account Level Risk
    max_account_drawdown: Optional[float] = None
    max_daily_loss: Optional[float] = None
    stop_loss_threshold: Optional[float] = None
```

Account-level field semantics:

*   `max_account_drawdown`: Maximum drawdown limit in 0~1 ratio. Drawdown is measured against historical peak equity; once breached, new order requests are rejected.
*   `max_daily_loss`: Daily loss limit in 0~1 ratio. Loss is measured against equity at the first risk check of the trading day; once breached, new order requests are rejected.
*   `stop_loss_threshold`: Equity stop-loss threshold in 0~1 ratio. If current equity falls below `baseline_equity_at_rule_activation * threshold`, new order requests are rejected.

Rejection reasons are available in `orders_df.reject_reason`.

## 6. Analysis

### `akquant.BacktestResult`

Backtest result object.

**Properties:**

*   `metrics_df`: Performance metrics DataFrame. Core trade-state fields include `closed_trade_count`, `execution_count`, and `open_position_count`.
*   `trades_df`: Trade history DataFrame.
*   `orders_df`: Order history DataFrame.
*   `executions_df`: Execution fills DataFrame (prefers Rust IPC/dict fast export path).
*   `positions_df`: Daily position details.
*   `equity_curve`: Equity curve.
*   `cash_curve`: Cash curve.
*   `margin_curve`: Margin curve (used margin over time).
*   `equity_curve_daily`: Daily end-of-day equity curve.
*   `cash_curve_daily`: Daily end-of-day cash curve.
*   `margin_curve_daily`: Daily end-of-day margin curve.

**Analysis Methods:**

*   `exposure_df(freq="D")`: Portfolio exposure decomposition (net/gross/leverage).
*   `attribution_df(by="symbol", use_net=True, top_n=None)`: Grouped attribution by symbol/tag.
*   `capacity_df(freq="D")`: Capacity proxy metrics (order count, fill rates, turnover).
*   `benchmark_analysis(benchmark=None, curve_freq="raw")`: Return a structured benchmark analysis payload for frontends and APIs.
*   `export_benchmark_analysis(path, benchmark=None, format="json", curve_freq="raw")`: Persist benchmark analysis as JSON or parquet artifacts.
*   `top_reject_reason_types(top_n=10)`: Aggregate reject counts by normalized reject type and include a sample detail row.
*   `orders_by_strategy()`: Strategy-ownership order aggregation by `owner_strategy_id`.
*   `executions_by_strategy()`: Strategy-ownership execution aggregation by `owner_strategy_id`.
*   `get_event_stats()`: Unified stream summary stats (for example `processed_events`, `dropped_event_count`, `callback_error_count`, `backpressure_policy`, `stream_mode`).
*   `report(..., curve_freq="D" | "raw")`: Generate HTML report with daily end-of-day curves by default, or switch back to raw frequency.

```python
orders_by_strategy = result.orders_by_strategy()
executions_by_strategy = result.executions_by_strategy()
benchmark_analysis = result.benchmark_analysis(
    benchmark=benchmark_returns,
    curve_freq="D",
)

# Common benchmark_analysis fields:
# - schema_version, available, reason
# - benchmark.label
# - summary.total_excess / annual_excess / tracking_error
# - summary.information_ratio / beta / alpha
# - series[*].date / strategy_return / benchmark_return / excess_return
# - series[*].strategy_cum_return / benchmark_cum_return / excess_cum_return

# Common columns
# orders_by_strategy:
# - owner_strategy_id, order_count, filled_order_count,
#   ordered_quantity, filled_quantity, ordered_value, filled_value,
#   fill_rate_qty, fill_rate_value
#
# executions_by_strategy:
# - owner_strategy_id, execution_count, total_quantity,
#   total_notional, total_commission, avg_fill_price

event_stats = result.get_event_stats()
# Common fields:
# - processed_events, dropped_event_count, callback_error_count,
#   backpressure_policy, stream_mode, reason
```
