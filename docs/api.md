# API 参考

`akquant` 模块暴露了以下核心类用于构建回测系统。

## 核心组件

### `akquant.Engine`

回测引擎的主入口。负责管理数据、资金、持仓，并驱动策略运行。市场规则与交易时段可由用户配置。

**方法:**

*   `__init__()`: 创建一个新的引擎实例。
*   `add_data(feed: DataFeed)`: 加载数据源。
*   `add_bars(bars: List[Bar])`: 批量加载 K 线数据 (推荐使用)。
*   `add_instrument(instrument: Instrument)`: 添加合约信息。
    *   `instrument.symbol`: 合约代码
    *   `instrument.asset_type`: 资产类型 (Stock, Fund, Futures, Option)
    *   `instrument.multiplier`: 合约乘数
    *   `instrument.margin_ratio`: 保证金比率
    *   `instrument.lot_size`: 最小交易单位 (股票默认为 100)
    *   `instrument.option_type`: 期权类型 (Call, Put, 可选)
    *   `instrument.strike_price`: 行权价 (可选)
    *   `instrument.expiry_date`: 到期日 (可选)
*   `run(strategy: object) -> BacktestResult`: 运行 K 线回测。`strategy` 对象必须实现 `on_bar(bar, ctx)` 方法。
*   `set_execution_mode(mode: ExecutionMode)`: 设置执行模式 (`CurrentClose` 或 `NextOpen`)。
*   `set_timezone(offset_secs: int)`: 设置时区偏移秒数 (例如 28800 为 UTC+8)。
*   `set_timezone_name(tz_name: str)`: 设置时区名称 (如 "Asia/Shanghai")，用于日志和时间处理（Python 封装层方法）。
*   `use_simple_market(commission_rate: float)`: 启用简单市场模式 (7x24小时, T+0, 无税, 简单佣金)。
*   `use_china_market()`: 启用中国市场模式 (支持 T+1/T+0, 印花税, 过户费, 交易时段等)。
*   `use_china_futures_market()`: 启用中国期货市场默认配置 (T+0, 需手动设置交易时段)。
*   `set_t_plus_one(enabled: bool)`: 设置股票 T+1/T+0 规则 (仅针对 ChinaMarket)。
*   `set_force_session_continuous(enabled: bool)`: 强制使用连续交易时段。
*   `set_stock_fee_rules(commission_rate: float, stamp_tax: float, transfer_fee: float, min_commission: float)`: 设置股票费率参数。
*   `set_fund_fee_rules(commission_rate: float, transfer_fee: float, min_commission: float)`: 设置基金费率参数。
*   `set_future_fee_rules(commission_rate: float)`: 设置期货费率参数。
*   `set_option_fee_rules(commission_per_contract: float)`: 设置期权费率参数 (按张收费)。
*   `set_slippage(type: str, value: float)`: 设置滑点模型。
    *   `type`: "fixed" (固定金额) 或 "percent" (百分比)。
    *   `value`: 滑点数值 (例如 0.01 表示 1分钱 或 1%)。
*   `set_volume_limit(limit: float)`: 设置成交量限制比例 (0.0 - 1.0)，限制单笔撮合数量不超过当根 K 线成交量的指定比例。
*   `set_market_sessions(sessions: List[Tuple[str, str, TradingSession]])`: 设置交易时段列表，时间格式为 `HH:MM` 或 `HH:MM:SS`。
*   `get_results() -> BacktestResult`: 获取回测结果。

### `akquant.BacktestResult`

回测结果对象，包含绩效指标、交易记录和权益曲线。

**属性:**

*   `metrics` (`PerformanceMetrics`): 包含核心绩效指标的对象。
    *   `total_return_pct`: 总收益率 (%)
    *   `annualized_return`: 年化收益率 (小数)
    *   `max_drawdown_pct`: 最大回撤 (%)
    *   `sharpe_ratio`: 夏普比率
    *   `sortino_ratio`: 索提诺比率
    *   `win_rate`: 胜率 (小数)
*   `metrics_df` (`pd.DataFrame`): 包含上述指标的 DataFrame (单行)。
*   `trades_df` (`pd.DataFrame`): 包含所有已平仓交易的详细记录 (Entry/Exit Time/Price, PnL, Commission 等)。
*   `daily_positions_df` (`pd.DataFrame`): 每日持仓快照 DataFrame。
*   `equity_curve` (`List[Tuple[int, float]]`): 权益曲线数据列表 `[(timestamp, equity), ...]`。
*   `trades` (`List[ClosedTrade]`): 原始交易记录对象列表。

### `akquant.DataLoader`

数据加载与缓存工具类。

**方法:**

*   `__init__(cache_dir: Optional[str] = None)`: 初始化数据加载器，指定缓存目录。
*   `load_akshare(symbol: str, start_date: str, end_date: str, adjust: str = "qfq", period: str = "daily") -> pd.DataFrame`: 加载 AKShare A 股历史数据并自动缓存。

### `akquant.config.StrategyConfig`

全局策略配置对象 (`akquant.config.strategy_config`)。

**属性:**

*   `initial_cash` (float): 初始资金。
*   `fee_mode` (str): 费率模式 ('per_order', 'per_share', 'percent')。
*   `fee_amount` (float): 费率数值。
*   `execution_mode` (ExecutionMode): 执行模式。
*   `max_order_size` (float): 最大订单比例 (0.0 - 1.0)。
*   `max_order_value` (float): 最大订单金额。
*   `max_position_size` (float): 最大持仓比例。
*   `bootstrap_samples` (int): Bootstrap 抽样次数 (默认 1000)。
*   `exit_on_last_bar` (bool): 是否在回测结束时强制平仓 (默认 True)。

### `akquant.indicator.IndicatorSet`

指标集合，用于向量化预计算。

**方法:**

*   `add(name: str, func: Callable, *args, **kwargs)`: 添加指标计算函数。
*   `calculate_all(df: pd.DataFrame, symbol: str) -> Dict[str, pd.Series]`: 编译并计算所有指标，返回字典。

### `akquant.DataFeed`

数据容器，用于存储和管理 K 线 (Bar) 和 Tick 数据。

**方法:**

*   `__init__()`: 创建一个空的数据源。
*   `add_bar(bar: Bar)`: 添加一条 K 线数据。
*   `add_tick(tick: Tick)`: 添加一条 Tick 数据。
*   `get_bars() -> List[Bar]`: 获取所有 K 线数据。

### `akquant.StrategyContext`

策略上下文对象，在 `on_bar` 回调中传递给策略，用于查询状态和执行交易。

**属性:**

*   `cash`: 当前可用资金 (float, 只读)。
*   `orders`: 当前挂单列表 (List[Order], 只读)。
*   `positions`: 当前持仓字典 `{symbol: quantity}` (Dict[str, float], 只读)。
*   `available_positions`: 当前可用持仓字典 (Dict[str, float], 只读)。
*   `session`: 当前交易时段 (`TradingSession`, 只读)。

**方法:**

*   `buy(...)`: 发送买单 (底层接口)。
*   `sell(...)`: 发送卖单 (底层接口)。
*   `get_position(symbol: str) -> float`: 获取指定标的的持仓数量。
*   `schedule(timestamp: int, payload: str)`: 注册定时事件，将在指定时间戳触发 `on_timer(payload)` 回调。
*   `cancel_order(order_id: str)`: 取消指定订单。

### `akquant.Strategy`

Python 策略基类 (`akquant.strategy.Strategy`)。

**回调方法 (需重写):**

*   `on_bar(bar)`: K 线数据到达时触发。
*   `on_tick(tick)`: Tick 数据到达时触发。
*   `on_timer(payload)`: 定时器触发时调用。

**方法:**

*   `buy(symbol=None, quantity=None, price=None, time_in_force=None, trigger_price=None)`: 发送买单。
*   `sell(symbol=None, quantity=None, price=None, time_in_force=None, trigger_price=None)`: 发送卖单。
*   `buy_all(symbol=None)`: 全仓买入。
*   `close_position(symbol=None)`: 平仓当前标的。
*   `cancel_order(order_or_id)`: 取消订单。
*   `cancel_all_orders(symbol=None)`: 取消所有订单。
*   `get_open_orders(symbol=None) -> List[Order]`: 获取未完成订单。
*   `schedule(timestamp, payload)`: 注册定时事件。
*   `set_history_depth(depth: int)`: 设置自动维护的历史数据长度 (0 表示禁用)。
*   `get_history(count, symbol=None, field="close") -> np.ndarray`: 获取最近 `count` 个 Bar 的历史数据。
*   `set_sizer(sizer: Sizer)`: 设置仓位管理器。

### `akquant.run_backtest`

简化版回测入口函数，支持类和函数式策略。

**参数:**

*   `data`: 回测数据 (DataFrame, Dict[str, DataFrame] 或 List[Bar])。
*   `strategy`: 策略类、实例或回调函数。
*   `symbol`: 标的代码 (str 或 List[str])。
*   `cash`: 初始资金。
*   `commission`, `stamp_tax`, `transfer_fee`, `min_commission`: 费率参数。
*   `execution_mode`: 执行模式 ("next_open" 或 ExecutionMode)。
*   `timezone`: 时区名称 (默认 "Asia/Shanghai")。
*   `history_depth`: 自动维护历史数据的长度 (默认为 0)。
*   `lot_size`: 最小交易单位 (int 或 Dict[str, int])。
*   `initialize`, `context`: 函数式策略专用参数。

**返回:**

*   `BacktestResult`: 回测结果对象 (Python 包装器)。

## 数据结构

### `akquant.Bar`

代表一根 K 线数据。

**属性:**

*   `timestamp` (int): Unix 时间戳（纳秒）。
*   `open` (float): 开盘价。
*   `high` (float): 最高价。
*   `low` (float): 最低价。
*   `close` (float): 收盘价。
*   `volume` (float): 成交量。
*   `symbol` (str): 标的代码。

### `akquant.BacktestResult`

回测运行的最终结果对象 (Python 包装类)。

**属性 (DataFrame):**

*   `metrics_df` (pd.DataFrame): 包含详细的绩效指标。
*   `trades_df` (pd.DataFrame): 包含所有平仓交易记录。
*   `daily_positions_df` (pd.DataFrame): 每日持仓快照。

**属性 (原始数据):**

*   `metrics` (PerformanceMetrics): 详细的绩效指标对象。
*   `trade_metrics` (TradePnL): 详细的交易盈亏统计对象。
*   `trades` (List[ClosedTrade]): 交易列表。
*   `equity_curve` (List[Tuple[int, float]]): 权益曲线。

### `akquant.PerformanceMetrics`

包含详细的策略绩效评估指标。

**属性:**

*   `total_return` (float): 总收益率。
*   `annualized_return` (float): 年化收益率。
*   `max_drawdown` (float): 最大回撤金额。
*   `sharpe_ratio` (float): 夏普比率。
*   `sortino_ratio` (float): 索提诺比率。
*   `ulcer_index` (float): 溃疡指数。
*   `equity_r2` (float): 权益曲线回归 R²。
*   `win_rate` (float): 胜率。

### `akquant.TradePnL`

基于 FIFO (先进先出) 规则计算的交易盈亏统计。

**属性:**

*   `win_rate` (float): 胜率 (0.0 - 1.0)。
*   `profit_factor` (float): 盈亏比。
*   `max_wins` (int): 最大连续盈利次数。
*   `max_losses` (int): 最大连续亏损次数。
*   `unrealized_pnl` (float): 未实现盈亏。
