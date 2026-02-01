# API 参考

`AKQuant` 模块暴露了以下核心类用于构建回测系统。

## 核心组件

### `akquant.Engine`

回测引擎的主入口。负责管理数据、资金、持仓，并驱动策略运行。

**属性:**

*   `risk_manager` (`RiskManager`): 访问引擎内部的风控管理器。

**方法:**

*   `__init__()`: 创建一个新的引擎实例。
*   `add_data(feed: DataFeed)`: 加载数据源。
*   `add_bars(bars: List[Bar])`: 批量加载 K 线数据 (推荐使用)。
*   `add_instrument(instrument: Instrument)`: 添加合约信息。
*   `run(strategy: object, show_progress: bool = True) -> str`: 运行回测。
*   `set_execution_mode(mode: ExecutionMode)`: 设置执行模式 (`CurrentClose` 或 `NextOpen`)。
*   `set_timezone(offset_secs: int)`: 设置时区偏移秒数 (例如 28800 为 UTC+8)。
*   `use_simple_market(commission_rate: float)`: 启用简单市场模式 (7x24小时, T+0, 无税, 简单佣金)。
*   `use_china_market()`: 启用中国市场模式 (支持 T+1/T+0, 印花税, 过户费, 交易时段等)。
*   `use_china_futures_market()`: 启用中国期货市场默认配置 (T+0, 需手动设置交易时段)。
*   `set_t_plus_one(enabled: bool)`: 设置股票 T+1/T+0 规则 (仅针对 ChinaMarket)。
*   `set_force_session_continuous(enabled: bool)`: 强制使用连续交易时段。
*   `set_stock_fee_rules(commission_rate: float, stamp_tax: float, transfer_fee: float, min_commission: float)`: 设置股票费率参数。
*   `set_future_fee_rules(commission_rate: float)`: 设置期货费率参数。
*   `set_fund_fee_rules(commission_rate: float, transfer_fee: float, min_commission: float)`: 设置基金费率参数。
*   `set_option_fee_rules(commission_per_contract: float)`: 设置期权费率参数 (按张收费)。
*   `set_slippage(type: str, value: float)`: 设置滑点模型 (`fixed` 或 `percent`)。
*   `set_volume_limit(limit: float)`: 设置成交量限制比例 (0.0 - 1.0)。
*   `set_market_sessions(sessions: List[Tuple[str, str, TradingSession]])`: 设置交易时段列表。
*   `get_results() -> BacktestResult`: 获取回测结果。
*   `set_history_depth(depth: int)`: 设置引擎层面的历史数据缓存深度（通常由策略自动设置）。

### `akquant.Strategy`

Python 策略基类 (`akquant.strategy.Strategy`)。

**回调方法 (需重写):**

*   `on_bar(bar: Bar)`: K 线数据到达时触发。
*   `on_tick(tick: Tick)`: Tick 数据到达时触发。
*   `on_timer(payload: str)`: 定时器触发时调用。
*   `on_start()`: 策略启动时调用，可用于订阅行情或注册指标。

**交易方法:**

*   `buy(symbol, quantity, price, time_in_force, trigger_price)`: 发送买单。
*   `sell(symbol, quantity, price, time_in_force, trigger_price)`: 发送卖单。
*   `short(symbol, quantity, price, time_in_force, trigger_price)`: 卖出开空 (Short Sell)。
*   `cover(symbol, quantity, price, time_in_force, trigger_price)`: 买入平空 (Buy to Cover)。
*   `stop_buy(symbol, trigger_price, quantity, price, time_in_force)`: 发送止损买入单。
*   `stop_sell(symbol, trigger_price, quantity, price, time_in_force)`: 发送止损卖出单。
*   `order_target_value(target_value, symbol, price, **kwargs)`: 调整仓位到目标市值。
*   `order_target_percent(target_percent, symbol, price, **kwargs)`: 调整仓位到目标账户占比。
*   `buy_all(symbol)`: 全仓买入。
*   `close_position(symbol)`: 平仓当前标的。
*   `cancel_order(order_or_id)`: 取消指定订单。
*   `cancel_all_orders(symbol)`: 取消所有订单。

**数据与状态方法:**

*   `get_position(symbol) -> float`: 获取指定标的持仓数量。
*   `get_cash() -> float`: 获取当前可用资金。
*   `get_open_orders(symbol) -> List[Order]`: 获取未完成订单。
*   `get_history(count, symbol, field="close") -> np.ndarray`: 获取历史数据。
*   `set_history_depth(depth: int)`: 设置历史数据回溯长度 (0 表示禁用)。
*   `set_sizer(sizer: Sizer)`: 设置仓位管理器。
*   `register_indicator(name, indicator)`: 注册指标，可通过 `self.name` 访问。
*   `subscribe(instrument_id: str)`: 订阅合约行情。
*   `schedule(timestamp: int, payload: str)`: 注册定时事件。

### `akquant.VectorizedStrategy`

向量化策略基类 (`akquant.strategy.VectorizedStrategy`)，继承自 `Strategy`。
用于支持基于预计算指标的高速回测模式。

**方法:**

*   `__init__(precalculated_data: Dict[str, Dict[str, np.ndarray]])`: 初始化策略。
*   `get_value(indicator_name: str, symbol: str) -> float`: 获取当前 Bar 对应的预计算指标值 (O(1) 访问)。

### `akquant.RiskManager` & `akquant.RiskConfig`

风控管理模块。

**`RiskConfig` 属性:**

*   `max_order_size` (float): 单笔最大下单数量。
*   `max_order_value` (float): 单笔最大下单金额。
*   `max_position_size` (float): 最大持仓数量 (绝对值)。
*   `restricted_list` (List[str]): 限制交易标的列表。
*   `active` (bool): 是否启用风控。

**`RiskManager` 方法:**

*   `check(order: Order, portfolio: Portfolio) -> Optional[str]`: 检查订单是否合规，不合规返回错误信息。

**辅助函数:**

*   `akquant.risk.apply_risk_config(engine: Engine, config: RiskConfig)`: 将 Python 侧的风控配置应用到 Rust 引擎。

## 数据结构与枚举

### 枚举类型

*   `AssetType`: `Stock`, `Future`, `Option`, `Fund`, `Crypto`, `Forex`, `Index`, `Bond`
*   `ExecutionMode`: `CurrentClose`, `NextOpen`
*   `OrderSide`: `Buy`, `Sell`
*   `OrderType`: `Market`, `Limit`, `Stop`, `StopLimit`
*   `OrderStatus`: `New`, `Submitted`, `PartiallyFilled`, `Filled`, `Canceled`, `Rejected`, `Expired`
*   `TimeInForce`: `GTC`, `Day`, `IOC`, `FOK`, `GTD`
*   `TradingSession`: `PreMarket`, `Regular`, `PostMarket`
*   `OptionType`: `Call`, `Put`

### `akquant.Instrument`

交易标的定义。

**属性:**

*   `symbol` (str): 代码
*   `asset_type` (AssetType): 资产类型
*   `multiplier` (float): 合约乘数
*   `margin_ratio` (float): 保证金比率
*   `tick_size` (float): 最小变动价位
*   `option_type` (OptionType, optional): 期权类型
*   `strike_price` (float, optional): 行权价
*   `expiry_date` (int, optional): 到期日
*   `lot_size` (float, optional): 最小交易单位

### `akquant.Bar`

K 线数据结构。

**属性:** `timestamp` (int), `symbol` (str), `open` (float), `high` (float), `low` (float), `close` (float), `volume` (float), `extra` (Dict[str, float])

### `akquant.Tick`

Tick 数据结构。

**属性:** `timestamp` (int), `symbol` (str), `price` (float), `volume` (float)

### `akquant.Order`

订单对象。

**属性:**

*   `id` (str): 订单ID
*   `symbol` (str): 标的代码
*   `side` (OrderSide): 交易方向
*   `order_type` (OrderType): 订单类型
*   `quantity` (float): 数量
*   `price` (float, optional): 价格
*   `time_in_force` (TimeInForce): 有效期
*   `trigger_price` (float, optional): 触发价
*   `status` (OrderStatus): 状态
*   `filled_quantity` (float): 已成交数量
*   `average_filled_price` (float): 成交均价

### `akquant.Trade`

成交记录。

**属性:** `id`, `order_id`, `symbol`, `side`, `quantity`, `price`, `commission`, `timestamp`, `bar_index`

### `akquant.ClosedTrade`

平仓交易记录 (Entry + Exit)。

**属性:**

*   `symbol`, `entry_time`, `exit_time`
*   `entry_price`, `exit_price`
*   `quantity`, `direction` (str)
*   `pnl`, `net_pnl`, `return_pct`
*   `commission`
*   `duration_bars`

### `akquant.BacktestResult`

回测结果对象。

**属性:**

*   `metrics` (`PerformanceMetrics`): 核心绩效指标。
*   `trade_metrics` (`TradePnL`): 交易统计指标。
*   `trades` (`List[ClosedTrade]`): 所有平仓交易记录列表。
*   `equity_curve` (`List[Tuple[int, float]]`): 权益曲线数据。
*   `daily_positions` (`List[Tuple[int, Dict[str, float]]]`): 每日持仓快照。

### `akquant.PerformanceMetrics`

核心绩效指标。

**属性:**

*   `akquant.SMA(period)`: 简单移动平均线计算器 (流式更新)。
    *   `update(value) -> Optional[float]`: 更新数据并获取当前均值。

*   `total_return` (float): 总收益
*   `total_return_pct` (float): 总收益率
*   `annualized_return` (float): 年化收益率
*   `max_drawdown` (float): 最大回撤金额
*   `max_drawdown_pct` (float): 最大回撤比例
*   `sharpe_ratio` (float): 夏普比率
*   `sortino_ratio` (float): 索提诺比率
*   `volatility` (float): 波动率
*   `win_rate` (float): 胜率 (基于天数或周期，非交易笔数)
*   `initial_market_value` (float): 初始市值
*   `end_market_value` (float): 结束市值

### `akquant.TradePnL`

交易盈亏统计 (FIFO)。

**属性:**

*   `net_pnl` (float): 净盈亏
*   `total_commission` (float): 总手续费
*   `total_closed_trades` (int): 总平仓交易数
*   `won_count` (int): 盈利交易数
*   `lost_count` (int): 亏损交易数
*   `win_rate` (float): 胜率 (盈利次数 / 总次数)
*   `profit_factor` (float): 盈亏比 (总盈利 / 总亏损绝对值)
*   `avg_profit` (float): 平均盈利
*   `avg_loss` (float): 平均亏损
*   `largest_win` (float): 最大单笔盈利
*   `largest_loss` (float): 最大单笔亏损
*   `unrealized_pnl` (float): 未实现盈亏

### `akquant.StrategyContext`

策略上下文，提供给策略逻辑使用的状态和操作接口。

**属性:**

*   `cash` (float): 当前现金
*   `positions` (Dict[str, float]): 当前持仓
*   `active_orders` (List[Order]): 活动订单
*   `session` (TradingSession): 当前交易时段

**方法:**

*   `buy(...)`, `sell(...)`: 交易指令
*   `cancel_order(order_id)`: 取消订单
*   `history(...)`: 获取历史数据
*   `get_position(symbol)`: 获取持仓


## 工具函数

*   `akquant.from_arrays(timestamps, opens, highs, lows, closes, volumes, symbol=None, ...)`: 从 Numpy 数组高效批量创建 `Bar` 列表。
*   `akquant.run_backtest(data, strategy, ...)`: 简化版回测入口函数。
*   `akquant.DataLoader`: 数据加载工具，支持 AKShare 数据自动缓存。
*   `akquant.DataFeed`: 数据容器，支持 CSV 流式读取和实时模式。
