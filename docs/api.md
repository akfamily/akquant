# API 参考

本 API 文档涵盖了 AKQuant 的核心类和方法。

## 1. 核心引擎 (Core)

### `akquant.Engine`

回测引擎的主入口。

```python
engine = akquant.Engine()
```

**配置方法:**

*   `set_timezone(offset: int)`: 设置时区偏移 (秒)。例如 UTC+8 为 28800。
*   `use_simulated_execution()`: (默认) 启用内存撮合模拟执行。
*   `use_realtime_execution()`: 启用实盘/仿真执行 (订单发送至外部 Broker)。
*   `set_execution_mode(mode: ExecutionMode)`: 设置撮合模式。
    *   `ExecutionMode.CurrentClose`: 当前 Bar 收盘价撮合 (默认)。
    *   `ExecutionMode.NextOpen`: 下一 Bar 开盘价撮合。
*   `set_history_depth(depth: int)`: 设置引擎层面的历史数据缓存长度。

**市场与费率配置:**

*   `use_simple_market(commission_rate: float)`: 启用简单市场 (T+0, 7x24, 无税)。
*   `use_china_market()`: 启用中国市场 (T+1/T+0, 交易时段, 税费)。
*   `use_china_futures_market()`: 启用中国期货市场 (T+0, 需手动配置时段)。
*   `set_t_plus_one(enabled: bool)`: 开启/关闭 T+1 规则 (仅限 ChinaMarket)。
*   `set_stock_fee_rules(commission_rate, stamp_tax, transfer_fee, min_commission)`: 设置股票费率。
*   `set_slippage(type: str, value: float)`: 设置滑点。`type` 可为 `"fixed"` (固定金额) 或 `"percent"` (百分比)。

**运行方法:**

*   `add_instrument(instrument: Instrument)`: 添加合约定义。
*   `add_data(feed: DataFeed)`: 添加数据源。
*   `add_bars(bars: List[Bar])`: 批量添加 Bar 数据。
*   `run(strategy: Strategy, show_progress: bool) -> str`: 运行回测。
*   `get_results() -> BacktestResult`: 获取详细回测结果。

### `akquant.DataFeed`

数据容器。

*   `add_bars(bars: List[Bar])`: 添加数据。
*   `sort()`: 按时间戳排序数据。

## 2. 策略开发 (Strategy)

### `akquant.Strategy`

策略基类。用户应继承此类并重写回调方法。

**回调方法:**

*   `on_start()`: 策略启动时触发。用于订阅 (`subscribe`) 和注册指标。
*   `on_bar(bar: Bar)`: K 线闭合时触发。
*   `on_tick(tick: Tick)`: Tick 到达时触发。
*   `on_timer(payload: str)`: 定时器触发。

**交易方法:**

*   `buy(symbol, quantity, price=None, ...)`: 买入。不指定 `price` 则为市价单。
*   `sell(symbol, quantity, price=None, ...)`: 卖出。
*   `short(symbol, quantity, price=None, ...)`: 卖空。
*   `cover(symbol, quantity, price=None, ...)`: 平空。
*   `stop_buy(symbol, trigger_price, quantity, ...)`: 止损买入。
*   `stop_sell(symbol, trigger_price, quantity, ...)`: 止损卖出。
*   `order_target_value(target_value, symbol, price=None)`: 调整至目标持仓市值。
*   `order_target_percent(target_percent, symbol, price=None)`: 调整至目标账户占比。
*   `close_position(symbol)`: 平仓指定标的。
*   `cancel_all_orders(symbol)`: 取消指定标的的所有挂单。

**数据访问:**

*   `get_history(count, symbol, field="close") -> np.ndarray`: 获取历史数据数组 (Zero-Copy)。
*   `get_position(symbol) -> float`: 获取当前持仓量。
*   `get_cash() -> float`: 获取当前可用资金。
*   `subscribe(instrument_id: str)`: 订阅行情。

### `akquant.Bar`

K 线数据对象。

*   `timestamp`: Unix 时间戳 (纳秒)。
*   `open`, `high`, `low`, `close`, `volume`: OHLCV 数据。
*   `symbol`: 标的代码。

## 3. 交易对象 (Trading Objects)

### `akquant.Order`

订单对象。

*   `id`: 订单 ID。
*   `symbol`: 标的代码。
*   `side`: `OrderSide.Buy` 或 `OrderSide.Sell`。
*   `order_type`: `OrderType.Market`, `OrderType.Limit`, `OrderType.StopMarket` 等。
*   `status`: `OrderStatus.New`, `Submitted`, `Filled`, `Cancelled`, `Rejected` 等。
*   `quantity`: 委托数量。
*   `filled_quantity`: 已成交数量。
*   `average_filled_price`: 成交均价。

### `akquant.Instrument`

合约定义。

```python
Instrument(
    symbol="AAPL",
    asset_type=AssetType.Stock,
    multiplier=1.0,
    margin_ratio=1.0,
    tick_size=0.01
)
```

## 4. 投资组合与风控 (Portfolio & Risk)

### `akquant.Portfolio`

*   `cash`: 当前现金。
*   `positions`: 持仓字典 `{symbol: quantity}`。
*   `available_positions`: 可用持仓 (考虑 T+1 和冻结)。

### `akquant.RiskConfig`

风控配置。

*   `active`: 是否启用。
*   `max_order_size`: 单笔最大数量。
*   `max_order_value`: 单笔最大金额。
*   `max_position_size`: 最大持仓数量。
*   `restricted_list`: 限制交易名单 (List[str])。

## 5. 结果分析 (Analysis)

### `akquant.BacktestResult`

回测结果容器。

*   `metrics_df`: 包含各项绩效指标的 DataFrame (Total Return, Sharpe, Max Drawdown 等)。
*   `daily_positions_df`: 每日持仓 DataFrame。
*   `trades`: 交易记录列表 (`ClosedTrade` 对象)。
*   `equity_curve`: 权益曲线数据。

## 6. 内置指标 (Indicators)

位于 `akquant.indicators` 模块。

*   `SMA(period)`
*   `EMA(period)`
*   `MACD(fast, slow, signal)`
*   `RSI(period)`
*   `BollingerBands(period, multiplier)`
*   `ATR(period)`

所有指标均有 `value` 属性获取当前值，且在注册到 Strategy 后会自动更新。
