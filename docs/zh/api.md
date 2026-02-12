# API 参考

本 API 文档涵盖了 AKQuant 的核心类和方法。

## 1. 高级入口 (High-Level API)

### `akquant.run_backtest`

最常用的回测入口函数，封装了引擎的初始化和配置过程。

```python
def run_backtest(
    data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame], List[Bar]]] = None,
    strategy: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None] = None,
    symbol: Union[str, List[str]] = "BENCHMARK",
    cash: float = 1_000_000.0,
    commission: float = 0.0003,
    instruments_config: Optional[Union[List[InstrumentConfig], Dict[str, InstrumentConfig]]] = None,
    warmup_period: int = 0,
    # ... 其他参数
) -> BacktestResult
```

**关键参数:**

*   `data`: 回测数据。支持单个 DataFrame，或 `{symbol: DataFrame}` 字典。
*   `warmup_period`: **(新增)** 策略预热期。指定需要预加载的历史数据长度（Bar 数量），用于计算指标。
*   `instruments_config`: **(新增)** 标的配置。用于设置期货/期权等非股票资产的参数（如乘数、保证金）。
    *   接收 `List[InstrumentConfig]` 或 `{symbol: InstrumentConfig}`。

### `akquant.InstrumentConfig`

用于配置单个标的属性的数据类。

```python
@dataclass
class InstrumentConfig:
    symbol: str
    asset_type: str = "STOCK"  # "STOCK", "FUTURES", "FUND", "OPTION"
    multiplier: float = 1.0    # 合约乘数
    margin_ratio: float = 1.0  # 保证金率 (0.1 表示 10% 保证金)
    tick_size: float = 0.01    # 最小变动价位
    lot_size: int = 1          # 最小交易单位
```

## 2. 核心引擎 (Core)

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
    *   `ExecutionMode.NextOpen`: 下一 Bar 开盘价撮合 (默认)。
    *   `ExecutionMode.CurrentClose`: 当前 Bar 收盘价撮合。
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

### `akquant.BarAggregator`

实时 Tick 聚合器，用于将 Tick 流转换为 Bar 数据并自动注入 DataFeed。

```python
aggregator = akquant.BarAggregator(feed: DataFeed, interval_min: int = 1)
```

**方法:**

*   `on_tick(symbol: str, price: float, volume: float, timestamp_ns: int)`: 处理新的 Tick 数据。
    *   `volume`: 这里的 volume 应该是累计成交量 (TotalVolume)，聚合器会自动计算增量。

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

回测运行后返回的结果对象，包含账户权益曲线、交易记录和详细的绩效指标。

**主要属性:**

*   `metrics_df`: (pd.DataFrame) 包含所有绩效指标的表格，以指标名称为索引。
*   `trades_df`: (pd.DataFrame) 包含所有平仓交易记录的表格。
*   `orders_df`: (pd.DataFrame) 包含所有委托记录的表格。
*   `positions_df`: (pd.DataFrame) 包含每日持仓详情，包括持仓数量、市值、浮动盈亏、**持仓均价 (entry_price)** 等。
*   `equity_curve`: (pd.Series) 权益曲线，索引为时间，值为账户总权益。
*   `cash_curve`: (pd.Series) 现金曲线，索引为时间，值为账户可用现金。

**绩效指标详解 (Performance Metrics):**

详细的绩效指标说明、单位及计算公式，请参考 **[绩效指标详解](metrics.md)**。


## 6. 内置指标 (Indicators)

位于 `akquant.indicators` 模块。

*   `SMA(period)`
*   `EMA(period)`
*   `MACD(fast, slow, signal)`
*   `RSI(period)`
*   `BollingerBands(period, multiplier)`
*   `ATR(period)`

所有指标均有 `value` 属性获取当前值，且在注册到 Strategy 后会自动更新。

## 7. 机器学习 (Machine Learning)

AKQuant 提供了专门的机器学习支持模块 `akquant.ml`。详细使用说明请参考 [机器学习指南](ml_guide.md)。

### 核心类

*   `akquant.ml.QuantModel`: 所有 ML 模型的统一接口。
*   `akquant.ml.SklearnAdapter`: 用于适配 Scikit-learn 风格的模型 (如 XGBoost, LightGBM)。
*   `akquant.ml.PyTorchAdapter`: 用于适配 PyTorch 深度学习模型。

主要方法:

*   `set_validation(method='walk_forward', verbose=False, ...)`: 配置滚动验证/训练参数。
*   `predict(X)`: 执行预测。
