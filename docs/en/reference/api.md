# API Reference

This API documentation covers the core classes and methods of AKQuant.

## 1. High-Level API

### `akquant.run_backtest`

The most commonly used backtest entry function, encapsulating the initialization and configuration process of the engine.

```python
def run_backtest(
    data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame], List[Bar]]] = None,
    strategy: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None] = None,
    symbol: Union[str, List[str]] = "BENCHMARK",
    cash: float = 1_000_000.0,
    commission: float = 0.0003,
    instruments_config: Optional[Union[List[InstrumentConfig], Dict[str, InstrumentConfig]]] = None,
    warmup_period: int = 0,
    # ... other parameters
) -> BacktestResult
```

**Key Parameters:**

*   `data`: Backtest data. Supports a single DataFrame, or a `{symbol: DataFrame}` dictionary.
*   `warmup_period`: **(New)** Strategy warmup period. Specifies the length of historical data (number of Bars) to preload for indicator calculation.
*   `instruments_config`: **(New)** Instrument configuration. Used to set parameters for non-stock assets like futures/options (e.g., multiplier, margin ratio).
    *   Accepts `List[InstrumentConfig]` or `{symbol: InstrumentConfig}`.

### `akquant.InstrumentConfig`

A data class used to configure the properties of a single instrument.

```python
@dataclass
class InstrumentConfig:
    symbol: str
    asset_type: str = "STOCK"  # "STOCK", "FUTURES", "FUND", "OPTION"
    multiplier: float = 1.0    # Contract multiplier
    margin_ratio: float = 1.0  # Margin ratio (0.1 means 10% margin)
    tick_size: float = 0.01    # Minimum price variation
    lot_size: int = 1          # Minimum trading unit
```

## 2. Core Engine

### `akquant.Engine`

The main entry point for the backtesting engine.

```python
engine = akquant.Engine()
```

**Configuration Methods:**

*   `set_timezone(offset: int)`: Set timezone offset (seconds). E.g., 28800 for UTC+8.
*   `use_simulated_execution()`: (Default) Enable in-memory matching simulation execution.
*   `use_realtime_execution()`: Enable real-time/paper trading execution (orders sent to external Broker).
*   `set_execution_mode(mode: ExecutionMode)`: Set matching mode.
    *   `ExecutionMode.NextOpen`: Match at next Bar Open (Default).
    *   `ExecutionMode.CurrentClose`: Match at current Bar Close.
*   `set_history_depth(depth: int)`: Set the history data cache length at the engine level.

**Market & Fee Configuration:**

*   `use_simple_market(commission_rate: float)`: Enable simple market (T+0, 7x24).
    *   **Update**: Now supports stamp tax, transfer fee, and min commission configuration (via `set_stock_fee_rules`).
*   `use_china_market()`: Enable China market (T+1/T+0, trading sessions, taxes).
*   `use_china_futures_market()`: Enable China futures market (T+0, manual session config required).
*   `set_t_plus_one(enabled: bool)`: Enable/Disable T+1 rule (ChinaMarket only).
*   `set_stock_fee_rules(commission_rate, stamp_tax, transfer_fee, min_commission)`: Set stock fee rules (Applicable to both SimpleMarket and ChinaMarket).
*   `set_slippage(type: str, value: float)`: Set slippage. `type` can be `"fixed"` (fixed amount) or `"percent"` (percentage).

**Runtime Methods:**

*   `add_instrument(instrument: Instrument)`: Add instrument definition.
*   `add_data(feed: DataFeed)`: Add data source.
*   `add_bars(bars: List[Bar])`: Batch add Bar data.
*   `run(strategy: Strategy, show_progress: bool) -> str`: Run backtest.
*   `get_results() -> BacktestResult`: Get detailed backtest results.

### `akquant.DataFeed`

Data container.

*   `add_bars(bars: List[Bar])`: Add data.
*   `sort()`: Sort data by timestamp.

### `akquant.BarAggregator`

Real-time Tick aggregator, used to convert Tick streams into Bar data and automatically inject into DataFeed.

```python
aggregator = akquant.BarAggregator(feed: DataFeed, interval_min: int = 1)
```

**Methods:**

*   `on_tick(symbol: str, price: float, volume: float, timestamp_ns: int)`: Process new Tick data.
    *   `volume`: Here volume should be accumulated volume (TotalVolume), the aggregator will automatically calculate the increment.

## 2. Strategy Development (Strategy)

### `akquant.Strategy`

Strategy base class. Users should inherit from this class and override callback methods.

**Callback Methods:**

*   `on_start()`: Triggered when the strategy starts. Used for subscription (`subscribe`) and indicator registration.
*   `on_bar(bar: Bar)`: Triggered when a Bar closes.
*   `on_tick(tick: Tick)`: Triggered when a Tick arrives.
*   `on_timer(payload: str)`: Triggered by timer.

**Trading Methods:**

*   `buy(symbol, quantity, price=None, ...)`: Buy. Market order if `price` is not specified.
*   `sell(symbol, quantity, price=None, ...)`: Sell.
*   `short(symbol, quantity, price=None, ...)`: Short sell.
*   `cover(symbol, quantity, price=None, ...)`: Buy to cover.
*   `stop_buy(symbol, trigger_price, quantity, ...)`: Stop buy.
*   `stop_sell(symbol, trigger_price, quantity, ...)`: Stop sell.
*   `order_target_value(target_value, symbol, price=None)`: Adjust position to target value.
*   `order_target_percent(target_percent, symbol, price=None)`: Adjust position to target account percentage.
*   `close_position(symbol)`: Close position for a specific instrument.
*   `cancel_all_orders(symbol)`: Cancel all pending orders for a specific instrument.

**Data Access:**

*   `get_history(count, symbol, field="close") -> np.ndarray`: Get history data array (Zero-Copy).
*   `get_position(symbol) -> float`: Get current position size.
*   `get_cash() -> float`: Get current available cash.
*   `subscribe(instrument_id: str)`: Subscribe to market data.

### `akquant.Bar`

Bar data object.

*   `timestamp`: Unix timestamp (nanoseconds).
*   `open`, `high`, `low`, `close`, `volume`: OHLCV data.
*   `symbol`: Instrument symbol.

## 3. Trading Objects

### `akquant.Order`

Order object.

*   `id`: Order ID.
*   `symbol`: Instrument symbol.
*   `side`: `OrderSide.Buy` or `OrderSide.Sell`.
*   `order_type`: `OrderType.Market`, `OrderType.Limit`, `OrderType.StopMarket` etc.
*   `status`: `OrderStatus.New`, `Submitted`, `Filled`, `Cancelled`, `Rejected` etc.
*   `quantity`: Order quantity.
*   `filled_quantity`: Filled quantity.
*   `average_filled_price`: Average filled price.

### `akquant.Instrument`

Contract definition.

```python
Instrument(
    symbol="AAPL",
    asset_type=AssetType.Stock,
    multiplier=1.0,
    margin_ratio=1.0,
    tick_size=0.01
)
```

## 4. Portfolio & Risk

### `akquant.Portfolio`

*   `cash`: Current cash.
*   `positions`: Position dictionary `{symbol: quantity}`.
*   `available_positions`: Available positions (considering T+1 and frozen).

### `akquant.RiskConfig`

Risk configuration.

*   `active`: Whether enabled.
*   `max_order_size`: Max single order size.
*   `max_order_value`: Max single order value.
*   `max_position_size`: Max position size.
*   `restricted_list`: Restricted trading list (List[str]).

## 5. Analysis

### `akquant.BacktestResult`

Backtest result container.

*   `metrics_df`: (pd.DataFrame) DataFrame containing performance metrics (Total Return, Sharpe, Max Drawdown, Ulcer Index, UPI, **SQN**, **Kelly**, **VaR/CVaR**, etc.).
*   `trades_df`: (pd.DataFrame) DataFrame containing all closed trades.
*   `orders_df`: (pd.DataFrame) DataFrame containing all order records.
*   `positions_df`: (pd.DataFrame) DataFrame containing daily position details, including quantity, market value, unrealized PnL, **average entry price (entry_price)**, etc.
*   `equity_curve`: (pd.Series) Equity curve, indexed by time, values are total account equity.
*   `cash_curve`: (pd.Series) Cash curve, indexed by time, values are available cash.
*   `trades`: (List[ClosedTrade]) Raw list of closed trade objects.

## 6. Built-in Indicators

Located in `akquant.indicators` module.

*   `SMA(period)`
*   `EMA(period)`
*   `MACD(fast, slow, signal)`
*   `RSI(period)`
*   `BollingerBands(period, multiplier)`
*   `ATR(period)`

All indicators have a `value` property to get the current value, and will automatically update after being registered to a Strategy.

## 7. Machine Learning

AKQuant provides a dedicated machine learning support module `akquant.ml`. For detailed usage, please refer to the [Machine Learning Guide](ml_guide.md).

### Core Classes

*   `akquant.ml.QuantModel`: Unified interface for all ML models.
*   `akquant.ml.SklearnAdapter`: Adapter for Scikit-learn style models (e.g., XGBoost, LightGBM).
*   `akquant.ml.PyTorchAdapter`: Adapter for PyTorch deep learning models.

Key Methods:

*   `set_validation(method='walk_forward', verbose=False, ...)`: Configure rolling validation/training parameters.
*   `predict(X)`: Execute prediction.
