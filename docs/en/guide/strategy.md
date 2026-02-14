# Strategy Guide

This document aims to help strategy developers quickly master how to write strategies in AKQuant.

## 1. Core Concepts (Glossary)

For those new to quantitative trading, here are some basic terms:

*   **Bar (Candlestick)**: Contains market data for a specific time period (e.g., 1 minute, 1 day), primarily including 5 data points:
    *   **Open**: Opening price
    *   **High**: Highest price
    *   **Low**: Lowest price
    *   **Close**: Closing price
    *   **Volume**: Trading volume
*   **Strategy**: Your trading robot. Its core job is to continuously watch the market (`on_bar`) and then decide whether to `buy` or `sell`.
*   **Context**: The robot's "notebook" and "toolbox". It records how much cash and how many positions are currently held, and provides tools for placing orders.
*   **Position**: The quantity of stocks or futures you currently hold. A positive number indicates a long position (buying to hold), and a negative number indicates a short position (selling borrowed securities).
*   **Backtest**: Historical simulation. Testing your strategy using past data to see how much money it would have made if executed in the past.

## 2. Strategy Lifecycle

A strategy goes through the following stages from start to finish:

*   `__init__`: Python object initialization, suitable for defining parameters.
*   `on_start`: Called when the strategy starts. You **must** use `self.subscribe()` here to subscribe to data, and you can also register indicators here.
*   `on_bar`: Triggered when each Bar closes (core trading logic).
*   `on_tick`: Triggered when each Tick arrives (high-frequency/order book strategies).
*   `on_order`: Triggered when order status changes (e.g., Submitted, Filled, Cancelled).
*   `on_trade`: Triggered when a trade execution report is received.
*   `on_timer`: Called when a timer triggers (needs manual registration).
    > Recommended: Use `self.add_daily_timer("14:55:00", "payload")`.
*   `on_stop`: Called when the strategy stops, suitable for resource cleanup or result statistics (refer to Backtrader `stop` / Nautilus `on_stop`).
*   `on_train_signal`: Triggered for rolling training signals (only in ML mode).

## 3. Utilities

AKQuant provides a set of utilities to simplify strategy development.

### 3.1 Logging

Use `self.log()` to output logs with the current **backtest timestamp**, which is useful for debugging.

```python
def on_bar(self, bar):
    # Automatically adds timestamp, e.g., [2023-01-01 09:30:00] Signal: Buy
    self.log("Signal: Buy")

    # Support logging level
    import logging
    self.log("Insufficient funds", level=logging.WARNING)
```

### 3.2 Data Access (Syntactic Sugar)

The `Strategy` class provides properties for quick access to current Bar/Tick data:

| Property | Description | Original Code |
| :--- | :--- | :--- |
| `self.symbol` | Current symbol | `bar.symbol` / `tick.symbol` |
| `self.close` | Current price | `bar.close` / `tick.price` |
| `self.open` | Current open price | `bar.open` (0 in Tick mode) |
| `self.high` | Current high price | `bar.high` (0 in Tick mode) |
| `self.low` | Current low price | `bar.low` (0 in Tick mode) |
| `self.volume` | Current volume | `bar.volume` / `tick.volume` |

**Example**:
```python
def on_bar(self, bar):
    # Old way
    if bar.close > bar.open: ...

    # New way (Cleaner)
    if self.close > self.open:
        self.buy(self.symbol, 100)
```

### 3.3 Timer

In addition to the low-level `schedule` method, AKQuant provides more convenient ways to register timers:

*   **`add_daily_timer(time_str, payload)`**: Triggers daily at a specified time.
*   **`schedule(trigger_time, payload)`**: Triggers once at a specified datetime.

```python
def on_start(self):
    # Daily check at 14:55:00
    self.add_daily_timer("14:55:00", "daily_check")

    # Specific event
    self.schedule("2023-01-01 09:30:00", "special_event")

def on_timer(self, payload):
    if payload == "daily_check":
        self.log("Running daily check...")
```

## 4. Choosing a Strategy Style {: #style-selection }

AKQuant provides two styles of strategy development interfaces:

| Feature | Class-based Style (Recommended) | Function-based Style |
| :--- | :--- | :--- |
| **Definition** | Inherit from `akquant.Strategy` | Define `initialize` and `on_bar` functions |
| **Scenarios** | Complex strategies, need to maintain internal state, production | Rapid prototyping, migrating Zipline/Backtrader strategies |
| **Structure** | Object-oriented, good logic encapsulation | Script-like, simple and intuitive |
| **API Call** | `self.buy()`, `self.ctx` | `ctx.buy()`, pass `ctx` as parameter |

## 4. Writing Class-based Strategies {: #class-based }

This is the recommended way to write strategies in AKQuant, offering a clear structure and easy extensibility.

```python
from akquant import Strategy, Bar
import numpy as np

class MyStrategy(Strategy):
    def __init__(self, ma_window=20):
        # Note: The Strategy class uses __new__ for initialization, subclasses do not need to call super().__init__()
        self.ma_window = ma_window

    def on_start(self):
        # Explicitly subscribe to data
        self.subscribe("600000")

    def on_bar(self, bar: Bar):
        # 1. Get historical data (Online mode)
        # Get the last N closing prices
        history = self.get_history(count=self.ma_window, symbol=bar.symbol, field="close")

        # Check if data is sufficient
        if len(history) < self.ma_window:
            return

        # Calculate Moving Average
        ma_value = np.mean(history)

        # 2. Trading Logic
        # Get current position
        pos = self.get_position(bar.symbol)

        if bar.close > ma_value and pos == 0:
            self.buy(symbol=bar.symbol, quantity=100)
        elif bar.close < ma_value and pos > 0:
            self.sell(symbol=bar.symbol, quantity=100)
```

## 5. Orders & Execution

### 4.1 Order Lifecycle

In AKQuant, order status transitions are as follows:

1.  **New**: Order object is created.
2.  **Submitted**: Order has been sent to the exchange/simulation matching engine.
3.  **Accepted**: (Live mode) Exchange confirms receipt of the order.
4.  **Filled**: Order is fully filled.
    *   **PartiallyFilled**: Partially filled (currently unified as Filled status code, check `filled_quantity`).
5.  **Cancelled**: Order has been cancelled.
6.  **Rejected**: Order rejected by risk control or exchange (e.g., insufficient funds, exceeding price limits).

### 5.2 Common Trading Commands

*   **Market Order**:
    ```python
    self.buy(symbol="AAPL", quantity=100) # Market Buy
    self.sell(symbol="AAPL", quantity=100) # Market Sell
    ```
*   **Limit Order**:
    Executes at a specified price, only when the market price is at or better than the specified price.
    ```python
    self.buy(symbol="AAPL", quantity=100, price=150.0) # Limit Buy at 150
    ```
*   **Stop Order**:
    Converts to a market order when the market price touches the trigger price (`trigger_price`).
    ```python
    # Stop Sell (Market) when price drops below 140
    self.stop_sell(symbol="AAPL", quantity=100, trigger_price=140.0)
    ```
*   **Target Orders**:
    Automatically calculates buy/sell quantities to adjust the position to a target value.
    ```python
    # Adjust position to 50% of total assets
    self.order_target_percent(target_percent=0.5, symbol="AAPL", price=None)

    # Adjust holding to 1000 shares (Buy 1000 if 0, Sell 1000 if 2000)
    self.order_target_value(target_value=1000 * price, symbol="AAPL") # Note: API does not support target_share directly yet, simulate with value
    ```
*   **Cancel Order**:
    ```python
    self.cancel_order(order_id) # Cancel specific order
    self.cancel_all_orders()    # Cancel all open orders
    ```

### 5.3 Execution Modes

Set via `engine.set_execution_mode(mode)` (or pass `execution_mode` parameter in `run_backtest`):

*   **NextOpen (Default)**: Signals are matched at the Open of the *next* Bar. This is a more rigorous backtesting method, aligning with live trading logic (place order after today's close, match at tomorrow's open).
*   **CurrentClose**: Signals are matched immediately at the Close of the *current* Bar. Suitable for special strategies using closing prices for settlement, or scenarios where next-day data is unavailable.

### 5.4 Event Callbacks {: #callbacks }

AKQuant provides a callback mechanism similar to Backtrader for tracking order status and trade records.

#### 5.4.1 Order Status Callback (`on_order`)

Triggered when order status changes (e.g., from `New` to `Submitted`, or to `Filled`).

```python
from akquant import OrderStatus

def on_order(self, order):
    if order.status == OrderStatus.Filled:
        print(f"Order Filled: {order.symbol} Side: {order.side} Qty: {order.filled_quantity}")
    elif order.status == OrderStatus.Cancelled:
        print(f"Order Cancelled: {order.id}")
```

#### 5.4.2 Trade Execution Callback (`on_trade`)

Triggered when a real trade occurs. Unlike `on_order`, `on_trade` contains specific execution price, quantity, and commission information.

```python
def on_trade(self, trade):
    print(f"Trade Execution: {trade.symbol} Price: {trade.price} Qty: {trade.quantity} Comm: {trade.commission}")
```

### 5.5 Account & Position Query

In addition to `get_position`, you can query more account information:

*   **`self.equity`**: Current account equity (Cash + Market Value of Positions).
*   **`self.get_trades()`**: Get all historical closed trades.
*   **`self.get_open_orders()`**: Get current open orders.
*   **`self.get_available_position(symbol)`**: Get available position (considering T+1 rule).

## 6. Risk Management

AKQuant has a built-in Rust-level risk manager that can simulate exchange or broker risk control rules during backtesting.

```python
from akquant import RiskConfig

# Set after Engine initialization
risk_config = RiskConfig()
risk_config.active = True
risk_config.max_order_value = 1_000_000.0  # Max 1 million per order
risk_config.max_position_size = 5000       # Max 5000 shares per symbol
risk_config.restricted_list = ["ST_STOCK"] # Blacklist (Symbol)

engine.risk_manager.config = risk_config # Apply config
```

If an order violates risk rules, functions like `self.buy()` will return `None` or the generated order status will be directly `Rejected`, and the reason will be recorded in the logs.

## 6. Using High-Performance Indicators {: #indicatorset }

AKQuant includes commonly used technical indicators built into the Rust layer. They use Incremental Calculation to avoid repeated full recalculations, resulting in extremely high performance.

Supported Indicators: `SMA`, `EMA`, `MACD`, `RSI`, `BollingerBands`, `ATR`.

### 7.1 Registration and Usage

AKQuant supports **Auto-Discovery**. You can simply assign indicators to `self` in `__init__`, and they will be registered automatically.

```python
from akquant import Strategy
from akquant.indicators import SMA, RSI

class IndicatorStrategy(Strategy):
    def __init__(self):
        # Method 1: Auto-Discovery (Recommended)
        # Assign to self.xxx, automatically registered and calculated
        self.sma20 = SMA(20)
        self.rsi14 = RSI(14)

    def on_start(self):
        self.subscribe("AAPL")

        # Method 2: Manual Registration
        # self.register_indicator("sma20", SMA(20))

    def on_bar(self, bar: Bar):
        # Access value via property
        if bar.close > self.sma20.value:
            self.buy(bar.symbol, 100)

        # Or get historical value
        # val = self.sma20.get_value(bar.symbol, bar.timestamp)
```

## 7. Strategy Cookbook

### 7.1 Trailing Stop

```python
class TrailingStopStrategy(Strategy):
    def __init__(self):
        self.highest_price = 0.0
        self.trailing_percent = 0.05 # 5% trailing stop

    def on_bar(self, bar):
        pos = self.get_position(bar.symbol)

        if pos > 0:
            # Update highest price
            self.highest_price = max(self.highest_price, bar.high)

            # Check drawdown
            drawdown = (self.highest_price - bar.close) / self.highest_price
            if drawdown > self.trailing_percent:
                print(f"Trailing Stop Triggered: High {self.highest_price}, Current {bar.close}")
                self.close_position(bar.symbol)
                self.highest_price = 0.0 # Reset
        else:
            # Entry logic (Example)
            if bar.close > 100:
                self.buy(bar.symbol, 100)
                self.highest_price = bar.close # Initialize highest price
```

### 7.2 Intraday Exit

```python
class IntradayStrategy(Strategy):
    def on_bar(self, bar):
        # Assuming bar.timestamp is nanosecond timestamp
        # Convert to datetime (requires import datetime)
        dt = datetime.fromtimestamp(bar.timestamp / 1e9)

        # Force exit at 14:55 daily
        if dt.hour == 14 and dt.minute >= 55:
            if self.get_position(bar.symbol) != 0:
                self.close_position(bar.symbol)
            return

        # Other trading logic...
```

### 7.3 Multi-Asset Rotation {: #multi-asset }

```python
class RotationStrategy(Strategy):
    def on_bar(self, bar):
        # Note: on_bar is triggered for each symbol
        # If cross-sectional comparison is needed, it is recommended to process in on_timer or after collecting all bars
        # This shows simple independent processing
        pass

    def on_timer(self, payload):
        # Assume a daily timer is registered
        # Get current prices of all subscribed symbols
        scores = {}
        # Actually should iterate over watchlist or subscribed symbols
        # Note: self.ctx.positions contains current positions, but we might want to check all watched symbols
        for symbol in self.ctx.positions.keys():
             hist = self.get_history(20, symbol)
             scores[symbol] = hist[-1] / hist[0] # 20-day momentum

        # Sort and rebalance...
```

## 8. Mixed Asset Backtesting Configuration

AKQuant supports mixed trading of multiple assets such as stocks, futures, and options within the same strategy. Different assets usually have different attributes (e.g., contract multiplier, margin ratio, tick size).

Using `InstrumentConfig` allows you to conveniently configure these attributes for each instrument.

### 8.1 Configuration Steps

1.  **Prepare Data**: Prepare data (DataFrame or CSV) for each instrument.
2.  **Create Config**: Use `InstrumentConfig` to define parameters for non-stock assets.
3.  **Run Backtest**: Pass the configuration to the `instruments_config` parameter of `run_backtest`.

### 8.2 Configuration Example

Suppose we want to backtest a portfolio containing "Stock A" and "Stock Index Futures IF":

```python
from akquant import InstrumentConfig, run_backtest

# 1. Define Futures Configuration
future_config = InstrumentConfig(
    symbol="IF2301",          # Instrument Symbol
    asset_type="FUTURES",     # Asset Type: STOCK, FUTURES, OPTION
    multiplier=300.0,         # Contract Multiplier (300 per point)
    margin_ratio=0.1,         # Margin Ratio (10%)
    tick_size=0.2             # Tick Size
)

# 2. Run Backtest
# Note: Unconfigured instruments (e.g., STOCK_A) will use default parameters (Stock, Multiplier 1, Margin 100%)
run_backtest(
    data=data_dict,
    strategy=MyStrategy,
    instruments_config=[future_config], # Pass config list
    # ...
)
```

For detailed code, please refer to the [Mixed Asset Backtest Example](examples.md#mixed-asset).
