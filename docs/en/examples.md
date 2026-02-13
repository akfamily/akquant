# Examples Collection

## 1. Basic Examples

*   [Quick Start](quickstart.md): Complete workflow covering manual data backtesting and AKShare data backtesting.
*   [Simple SMA Strategy](strategy_guide.md#class-based): Demonstrates how to write a strategy in class style and perform simple trading logic in `on_bar`.

## 2. Advanced Examples

*   **Zipline Style Strategy**: Demonstrates how to write strategies using functional API (`initialize`, `on_bar`), suitable for users migrating from Zipline.
    *   Refer to [Strategy Guide](strategy_guide.md#style-selection).

*   **Multi-Asset Backtest**:
    *   **Futures Strategy**: Demonstrates futures backtest configuration (margin, multiplier). Refer to [Strategy Guide](strategy_guide.md#multi-asset).
    *   **Option Strategy**: Demonstrates option backtest configuration (premium, per contract fee). Refer to [Strategy Guide](strategy_guide.md#multi-asset).

*   **Vectorized Indicators**:
    *   Demonstrates how to use `IndicatorSet` to pre-calculate indicators to improve backtest speed. Refer to [Strategy Guide](strategy_guide.md#indicatorset).

## 3. Common Strategies

Here are some common quantitative strategy implementations that you can use directly in your projects. We provide detailed logic explanations for each strategy to help you understand the core concepts.

### 3.1 Dual Moving Average Strategy

**Core Concept**:
The Dual Moving Average strategy uses two moving averages (SMA) with different periods to determine market trends.
*   **Short-term SMA** (e.g., 5 days): Sensitive, closely follows price fluctuations.
*   **Long-term SMA** (e.g., 20 days): Lagging, represents the long-term trend.

**Trading Signals**:
*   **Golden Cross**: When the short-term SMA **crosses above** the long-term SMA, it indicates a strengthening short-term trend, which is a **Buy** signal.
*   **Death Cross**: When the short-term SMA **crosses below** the long-term SMA, it indicates a weakening short-term trend, which is a **Sell** signal.

This example uses high-performance incremental indicators `aq.SMA` implemented in Rust.

```python
import akquant as aq

class DualSMAStrategy(aq.Strategy):
    def __init__(self, short_window=5, long_window=20):
        # Initialize two indicators: Short SMA and Long SMA
        # Use Rust implemented high-performance incremental SMA indicators
        self.sma_short = aq.SMA(short_window)
        self.sma_long = aq.SMA(long_window)

    def on_bar(self, bar: aq.Bar):
        # 1. Update indicator status
        # The update method accepts the current closing price and returns the latest MA value
        short_val = self.sma_short.update(bar.close)
        long_val = self.sma_long.update(bar.close)

        # 2. Skip if indicator data is insufficient (e.g., cannot calculate 20-day MA at the start)
        if short_val is None or long_val is None:
            return

        # Get current position quantity
        position = self.get_position(bar.symbol)

        # 3. Generate Trading Signals

        # Golden Cross (Short MA crosses above Long MA) -> And no position -> Buy
        if short_val > long_val and position == 0:
            self.buy(bar.symbol, 100)

        # Death Cross (Short MA crosses below Long MA) -> And holding position -> Sell to Close
        elif short_val < long_val and position > 0:
            self.sell(bar.symbol, 100)
```

### 3.2 RSI Mean Reversion Strategy

**Core Concept**:
RSI (Relative Strength Index) is a momentum indicator ranging from 0 to 100, measuring the magnitude of recent price changes.
*   **Mean Reversion**: This strategy assumes that prices will not rise or fall indefinitely and will eventually revert to a normal level after excessive deviation.
*   **Oversold**: RSI below a threshold (e.g., 30) implies the recent drop is excessive, suggesting a potential rebound -> **Buy**.
*   **Overbought**: RSI above a threshold (e.g., 70) implies the recent rise is excessive, suggesting a potential pullback -> **Sell**.

This example demonstrates how to use `get_history_df` to retrieve historical data and combine it with `pandas` to calculate complex indicators.

```python
import akquant as aq
import pandas as pd
import numpy as np

class RSIStrategy(aq.Strategy):
    def __init__(self, period=14, buy_threshold=30, sell_threshold=70):
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        # IMPORTANT: Set historical data lookback depth
        # Since calculating RSI requires past N days of data, sufficient history window must be reserved
        self.set_history_depth(period + 20)

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI using pandas."""
        delta = prices.diff()
        # Simple RSI algorithm implementation
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def on_bar(self, bar: aq.Bar):
        # 1. Get historical closing price DataFrame
        # get_history_df returns data for the past N bars
        history = self.get_history_df(self.period + 20, bar.symbol)

        # Return if data is insufficient
        if len(history) < self.period + 1:
            return

        # 2. Calculate RSI
        rsi_series = self.calculate_rsi(history['close'])
        current_rsi = rsi_series.iloc[-1] # Get the latest RSI value

        if np.isnan(current_rsi):
            return

        position = self.get_position(bar.symbol)

        # 3. Trading Logic

        # RSI < 30 (Oversold) -> Expect Rebound -> Buy
        if current_rsi < self.buy_threshold and position == 0:
            self.buy(bar.symbol, 100)

        # RSI > 70 (Overbought) -> Expect Drop -> Sell
        elif current_rsi > self.sell_threshold and position > 0:
            self.sell(bar.symbol, 100)
```

### 3.3 Bollinger Bands Strategy

**Core Concept**:
Bollinger Bands consist of three lines:
*   **Middle Band**: N-day Moving Average.
*   **Upper Band**: Middle Band + K * Standard Deviation.
*   **Lower Band**: Middle Band - K * Standard Deviation.

According to statistical principles, prices have a high probability (e.g., 95%) of falling between the upper and lower bands.
*   When price **breaks below the lower band**, it is often seen as an irrational **oversold** state, and price may revert to the middle band -> **Buy**.
*   When price **breaks above the upper band**, it is often seen as an irrational **overbought** state, and price may pullback -> **Sell**.

```python
import akquant as aq
import pandas as pd

class BollingerStrategy(aq.Strategy):
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
        # Set history depth to ensure enough data for mean and std calculation
        self.set_history_depth(window + 5)

    def on_bar(self, bar: aq.Bar):
        # 1. Get historical data
        history = self.get_history_df(self.window, bar.symbol)
        if len(history) < self.window:
            return

        # 2. Calculate Bollinger Bands
        close_prices = history['close']
        ma = close_prices.mean()          # Middle Band (Mean)
        std = close_prices.std()          # Standard Deviation
        upper_band = ma + self.num_std * std # Upper Band
        lower_band = ma - self.num_std * std # Lower Band

        position = self.get_position(bar.symbol)
        current_price = bar.close

        # 3. Trading Logic

        # Price breaks below lower band -> Oversold reversal signal -> Buy
        if current_price < lower_band and position == 0:
            self.buy(bar.symbol, 100)

        # Price breaks above upper band -> Overbought reversal signal -> Sell
        elif current_price > upper_band and position > 0:
            self.sell(bar.symbol, 100)
```

### 3.4 Mixed Asset Backtest {: #mixed-asset }

**Core Concept**:
In real trading, strategies may involve multiple assets like stocks, futures, and options simultaneously. Different assets have different trading attributes:
*   **Stocks**: Usually 1 lot = 100 shares, fully paid trading.
*   **Futures**: Have **Contract Multiplier** (e.g., 1 point = $300) and **Margin Ratio** (e.g., buy contract with 10% funds).

This example demonstrates how to use `InstrumentConfig` to configure special attributes for futures and mix trade stocks and futures in the same strategy.

```python
import akquant as aq
from akquant import InstrumentConfig
import pandas as pd
import numpy as np

# 1. Prepare data (Mock data)
def create_dummy_data(symbol, start_time, n_bars, price=100.0):
    dates = pd.date_range(start_time, periods=n_bars, freq="B")
    np.random.seed(42)
    changes = np.random.randn(n_bars)
    prices = price + np.cumsum(changes)

    df = pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1, "close": prices,
        "volume": 1000, "symbol": symbol
    }, index=dates)
    return df

class TestStrategy(aq.Strategy):
    def __init__(self):
        self.count = 0

    def on_bar(self, bar: aq.Bar):
        # Simple logic: Buy stock and future respectively on first two bars
        if self.count < 2:
            print(f"[{bar.timestamp}] Buying {bar.symbol}")
            self.buy(bar.symbol, 1)
        self.count += 1

# 2. Generate data
df_stock = create_dummy_data("STOCK_A", "2023-01-01", 100, 100.0)
df_future = create_dummy_data("FUTURE_B", "2023-01-01", 100, 3500.0)
data = {"STOCK_A": df_stock, "FUTURE_B": df_future}

# 3. Configure futures parameters
# Tell backtest engine: FUTURE_B is a future, multiplier 300, margin 10%
future_config = InstrumentConfig(
    symbol="FUTURE_B",
    asset_type="FUTURES",
    multiplier=300, # Index future multiplier
    margin_ratio=0.1 # 10% Margin
)

# 4. Run
run_backtest(
    data=data,
    strategy=TestStrategy,
    instruments_config=[future_config]
)
```

## 4. Complex Orders & Risk Control {: #complex-orders }

**Core Concept**:
Advanced trading often requires precise order management.
*   **Bracket Order**: A combination of "Entry + Stop Loss + Take Profit". When you open a position (Entry), you immediately set a "Stop Loss" and "Take Profit" order for this position, bracketing the price like a pair of brackets.
*   **OCO (One-Cancels-Other)**: Refers to the relationship between the "Stop Loss" and "Take Profit" orders. If price rises and triggers Take Profit, the Stop Loss order should be automatically cancelled (since the position is closed, no need to stop loss anymore), and vice versa.

Although AKQuant's core matching engine does not natively support OCO order types yet, you can easily implement these advanced logic via strategy callback functions (`on_trade`, `on_order`).

### 4.1 OCO and Bracket Order Implementation

**Logic Flow**:
1.  **Entry**: Strategy sends an open position signal.
2.  **Trade Callback (`on_trade`)**: Once entry order is filled, immediately send two closing orders:
    *   **Stop Loss**: Sell if price drops to X (protect principal).
    *   **Take Profit**: Sell if price rises to Y (lock in profit).
3.  **Subsequent Trades**:
    *   If Stop Loss filled -> Immediately cancel Take Profit.
    *   If Take Profit filled -> Immediately cancel Stop Loss.

```python
def on_trade(self, trade):
    # 1. Entry order filled -> Immediately place SL and TP
    if trade.order_id == self.entry_order_id:
        # Place Stop Loss (Stop Market: Sell at market price after trigger)
        self.stop_loss_id = self.sell(
            trade.symbol, trade.quantity,
            trigger_price=trade.price * 0.98, # Stop Price (Cost - 2%)
            price=None # None means Market Sell after trigger
        )

        # Place Take Profit (Limit Sell: Sell at specified price)
        self.take_profit_id = self.sell(
            trade.symbol, trade.quantity,
            price=trade.price * 1.05 # Take Profit Price (Cost + 5%)
        )

    # 2. Stop Loss filled -> Cancel Take Profit
    # We have exited with stop loss, cancel the previous take profit order
    elif trade.order_id == self.stop_loss_id:
        self.cancel_order(self.take_profit_id)

    # 3. Take Profit filled -> Cancel Stop Loss
    # We have exited with profit, cancel the previous stop loss order
    elif trade.order_id == self.take_profit_id:
        self.cancel_order(self.stop_loss_id)
```

!!! tip "Parameter Optimization"
    The `stop_loss_pct` and `take_profit_pct` parameters of this strategy can be optimized via grid search using `akquant.run_optimization`.

    ```python
    from akquant import run_optimization
    from examples.complex_orders import BracketStrategy

    param_grid = {
        "stop_loss_pct": [0.01, 0.02, 0.03],
        "take_profit_pct": [0.03, 0.05, 0.08]
    }

    results = run_optimization(BracketStrategy, param_grid, data=df)
    ```

Please refer to [examples/complex_orders.py](file:///examples/complex_orders.py) for the full code.

> **Note**: Methods `buy` / `sell` / `stop_buy` / `stop_sell` all return a unique `order_id` (str). You can use this ID to precisely track the status of each order in `on_trade` and `on_order` callbacks.
