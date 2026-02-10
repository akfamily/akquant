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

Here are some common quantitative strategy implementations that you can use directly in your projects.

### 3.1 Dual Moving Average Strategy

Classic trend following strategy using the crossover of long and short period moving averages to generate buy/sell signals. This example uses high-performance incremental indicators `aq.SMA` implemented in Rust.

```python
import akquant as aq

class DualSMAStrategy(aq.Strategy):
    def __init__(self, short_window=5, long_window=20):
        # Use Rust implemented high-performance incremental SMA indicators
        self.sma_short = aq.SMA(short_window)
        self.sma_long = aq.SMA(long_window)

    def on_bar(self, bar: aq.Bar):
        # Update indicators
        short_val = self.sma_short.update(bar.close)
        long_val = self.sma_long.update(bar.close)

        # Return if indicators are not ready
        if short_val is None or long_val is None:
            return

        position = self.get_position(bar.symbol)

        # Golden Cross (Short MA crosses above Long MA) -> Buy
        if short_val > long_val and position == 0:
            self.buy(bar.symbol, 100)

        # Death Cross (Short MA crosses below Long MA) -> Sell to Close
        elif short_val < long_val and position > 0:
            self.sell(bar.symbol, 100)
```

### 3.2 RSI Mean Reversion Strategy

Uses Relative Strength Index (RSI) to determine overbought and oversold conditions. This example demonstrates how to use `get_history_df` combined with pandas to calculate complex indicators.

```python
import akquant as aq
import pandas as pd
import numpy as np

class RSIStrategy(aq.Strategy):
    def __init__(self, period=14, buy_threshold=30, sell_threshold=70):
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        # Set sufficient history depth to calculate RSI (period + warmup data)
        self.set_history_depth(period + 20)

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI using pandas."""
        delta = prices.diff()
        # Simple RSI algorithm
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def on_bar(self, bar: aq.Bar):
        # Get historical closing price DataFrame
        history = self.get_history_df(self.period + 20, bar.symbol)

        # Return if data is insufficient
        if len(history) < self.period + 1:
            return

        # Calculate RSI
        rsi_series = self.calculate_rsi(history['close'])
        current_rsi = rsi_series.iloc[-1]

        if np.isnan(current_rsi):
            return

        position = self.get_position(bar.symbol)

        # RSI < 30 (Oversold) -> Buy
        if current_rsi < self.buy_threshold and position == 0:
            self.buy(bar.symbol, 100)

        # RSI > 70 (Overbought) -> Sell
        elif current_rsi > self.sell_threshold and position > 0:
            self.sell(bar.symbol, 100)
```

### 3.3 Bollinger Bands Strategy

Uses Bollinger Bands upper and lower rails as trading signals. This example demonstrates how to calculate statistical indicators via pandas.

```python
import akquant as aq
import pandas as pd

class BollingerStrategy(aq.Strategy):
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
        # Set history depth
        self.set_history_depth(window + 5)

    def on_bar(self, bar: aq.Bar):
        # Get historical data
        history = self.get_history_df(self.window, bar.symbol)
        if len(history) < self.window:
            return

        # Calculate Bollinger Bands
        close_prices = history['close']
        ma = close_prices.mean()
        std = close_prices.std()
        upper_band = ma + self.num_std * std
        lower_band = ma - self.num_std * std

        position = self.get_position(bar.symbol)
        current_price = bar.close

        # Price breaks below lower band -> Oversold reversal signal -> Buy
        if current_price < lower_band and position == 0:
            self.buy(bar.symbol, 100)
        # Price breaks above upper band -> Overbought reversal signal -> Sell
        elif current_price > upper_band and position > 0:
            self.sell(bar.symbol, 100)
```

### 3.4 Mixed Asset Backtest {: #mixed-asset }

Demonstrates how to trade stocks and futures in the same strategy, using `InstrumentConfig` to configure futures parameters.

```python
import akquant as aq
from akquant import InstrumentConfig
import pandas as pd
import numpy as np

# 1. Prepare data (Mock data)
def create_dummy_data(symbol, start_date, n_bars, price=100.0):
    dates = pd.date_range(start_date, periods=n_bars, freq="B")
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
        # Simple logic: Buy on first two bars
        if self.count < 2:
            print(f"[{bar.timestamp}] Buying {bar.symbol}")
            self.buy(bar.symbol, 1)
        self.count += 1

# 2. Generate data
df_stock = create_dummy_data("STOCK_A", "2023-01-01", 100, 100.0)
df_future = create_dummy_data("FUTURE_B", "2023-01-01", 100, 3500.0)
data = {"STOCK_A": df_stock, "FUTURE_B": df_future}

# 3. Configure futures parameters
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

Although AKQuant's core matching engine does not natively support OCO (One-Cancels-Other) or Bracket Order types yet, you can easily implement these advanced logic via strategy callback functions (`on_trade`, `on_order`).

### 4.1 OCO and Bracket Order

A Bracket Order typically consists of three parts:

1.  **Entry Order**: Initial order (e.g., Breakout Buy).
2.  **Stop Loss**: Protective sell order.
3.  **Take Profit**: Profit-taking sell order.

The Stop Loss and Take Profit form an **OCO** group: if one executes, the other should be automatically cancelled immediately.

**Core Logic Implementation:**

```python
def on_trade(self, trade):
    # 1. Entry order filled -> Immediately place SL and TP
    if trade.order_id == self.entry_order_id:
        # Place Stop Loss (Stop Market)
        self.stop_loss_id = self.sell(
            trade.symbol, trade.quantity,
            trigger_price=trade.price * 0.98, # Stop Price
            price=None # None means Market Sell after trigger
        )

        # Place Take Profit (Limit Sell)
        self.take_profit_id = self.sell(
            trade.symbol, trade.quantity,
            price=trade.price * 1.05 # Take Profit Price
        )

    # 2. Stop Loss filled -> Cancel Take Profit
    elif trade.order_id == self.stop_loss_id:
        self.cancel_order(self.take_profit_id)

    # 3. Take Profit filled -> Cancel Stop Loss
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
