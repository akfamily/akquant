<p align="center">
  <img src="../assets/akquant-logo.svg" alt="AKQuant Logo" width="800" />
</p>

---

**AKQuant** is a high-performance quantitative research framework built on **Rust** and **Python**. It combines the extreme performance of Rust with the ease of use of Python, providing powerful backtesting and research tools for quantitative traders.

The latest version features a modular design, independent portfolio management, advanced order type support, and convenient data loading and caching mechanisms.

## Core Features

*   **High-Performance Core**: The core backtesting engine is written in Rust and exposed to Python via PyO3.
    *   **Practical Note**: The Rust core is designed to reduce Python-side overhead in event-driven backtests. Actual runtime depends on strategy logic, data size, callback frequency, and the execution environment.
    *   **Efficient Data Path**: Data ingest (`add_arrays`) borrows NumPy buffers into the engine with zero copy. Historical reads (`get_history`) return a safe snapshot copy of the Rust rolling buffer (the window is small, so the cost is negligible); use `get_history_multi` to fetch several fields in a single FFI crossing.
*   **Modular Architecture**:
    *   **Engine**: Event-driven core matching engine using BinaryHeap for event queue management.
    *   **Clock**: Trading clock precisely managing TradingSessions and time flow.
    *   **Portfolio**: Independent portfolio management supporting real-time equity calculation.
    *   **MarketModel**: Pluggable market models with built-in A-share T+1 and Futures T+0 rules.
        *   **T+1 Strict Risk Control**: For stocks/funds, strictly enforces T+1 available position checks to prevent day trading (unless configured as T+0 market).
        *   **Available Position Management**: Automatically maintains `available_positions` and deducts frozen quantities from pending sell orders to prevent overselling.
*   **Event System**:
    *   **Timer**: Supports `schedule(timestamp, payload)` to register timed events, triggering `on_timer` callbacks for complex intraday timing logic.
*   **Risk Control System (New)**:
    *   **Independent Interception Layer**: Built-in `RiskManager` intercepts violating orders directly at the Rust engine layer.
    *   **Available Position Check**: Checks available positions (Available - Pending Sell) in real-time before ordering.
    *   **Flexible Configuration**: `RiskConfig` allows configuring max order amount, max position ratio, blocklists, etc.
*   **Data Ecosystem**:
    *   **Streaming CSV (New)**: Supports streaming loading of large CSV files (`DataFeed.from_csv`) to minimize memory usage.
    *   **Pandas Integration**: Supports loading Pandas DataFrame directly.
    *   **Smart Caching**: Supports local data caching (Pickle) to avoid repeated downloads and accelerate strategy iteration.
*   **Machine Learning (New)**:
    *   **ML Framework**: Built-in high-performance ML training framework supporting Walk-forward Validation.
    *   **Adapter Pattern**: Unifies Scikit-learn and PyTorch interfaces.
    *   **📖 [Machine Learning Guide](advanced/ml.md)**: Learn how to build AI-driven strategies.
*   **Flexible Configuration**:
    *   **StrategyConfig**: Global strategy configuration.
    *   **Three-Axis Fill Policy**: `fill_policy` uses `price_basis`, `bar_offset`, and `temporal` for unified execution semantics.
*   **Rich Analysis Tools**:
    *   **PerformanceMetrics**:
        *   **Return**: Total Return, Annualized Return, Alpha, Win Rate.
        *   **Risk**: Max Drawdown, Sharpe Ratio, Sortino Ratio, Ulcer Index, UPI.
        *   **Fit**: Equity R².
    *   **TradeAnalyzer**: Detailed trade statistics including win rate, PnL ratio, max consecutive PnL, etc.
*   **Simulation Enhancements**:
    *   **Slippage Model**: Supports Fixed and Percent slippage models.
    *   **Volume Limit**: Supports limiting order fill quantity by bar volume ratio and partial fills.

## Why Choose AKQuant?

AKQuant aims to solve the performance bottlenecks of traditional Python backtesting frameworks (like Backtrader) and the high development barriers of pure C++/Rust frameworks. We have achieved breakthroughs in five core dimensions through our hybrid architecture:

### 1. Extreme Performance: Rust Core + Python Ecosystem
*   **Hybrid Architecture**: The core computation layer (matching, capital, risk control) is written in **Rust** and exposed to Python via PyO3.
*   **Efficient Data Path**: Bulk data ingest (`add_arrays`) borrows NumPy buffers into the engine with zero copy. Strategy-facing reads such as `get_history` / `ctx.positions` return **safe snapshot copies** (small windows, negligible cost) rather than views into the mutable engine state; batch multi-field reads with `get_history_multi`.
*   **Performance Positioning**: The Rust core can reduce interpreter overhead in some event-driven workloads, but measured speedups are workload-dependent and should be validated with your own strategy and environment.
*   **Incremental Calculation**: Internal indicator calculations use incremental update algorithms instead of full recalculation, suitable for ultra-long history backtesting.

### 2. Machine Learning First
*   **Built-in Training Framework**: Unlike traditional frameworks that only support simple technical indicators, AKQuant features a built-in full ML Pipeline.
*   **Walk-forward Validation**: Natively supports rolling window training (Walk-forward) to effectively prevent look-ahead bias and overfitting.
*   **Adapter Pattern**: Provides unified adapters (`QuantModel`) for Scikit-learn and PyTorch, allowing AI model integration into strategies with just a few lines of code.
*   **Feature Engineering**: `DataFeed` supports dynamic feature calculation, facilitating integration with Talib or Pandas for feature preprocessing.

### 3. Precise and Flexible Event-Driven Engine
*   **Precise Simulation**: Featuring a precise time flow model and order lifecycle management.
*   **Complex Order Support**: Supports Market, Limit, Stop, TakeProfit, and other order types.
*   **Multi-Asset Mixing**: Supports mixed backtesting of stocks, futures, ETFs, etc., with independent fee, slippage, and trading session configurations for each asset.
*   **Intraday Scheduled Tasks**: Supports `schedule` to register intraday timed events (e.g., close positions daily at 14:50), offering more flexibility than simple `on_bar`.

### 4. Production-Grade Risk Control and Live Trading
*   **Built-in Risk Manager**: The engine layer includes a `RiskManager` supporting hard limits on capital, position ratios, blocklists, etc., to prevent runaway strategies.
*   **Seamless Live Trading Switch**: Strategy code is decoupled from live trading interfaces. Theoretically, switching to live trading only requires replacing `Broker` and `DataFeed` adapters (live interface under development).
*   **Data Aggregator**: `DataFeed` supports multi-source aggregation, handling data alignment issues for different frequencies.

### 5. Ultimate Developer Experience
*   **LLM Friendly**: Clear code structure, detailed documentation, and optimized Type Hints facilitate strategy writing with Copilot or GPT.
*   **Dual-Style API**: Supports both **Class-based** and **Functional (Zipline-style)** strategy writing styles to suit different user habits.
*   **Strict Type Checking**: Core logic is strictly checked by the Rust compiler, and Python code is checked via `mypy`, minimizing runtime errors.

## Installation

For detailed installation steps, please refer to the **[Installation Guide](start/installation.md)**.

## Quick Start

### 1. Quick Backtest using helper (Recommended)

`AKQuant` provides a convenient entry point `run_backtest` similar to Zipline.

```python
import pandas as pd
import numpy as np
from akquant import Strategy, run_backtest

# 1. Prepare data (example using random data)
# Real scenario: pd.read_csv("data.csv")
def generate_data():
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    n = len(dates)
    price = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
    return pd.DataFrame({
        "date": dates,
        "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
        "volume": 10000,
        "symbol": "600000"
    })

# 2. Define Strategy
class MyStrategy(Strategy):
    def on_bar(self, bar):
        # Simple strategy logic (example)
        # For real backtests, using IndicatorSet for vectorized calculation is recommended
        position = self.get_position(bar.symbol)
        if position == 0:
            self.buy(symbol=bar.symbol, quantity=100)
        elif position > 0:
            self.sell(symbol=bar.symbol, quantity=100)

# 3. Run Backtest
df = generate_data()
result = run_backtest(
    strategy=MyStrategy,  # Pass class or instance
    data=df,              # Explicitly pass data
    symbols="600000",      # SPD Bank
    initial_cash=500_000.0,       # Initial cash
    commission_rate=0.0003     # 0.03% commission
)

# 4. View Results
print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")

# 5. Get Detailed Data (DataFrame)
# Performance metrics table
print(result.metrics_df)
# Trade record table
print(result.trades_df)
# Daily position table
print(result.positions_df)
```

### 2. Functional API (Zipline Style)

If you are used to Zipline or Backtrader's functional style, you can also use:

```python
from akquant import run_backtest

def initialize(ctx):
    ctx.stop_loss_pct = 0.05

def on_bar(ctx, bar):
    position = ctx.get_position(bar.symbol)
    if position == 0:
        ctx.buy(symbol=bar.symbol, quantity=100)
    elif position > 0:
        ctx.sell(symbol=bar.symbol, quantity=100)

run_backtest(
    strategy=on_bar,
    initialize=initialize,
    data=df, # Use data generated above
    symbols="600000"
)
```

### 3. Using Custom Factors

AKQuant supports passing any number of custom numeric fields (such as factors, signals, etc.) in the `DataFrame`, which can be accessed via the `bar.extra` dictionary in `on_bar`.

```python
import pandas as pd
import numpy as np
from akquant import Strategy, run_backtest

# 1. Prepare data
def generate_data():
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    n = len(dates)
    price = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
    return pd.DataFrame({
        "date": dates,
        "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
        "volume": 10000,
        "symbol": "600000"
    })

df = generate_data()

# 2. Add custom factors (must be numeric)
df["momentum"] = df["close"] / df["open"]       # Factor 1
df["volatility"] = df["high"] - df["low"]       # Factor 2
df["sentiment_score"] = np.random.rand(len(df)) # Factor 3

# 3. Access these fields simultaneously in the strategy
class MyStrategy(Strategy):
    def on_bar(self, bar):
        # Access by key name (returns float type)
        mom = bar.extra.get("momentum", 0.0)
        vol = bar.extra.get("volatility", 0.0)
        score = bar.extra.get("sentiment_score", 0.0)

        # Comprehensive judgment
        if mom > 1.02 and score > 0.8:
            self.buy(bar.symbol, 100)

# 4. Run Backtest
run_backtest(strategy=MyStrategy, data=df, symbols="600000")
```

For more examples, please refer to the `examples/` directory.

## Parameter Modeling Entry

*   Optimization guide (recommended workflow): [Parameter-Model-Driven Optimization](guide/optimization.md)
*   Examples guide (UI-driven setup): [UI-Driven Strategy Parameterization (Inline Param Fields)](guide/examples.md)
*   API best practices (frontend-backend consistency): [UI-Driven Parameter Input](reference/api.md)
*   Example script entry: [examples/02_parameter_optimization.py](https://github.com/akfamily/akquant/blob/main/examples/02_parameter_optimization.py)

## Quick Links

*   Phase-5 migration FAQ (quickstart): [Phase-5 Migration FAQ](start/quickstart.md)
*   Full compatibility notes (API reference): [Compatibility & Migration Notes](reference/api.md)
*   Understand AKQuant time semantics (quant basics): [AKQuant Time and Timezones](guide/quant_basics.md)
*   Timezone FAQ and data preparation advice (advanced): [Timezone Handling Guide](advanced/timezone.md)
*   Multi-strategy checklist (advanced): [Multi-Strategy Guide](advanced/multi_strategy_guide.md)
*   Dynamic strategy loading guide (advanced): [Runtime Config Guide](advanced/runtime_config.md)
*   Dynamic strategy loading example (examples): [44_strategy_source_loader_demo.py](https://github.com/akfamily/akquant/blob/main/examples/44_strategy_source_loader_demo.py)
*   Dynamic strategy loading textbook chapter (examples/textbook): [ch15_strategy_loader.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_strategy_loader.py)
*   Warm-start state persistence and resume: [Warm Start Guide](advanced/warm_start.md)

## Strategy Playbook Entry

*   Cross-section strategy checklist: [Cross-Section Strategy Playbook Checklist](guide/cross_section_checklist.md)
*   Primary custom indicator entry: [Custom Indicator Guide](guide/custom_indicator.md)
*   Indicator registration and mode selection: [Strategy Guide](guide/strategy.md)
