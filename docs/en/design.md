# AKQuant Design and Development Guide

This document details the internal design principles, core component architecture, and extension development guide of `AKQuant`. It aims to help developers deeply understand the project structure for secondary development and functional expansion.

## 1. Project Overview

### 1.1 Design Philosophy

`AKQuant` follows these core design principles:

1.  **Core Calculation Sinking (Rust Core)**: All computationally intensive tasks (event loop, order matching, risk checks, data management, performance calculation, historical data maintenance) are implemented in the Rust layer to ensure high performance and memory safety.
2.  **Strategy Logic Floating (Python API)**: User interaction layers such as strategy writing, parameter configuration, data analysis, and machine learning model definition remain in Python, leveraging its dynamic features and rich ecosystem (Pandas, Scikit-learn, PyTorch, etc.).
3.  **Modularity & Decoupling**: Drawing from mature frameworks like `NautilusTrader`, modules such as data, execution, strategy, risk control, and machine learning are strictly separated and interact through clear interfaces (Traits).

### 1.2 Project Directory Structure

```text
akquant/
├── Cargo.toml              # Rust project dependencies and config
├── pyproject.toml          # Python project build config (Maturin)
├── Makefile                # Project management commands
├── src/                    # Rust core source (Low-level implementation)
│   ├── lib.rs              # PyO3 module entry, registers Python module
│   ├── engine.rs           # Backtest engine: drives timeline and event loop
│   ├── execution.rs        # Execution layer: simulates exchange matching logic
│   ├── market.rs           # Market layer: defines commissions, stamp tax, T+1 rules
│   ├── portfolio.rs        # Account layer: manages funds, positions, and available positions
│   ├── data.rs             # Data layer: manages Bar/Tick data streams
│   ├── analysis.rs         # Analysis layer: calculates performance metrics (Sharpe, Drawdown)
│   ├── context.rs          # Context: data snapshot for Python callbacks
│   ├── clock.rs            # Clock module: unified time management
│   ├── event.rs            # Event system: defines internal system events
│   ├── history.rs          # Historical data: efficient ring buffer management
│   ├── indicators.rs       # Technical indicators: Rust native implementation (e.g., SMA)
│   └── model/              # Data model: defines basic data structures
│       ├── mod.rs          # Model module definition
│       ├── order.rs        # Order and Trade
│       ├── instrument.rs   # Instrument information
│       ├── market_data.rs  # Market data (Bar, Tick)
│       ├── timer.rs        # Timer events
│       └── types.rs        # Basic enums (Side, Type, ExecutionMode)
├── python/
│   └── akquant/            # Python package source (User Interface)
│       ├── __init__.py     # Exports public API
│       ├── strategy.py     # Strategy base class: encapsulates context, provides ML training & trading interface
│       ├── backtest.py     # BacktestResult analysis & DataFrame conversion
│       ├── config.py       # Config definitions: BacktestConfig, StrategyConfig, RiskConfig
│       ├── risk.py         # Risk config adapter layer
│       ├── data.py         # Data loading & catalog service (DataCatalog)
│       ├── sizer.py        # Sizer base class: provides various position sizing implementations
│       ├── indicator.py    # Python indicator interface
│       ├── optimize.py     # Parameter optimization tool
│       ├── plot.py         # Plotting tool
│       ├── log.py          # Logging module
│       ├── utils.py        # Utility functions
│       ├── ml/             # Machine Learning framework (New)
│       │   ├── __init__.py
│       │   └── model.py    # QuantModel, SklearnAdapter, ValidationConfig
│       └── akquant.pyi     # Type hint file (IDE completion support)
├── tests/                  # Test cases
└── examples/               # Example code
    ├── benchmark_akquant_multi.py  # Multi-threaded Benchmark
    ├── ml_framework_demo.py        # ML framework basic example
    ├── ml_walk_forward_demo.py     # Walk-forward training example
    ├── optimization_demo.py        # Parameter optimization example
    └── plot_demo.py                # Plotting example
```

## 2. Core Component Architecture Details

### 2.1 Data Model Layer (`src/model/`)

To ensure cross-language interaction performance and type safety, core data structures are defined in Rust and exported.

*   **`types.rs`**:
    *   `ExecutionMode`: `CurrentClose` (Match on current close, i.e., Cheat-on-Close) vs `NextOpen` (Match on next open, more realistic).
    *   `OrderSide`: `Buy` / `Sell`.
    *   `OrderType`: `Market`, `Limit`.
    *   `TimeInForce`: `Day`, `GTC`, `IOC`/`FOK`.
*   **`instrument.rs`**: `Instrument` contains `multiplier` (contract multiplier) and `tick_size`.
*   **`market_data.rs`**: `Bar` (OHLCV) and `Tick` (Last Price/Volume).

### 2.2 Execution Layer (`src/execution.rs`)

`ExchangeSimulator` is the core of backtest accuracy, responsible for simulating exchange matching logic.

*   **Matching Mechanism**:
    *   **Limit Order**: Buy requires `Low <= Price`, Sell requires `High >= Price`.
    *   **Market Order**: Matches at `Close` or `Open` based on `ExecutionMode`.
*   **Trigger Mechanism**: Supports `trigger_price` (Stop Loss/Take Profit orders).

### 2.3 Market Rule Layer (`src/market.rs`)

Isolates rules of different markets via `MarketModel` Trait. Currently built-in `ChinaMarket` (A-share market rules):

*   **Commission Calculation**: Supports stocks (Stamp Tax, Transfer Fee, Commission) and Futures (Per Lot or Per Amount).
*   **Trading Restrictions**: Strict T+1 (Stocks) and T+0 (Futures) available position management.

### 2.4 Risk Control Layer (`src/risk.rs`)

`RiskManager` is independent of the execution layer, intercepting every order:

*   **Check Rules**: Restricted list, max single order size/value, max position ratio.
*   **Configuration**: Python side `RiskConfig` automatically injected into Rust engine.

### 2.5 Account Layer (`src/portfolio.rs`)

`Portfolio` struct maintains account status:

*   `cash`: Available funds.
*   `positions`: Total positions.
*   `available_positions`: Sellable positions (Core of T+1 logic).
*   **Equity Calculation**: Real-time Mark-to-Market calculation.

### 2.6 Engine Layer (`src/engine.rs` & `src/history.rs`)

`Engine` is the system driver:

*   **Event Loop**: Consumes `Bar` or `Tick` events.
*   **History Data Management**: `Engine` internally maintains a `History` module, an efficient ring buffer storing data for the last N bars, allowing strategies fast access via `get_history` without accumulating data on the Python side.
*   **Day Cut Processing**: Triggers T+1 unlocking, expired order cleanup.

### 2.7 Analysis Layer (`src/analysis.rs`)

Follows standard PnL calculation: `Gross PnL`, `Net PnL`, `Commission`.

### 2.8 Python Abstraction Layer (`python/akquant/`)

*   **`Strategy` (`strategy.py`)**:
    *   **History Data Access**:
        *   `set_history_depth(depth)`: Enable Rust side history recording.
        *   `get_history(count)` / `get_history_df(count)`: Get OHLCV data for the last N bars (Numpy/DataFrame).
    *   **ML Integration**:
        *   `set_rolling_window(train, step)`: Configure rolling training parameters.
        *   `on_train_signal(context)`: Periodically trigger model training.
        *   `prepare_features(df)`: Feature engineering interface.
*   **`BacktestResult` (`backtest.py`)**:
    *   Encapsulates `BacktestResult` returned by Rust.
    *   Provides convenient properties like `metrics_df`, `daily_positions_df`.
*   **`Sizer` (`sizer.py`)**: Position sizing base class.

### 2.9 Machine Learning Framework (`python/akquant/ml/`)

`AKQuant` provides a standardized ML interface aimed at simplifying the "Rolling Train-Predict" workflow.

*   **`QuantModel` (`model.py`)**:
    *   Abstract base class for all models.
    *   Interface: `fit(X, y)`, `predict(X)`, `save(path)`, `load(path)`.
    *   **`set_validation`**: Configure Walk-forward Validation parameters (Train window, Test window, Rolling step).
*   **`SklearnAdapter`**:
    *   Encapsulates Scikit-learn style models (e.g., RandomForest, LinearRegression) to adapt to `QuantModel` interface.
*   **Workflow**:
    1.  User defines model in strategy `self.model = SklearnAdapter(RandomForestClassifier())`.
    2.  Set rolling parameters `self.set_rolling_window(train_window=250, step=20)`.
    3.  Override `prepare_features` to convert raw OHLCV to Features (X) and Labels (y).
    4.  During backtest, engine automatically triggers `on_train_signal` at specified steps, strategy retrieves history data and trains model.
    5.  Call `self.model.predict` in `on_bar` to generate signals.

## 3. Key Workflow Details

### 3.1 Backtest Main Loop & Execution Mode

`Engine::run` flow depends on `ExecutionMode`:

*   **NextOpen**: Recommended mode. Bar Close generates signal -> Next Bar Open matches.
*   **CurrentClose**: Simplified mode. Bar Close generates signal -> Current Bar Close matches (Cheat-on-Close).

### 3.2 Order Lifecycle

Signal -> Creation -> Submission -> Risk Check (Rust) -> Matching (Rust) -> Settlement (Rust) -> Reporting.

## 4. Extension Development Guide

### 4.1 How to Add New Order Types

1.  `src/model/types.rs`: Add enum.
2.  `src/model/order.rs`: Update struct.
3.  `src/execution.rs`: Implement matching logic.
4.  `akquant.pyi`: Update type hints.

### 4.2 How to Customize Indicators

1.  **Python Side (Rapid Prototype)**: Inherit `akquant.Indicator`, calculate in `on_bar`.
2.  **Rust Side (High Performance)**:
    *   Implement `Indicator` Trait in `src/indicators.rs`.
    *   Export to Python via `#[pyclass]`.

### 4.3 How to Access New Data Sources

Convert data to pandas DataFrame, construct `akquant.Bar` object list, call `engine.add_feed`.

## 5. Rust & Python Interaction Notes

*   **GIL**: Rust only acquires GIL when calling back into Python; computationally intensive tasks release GIL (implementation dependent, currently mainly single-threaded model).
*   **Data Copying**: Minimize large-scale data transfer between Python and Rust. `get_history` returns Numpy view or copy, far more efficient than Python list.
