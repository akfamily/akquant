# AKQuant - High-Performance Rust/Python Quantitative Research Framework

## 1. Project Overview

**AKQuant** is a hybrid language quantitative research framework designed for high-performance backtesting. It leverages **Rust** to handle heavy computational tasks (such as event loops, numerical calculations, and memory management) while using **Python** for strategy definition, data analysis, and visualization.

## 2. Architecture Design

### 2.1 System Layering

1.  **Rust Core Layer (`akquant_core`)**:
    *   **Data Engine**: Uses `polars` (Arrow format) to manage OHLCV data, aiming for zero-copy memory mapping.
    *   **Backtest Engine**: Event-driven execution engine.
    *   **Event Bus**:
        *   Inspired by mature event-driven message bus concepts, implemented based on Rust Channel (`mpsc`).
        *   Decouples strategy, risk control, execution, and data components.
        *   Supports asynchronous flow of events like `OrderRequest` (request), `OrderValidated` (risk passed), `ExecutionReport` (execution report).
        *   Handles control events with high priority, supporting future expansion to multi-strategy parallelism or asynchronous risk checks.
    *   **Order Matching**: Simulates limit/market orders, slippage, and commissions.
    *   **Risk Control Module**: Built-in `RiskManager` supporting pre-trade risk control (T+1 available positions, fund limits, etc.).
    *   **Indicator Calculation**: Rapid calculation of metrics like Sharpe ratio and Max Drawdown.
2.  **Interface Layer (PyO3)**:
    *   Exposes Rust structs (`Engine`, `DataFeed`, `StrategyContext`) as Python classes.
    *   Handles type conversion (e.g., Rust `DataFrame` <-> Python `pandas`/`polars`).
3.  **Python User Layer**:
    *   **Strategy API**: Abstract base class inherited by users.
    *   **Data API**: Data connectors for Tushare, AKShare, and Parquet files.
    *   **Visualization**: Integration with Plotly/Matplotlib.

### 2.2 Directory Structure

```
akquant/
├── Cargo.toml          # Rust dependency management
├── pyproject.toml      # Python build system (maturin)
├── src/                # Rust source code
│   ├── lib.rs          # PyO3 entry point
│   ├── model/          # Data models (Order, Trade, Instrument, Bar, etc.)
│   ├── data.rs         # Data source (DataFeed)
│   ├── engine.rs       # Core backtest engine
│   ├── event.rs        # Event definitions and bus messages
│   ├── clock.rs        # Trading clock
│   ├── execution.rs    # Exchange simulation and order matching
│   ├── market.rs       # Market rules (Fees, T+1/T+0)
│   ├── portfolio.rs    # Fund and position management
│   ├── risk.rs         # Risk management (RiskManager)
│   ├── context.rs      # Strategy interaction context
│   ├── history.rs      # Historical data management (Zero-Copy View)
│   ├── analysis.rs     # Performance metric calculation
│   └── indicators.rs   # High-performance indicator implementation
├── python/             # Python source code
│   └── akquant/
│       ├── ml/         # Machine Learning adapters
│       ├── __init__.py
│       ├── akquant.pyi # Type hint file
│       ├── backtest.py # Backtest entry
│       ├── strategy.py # Strategy base class
│       ├── indicator.py# Indicator wrapper
│       ├── optimize.py # Parameter optimization
│       └── ...         # Other helper modules
└── examples/           # Usage examples
```

### 2.3 Core Interaction Flow (Event Bus)

1.  **Strategy Layer**: Strategy calls `self.buy()`, generating an `OrderRequest` event in the Rust layer and sending it to the `event_tx` channel.
2.  **Engine Layer**:
    *   The main loop prioritizes checking the `event_rx` channel.
    *   Upon receiving `OrderRequest`, calls **Risk Module** (`RiskManager`) for checking.
    *   If risk check passes, generates `OrderValidated` event and resends to the channel.
    *   If risk check fails, generates `ExecutionReport` with rejected status and sends to the channel.
3.  **Execution Layer**:
    *   Receives `OrderValidated` event.
    *   **Simulation Mode**: Immediately matches or adds to order book, generating `ExecutionReport`.
    *   **Live Mode**: Sends order to external gateway and generates `ExecutionReport` upon receiving return.
4.  **Strategy Layer**: Strategy updates order status (`pending_orders`) and positions via callbacks.

## 3. Technology Stack Selection

*   **Rust**:
    *   `pyo3`: Generates Python bindings.
    *   `polars` / `arrow`: High-performance columnar data storage.
    *   `rayon`: Parallel processing for multi-asset backtesting.
    *   `serde`: Serialization support.
*   **Python**:
    *   `maturin`: Backend build system.
    *   `pandas` / `numpy`: User-facing data processing.
    *   `plotly`: Interactive visualization.

## 4. Development Roadmap

### Phase 1: Prototype Verification (Current)

*   [ ] Configure project build system (Maturin).
*   [ ] Define `Candle` (K-line) and `Feed` (Data Stream) structs in Rust.
*   [ ] Expose basic data loading functions to Python.
*   [ ] Return Pandas DataFrame from Rust.

### Phase 2: Core Engine

*   [ ] Implement event loop in Rust.
*   [ ] Create `Strategy` trait in Rust and wrap it for Python inheritance.
*   [ ] Basic order matching (Market/Limit orders).

### Phase 3: Analysis & Visualization

*   [ ] Implement performance metric calculation (Rust side).
*   [ ] Create visualization module (Python side).

### Phase 4: Production Ready

*   [ ] Connect to external data sources (Tushare/SQL).
*   [ ] Support parallel backtesting.
*   [ ] Refine documentation.
