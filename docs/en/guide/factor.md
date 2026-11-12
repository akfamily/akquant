# Factor Expression Engine

AKQuant includes a high-performance factor expression engine (`akquant.factor`) that allows users to define complex Alpha factors using concise string formulas. It is built on **Polars** (Rust), providing extreme performance and parallel processing capabilities.

## Key Features

*   **High Performance**: Based on Polars Lazy API, optimizing query plans and utilizing Rust multi-threaded execution.
*   **Concise Syntax**: Uses WorldQuant Alpha101-like syntax, e.g., `Rank(Ts_Mean(Close, 5))`.
*   **Future-Data Prevention**: Encapsulated time-series operators (like `Ts_Mean`) handle windows and shifts automatically, reducing the risk of look-ahead bias.
*   **Automatic Alignment**: The engine automatically handles panel data alignment and grouping (Group By Symbol/Date).

## Design Philosophy

AKQuant's factor engine adopts the classic **DSL (Domain Specific Language)** design pattern, converting user business logic into an underlying computation graph via AST Parsing and Operator Mapping. This design is widely used in quantitative finance:

1.  **WorldQuant BRAIN**: As the pioneer of Formulaic Alpha, its Alpha Compiler defined the industry standard for operator naming (e.g., `Ts_Mean`, `Rank`). AKQuant follows this convention to lower the learning curve for users.
2.  **DolphinDB**: A high-performance time-series database that uses `streamEngineParser` to parse similar factor expressions and build unified stream/batch processing pipelines.
3.  **Microsoft Qlib**: An AI-oriented quantitative platform with an Expression Engine supporting string syntax like `Ref($close, 1)`, leveraging caching and vectorized computation for efficiency.
4.  **Zipline**: While primarily using Python objects, its Pipeline API's concept of building a dependency graph aligns with AKQuant's use of Polars Lazy Execution to construct computation graphs.

By standing on the shoulders of giants and integrating the modern **Polars (Rust)** high-performance engine, AKQuant achieves factor calculation efficiency close to C++ within Python, while maintaining code simplicity and flexibility.

## Quick Start

### 1. Prepare Data

The engine uses `ParquetDataCatalog` by default. To ensure the factor engine works correctly, you need to organize your data in a panel format and ensure it includes the following key columns:

*   **`symbol`** (String): Ticker symbol, used to distinguish different assets (for group-by operations).
*   **`date`** (Datetime): Date or timestamp, used for time-series sorting and alignment.
*   **OHLCV Fields**: Such as `open`, `high`, `low`, `close`, `volume`, etc., used for factor calculation.

**Example: Building Simulation Data and Writing**

```python
import pandas as pd
import numpy as np
from akquant.data import ParquetDataCatalog

# 1. Create simulation data
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
symbols = ["AAPL", "MSFT", "GOOG"]

data_list = []
for symbol in symbols:
    # Simulate random prices
    price = 100 + np.cumsum(np.random.randn(len(dates)))
    df = pd.DataFrame({
        "date": dates,
        "symbol": symbol,
        "open": price + np.random.randn(len(dates)),
        "high": price + 5,
        "low": price - 5,
        "close": price,
        "volume": np.random.randint(1000, 10000, len(dates))
    })
    # Ensure date is datetime type
    df["date"] = pd.to_datetime(df["date"])
    data_list.append(df)

# Write to catalog
# ParquetDataCatalog.write accepts a DataFrame for a single symbol
catalog = ParquetDataCatalog("./data_catalog")

for df in data_list:
    symbol = df["symbol"].iloc[0]
    # Note: write method handles index automatically.
    # Recommended to set date as index or keep as a column.
    df.set_index("date", inplace=True)
    catalog.write(symbol, df)

print("Data preparation complete! Directory: ./data_catalog")
```

### 2. Calculate Factors

```python
from akquant.factor import FactorEngine

# Initialize engine
engine = FactorEngine(catalog)

# Run a single expression
# Returns a DataFrame with [date, symbol, factor_value]
df = engine.run("Rank(Ts_Mean(Close, 10))")
print(df.head())

# Run batch expressions
df_batch = engine.run_batch([
    "Ts_Mean(Close, 5)",
    "Rank(Volume)",
    "If(Close > Open, 1, 0)"
])
```

## Operators Reference

Expressions support variables (column names, case-insensitive, e.g., `Close`, `open`) and constants.

### Arithmetic, Comparison & Logical Operators

- Arithmetic: `+`, `-`, `*`, `/`, `**`, `%`
- Comparison: `>`, `<`, `>=`, `<=`, `==`, `!=`
- Logical: `&` (and), `|` (or), `~` (not); the keywords `and`, `or`, `not` are equivalent

```python
If((Close > Open) & (Volume > Ts_Mean(Volume, 20)), 1, 0)
If(Close > Open and Volume > Ts_Mean(Volume, 20), 1, 0)
```

Notes:

- **Parenthesize each comparison around `&` and `|`.** They bind tighter than comparisons, so `Close > Open & Volume > 0` is read as `Close > (Open & Volume) > 0` and raises an error. `and` / `or` bind looser than comparisons and need no parentheses.
- **Element-wise, no short-circuit.** `and` / `or` behave exactly like `&` / `|` and are evaluated row by row.
- **Nulls follow three-valued logic.** A comparison against a rolling window that is not yet full is null; `null & False` is `False`, `null | True` is `True`, and other combinations stay null. `If` takes its third argument when the condition is null.
- Chained comparisons such as `1 < Close < 5` are not supported; write `(1 < Close) & (Close < 5)`.

Logical operators are available since 0.3.87. On 0.3.86 and earlier, nest instead: `If(A, If(B, 1, 0), 0)`.

### Time Series Operators

Calculated independently for each symbol over time.

| Operator | Description | Example |
| :--- | :--- | :--- |
| `Ts_Mean(X, d)` | Rolling mean over past `d` days | `Ts_Mean(Close, 5)` |
| `Ts_Std(X, d)` | Rolling std dev over past `d` days | `Ts_Std(Close, 20)` |
| `Ts_Max(X, d)` | Rolling max over past `d` days | `Ts_Max(High, 10)` |
| `Ts_Min(X, d)` | Rolling min over past `d` days | `Ts_Min(Low, 10)` |
| `Ts_Sum(X, d)` | Rolling sum over past `d` days | `Ts_Sum(Volume, 5)` |
| `Ts_ArgMax(X, d)` | Days since max value over past `d` days (0=today) | `Ts_ArgMax(Close, 5)` |
| `Ts_ArgMin(X, d)` | Days since min value over past `d` days (0=today) | `Ts_ArgMin(Close, 5)` |
| `Ts_Rank(X, d)` | Rolling percentile rank (0 to 1) of current value | `Ts_Rank(Close, 5)` |
| `Ts_Corr(X, Y, d)` | Rolling correlation of X and Y | `Ts_Corr(Close, Volume, 20)` |
| `Ts_Cov(X, Y, d)` | Rolling covariance of X and Y | `Ts_Cov(Close, Open, 20)` |
| `Delay(X, d)` | Lagged value by `d` days (Ref) | `Delay(Close, 1)` |
| `Delta(X, d)` | Difference: `X(t) - X(t-d)` | `Delta(Close, 1)` |

### Cross Sectional Operators

Calculated across all symbols at the same timestamp.

| Operator | Description | Example |
| :--- | :--- | :--- |
| `Rank(X)` | Percentile rank (0 to 1) | `Rank(Ts_Mean(Close, 5))` |
| `Scale(X)` | Scale such that `sum(abs(X)) = 1` | `Scale(Close)` |

### Math & Logic Operators

| Operator | Description | Example |
| :--- | :--- | :--- |
| `Log(X)` | Natural logarithm | `Log(Volume)` |
| `Abs(X)` | Absolute value | `Abs(Return)` |
| `Sign(X)` | Sign function (1, 0, -1) | `Sign(Close - Open)` |
| `SignedPower(X, e)` | Power preserving sign | `SignedPower(Close, 2)` |
| `If(Cond, A, B)` | Conditional (If-Else) | `If(Close > Open, 1, -1)` |

### Basic Operations

Supports arithmetic `+`, `-`, `*`, `/` and comparison `>`, `<`, `>=`, `<=`, `==`, `!=`.

Example:
```
(Close - Open) / Open
```

## Advanced Tips

### 1. Alignment & Padding

The engine uses Polars `LazyFrame` for calculation. For time-series operations (e.g., `Ts_Mean`), Polars operates on data grouped by `symbol`.
**Note**: If data is missing for certain dates (e.g., trading halts), `rolling` window calculations might be based on physical rows rather than time. To ensure precision, it is recommended to reindex/pad your data against a trading calendar before writing it to the catalog.

### 2. Smart Date Column Detection

`FactorEngine` automatically detects the date/time column when loading data. If your data contains one of the following columns, it will be automatically renamed to `date` and used for alignment:
*   `date`
*   `index`
*   `datetime`
*   `__index_level_0__` (Pandas default index name)

This means you can directly use Parquet files exported via Pandas `to_parquet` (which include the index) without manually resetting the index column.

### 3. Memory Optimization

`FactorEngine` uses Lazy Evaluation:
1.  `engine.run()` does not load all data into memory immediately.
2.  Polars constructs a query plan and only reads the columns required for calculation (Projection Pushdown).
3.  For example, when calculating `Ts_Mean(Close, 5)`, the engine only reads `symbol`, `date`, and `close` columns, ignoring `open`, `high`, etc., saving significant memory.

### 4. Execution Plan & Materialization

For nested window formulas such as `Rank(Ts_Mean(Close, 5))`, the engine does not force everything into one pass. It first creates an execution plan, then runs step by step:

1. Parse the expression AST and classify operators as TS / CS / EL.
2. If it detects `CS(TS(...))` or `TS(CS(...))`, extract the inner part into an intermediate step (e.g. `_auto_var_0`).
3. Compute that intermediate column with `with_columns`, materialize it, and then run the outer step.

This keeps semantics correct across different partition dimensions and avoids ambiguous nested `over()` behavior.

### 5. Partition Semantics

Understanding partitions helps you design predictable factors:

*   **TS (Time-Series)**: grouped by `symbol`, e.g. `Ts_Mean`, `Ts_Std`, `Delay`, `Ts_Rank`.
*   **CS (Cross-Sectional)**: grouped by `date`, e.g. `Rank`, `Scale`, `Standardize`.
*   **EL (Element-wise)**: row-wise transform with no window partition, e.g. `Log`, `Abs`, `Sign`, `If`.

Recommended workflow:
*   Validate a single-layer expression first (for example `Ts_Mean` alone).
*   Then wrap with CS or logic operators to build composite formulas incrementally.

### 6. Composite Factor Examples

You can nest multiple operators to build complex Alpha factors:

*   **Price-Volume Divergence**: `Ts_Corr(Close, Volume, 20)` (Correlation between price and volume)
*   **Momentum Reversal**: `Rank(Ts_Mean(Close, 5)) - Rank(Ts_Mean(Close, 20))` (Short-term momentum minus long-term momentum)
*   **Volatility Adjusted Momentum**: `Ts_Mean(Close, 20) / Ts_Std(Close, 20)`

## FAQ

**Q: Why can `Rank(Ts_Mean(...))` be slower than a single-layer expression?**
A: This is a cross-partition nested expression (CS over TS). The engine may split execution into multiple planned steps and materialize intermediate columns to preserve correct semantics.

**Q: How can I debug a complex formula reliably?**
A: Break it into stages and validate each stage separately (inner TS, then outer CS, then logic layer). This mirrors the engine execution plan and makes mismatched assumptions easy to locate.

**Q: Is intraday data (minute bars) supported?**
A: Yes. As long as the `date` column contains timestamps. Time-series operators (e.g., `Ts_Mean`) are based on window length (number of rows). For minute bars, `d=60` represents the past 60 minute bars.

**Q: How to handle trading halts?**
A: It is recommended to pad records for halted days (using forward fill or NaN) before storing data. If the date is missing from the data, `rolling` functions will skip that date and take the previous row as `t-1`, which might not be intended behavior.

**Q: Why is my result full of NaNs?**
A:
1.  Check if window size `d` is larger than data length (Warmup period).
2.  Check if data contains many NaNs.
3.  Check if column names are correct (e.g., `Close` vs `close`; the engine auto-converts to lowercase, but `ClosePrice` won't match `close`).
