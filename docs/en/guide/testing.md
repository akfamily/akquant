# Testing Guide

To ensure the accuracy and stability of the AKQuant backtesting engine, we have established a layered testing system. This guide will help you understand how to run tests, interpret results, and add new test cases.

## Testing System Overview

AKQuant's tests are divided into the following levels:

1.  **Unit Tests**
    *   **Rust**: Run with `cargo test` to cover core algorithms (matching, fee calculation, risk control).
    *   **Python**: Run with `pytest` to cover Python interfaces, data processing, and configuration parsing.
2.  **Golden Tests / Integration Tests**
    *   Use **Synthetic Data** to simulate specific market scenarios (e.g., T+1, Limit Up/Down, Option Exercise).
    *   Compare backtest results (equity curve, trade records, metrics) strictly against a locked **Baseline**.
    *   Ensure that any algorithmic changes do not unintentionally alter backtest results.

---

## 🚀 Quick Start

### 1. Prepare Environment

Ensure you have installed development dependencies:

```bash
pip install -e ".[dev]"
```

### 2. Run All Tests

```bash
pytest
```

### 3. Run Only Golden Tests

```bash
pytest tests/golden/test_golden.py
```

---

## 🏆 Golden Tests

Golden Tests are the core of AKQuant's quality assurance. They are located in the `tests/golden/` directory.

### Directory Structure

```text
tests/golden/
├── strategies/         # Strategy code for testing (e.g., stock_t1.py)
├── data/               # Auto-generated synthetic data (Parquet format)
├── baselines/          # Locked baseline results (Standard Answers)
├── runner.py           # Test executor and comparison logic
├── gen_data.py         # Data generation script
└── test_golden.py      # Pytest entry point
```

### Included Scenarios

| Test Case | Description | Verification Points |
| :--- | :--- | :--- |
| **stock_t1** | Stock T+1 Trading | Day 1 buy -> Day 2 sell allowed; Day 3 buy -> Day 3 sell rejected. |
| **futures_margin** | Futures Margin | Initial capital sufficient for 1 lot; attempting to buy 2nd lot rejected due to insufficient margin. |
| **option_basic** | Option Basic Trading | Verify option contract buying, holding, selling (closing position), and PnL calculation. |

### How to Update Baseline?

If you modify core algorithms (e.g., improved matching logic or fee calculation) causing backtest results to change **expectedly**, the Golden Tests will fail. You need to update the baseline:

1.  **Confirm Differences**: Carefully check the failure output to ensure the differences are caused by your code changes and are as expected.
2.  **Regenerate Baseline**:
    ```bash
    python tests/golden/runner.py --generate-baseline
    ```
3.  **Submit Changes**: Commit the updated `tests/golden/baselines/` files along with your code.

### Streaming Regression Checklist

After changes to `run_backtest(..., on_event=...)`, event protocol, or streaming pipeline,
run this minimum checklist:

1.  **Semantic parity**: Stream and non-stream runs keep identical semantics
    (equity length, trade count, key metrics).
2.  **Event protocol correctness**: Event sequence starts with `started`, ends with
    `finished`, and `seq` is monotonic.
3.  **Sampling and backpressure behavior**: Under high-frequency data, sampled
    `progress`/`equity` volumes are controlled while critical events are preserved.
4.  **Callback exception policy**:
    *   `stream_error_mode="continue"`: callback exceptions do not stop backtest,
        and `finished` payload includes `callback_error_count`.
    *   `stream_error_mode="fail_fast"`: first callback exception fails immediately.
5.  **Parameter validation**: `stream_progress_interval`, `stream_equity_interval`,
    `stream_batch_size`, and `stream_max_buffer` must be positive integers, and
    `stream_error_mode` must be valid.

Recommended commands:

```bash
# Streaming and engine-related unit tests
pytest -q tests/test_engine.py

# Golden suite (includes stream/non-stream consistency checks)
pytest -q tests/golden/test_golden.py

# Generate baseline regression markdown report
python scripts/golden_baseline_report.py

# Docs link compliance check
python scripts/check_docs_links.py

# Docs API example compliance check
python scripts/check_docs_api_examples.py

# Check changed docs only (optional)
python scripts/check_docs_links.py --changed-only --from-rev origin/main --to-rev HEAD
python scripts/check_docs_api_examples.py --changed-only --from-rev origin/main --to-rev HEAD

# Docs strict build
mkdocs build --strict
```

---

## ➕ How to Add New Tests?

### Add Python Unit Tests

Create a new `test_*.py` file in the `tests/` directory and write tests using `pytest` style:

```python
def test_my_feature():
    assert 1 + 1 == 2
```

### Add New Golden Test Case

1.  **Generate Data**: Modify `tests/golden/gen_data.py` to add new data generation logic.
2.  **Write Strategy**: Create a new strategy file in `tests/golden/strategies/` (e.g., `my_test.py`).
3.  **Register Test**: Modify the `main` function in `tests/golden/runner.py` to add a new `run_test` call.
4.  **Generate Baseline**: Run `python tests/golden/runner.py --generate-baseline`.

---

## FAQ

**Q: Why do tests pass locally but fail in CI?**
A: This might be due to environmental differences (e.g., Pandas version, OS floating-point precision). Golden tests have a certain Tolerance, but large discrepancies will still cause errors. Check the CI logs for details.

**Q: How to debug Rust parts?**
A: Use `cargo test` to run native Rust tests. If you need to debug Python calls into Rust, refer to the mixed debugging guide in the developer documentation.
