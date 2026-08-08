# 测试指南

为了确保 AKQuant 回测引擎的准确性和稳定性，我们建立了一套分层测试体系。本指南将帮助你了解如何运行测试、解读结果以及添加新的测试用例。

## 测试体系概览

AKQuant 的测试分为以下几个层级：

1.  **单元测试 (Unit Tests)**
    *   **Rust**: 使用 `cargo test` 运行，覆盖核心算法（撮合、费用计算、风险控制）。
    *   **Python**: 使用 `pytest` 运行，覆盖 Python 接口、数据处理和配置解析。
2.  **黄金测试 (Golden Tests / Integration Tests)**
    *   使用合成数据（Synthetic Data）模拟特定市场场景（如 T+1、涨跌停、期权行权）。
    *   将回测结果（权益曲线、成交记录、指标）与锁定的**基线 (Baseline)** 进行严格比对。
    *   确保任何算法变更都不会意外改变回测结果。

---

## 🚀 快速开始

### 1. 准备环境

确保你已经安装了开发依赖：

```bash
pip install -e ".[dev]"
```

### 2. 运行所有测试

```bash
pytest
```

### 3. 仅运行黄金测试

```bash
pytest tests/golden/test_golden.py
```

---

## 🏆 黄金测试 (Golden Tests)

黄金测试是 AKQuant 质量保证的核心。它位于 `tests/golden/` 目录下。

### 目录结构

```text
tests/golden/
├── strategies/         # 测试策略代码 (e.g., stock_t1.py)
├── data/               # 自动生成的合成数据 (Parquet 格式)
├── baselines/          # 锁定的基线结果 (标准答案)
├── runner.py           # 测试执行器与对比逻辑
├── gen_data.py         # 数据生成脚本
└── test_golden.py      # Pytest 入口
```

### 包含的场景

| 测试用例 | 描述 | 验证点 |
| :--- | :--- | :--- |
| **stock_t1** | 股票 T+1 交易 | Day 1 买入 Day 2 可卖；Day 3 买入当日卖出被拒。 |
| **futures_margin** | 期货保证金 | 初始资金仅够 1 手；尝试买入第 2 手时因保证金不足被拒。 |
| **option_basic** | 期权基础交易 | 验证期权合约的买入、持仓与卖出平仓流程及 PnL。 |

### 如何更新基线？

如果你修改了核心算法（例如改进了撮合逻辑或费用计算），导致回测结果发生**预期内**的变化，黄金测试会失败。此时你需要更新基线：

1.  **确认差异**: 仔细检查测试失败的输出，确认差异是你修改代码导致的，且符合预期。
2.  **重新生成基线**:
    ```bash
    python tests/golden/runner.py --generate-baseline
    ```
3.  **提交变更**: 将更新后的 `tests/golden/baselines/` 文件随你的代码一起提交。

### 流式回测回归清单

在修改 `run_backtest(..., on_event=...)`、事件协议或引擎流式通道后，建议至少执行以下检查：

1.  **核心语义一致性**: 流式与非流式结果保持一致（权益长度、成交数量、关键指标）。
2.  **事件协议正确性**: 事件序列从 `started` 开始，以 `finished` 结束，`seq` 单调递增。
3.  **采样与背压行为**: 高频场景下 `progress`/`equity` 事件量受采样参数控制，关键事件不丢失。
4.  **回调异常策略**:
    *   `stream_error_mode="continue"`: 回调报错不终止回测，结束事件包含 `callback_error_count`。
    *   `stream_error_mode="fail_fast"`: 回调首次报错立即失败并抛异常。
5.  **参数校验**: `stream_progress_interval`、`stream_equity_interval`、`stream_batch_size`、`stream_max_buffer` 必须为正整数，`stream_error_mode` 取值合法。

推荐命令：

```bash
# 流式与回测引擎相关单元测试
pytest -q tests/test_engine.py

# 黄金测试（含流式与非流式一致性套件）
pytest -q tests/golden/test_golden.py

# 生成黄金基线回归对比报告（Markdown）
python scripts/golden_baseline_report.py

# 文档链接合规检查
python scripts/check_docs_links.py

# 文档示例 API 合规检查
python scripts/check_docs_api_examples.py

# 仅检查本次变更文档（可选）
python scripts/check_docs_links.py --changed-only --from-rev origin/main --to-rev HEAD
python scripts/check_docs_api_examples.py --changed-only --from-rev origin/main --to-rev HEAD

# 文档严格构建
mkdocs build --strict
```

---

## ➕ 如何添加新测试？

### 添加 Python 单元测试

在 `tests/` 目录下创建新的 `test_*.py` 文件，使用 `pytest` 风格编写：

```python
def test_my_feature():
    assert 1 + 1 == 2
```

### 添加新的黄金测试用例

1.  **生成数据**: 修改 `tests/golden/gen_data.py`，添加新的数据生成逻辑。
2.  **编写策略**: 在 `tests/golden/strategies/` 下创建新的策略文件（如 `my_test.py`）。
3.  **注册测试**: 修改 `tests/golden/runner.py` 中的 `main` 函数，添加新的 `run_test` 调用。
4.  **生成基线**: 运行 `python tests/golden/runner.py --generate-baseline`。

---

## 常见问题 (FAQ)

**Q: 为什么本地测试通过，CI 却失败了？**
A: 可能是环境差异（如 Pandas 版本、操作系统浮点数精度）导致的。黄金测试设置了一定的容忍度（Tolerance），但如果差异过大仍会报错。请检查 CI 日志中的差异详情。

**Q: 如何调试 Rust 部分的测试？**
A: 使用 `cargo test` 运行 Rust 原生测试。如果需要调试 Python 调用 Rust 的部分，可以参考开发文档中的混合调试指南。
