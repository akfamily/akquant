# 示例代码

`akquant` 提供了多个示例脚本，涵盖了从基础回测到性能基准测试的场景。您可以在 `examples/` 目录下找到这些文件的完整源码。

> **注意**: 更多入门级示例和教程请参考 [快速开始 (Quickstart)](quickstart.md)。

## 1. 基础回测 (`kan.py`)

这是一个完整的基础示例，展示了如何：
1.  使用 Pandas 生成模拟的价格数据。
2.  定义一个简单的策略 (`SmaStrategy`)。
3.  使用 `run_backtest` 运行回测。
4.  访问并打印回测结果，包括 `metrics_df` (绩效指标), `trades_df` (交易记录) 和 `daily_positions_df` (每日持仓)。

[查看源码](../examples/kan.py)

## 2. 多标的性能测试 (`benchmark_akquant_multi.py`)

展示了 `akquant` 在处理多标的回测时的性能和用法：
*   **多标的支持**: 同时回测 3 只股票 (STOCK_A, STOCK_B, STOCK_C)。
*   **数据生成**: 使用 `benchmark_utils` 生成大规模测试数据。
*   **独立指标**: 演示如何为每个标的维护独立的指标状态。
*   **性能日志**: 记录数据生成和策略运行的耗时。

[查看源码](../examples/benchmark_akquant_multi.py)

## 3. Backtrader 对比测试 (`kan_bt.py`)

为了验证策略逻辑和性能差异，我们提供了一个等价的 Backtrader 实现：
*   实现了与 `kan.py` 相同的数据生成逻辑。
*   实现了相同的策略逻辑和佣金模型。
*   用于对比两个框架的运行结果和效率。

[查看源码](../examples/kan_bt.py)
