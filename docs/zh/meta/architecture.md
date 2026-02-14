# 架构设计

**AKQuant** 采用高性能的 **Rust + Python 混合架构**。Rust 负责底层的密集计算、内存管理、事件调度和风控拦截，Python 负责上层的策略定义、数据交互和分析可视化。这种设计在保证极致执行效率的同时，维持了 Python 生态的灵活性。

## 1. 系统分层

### Rust 核心层 (`akquant_core`)
*   **Engine**: 事件驱动的核心撮合引擎，采用二进制堆 (BinaryHeap) 管理事件队列，确保时间精确性。
*   **DataFeed**: 高性能数据引擎，支持流式加载 (Streaming CSV) 和 Pandas DataFrame 直接映射，尽可能实现零拷贝内存访问。
*   **Event Bus**: 基于 Rust Channel (`mpsc`) 实现的事件总线，解耦策略、风控、执行和数据组件。支持 `OrderRequest` (请求), `OrderValidated` (风控通过), `ExecutionReport` (执行报告) 等事件的异步流转。
*   **RiskManager**: 独立拦截层，在 Rust 引擎层直接拦截违规订单，支持 T+1 可用持仓检查、资金限制等。
*   **Portfolio**: 独立的投资组合管理，实时维护持仓 (`Position`) 和 账户资金 (`Account`)，自动处理盈亏计算。
*   **MarketModel**: 可插拔的市场模型，内置 A 股 (T+1) 和 期货 (T+0) 规则，支持多资产混合回测。

### 接口层 (PyO3)
*   利用 `PyO3` 将 Rust 的核心结构体 (`Engine`, `StrategyContext`, `Bar`, `Order`) 暴露为 Python 类。
*   **Zero-Copy Access**: 利用 Rust 的 `arrow` 和 `numpy` 视图技术，历史数据 (`ctx.history`) 通过 PyO3 Buffer Protocol 直接映射 Rust 内存，Python 端访问 OHLCV 和计算指标时无需内存复制，大幅提升性能。

### Python 用户层
*   **Strategy API**: 提供简洁的 `Strategy` 基类，支持 `on_bar`, `on_order`, `on_trade` 等回调。
*   **ML Framework**: 内置 `akquant.ml` 模块，提供 Walk-forward Validation (滚动训练) 框架，统一 Scikit-learn 和 PyTorch 接口。
*   **Data API**: 兼容 Pandas 生态，支持从 CSV、Parquet 或内存 DataFrame 加载数据。
*   **Analysis**: 集成 Plotly/Matplotlib 进行交互式可视化，提供丰富的绩效指标 (`PerformanceMetrics`) 和交易分析 (`TradeAnalyzer`)。

## 2. 核心模块与特性

### 2.1 极致性能
*   **Rust 核心**: 核心回测引擎采用 Rust 编写，避免了 Python GIL 的限制。
*   **基准测试**: 在 200k K线数据的 SMA 策略回测中，AKQuant 耗时仅 **1.31s** (吞吐量 ~152k bars/sec)，相比 Backtrader (26.55s) 快约 **20倍**。
*   **增量计算**: 内部状态和指标计算采用增量更新算法，而非全量重算，适合超长历史回测。

### 2.2 事件驱动与仿真
*   **Timer**: 支持 `schedule(timestamp, payload)` 注册定时事件，触发 `on_timer` 回调，实现复杂的盘中定时逻辑（如：每日 14:50 平仓）。
*   **ExecutionMode**: 支持 `CurrentClose` (信号当根K线收盘成交) 和 `NextOpen` (次日开盘成交) 模式。
*   **滑点模型 (Slippage)**: 支持 Fixed (固定金额) 和 Percent (百分比) 滑点模型，模拟真实交易成本。
*   **成交量限制 (Volume Limit)**: 支持按 K 线成交量比例限制单笔撮合数量，并实现分批成交 (Partial Fill)。

### 2.3 严密的风控系统
*   **T+1 严格风控**: 针对股票/基金，严格执行 T+1 可用持仓检查，防止当日买入当日卖出（除非配置为 T+0 市场）。
*   **可用持仓管理**: 自动维护 `available_positions`，并扣除未成交的卖单冻结数量，防止超卖。
*   **灵活配置**: 通过 `RiskConfig` 可配置最大单笔金额、最大持仓比例、黑名单等。

### 2.4 机器学习 (Machine Learning First)
*   **内置训练框架**: 内置完整的 ML Pipeline，不同于传统框架仅支持简单的技术指标。
*   **Walk-forward Validation**: 原生支持滚动窗口训练，有效防止未来函数和过拟合。
*   **Adapter Pattern**: 提供了 `QuantModel` 适配器，解耦模型与策略逻辑，只需几行代码即可接入 AI 模型。

### 2.5 数据生态
*   **Streaming CSV**: 支持流式加载超大 CSV 文件 (`DataFeed.from_csv`)，极大降低内存占用。
*   **Pandas 集成**: 支持直接加载 Pandas DataFrame 数据，兼容各类数据源。
*   **智能缓存**: 支持数据本地缓存 (Pickle)，避免重复下载，加速策略迭代。

## 3. 目录结构

```
akquant/
├── Cargo.toml          # Rust 依赖管理
├── pyproject.toml      # Python 构建系统 (maturin)
├── src/                # Rust 源代码
│   ├── lib.rs          # PyO3 入口点
│   ├── model/          # 数据模型 (Order, Trade, Instrument, Bar 等)
│   ├── data.rs         # 数据源 (DataFeed)
│   ├── engine.rs       # 回测核心引擎
│   ├── event.rs        # 事件定义与总线消息
│   ├── clock.rs        # 交易时钟
│   ├── execution.rs    # 交易所模拟与订单撮合
│   ├── market.rs       # 市场规则 (费率、T+1/T+0)
│   ├── portfolio.rs    # 资金与持仓管理
│   ├── risk.rs         # 风控管理 (RiskManager)
│   ├── context.rs      # 策略交互上下文
│   ├── history.rs      # 历史数据管理 (Zero-Copy View)
│   ├── analysis.rs     # 绩效指标计算
│   └── indicators.rs   # 高性能指标实现
├── python/             # Python 源代码
│   └── akquant/
│       ├── ml/         # 机器学习适配器
│       ├── __init__.py
│       ├── akquant.pyi # 类型提示文件
│       ├── backtest.py # 回测入口
│       ├── strategy.py # 策略基类
│       ├── indicator.py# 指标包装
│       ├── optimize.py # 参数优化
│       └── ...         # 其他辅助模块
└── examples/           # 使用示例
```

## 4. 技术栈选型
*   **Rust**:
    *   `pyo3`: 生成 Python 绑定。
    *   `polars` / `arrow`: 高性能列式数据存储。
    *   `rayon`: 多资产回测的并行处理。
    *   `serde`: 序列化支持。
*   **Python**:
    *   `maturin`: 构建后端。
    *   `pandas` / `numpy`: 面向用户的数据处理。
    *   `plotly`: 交互式可视化。
