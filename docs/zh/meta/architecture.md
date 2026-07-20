# 架构设计

**AKQuant** 采用高性能的 **Rust + Python 混合架构**。Rust 负责底层的密集计算、内存管理、事件调度和风控拦截，Python 负责上层的策略定义、数据交互和分析可视化。这种设计在保证极致执行效率的同时，维持了 Python 生态的灵活性。

## 1. 系统分层

### Rust 核心层 (`akquant_core`)
*   **Engine**: 事件驱动的核心引擎，采用二进制堆 (BinaryHeap) 管理定时器，基于 Pipeline 模式处理事件流。支持 `Event::Bar`, `Event::Tick`, `Event::Timer` 等事件。
*   **DataFeed**: 高性能数据引擎，支持模拟数据 (Memory/CSV) 和实时数据流 (Channel)。支持零拷贝数组导入 (`add_arrays`)。
*   **EventManager**: 基于 Rust Channel (`mpsc`) 实现的事件总线，负责事件的分发与流转。
*   **RiskManager**: 独立拦截层，在 Rust 引擎层直接拦截违规订单。支持多层级风控规则 (Common, Asset-Specific) 和自动调仓逻辑 (Check and Adjust)。
*   **Portfolio**: 独立的投资组合管理，实时维护持仓 (`Position`) 和 账户资金 (`Account`)，自动处理盈亏计算和权益更新。
*   **MarketManager**: 市场规则管理器，支持多种市场模型 (`SimpleMarket`, `ChinaMarket`) 的切换和组合。
*   **ExecutionClient**: 执行客户端抽象，支持模拟执行 (`SimulatedExecutionClient`) 和实盘执行 (`RealtimeExecutionClient`)。

### 接口层 (PyO3)
*   利用 `PyO3` 将 Rust 的核心结构体 (`Engine`, `StrategyContext`, `Bar`, `Order`, `RiskManager`) 暴露为 Python 类。
*   **数据读取语义**: 历史数据 (`ctx.history` / `get_history`) 返回滚动缓冲的安全快照拷贝——底层为可变环形缓冲 (`VecDeque`)，直接映射的视图会在下一根 Bar 后失效，故按拷贝返回（窗口小，开销可忽略）。真正的零拷贝在数据导入侧 (`add_arrays` 借用 NumPy 缓冲)。
*   **PyExecutionMatcher**: 支持在 Python 端实现自定义撮合逻辑，并注册到 Rust 引擎中。

### Python 用户层
*   **Strategy API**: 提供简洁的 `Strategy` 基类，支持 `on_bar`, `on_tick`, `on_order`, `on_trade`, `on_timer` 等回调。内置自动指标发现 (`_discover_indicators`)。
*   **ML Framework**: 内置 `akquant.ml` 模块，提供 `QuantModel` 抽象基类和 `SklearnAdapter` / `PyTorchAdapter`，支持 Walk-forward Validation (滚动训练) 框架。
*   **Data API**: 兼容 Pandas 生态，支持从 CSV、Parquet 或内存 DataFrame 加载数据。
*   **Analysis**: 提供 `plot` 模块，基于 Plotly 实现交互式可视化，包括资金曲线、盈亏分布、年度收益等图表。

## 2. 核心模块与特性

### 2.1 极致性能
*   **Rust 核心**: 核心回测引擎采用 Rust 编写，避免了 Python GIL 的限制。
*   **Pipeline 架构**: 采用流水线模式 (`DataProcessor` -> `ExecutionProcessor` -> `StrategyProcessor`) 处理事件，逻辑清晰且高效。
*   **增量计算**: 内部状态和指标计算采用增量更新算法，适合超长历史回测。

### 2.2 事件驱动与仿真
*   **Timer**: 支持 `schedule(timestamp, payload)` 注册定时事件，触发 `on_timer` 回调。支持每日定时任务 (`schedule_daily`)。
*   **三轴执行语义**: 通过 `fill_policy` 的 `price_basis`、`bar_offset`、`temporal` 统一表达成交价格与撮合时序。
*   **滑点模型 (Slippage)**: 支持 Fixed (固定金额) 和 Percent (百分比) 滑点模型。
*   **成交量限制 (Volume Limit)**: 支持按 K 线成交量比例限制单笔撮合数量。

### 2.3 严密的风控系统
*   **T+1 严格风控**: 针对股票/基金，严格执行 T+1 可用持仓检查。
*   **资金/持仓限制**: 支持最大单笔金额、最大持仓比例、限制名单 (`RestrictedListRule`) 等检查。
*   **自动调整**: 支持在资金不足时自动调整订单数量 (`check_and_adjust`)。

### 2.4 机器学习 (Machine Learning First)
*   **内置训练框架**: 内置完整的 ML Pipeline，支持 `fit`, `predict`, `save`, `load` 标准接口。
*   **滚动训练**: 策略中通过 `set_rolling_window` 配置滚动参数，引擎自动触发 `on_train_signal` 事件。
*   **深度学习支持**: 内置 `PyTorchAdapter`，支持增量训练 (`incremental`) 和 GPU 加速。

### 2.5 数据生态
*   **Streaming CSV**: 支持流式加载超大 CSV 文件，降低内存占用。
*   **Pandas 集成**: 支持直接加载 Pandas DataFrame 数据，兼容各类数据源。
*   **实时数据**: `DataFeed` 支持 Channel 模式，可对接实时行情源。

## 3. 目录结构

```
akquant/
├── Cargo.toml          # Rust 依赖管理
├── pyproject.toml      # Python 构建系统 (maturin)
├── src/                # Rust 源代码
│   ├── lib.rs          # PyO3 入口点
│   ├── engine/         # 引擎核心 (Core, State, Python Bindings)
│   ├── data/           # 数据层 (Feed, Client, Aggregator)
│   ├── event.rs        # 事件定义
│   ├── event_manager.rs # 事件管理器
│   ├── execution/      # 执行层 (Matcher, Simulated, Realtime, Slippage)
│   ├── market/         # 市场层 (Manager, China, Simple, Stock, Futures)
│   ├── risk/           # 风控层 (Manager, Rule, Config, Common, Asset-Specific)
│   ├── portfolio.rs    # 资金与持仓管理
│   ├── context.rs      # 策略交互上下文
│   ├── history.rs      # 历史数据管理 (环形缓冲, 读取按安全快照拷贝返回)
│   ├── analysis/       # 绩效分析
│   ├── pipeline/       # 流水线处理架构
│   ├── settlement/     # 结算逻辑
│   ├── statistics/     # 统计模块
│   └── model/          # 数据模型 (Order, Trade, Instrument, Bar 等)
├── python/             # Python 源代码
│   └── akquant/
│       ├── ml/         # 机器学习框架 (Model, Adapters)
│       ├── plot/       # 绘图分析模块
│       ├── backtest/   # 回测引擎封装
│       ├── __init__.py
│       ├── akquant.pyi # 类型提示文件
│       ├── strategy.py # 策略基类
│       ├── config.py   # 配置定义
│       └── ...         # 其他辅助模块
└── examples/           # 使用示例
```

## 4. 技术栈选型
*   **Rust**:
    *   `pyo3`: 生成 Python 绑定。
    *   `chrono`: 时间处理。
    *   `rust_decimal`: 高精度金额计算。
    *   `indicatif`: 进度条显示。
    *   `serde`: 序列化支持。
*   **Python**:
    *   `maturin`: 构建后端。
    *   `pandas` / `numpy`: 面向用户的数据处理。
    *   `plotly`: 交互式可视化。
    *   `scikit-learn` / `torch`: 机器学习支持。
