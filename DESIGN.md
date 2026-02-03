# AKQuant 设计与开发指南

本文档详细介绍了 `AKQuant` 的内部设计原理、核心组件架构以及扩展开发指南。旨在帮助开发者深入理解项目结构，以便进行二次开发和功能扩展。

## 1. 项目概览

### 1.1 设计理念

`AKQuant` 遵循以下核心设计原则：

1.  **核心计算下沉 (Rust Core)**: 所有的计算密集型任务（事件循环、订单撮合、风控检查、数据管理、绩效计算、历史数据维护）都在 Rust 层实现，以确保高性能和内存安全。
2.  **策略逻辑上浮 (Python API)**: 策略编写、参数配置、数据分析、机器学习模型定义等用户交互层保留在 Python 中，利用其动态特性和丰富的生态系统 (Pandas, Scikit-learn, PyTorch 等)。
3.  **模块化与解耦 (Modularity)**: 借鉴 `NautilusTrader` 等成熟框架，将数据、执行、策略、风控、机器学习等模块严格分离，通过清晰的接口（Traits）交互。

### 1.2 项目目录结构

```text
akquant/
├── Cargo.toml              # Rust 项目依赖与配置
├── pyproject.toml          # Python 项目构建配置 (Maturin)
├── src/                    # Rust 核心源码 (底层实现)
│   ├── lib.rs              # PyO3 模块入口，注册 Python 模块
│   ├── engine.rs           # 回测引擎：驱动时间轴与事件循环
│   ├── execution.rs        # 执行层：模拟交易所撮合逻辑
│   ├── market.rs           # 市场层：定义佣金、印花税、T+1 规则
│   ├── portfolio.rs        # 账户层：管理资金、持仓与可用头寸
│   ├── data.rs             # 数据层：管理 Bar/Tick 数据流
│   ├── analysis.rs         # 分析层：计算绩效指标 (Sharpe, Drawdown)
│   ├── context.rs          # 上下文：用于 Python 回调的数据快照
│   ├── clock.rs            # 时钟模块：统一时间管理
│   ├── event.rs            # 事件系统：定义系统内部事件
│   ├── history.rs          # 历史数据：高效的环形缓冲区管理
│   ├── indicators.rs       # 技术指标：Rust 原生指标实现 (如 SMA)
│   └── model/              # 数据模型：定义基础数据结构
│       ├── order.rs        # 订单 (Order) 与成交 (Trade)
│       ├── instrument.rs   # 标的物信息 (Instrument)
│       ├── timer.rs        # 定时器事件
│       └── types.rs        # 基础枚举 (Side, Type, ExecutionMode)
├── python/
│   └── akquant/            # Python 包源码 (用户接口)
│       ├── __init__.py     # 导出公共 API
│       ├── strategy.py     # Strategy 基类：封装上下文，提供 ML 训练与交易接口
│       ├── config.py       # 配置定义：BacktestConfig, StrategyConfig, RiskConfig
│       ├── risk.py         # 风控配置适配层
│       ├── data.py         # 数据加载与目录服务 (DataCatalog)
│       ├── sizer.py        # Sizer 基类：提供多种仓位管理实现
│       ├── analyzer.py     # TradeAnalyzer：交易记录分析工具
│       ├── indicator.py    # Python 指标接口
│       ├── log.py          # 日志模块
│       ├── ml/             # 机器学习框架 (New)
│       │   ├── __init__.py
│       │   └── model.py    # QuantModel, SklearnAdapter, ValidationConfig
│       └── akquant.pyi     # 类型提示文件 (IDE 补全支持)
└── examples/               # 示例代码
    ├── ml_framework_demo.py      # ML 框架基础示例
    └── ml_walk_forward_demo.py   # 滚动训练示例
```

## 2. 核心组件架构详解

### 2.1 数据模型层 (`src/model/`)

为了保证跨语言交互的性能与类型安全，核心数据结构均在 Rust 中定义并导出。

*   **`types.rs`**:
    *   `ExecutionMode`: `CurrentClose` (当前收盘价成交，即 Cheat-on-Close) vs `NextOpen` (次日开盘价成交，更真实)。
    *   `OrderSide`: `Buy` / `Sell`。
    *   `OrderType`: `Market` (市价), `Limit` (限价)。
    *   `TimeInForce`: `Day` (当日有效), `GTC` (撤前有效), `IOC`/`FOK`。
*   **`instrument.rs`**: `Instrument` 包含 `multiplier` (合约乘数) 和 `tick_size`。
*   **`market_data.rs`**: `Bar` (OHLCV) 和 `Tick` (最新价/量)。

### 2.2 执行层 (`src/execution.rs`)

`ExchangeSimulator` 是回测准确性的核心，负责模拟交易所的撮合逻辑。

*   **撮合机制**:
    *   **限价单 (Limit)**: 买入需 `Low <= Price`，卖出需 `High >= Price`。
    *   **市价单 (Market)**: 根据 `ExecutionMode` 决定按 `Close` 或 `Open` 成交。
*   **触发机制**: 支持 `trigger_price` (止损/止盈单)。

### 2.3 市场规则层 (`src/market.rs`)

通过 `MarketModel` Trait 实现不同市场的规则隔离。目前内置 `ChinaMarket` (A股市场规则)：

*   **佣金计算**: 支持股票 (印花税、过户费、佣金) 和期货 (按手或按金额)。
*   **交易限制**: 严格的 T+1 (股票) 与 T+0 (期货) 可用持仓管理。

### 2.4 风控层 (`src/risk.rs`)

`RiskManager` 独立于执行层，拦截每一笔订单：

*   **检查规则**: 限制名单、最大单笔数量/金额、最大持仓比例。
*   **配置**: Python 端 `RiskConfig` 自动注入 Rust 引擎。

### 2.5 账户层 (`src/portfolio.rs`)

`Portfolio` 结构体维护账户状态：

*   `cash`: 可用资金。
*   `positions`: 总持仓。
*   `available_positions`: 可卖持仓 (T+1 逻辑的核心)。
*   **权益计算**: 实时 Mark-to-Market 计算。

### 2.6 引擎层 (`src/engine.rs` & `src/history.rs`)

`Engine` 是系统的驱动器：

*   **事件循环**: 消费 `Bar` 或 `Tick` 事件。
*   **历史数据管理**: `Engine` 内部维护 `History` 模块，这是一个高效的环形缓冲区，用于存储最近 N 个 Bar 的数据，供策略通过 `get_history` 快速访问，无需在 Python 端累积数据。
*   **日切处理**: 触发 T+1 解锁、过期订单清理。

### 2.7 分析层 (`src/analysis.rs`)

遵循标准 PnL 计算：`Gross PnL` (毛利), `Net PnL` (净利), `Commission` (佣金)。

### 2.8 Python 抽象层 (`python/akquant/`)

*   **`Strategy` (`strategy.py`)**:
    *   **历史数据访问**:
        *   `set_history_depth(depth)`: 开启 Rust 端历史数据记录。
        *   `get_history(count)` / `get_history_df(count)`: 获取最近 N 个 Bar 的 OHLCV 数据 (Numpy/DataFrame)。
    *   **ML 集成**:
        *   `set_rolling_window(train, step)`: 配置滚动训练参数。
        *   `on_train_signal(context)`: 周期性触发模型训练。
        *   `prepare_features(df)`: 特征工程接口。
*   **`Sizer` (`sizer.py`)**: 仓位管理基类。
*   **`TradeAnalyzer` (`analyzer.py`)**: 交易记录分析。

### 2.9 机器学习框架 (`python/akquant/ml/`)

`AKQuant` 提供了一套标准化的 ML 接口，旨在简化“滚动训练-预测”流程。

*   **`QuantModel` (`model.py`)**:
    *   所有模型的抽象基类。
    *   接口: `fit(X, y)`, `predict(X)`, `save(path)`, `load(path)`。
    *   **`set_validation`**: 配置 Walk-forward Validation 参数 (训练窗口、测试窗口、滚动步长)。
*   **`SklearnAdapter`**:
    *   封装 Scikit-learn 风格的模型 (如 RandomForest, LinearRegression)，使其适配 `QuantModel` 接口。
*   **工作流**:
    1.  用户在策略中定义模型 `self.model = SklearnAdapter(RandomForestClassifier())`。
    2.  设置滚动参数 `self.set_rolling_window(train_window=250, step=20)`。
    3.  重写 `prepare_features` 将原始 OHLCV 转换为特征 (X) 和 标签 (y)。
    4.  回测过程中，引擎自动在指定步长触发 `on_train_signal`，策略自动获取历史数据并训练模型。
    5.  在 `on_bar` 中调用 `self.model.predict` 生成信号。

## 3. 关键工作流详解

### 3.1 回测主循环与执行模式

`Engine::run` 的流程依赖 `ExecutionMode`：

*   **NextOpen**: 推荐模式。Bar Close 生成信号 -> Next Bar Open 成交。
*   **CurrentClose**: 简化模式。Bar Close 生成信号 -> Current Bar Close 成交 (Cheat-on-Close)。

### 3.2 订单全生命周期

Signal -> Creation -> Submission -> Risk Check (Rust) -> Matching (Rust) -> Settlement (Rust) -> Reporting。

## 4. 扩展开发指南

### 4.1 如何添加新的订单类型

1.  `src/model/types.rs`: 添加枚举。
2.  `src/model/order.rs`: 更新结构体。
3.  `src/execution.rs`: 实现撮合逻辑。
4.  `akquant.pyi`: 更新类型提示。

### 4.2 如何自定义指标

1.  **Python 侧 (快速原型)**: 继承 `akquant.Indicator`，在 `on_bar` 中计算。
2.  **Rust 侧 (高性能)**:
    *   在 `src/indicators.rs` 中实现 `Indicator` Trait。
    *   通过 `#[pyclass]` 导出到 Python。

### 4.3 如何接入新的数据源

将数据转换为 pandas DataFrame，构造 `akquant.Bar` 对象列表，调用 `engine.add_feed` 即可。

## 5. Rust 与 Python 交互注意事项

*   **GIL**: Rust 仅在回调 Python 时获取 GIL，计算密集型任务释放 GIL (视实现而定，目前主要是单线程模型)。
*   **数据拷贝**: 尽量减少 Python 与 Rust 之间的大规模数据传输。`get_history` 返回的是 Numpy 视图或拷贝，效率远高于 Python list。
