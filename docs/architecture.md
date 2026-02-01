# AKQuant - 高性能 Rust/Python 量化投研框架

## 1. 项目概览
**AKQuant** 是一个专为高性能回测设计的混合语言量化投研框架。它利用 **Rust** 处理繁重的计算任务（如事件循环、数值计算、内存管理），并使用 **Python** 进行策略定义、数据分析和可视化。

## 2. 架构设计

### 2.1 系统分层
1.  **Rust 核心层 (`akquant_core`)**:
    *   **数据引擎**: 使用 `polars` (Arrow 格式) 管理 OHLCV 数据，尽可能实现零拷贝内存映射。
    *   **回测引擎**: 事件驱动的执行引擎。
    *   **订单撮合**: 模拟限价单/市价单、滑点和手续费。
    *   **指标计算**: 快速计算夏普比率、最大回撤等指标。
2.  **接口层 (PyO3)**:
    *   将 Rust 结构体 (`Engine`, `DataFeed`, `StrategyContext`) 暴露为 Python 类。
    *   处理类型转换（例如：Rust `DataFrame` <-> Python `pandas`/`polars`）。
3.  **Python 用户层**:
    *   **策略 API**: 用户继承的抽象基类。
    *   **数据 API**: 连接 Tushare, AKShare, Parquet 文件的数据连接器。
    *   **可视化**: 集成 Plotly/Matplotlib。

### 2.2 目录结构
```
akquant/
├── Cargo.toml          # Rust 依赖管理
├── pyproject.toml      # Python 构建系统 (maturin)
├── src/                # Rust 源代码
│   ├── lib.rs          # PyO3 入口点
│   ├── model/          # 数据模型 (Order, Trade, Instrument, Bar 等)
│   ├── data.rs         # 数据源 (DataFeed)
│   ├── engine.rs       # 回测核心引擎
│   ├── clock.rs        # 交易时钟 (NautilusTrader 风格)
│   ├── execution.rs    # 交易所模拟与订单撮合
│   ├── market.rs       # 市场规则 (费率、T+1/T+0)
│   ├── portfolio.rs    # 资金与持仓管理
│   └── context.rs      # 策略交互上下文
├── python/             # Python 源代码
│   └── akquant/
│       ├── __init__.py
│       └── akquant.pyi    # 类型提示文件
└── examples/           # 使用示例
```

## 3. 技术栈选型
*   **Rust**:
    *   `pyo3`: 生成 Python 绑定。
    *   `polars` / `arrow`: 高性能列式数据存储。
    *   `rayon`: 多资产回测的并行处理。
    *   `serde`: 序列化支持。
*   **Python**:
    *   `maturin`: 构建后端。
    *   `pandas` / `numpy`: 面向用户的数据处理。
    *   `plotly`: 交互式可视化。

## 4. 开发路线图

### 第一阶段：原型验证（当前）
*   [ ] 配置项目构建系统 (Maturin)。
*   [ ] 在 Rust 中定义 `Candle` (K线) 和 `Feed` (数据流) 结构体。
*   [ ] 向 Python 暴露基础的数据加载功能。
*   [ ] 从 Rust 返回 Pandas DataFrame。

### 第二阶段：核心引擎
*   [ ] 在 Rust 中实现事件循环。
*   [ ] 在 Rust 中创建 `Strategy` trait 并将其封装以供 Python 继承。
*   [ ] 基础订单撮合（市价单/限价单）。

### 第三阶段：分析与可视化
*   [ ] 实现绩效指标计算（Rust 端）。
*   [ ] 创建可视化模块（Python 端）。

### 第四阶段：生产就绪
*   [ ] 对接外部数据源 (Tushare/SQL)。
*   [ ] 支持并行回测。
*   [ ] 完善文档。
