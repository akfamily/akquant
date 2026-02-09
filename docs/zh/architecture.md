# AKQuant - 高性能 Rust/Python 量化投研框架

## 1. 项目概览
**AKQuant** 是一个专为高性能回测设计的混合语言量化投研框架。它利用 **Rust** 处理繁重的计算任务（如事件循环、数值计算、内存管理），并使用 **Python** 进行策略定义、数据分析和可视化。

## 2. 架构设计

### 2.1 系统分层
1.  **Rust 核心层 (`akquant_core`)**:
    *   **数据引擎**: 使用 `polars` (Arrow 格式) 管理 OHLCV 数据，尽可能实现零拷贝内存映射。
    *   **回测引擎**: 事件驱动的执行引擎。
    *   **事件总线 (Event Bus)**:
        *   参考成熟的事件驱动消息总线概念，基于 Rust Channel (`mpsc`) 实现。
        *   解耦了策略、风控、执行和数据组件。
        *   支持 `OrderRequest` (请求), `OrderValidated` (风控通过), `ExecutionReport` (执行报告) 等事件的异步流转。
        *   高优先级处理控制事件，支持未来扩展为多策略并行或异步风控检查。
    *   **订单撮合**: 模拟限价单/市价单、滑点和手续费。
    *   **风控模块**: 内置 `RiskManager`，支持预交易风控（T+1 可用持仓、资金限制等）。
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

### 2.3 核心交互流程 (事件总线)

1.  **策略层**: 策略调用 `self.buy()`，在 Rust 层生成 `OrderRequest` 事件并发送到 `event_tx` 通道。
2.  **引擎层**:
    *   主循环优先检查 `event_rx` 通道。
    *   接收到 `OrderRequest` 后，调用 **风控模块** (`RiskManager`) 进行检查。
    *   若风控通过，生成 `OrderValidated` 事件并重新发送到通道。
    *   若风控拒绝，生成拒绝状态的 `ExecutionReport` 并发送到通道。
3.  **执行层**:
    *   接收 `OrderValidated` 事件。
    *   **模拟模式**: 立即撮合或加入挂单列表，生成 `ExecutionReport`。
    *   **实盘模式**: 将订单发送到外部网关，并在收到回报时生成 `ExecutionReport`。
4.  **策略层**: 策略通过回调更新订单状态 (`pending_orders`) 和持仓。

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
