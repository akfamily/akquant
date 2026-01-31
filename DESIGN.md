# akquant 设计与开发指南

本文档详细介绍了 `akquant` 的内部设计原理、核心组件架构以及扩展开发指南。旨在帮助开发者深入理解项目结构，以便进行二次开发和功能扩展。

## 1. 项目概览

### 1.1 设计理念

`akquant` 遵循以下核心设计原则：

1.  **核心计算下沉 (Rust Core)**: 所有的计算密集型任务（事件循环、订单撮合、风控检查、数据管理、绩效计算）都在 Rust 层实现，以确保高性能和内存安全。
2.  **策略逻辑上浮 (Python API)**: 策略编写、参数配置、数据分析等用户交互层保留在 Python 中，利用其动态特性和丰富的生态系统 (Pandas, Matplotlib 等)。
3.  **模块化与解耦 (Modularity)**: 借鉴 `NautilusTrader` 等成熟框架，将数据、执行、策略、风控等模块严格分离，通过清晰的接口（Traits）交互。

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
│   └── model/              # 数据模型：定义基础数据结构
│       ├── order.rs        # 订单 (Order) 与成交 (Trade)
│       ├── instrument.rs   # 标的物信息 (Instrument)
│       └── types.rs        # 基础枚举 (Side, Type, ExecutionMode)
├── python/
│   └── akquant/            # Python 包源码 (用户接口)
│       ├── __init__.py     # 导出公共 API
│       ├── strategy.py     # Strategy 基类：封装上下文，提供 buy/sell 接口
│       ├── config.py       # 配置定义：BacktestConfig, StrategyConfig, RiskConfig
│       ├── risk.py         # 风控配置适配层
│       ├── data.py         # 数据加载与目录服务 (DataCatalog)
│       ├── sizer.py        # Sizer 基类：提供多种仓位管理实现
│       ├── analyzer.py     # TradeAnalyzer：交易记录分析工具
│       └── akquant.pyi     # 类型提示文件 (IDE 补全支持)
└── examples/               # 示例代码
```

## 2. 核心组件架构详解

### 2.1 数据模型层 (`src/model/`)

为了保证跨语言交互的性能与类型安全，核心数据结构均在 Rust 中定义并导出。

*   **`types.rs`**:
    *   `ExecutionMode`: `CurrentClose` (当前收盘价成交，即 Cheat-on-Close) vs `NextOpen` (次日开盘价成交，更真实)。
    *   `OrderSide`: `Buy` / `Sell`。
    *   `OrderType`: `Market` (市价), `Limit` (限价)。
    *   `TimeInForce`: `Day` (当日有效), `GTC` (撤前有效 - 暂未完全支持跨日保持), `IOC`/`FOK`。
*   **`instrument.rs`**: `Instrument` 包含 `multiplier` (合约乘数，如股票为1，期货可能为10/300等) 和 `tick_size`。
*   **`market_data.rs`**: `Bar` (OHLCV) 和 `Tick` (最新价/量)。

### 2.2 执行层 (`src/execution.rs`)

`ExchangeSimulator` 是回测准确性的核心，负责模拟交易所的撮合逻辑。

*   **撮合机制**:
    *   **限价单 (Limit)**:
        *   **买入**: 当 `Bar.Low <= LimitPrice` 时成交。成交价取 `min(LimitPrice, Bar.Open)` (如果在 Open 范围内) 或 `LimitPrice`。
        *   **卖出**: 当 `Bar.High >= LimitPrice` 时成交。成交价取 `max(LimitPrice, Bar.Open)` 或 `LimitPrice`。
    *   **市价单 (Market)**:
        *   `CurrentClose` 模式: 按 `Bar.Close` 成交。
        *   `NextOpen` 模式: 按 `Bar.Open` 成交。
*   **触发机制**: 支持 `trigger_price` (止损/止盈单)，当价格突破触发价后，订单转为市价或限价单进入撮合队列。

### 2.3 市场规则层 (`src/market.rs`)

通过 `MarketModel` Trait 实现不同市场的规则隔离。目前内置 `ChinaMarket` (A股市场规则)：

*   **佣金计算 (`calculate_commission`)**:
    *   **股票**:
        *   佣金: 成交金额 * 0.03% (最低 5.0 元)。
        *   印花税: 成交金额 * 0.05% (仅**卖出**方收取)。
        *   过户费: 成交金额 * 0.001%。
    *   **期货**:
        *   佣金: 成交金额 * 0.0023% (示例值，可配置)。
*   **交易限制 (`update_available_position`)**:
    *   **T+1 (股票)**: 买入当日增加总持仓 (`positions`)，但不增加可用持仓 (`available_positions`)。次日结算后解锁。
    *   **T+0 (期货)**: 买入立即增加可用持仓，允许当日平仓。

### 2.4 风控层 (`src/risk.rs` & `python/akquant/risk.py`) (New)

风控模块 (`RiskManager`) 是独立于执行层的拦截器，确保每一笔订单都符合预设的安全规则。

*   **拦截机制**: 在订单生成后、进入撮合队列前 (`Engine` 循环中)，调用 `RiskManager::check(order, portfolio)`。
*   **检查规则**:
    *   **限制名单 (Restricted List)**: 禁止交易特定标的。
    *   **最大单笔数量 (Max Order Size)**: 防止胖手指错误。
    *   **最大单笔金额 (Max Order Value)**: 控制单笔风险敞口。
    *   **最大持仓比例 (Max Position Size)**: 防止单标的仓位过重。
*   **Python 配置**: 用户在 Python 端通过 `RiskConfig` 配置参数，回测启动时自动注入到 Rust 引擎。

### 2.5 账户层 (`src/portfolio.rs`)

`Portfolio` 结构体维护账户状态：

*   `cash`: 可用资金。
*   `positions`: **总持仓** (Symbol -> Quantity)，包含冻结或未上市持仓。
*   `available_positions`: **可卖持仓** (Symbol -> Quantity)，用于下单时的风控检查。
*   **权益计算 (`calculate_equity`)**: 实时计算 `Cash + Σ(Position * CurrentPrice * Multiplier)`。

### 2.5 引擎层 (`src/engine.rs`)

`Engine` 是系统的驱动器，采用事件驱动模型：

*   **事件循环**: 消费 `DataFeed` 中的 `Bar` 或 `Tick` 事件。
*   **日切处理 (Day Close)**:
    *   检测到日期变更时，触发 `MarketModel::on_day_close`。
    *   对于 A 股，此时将昨日买入的冻结持仓移入 `available_positions` (T+1 解锁)。
    *   清理 `TimeInForce::Day` 的过期未成交订单。
*   **Python 回调**: 通过 `PyO3` 将 Rust 数据结构映射为 Python 对象，调用策略的 `on_bar` 方法。

### 2.6 分析层 (`src/analysis.rs`)

负责计算交易绩效和统计指标。为了与主流开源框架（如 Backtrader）保持一致，我们在 PnL（盈亏）计算上遵循以下标准：

*   **`pnl` (Gross PnL)**: 毛利，即 `(ExitPrice - EntryPrice) * Quantity * Multiplier` (多头)。不包含交易费用。
*   **`net_pnl` (Net PnL)**: 净利，即 `Gross PnL - Commission`。包含所有交易费用（佣金、印花税等）。
*   **`commission`**: 交易过程中产生的总费用。

这种设计确保了指标计算的准确性，并方便用户进行跨框架对比。

### 2.8 Python 抽象层 (`python/akquant/`)

*   **`Strategy` (`strategy.py`)**:
    *   **生命周期**:
        *   `on_start()`: 策略启动时调用，用于注册定时器或**显式订阅数据** (`self.subscribe("600000")`)。
        *   `on_bar(bar)`: 每个 Bar 到达时调用。
        *   `on_stop()`: 策略结束时调用。
    *   **Context 代理**: 自动注入 `self.context`，提供 `self.buy`, `self.sell`, `self.position` 等便捷属性。
    *   **Indicator 自动管理**: 在 `__init__` 中定义的指标 (如 `self.sma = SMA(10)`) 会被自动注册，并在 `on_bar` 之前自动计算最新值。
*   **`Sizer` (`sizer.py`)**:
    *   `FixedSize`: 每次交易固定股数。
    *   `PercentSizer`: 按当前资金百分比开仓 (默认 95% 防止资金不足)。
    *   `AllInSizer`: 满仓买入。
*   **`TradeAnalyzer` (`analyzer.py`)**:
    *   提供 Python 接口调用 Rust 端的 FIFO 盈亏计算逻辑，生成详细的交易清单和统计报表。

## 3. 关键工作流详解

### 3.1 回测主循环与执行模式

`Engine::run` 的流程高度依赖 `ExecutionMode` 配置：

#### 场景 A: `ExecutionMode.NextOpen` (推荐，更真实)
这是 Backtrader 的默认行为，模拟在当前 Bar 产生信号，在下一根 Bar 开盘时成交。

1.  **加载数据**: 获取当前 Bar (例如: 2023-01-01)。
2.  **撮合 (Phase 1)**: 使用当前 Bar 的 **Open** 价格，尝试撮合**上一时刻**遗留的待处理订单 (Pending Orders)。
    *   *逻辑*: 昨天的决策，今天开盘立即执行。
3.  **策略 (Phase 2)**: 调用 `strategy.on_bar(bar)`。
    *   策略基于当前 Bar (2023-01-01) 的 Close 等数据产生信号。
    *   生成的订单进入 Pending 队列，**不会**在当前 Bar 成交。
4.  **循环结束**: 进入下一个 Bar (2023-01-02)。

#### 场景 B: `ExecutionMode.CurrentClose` (默认，简化)
模拟“收盘价成交”，即信号产生和成交在同一根 Bar 完成 (Cheat-on-Close)。

1.  **加载数据**: 获取当前 Bar (2023-01-01)。
2.  **策略 (Phase 1)**: 调用 `strategy.on_bar(bar)`。
    *   策略产生买入信号。
3.  **撮合 (Phase 2)**: 使用当前 Bar 的 **Close** 价格，**立即**撮合刚才生成的订单。
    *   *逻辑*: 看到收盘价符合条件，瞬间以收盘价买入（实际交易中很难做到）。

### 3.2 订单全生命周期

1.  **Signal**: 策略调用 `self.buy(symbol='000001', price=10.0)`。
2.  **Creation**: `Strategy` 调用 `Sizer` 计算数量，构建 `Order` 对象 (Status: `New`)。
3.  **Submission**: 订单被推送到 `Engine` 的 `pending_orders` 队列。
4.  **Risk Check (Rust)**:
    *   `Engine` 调用 `RiskManager.check(order)`。
    *   如果违反风控规则（如超限额、黑名单），订单状态变为 `Rejected`，并记录错误日志，**终止后续流程**。
5.  **Matching (Rust)**:
    *   `ExchangeSimulator` 遍历 `pending_orders`。
    *   检查价格条件 (Limit/Stop) 和时间条件 (Open/Close)。
    *   若满足，生成 `Trade` 对象，修改 Order Status 为 `Filled`。
6.  **Settlement (Rust)**:
    *   `Trade` 触发 `Portfolio` 更新：
        *   `Cash` 减少 (含佣金)。
        *   `Positions` (总持仓) 增加。
        *   `MarketModel` 决定是否增加 `Available Positions` (T+0 vs T+1)。
7.  **Reporting**: 订单移入 `Engine.orders` 历史列表，成交记录移入 `Engine.trades`。

## 4. 扩展开发指南

### 4.1 如何添加新的订单类型（例如 Stop Limit）

1.  **Rust 定义**:
    *   `src/model/types.rs`: `OrderType` 枚举添加 `StopLimit`。
    *   `src/model/order.rs`: `Order` 结构体确认字段支持 (需 `trigger_price` 和 `price`)。
2.  **Rust 逻辑**:
    *   `src/execution.rs`: 在 `process_event` 的 `match` 语句中增加 `StopLimit` 分支。
    *   *逻辑*: 先判断是否触发 `trigger_price`，若触发，则将其视为 `Limit` 单进行价格判断。
3.  **Python 接口**:
    *   `python/akquant/akquant.pyi`: 更新 `OrderType` 和 `buy/sell` 函数签名提示。

### 4.2 如何接入新的数据源（例如 CSV 文件）

1.  **Rust 实现**:
    *   修改 `src/data.rs`，目前 `DataFeed` 使用 `Vec<Event>`。
    *   为了支持流式读取，可以定义一个 `Iterator` 接口，或者在 Python 端读取 CSV 转换为 `Vec<Bar>` 传入 (最简单)。
2.  **Python 端适配 (推荐)**:
    *   在 Python 中使用 Pandas 读取 CSV。
    *   转换为 `akquant.Bar` 对象列表。
    *   调用 `engine.add_feed(data)`。

### 4.3 如何自定义仓位管理 (Sizer)

`Sizer` 逻辑完全在 Python 层实现：

1.  **继承**: 继承 `akquant.Sizer`。
2.  **实现**: 重写 `get_size(self, price, cash, context, symbol)`。
3.  **使用**: `strategy.set_sizer(MySizer())`。

```python
class RiskSizer(akquant.Sizer):
    def get_size(self, price, cash, context, symbol):
        # 示例：根据 ATR 波动率控制风险
        # 假设 context 中可以访问 atr 指标 (需自行维护)
        atr = 1.5
        risk_per_share = atr  # 止损宽度
        total_risk = cash * 0.02 # 总资金 2% 风险
        return int(total_risk / risk_per_share)
```

## 5. Rust 与 Python 交互注意事项 (PyO3)

*   **对象生命周期**:
    *   Rust 中的 `Engine` 是长生命周期的。
    *   传递给 Python 的 `StrategyContext` 是**临时快照**，每次 `on_bar` 重新创建。不要在 Python 中长期持有 `context` 对象。
*   **GIL 管理**:
    *   Rust 调用 Python 回调时会自动获取 GIL。
    *   耗时的 Rust 计算 (如 `Engine::run`) 期间主要在 Rust 侧运行，释放了 GIL 压力的机会较少，但由于计算都在 Rust，Python 仅作为胶水，性能影响有限。
*   **类型映射**:
    *   Rust `Vec<T>` <-> Python `List[T]` (会有一次拷贝)。
    *   Rust `HashMap<K, V>` <-> Python `Dict[K, V]`.
    *   为了性能，`Bar` 和 `Order` 等结构体通过 `#[pyclass]` 暴露，避免了频繁的字典转换开销。
