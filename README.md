<p align="center">
  <img src="assets/logo.svg" alt="AKQuant" width="400">
</p>

<p align="center">
    <a href="https://pypi.org/project/akquant/">
        <img src="https://img.shields.io/pypi/v/akquant?style=flat-square&color=007ec6" alt="PyPI Version">
    </a>
    <a href="https://pypi.org/project/akquant/">
        <img src="https://img.shields.io/pypi/pyversions/akquant?style=flat-square" alt="Python Versions">
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
    </a>
</p>

# AKQuant

**AKQuant** 是一个基于 **Rust** 和 **Python** 构建的高性能量化投研框架。它旨在结合 Rust 的极致性能和 Python 的易用性，为量化交易者提供强大的回测和研究工具。

最新版本参考了 [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) 、 [PyBroker](https://github.com/edtechre/pybroker) 和 [Backtrader](https://github.com/mementum/backtrader) 的架构理念，引入了模块化设计、独立的投资组合管理、高级订单类型支持以及便捷的数据加载与缓存机制。

📖 **[设计与开发指南 (DESIGN.md)](DESIGN.md)**: 如果你想深入了解内部架构、学习如何设计此类系统或进行二次开发，请阅读此文档。

## 核心特性

*   **极致性能**: 核心回测引擎采用 Rust 编写，通过 PyO3 提供 Python 接口。
    *   **基准测试**: 在 200k K线数据的 SMA 策略回测中，AKQuant 耗时仅 **1.31s** (吞吐量 ~152k bars/sec)，相比 Backtrader (26.55s) 和 PyBroker (23.61s) 快约 **20倍**。
    *   **Zero-Copy Access (New)**: 历史数据 (`ctx.history`) 通过 PyO3 Buffer Protocol / Numpy View 直接映射 Rust 内存，实现零拷贝访问，大幅提升 Python 端指标计算性能。
*   **模块化架构**:
    *   **Engine**: 事件驱动的核心撮合引擎，采用二进制堆 (BinaryHeap) 管理事件队列。
    *   **Clock**: 参考 NautilusTrader 设计的交易时钟，精确管理交易时段 (TradingSession) 和时间流逝。
    *   **Portfolio**: 独立的投资组合管理，支持实时权益计算。
    *   **MarketModel**: 可插拔的市场模型，内置 A 股 T+1 和期货 T+0 规则。
        *   **T+1 严格风控**: 针对股票/基金，严格执行 T+1 可用持仓检查，防止当日买入当日卖出（除非配置为 T+0 市场）。
        *   **可用持仓管理**: 自动维护 `available_positions`，并扣除未成交的卖单冻结数量，防止超卖。
*   **事件系统**:
    *   **Callbacks (New)**: 支持 `on_order` (订单状态变化) 和 `on_trade` (成交回报) 回调，方便实现自定义日志、通知或复杂状态管理。
    *   **Timer**: 支持 `schedule(timestamp, payload)` 注册定时事件，触发 `on_timer` 回调，实现复杂的盘中定时逻辑。
*   **风控系统 (New)**:
    *   **独立拦截层**: 内置 `RiskManager`，在 Rust 引擎层直接拦截违规订单。
    *   **可用持仓检查**: 下单前实时检查可用持仓（Available - Pending Sell），防止超卖违规。
    *   **灵活配置**: 通过 `RiskConfig` 可配置最大单笔金额、最大持仓比例、黑名单等。
*   **机器学习 (New)**:
    *   **Walk-forward Validation**: 内置滚动训练框架，彻底杜绝未来函数。
    *   **统一适配器**: 提供 `SklearnAdapter` 和 `PyTorchAdapter`，统一 Scikit-learn 和 PyTorch 接口。
    *   **信号解耦**: 提倡 Signal (预测) 与 Action (执行) 分离的设计模式。
*   **数据生态**:
    *   **Streaming CSV (New)**: 支持流式加载超大 CSV 文件 (`DataFeed.from_csv`)，极大降低内存占用。
    *   **Live Trading (New)**: 支持通过 `DataFeed.create_live()` 创建实时数据源，支持 CTP/Gateway 实时数据推送。
    *   **Parquet Data Catalog (New)**: 采用 Apache Parquet 格式存储数据，相比 Pickle 读写速度更快，压缩率更高，便于跨语言使用。
    *   **Pandas 集成**: 支持直接加载 Pandas DataFrame 数据，兼容各类数据源。
    *   **显式订阅**: 策略通过 `subscribe` 方法明确声明所需数据，引擎自动按需加载。
*   **多资产支持**:
    *   **股票 (Stock)**: 默认支持 T+1，买入 100 股一手限制，印花税/过户费。
    *   **基金 (Fund)**: 支持基金特有费率配置。
    *   **期货 (Futures)**: 支持 T+0，保证金交易，合约乘数。
    *   **期权 (Option)**: 支持 Call/Put，行权价，按张收费模式。
*   **高级订单 (New)**:
    *   **Stop Orders**: Rust 引擎原生支持止损单触发，提供 StopMarket 和 StopLimit。
    *   **Target Position**: 内置 `order_target_value` 等辅助函数，自动计算调仓数量。
*   **架构抽象 (New)**:
    *   **ExecutionClient**: 抽象执行层，支持 `SimulatedExecutionClient` (内存撮合) 和 `RealtimeExecutionClient` (实盘对接)。
    *   **DataClient**: 抽象数据层，支持 `SimulatedDataClient` (内存/回放) 和 `RealtimeDataClient` (实时流)。
    *   **无缝切换**: 策略代码无需修改，仅需通过 `engine.use_realtime_execution()` 和 `DataFeed.create_live()` 即可切换至实盘模式。
*   **灵活配置**:
    *   **Typed Config (New)**: 引入 `BacktestConfig`, `StrategyConfig`, `RiskConfig` 类型化配置对象，替代散乱的 `**kwargs`，提供更好的 IDE 提示和参数校验。
    *   **ExecutionMode**: 支持 `CurrentClose` (信号当根K线收盘成交) 和 `NextOpen` (次日开盘成交) 模式。
*   **丰富的分析工具**:
    *   **PerformanceMetrics**:
        *   **收益**: Total Return, Annualized Return, Alpha.
        *   **风险**: Max Drawdown, Sharpe Ratio, Sortino Ratio, Ulcer Index, UPI (Ulcer Performance Index).
        *   **拟合**: Equity R² (线性回归拟合度).
    *   **TradeAnalyzer**: 包含胜率、盈亏比、最大连续盈亏等详细交易统计，支持未结盈亏 (Unrealized PnL) 计算。
*   **仿真增强**:
    *   **滑点模型 (Slippage)**: 支持 Fixed (固定金额) 和 Percent (百分比) 滑点模型，模拟真实交易成本。
    *   **成交量限制 (Volume Limit)**: 支持按 K 线成交量比例限制单笔撮合数量，并实现分批成交 (Partial Fill)。

## 为什么选择 AKQuant?

### 1. 极致性能：Rust 核心 + Python 生态 (Hybrid Architecture)
这是 AKQuant 最大的差异化优势。
*   **Rust 驱动引擎**：核心的回测循环、订单撮合、资金管理、风控检查以及技术指标（如 SMA, RSI）全部由 **Rust** 实现。这使得回测速度比纯 Python 框架（如 Backtrader, Zipline）快 10-100 倍。
*   **零拷贝数据访问**：利用 PyO3 的 Buffer Protocol，Python 策略可以直接读取 Rust 管理的内存数据（如 K 线历史），无需昂贵的数据复制开销。
*   **增量计算**：内置指标采用增量算法（Incremental Calculation），新来一个 Bar 只计算最新值，避免了全量重算。

### 2. 专为机器学习设计的原生支持 (Native ML Support)
大多数传统回测框架对 ML 的支持非常薄弱，而 AKQuant 将其作为一等公民：
*   **防未来函数机制**：框架层面强制执行 **Walk-forward Validation**（滚动/前向验证），彻底杜绝了使用未来数据训练模型的可能性。
*   **统一适配器**：通过 `QuantModel` 统一接口，无缝集成 **Scikit-learn** (XGBoost/LightGBM) 和 **PyTorch** (深度学习)，无需编写复杂的胶水代码。
*   **训练/推理分离**：明确区分 `prepare_features` (训练) 和 `on_bar` (推理)，符合工业界 ML 落地流程。

### 3. 精准且灵活的事件驱动引擎 (Event-Driven Engine)
*   **高保真模拟**：支持多种撮合模式（`NextOpen` 默认模式模拟T+1开盘撮合，`CurrentClose` 模拟收盘撮合），真实还原实盘交易环境。
*   **复杂订单支持**：支持市价单、限价单、止损单、目标仓位单（Target Orders），并内置了 T+1 可用持仓检查等 A 股特色规则。
*   **多资产混合**：可以在同一个策略中同时交易 **股票、期货、期权**，并支持为不同标的独立配置合约乘数、保证金率和最小变动价位。

### 4. 生产级风控与实盘能力 (Risk & Live Ready)
*   **内置风控管理器**：在 Rust 层实现了事前风控（Pre-trade Risk Check），包括资金上限、持仓限制、黑名单检查等。这不仅用于回测，也直接保护实盘交易。
*   **无缝实盘切换**：通过 `LiveRunner`，回测代码几乎无需修改即可切换到实盘/仿真模式（支持 CTP 接口）。
*   **数据聚合器**：内置 `BarAggregator`，能够将高频 Tick 数据实时聚合为分钟级 Bar，支持 Tick 级策略和 Bar 级策略的灵活切换。

### 5. 开发者体验优化 (Developer Experience)
*   **LLM 友好**：专门编写了 `llm_guide.md`，提供了标准化的 Prompt 模板，使得使用 ChatGPT/Claude 辅助编写策略变得非常容易和准确。
*   **双风格 API**：同时支持 **类风格 (Class-based)**（适合复杂逻辑、状态管理）和 **函数风格**（适合快速原型），满足不同层次开发者的需求。
*   **严格的类型检查**：代码库全面支持 Type Hints，通过 `mypy` 和 `ruff` 保证代码质量，减少运行时错误。

## 安装说明

**AKQuant** 已发布至 PyPI，支持 Windows, macOS 和 Linux。由于核心引擎预编译为 Wheel 包，**无需安装 Rust 环境**即可直接使用。

### 1. 快速安装 (推荐)

```bash
pip install akquant
```

### 2. 验证安装

```bash
python -c "import akquant; print(f'AKQuant v{akquant.__version__} installed successfully!')"
```

### 3. 本地开发 (贡献者)

如果你希望参与开发或修改源码，请参考 [贡献指南 (CONTRIBUTING.md)](CONTRIBUTING.md) 进行环境搭建（需要 Rust 工具链）。

## 快速开始

### 1. 使用 helper 快速回测 (推荐)

`AKQuant` 提供了一个类似 Zipline 的便捷入口 `run_backtest`，可以快速运行策略。

```python
import akquant
from akquant.backtest import run_backtest
from akquant import Strategy
from akquant.config import BacktestConfig

# 1. 定义策略
class MyStrategy(Strategy):
    def on_start(self):
        # 显式订阅数据
        self.subscribe("600000")

    def on_bar(self, bar):
        # 简单的双均线逻辑 (示例)
        # 实际回测推荐使用 IndicatorSet 进行向量化计算
        if self.ctx.position.size == 0:
            self.buy(symbol=bar.symbol, quantity=100)
        elif bar.close > self.ctx.position.avg_price * 1.1:
            self.sell(symbol=bar.symbol, quantity=100)

# 2. 配置回测
config = BacktestConfig(
    start_date="20230101",
    end_date="20241231",
    cash=500_000.0,
    commission=0.0003
)

# 3. 运行回测
# 自动加载数据、设置资金、费率等
result = run_backtest(
    strategy=MyStrategy,  # 传递类
    config=config         # 传递配置对象
)

# 4. 查看结果
print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")

# 4. 获取详细数据 (DataFrame)
# 绩效指标表
print(result.metrics_df)
# 交易记录表
print(result.trades_df)
# 每日持仓表
print(result.daily_positions_df)
```

### 2. 函数式 API (Zipline 风格)

如果你习惯 Zipline 或 Backtrader 的函数式写法，也可以直接使用：

```python
from akquant.backtest import run_backtest

def initialize(ctx):
    ctx.stop_loss_pct = 0.05

def on_bar(ctx, bar):
    if ctx.position.size == 0:
        ctx.buy(symbol=bar.symbol, quantity=100)
    elif bar.close < ctx.position.avg_price * (1 - ctx.stop_loss_pct):
        ctx.sell(symbol=bar.symbol, quantity=100)

run_backtest(
    strategy=on_bar,
    initialize=initialize,
    symbol="600000",
    start_date="20230101",
    end_date="20231231"
)
```

更多示例请参考 `examples/` 目录。

## 结果分析
`run_backtest` 返回的 `BacktestResult` 对象提供了丰富的数据用于后续分析：

*   **`result.metrics`**: 包含 Total Return, Sharpe, Max Drawdown 等核心指标的对象。
*   **`result.metrics_df`**: 包含上述指标的 Pandas DataFrame (单行)。
*   **`result.trades_df`**: 包含所有已平仓交易的详细记录 (Entry/Exit Time/Price, PnL, Commission 等)。
*   **`result.daily_positions_df`**: 包含每日持仓快照的 DataFrame。
*   **`result.equity_curve`**: 权益曲线数据列表 `[(timestamp, equity), ...]`。

## 快速链接
