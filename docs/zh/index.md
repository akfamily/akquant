# AKQuant

**AKQuant** 是一个基于 **Rust** 和 **Python** 构建的高性能量化投研框架。它旨在结合 Rust 的极致性能和 Python 的易用性，为量化交易者提供强大的回测和研究工具。

最新版本参考了 [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) 和 [PyBroker](https://github.com/edtechre/pybroker) 的架构理念，引入了模块化设计、独立的投资组合管理、高级订单类型支持以及便捷的数据加载与缓存机制。

📖 **[设计与开发指南 (DESIGN.md)](design.md)**: 如果你想深入了解内部架构、学习如何设计此类系统或进行二次开发，请阅读此文档。

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
    *   **Timer**: 支持 `schedule(timestamp, payload)` 注册定时事件，触发 `on_timer` 回调，实现复杂的盘中定时逻辑。
*   **风控系统 (New)**:
    *   **独立拦截层**: 内置 `RiskManager`，在 Rust 引擎层直接拦截违规订单。
    *   **可用持仓检查**: 下单前实时检查可用持仓（Available - Pending Sell），防止超卖违规。
    *   **灵活配置**: 通过 `RiskConfig` 可配置最大单笔金额、最大持仓比例、黑名单等。
*   **数据生态**:
    *   **Streaming CSV (New)**: 支持流式加载超大 CSV 文件 (`DataFeed.from_csv`)，极大降低内存占用。
    *   **Pandas 集成**: 支持直接加载 Pandas DataFrame 数据，兼容各类数据源。
    *   **智能缓存**: 支持数据本地缓存 (Pickle)，避免重复下载，加速策略迭代。
*   **机器学习 (New)**:
    *   **ML Framework**: 内置高性能机器学习训练框架，支持 Walk-forward Validation (滚动训练)。
    *   **Adapter Pattern**: 统一 Scikit-learn 和 PyTorch 接口，解耦模型与策略逻辑。
    *   **📖 [机器学习指南](ml_guide.md)**: 详细了解如何构建 AI 驱动的策略。
*   **灵活配置**:
    *   **StrategyConfig**: 全局策略配置 (类似 PyBroker)，支持资金管理、费率模式等设置。
    *   **ExecutionMode**: 支持 `CurrentClose` (信号当根K线收盘成交) 和 `NextOpen` (次日开盘成交) 模式。
*   **丰富的分析工具**:
    *   **PerformanceMetrics**:
        *   **收益**: Total Return, Annualized Return, Alpha, Win Rate.
        *   **风险**: Max Drawdown, Sharpe Ratio, Sortino Ratio, Ulcer Index, UPI (Ulcer Performance Index).
        *   **拟合**: Equity R² (线性回归拟合度).
    *   **TradeAnalyzer**: 包含胜率、盈亏比、最大连续盈亏等详细交易统计，支持未结盈亏 (Unrealized PnL) 计算。
*   **仿真增强**:
    *   **滑点模型 (Slippage)**: 支持 Fixed (固定金额) 和 Percent (百分比) 滑点模型，模拟真实交易成本。
    *   **成交量限制 (Volume Limit)**: 支持按 K 线成交量比例限制单笔撮合数量，并实现分批成交 (Partial Fill)。

## 为什么选择 AKQuant?

AKQuant 旨在解决传统 Python 回测框架（如 Backtrader）性能不足和纯 C++/Rust 框架开发门槛过高的问题。我们通过混合架构在五个核心维度实现了突破：

### 1. 极致性能：Rust 核心 + Python 生态
*   **混合架构**: 核心计算层（撮合、资金、风控）采用 **Rust** 编写，通过 PyO3 暴露给 Python。
*   **Zero-Copy Access**: 利用 Rust 的 `arrow` 和 `numpy` 视图技术，Python 端访问历史数据（OHLCV、指标）实现 **零拷贝**，避免了大量内存复制开销。
*   **基准测试**: 在 200k K线 SMA 策略测试中，耗时仅 **1.31s** (吞吐量 ~152k bars/sec)，比 Backtrader 快 **20倍**。
*   **增量计算**: 内部指标计算采用增量更新算法，而非全量重算，适合超长历史回测。

### 2. 原生支持机器学习 (Machine Learning First)
*   **内置训练框架**: 不同于传统框架仅支持简单的技术指标，AKQuant 内置了完整的 ML Pipeline。
*   **Walk-forward Validation**: 原生支持滚动窗口训练（Walk-forward），有效防止未来函数和过拟合。
*   **Adapter Pattern**: 提供了 Scikit-learn 和 PyTorch 的统一适配器 (`QuantModel`)，只需几行代码即可将 AI 模型接入策略。
*   **特征工程**: `DataFeed` 支持动态特征计算，方便接入 Talib 或 Pandas 进行特征预处理。

### 3. 精确且灵活的事件驱动引擎
*   **精确仿真**: 基于 **NautilusTrader** 的设计理念，拥有精确的时间流逝模型和订单生命周期管理。
*   **复杂订单支持**: 支持市价单 (Market)、限价单 (Limit)、止损单 (Stop)、止盈单 (TakeProfit) 等多种订单类型。
*   **多资产混合**: 支持股票、期货、ETF 等多资产混合回测，每个资产可独立配置费率、滑点和交易时段。
*   **盘中定时任务**: 支持 `schedule` 注册盘中定时事件（如：每天 14:50 平仓），比单纯的 `on_bar` 更加灵活。

### 4. 生产级风控与实盘能力
*   **内置风控器**: 引擎层内置 `RiskManager`，支持资金上限、持仓比例、黑名单等硬性风控，防止策略失控。
*   **无缝实盘切换**: 策略代码与实盘接口解耦，理论上只需替换 `Broker` 和 `DataFeed` 适配器即可切换至实盘（实盘接口开发中）。
*   **数据聚合器**: `DataFeed` 支持多数据源聚合，能够处理不同频率的数据对齐问题。

### 5. 极致的开发者体验
*   **LLM 友好**: 代码结构清晰，文档详尽，特别优化了类型提示 (Type Hints)，方便 Copilot 或 GPT 辅助编写策略。
*   **双风格 API**: 同时支持 **类 (Class-based)** 和 **函数式 (Zipline-style)** 两种策略编写风格，满足不同用户习惯。
*   **严格类型检查**: 核心逻辑经过 Rust 编译器严格检查，Python 端通过 `mypy` 类型检查，最大限度减少运行时错误。

## 前置要求

- **Rust**: [安装 Rust](https://www.rust-lang.org/tools/install)
- **Python**: 3.10+
- **Maturin**: `pip install maturin`

## 安装说明

### 开发模式（推荐）

如果你正在开发该项目并希望更改即时生效：

```bash
maturin develop
```

## 快速开始

### 1. 使用 helper 快速回测 (推荐)

`AKQuant` 提供了一个类似 Zipline 的便捷入口 `run_backtest`，可以快速运行策略。

```python
import pandas as pd
import numpy as np
from akquant import Strategy, run_backtest

# 1. 准备数据 (示例使用随机数据)
# 实际场景可使用 pd.read_csv("data.csv")
def generate_data():
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    n = len(dates)
    price = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
    return pd.DataFrame({
        "date": dates,
        "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
        "volume": 10000,
        "symbol": "600000"
    })

# 2. 定义策略
class MyStrategy(Strategy):
    def on_bar(self, bar):
        # 简单的策略逻辑 (示例)
        # 实际回测推荐使用 IndicatorSet 进行向量化计算
        position = self.ctx.get_position(bar.symbol)
        if position == 0:
            self.buy(symbol=bar.symbol, quantity=100)
        elif position > 0:
            self.sell(symbol=bar.symbol, quantity=100)

# 3. 运行回测
df = generate_data()
result = run_backtest(
    strategy=MyStrategy,  # 传递类或实例
    data=df,              # 显式传入数据
    symbol="600000",      # 浦发银行
    cash=500_000.0,       # 初始资金
    commission=0.0003     # 万三佣金
)

# 4. 查看结果
print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")

# 5. 获取详细数据 (DataFrame)
# 绩效指标表 (使用 .T 转置为竖排显示，方便阅读)
print(result.metrics_df.T)
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
    position = ctx.get_position(bar.symbol)
    if position == 0:
        ctx.buy(symbol=bar.symbol, quantity=100)
    elif position > 0:
        ctx.sell(symbol=bar.symbol, quantity=100)

run_backtest(
    strategy=on_bar,
    initialize=initialize,
    data=df, # 使用上文生成的数据
    symbol="600000"
)
```

### 3. 使用自定义因子数据 (Custom Factors)

AKQuant 支持在 `DataFrame` 中传入任意数量的自定义数值字段（如因子、信号等），并在 `on_bar` 中通过 `bar.extra` 字典访问。

```python
import pandas as pd
import numpy as np
from akquant import Strategy, run_backtest

# 1. 准备数据
def generate_data():
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    n = len(dates)
    price = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
    return pd.DataFrame({
        "date": dates,
        "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
        "volume": 10000,
        "symbol": "600000"
    })

df = generate_data()

# 2. 增加自定义因子 (必须是数值类型)
df["momentum"] = df["close"] / df["open"]       # 因子 1
df["volatility"] = df["high"] - df["low"]       # 因子 2
df["sentiment_score"] = np.random.rand(len(df)) # 因子 3

# 3. 在策略中同时访问这些字段
class MyStrategy(Strategy):
    def on_bar(self, bar):
        # 通过键名访问 (返回 float 类型)
        mom = bar.extra.get("momentum", 0.0)
        vol = bar.extra.get("volatility", 0.0)
        score = bar.extra.get("sentiment_score", 0.0)

        # 综合判断
        if mom > 1.02 and score > 0.8:
            self.buy(bar.symbol, 100)

# 4. 运行回测
run_backtest(strategy=MyStrategy, data=df, symbol="600000")
```

更多示例请参考 `examples/` 目录。

## 快速链接
