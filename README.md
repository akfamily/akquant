# akquant

**akquant** 是一个基于 **Rust** 和 **Python** 构建的高性能量化投研框架。它旨在结合 Rust 的极致性能和 Python 的易用性，为量化交易者提供强大的回测和研究工具。

最新版本参考了 [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) 和 [PyBroker](https://github.com/edtechre/pybroker) 的架构理念，引入了模块化设计、独立的投资组合管理、高级订单类型支持以及便捷的数据加载与缓存机制。

📖 **[设计与开发指南 (DESIGN.md)](DESIGN.md)**: 如果你想深入了解内部架构、学习如何设计此类系统或进行二次开发，请阅读此文档。

## 核心特性

*   **极致性能**: 核心回测引擎采用 Rust 编写，通过 PyO3 提供 Python 接口。
    *   **基准测试**: 在 200k K线数据的 SMA 策略回测中，akquant 耗时仅 **1.31s** (吞吐量 ~152k bars/sec)，相比 Backtrader (26.55s) 和 PyBroker (23.61s) 快约 **20倍**。
*   **模块化架构**:
    *   **Engine**: 事件驱动的核心撮合引擎，采用二进制堆 (BinaryHeap) 管理事件队列。
    *   **Clock**: 参考 NautilusTrader 设计的交易时钟，精确管理交易时段 (TradingSession) 和时间流逝。
    *   **Portfolio**: 独立的投资组合管理，支持实时权益计算。
    *   **MarketModel**: 可插拔的市场模型，内置 A 股 T+1 和期货 T+0 规则。
*   **事件系统**:
    *   **Timer**: 支持 `schedule(timestamp, payload)` 注册定时事件，触发 `on_timer` 回调，实现复杂的盘中定时逻辑。
*   **数据生态**:
    *   **AKShare 集成**: 内置 `DataLoader`，无缝支持 [AKShare](https://github.com/akfamily/akshare) 数据加载。
    *   **智能缓存**: 支持数据本地缓存 (Pickle)，避免重复下载，加速策略迭代。
    *   **自动降级**: 如果默认缓存目录 (`~/.akquant`) 不可写，自动降级到当前目录 (`.akquant_cache`)。
*   **多资产支持**:
    *   **股票 (Stock)**: 默认支持 T+1，买入 100 股一手限制，印花税/过户费。
    *   **基金 (Fund)**: 支持基金特有费率配置。
    *   **期货 (Futures)**: 支持 T+0，保证金交易，合约乘数。
    *   **期权 (Option)**: 支持 Call/Put，行权价，按张收费模式。
*   **灵活配置**:
    *   **StrategyConfig**: 全局策略配置 (类似 PyBroker)，支持资金管理、费率模式等设置。
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

## 为什么选择 akquant?

传统的 Python 回测框架（如 backtrader）在处理大规模数据或复杂逻辑时往往面临性能瓶颈。纯 C++/Rust 框架虽然性能优越，但开发和调试门槛较高。

**akquant** 试图在两者之间找到平衡点：

1.  **性能**: Rust 核心保证了回测速度，特别适合大规模参数优化。
2.  **易用**: 策略编写完全使用 Python，提供类似 PyBroker 的简洁 API。
3.  **专业**: 严格遵守中国市场交易规则（T+1、印花税、最低佣金等）。

## 前置要求

- **Rust**: [安装 Rust](https://www.rust-lang.org/tools/install)
- **Python**: 3.9+
- **Maturin**: `pip install maturin`

## 安装说明

### 开发模式（推荐）

如果你正在开发该项目并希望更改即时生效：

```bash
maturin develop
```

## 快速开始

### 1. 使用 helper 快速回测 (推荐)

`akquant` 提供了一个类似 Zipline 的便捷入口 `run_backtest`，可以快速运行策略。

```python
import akquant
from akquant.backtest import run_backtest
from akquant import Strategy

# 1. 定义策略
class MyStrategy(Strategy):
    def on_bar(self, bar):
        # 简单的双均线逻辑 (示例)
        # 实际回测推荐使用 IndicatorSet 进行向量化计算
        if self.ctx.position.size == 0:
            self.buy(symbol=bar.symbol, quantity=100)
        elif bar.close > self.ctx.position.avg_price * 1.1:
            self.sell(symbol=bar.symbol, quantity=100)

# 2. 运行回测
# 自动加载数据、设置资金、费率等
result = run_backtest(
    strategy=MyStrategy,  # 传递类或实例
    symbol="600000",      # 浦发银行
    start_date="20230101",
    end_date="20241231",
    cash=500_000.0,       # 初始资金
    commission=0.0003     # 万三佣金
)

# 3. 查看结果
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
