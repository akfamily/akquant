<p align="center">
  <img src="../assets/akquant-logo.svg" alt="AKQuant Logo" width="800" />
</p>

---

**AKQuant** 是一款专为 **量化投研 (Quantitative Research)** 打造的 **高性能混合架构引擎**。它以 **Rust** 铸造极速撮合内核，以 **Python** 链接数据与 AI 生态，旨在为量化投资者提供可靠高效的解决方案。

它超越了传统工具的范畴，将 **事件驱动**、**机器学习** 与 **生产级风控** 深度融合，让 **量化交易** 不再受限于计算性能，专注于策略本身的逻辑与价值。

## 核心特性

*   **高性能内核**: Rust 核心引擎 + Python 接口，旨在减少事件驱动回测中的 Python 开销；实际性能表现取决于策略实现、数据规模与运行环境。
*   **原生机器学习**: 内置 Walk-forward Validation 和 PyTorch/Scikit-learn 适配器。
*   **生产级风控**: 内置 Rust 层 RiskManager，严格执行 T+1 和资金风控。
*   **高效数据通路**: 数据导入 (`add_arrays`) 借用 NumPy 缓冲、零拷贝入引擎；历史数据读取 (`get_history`) 返回安全快照拷贝（窗口小、开销可忽略），多字段可用 `get_history_multi` 单次取回。
*   **灵活架构**: 事件驱动设计，支持盘中定时任务和多资产混合回测。

👉 **[查看完整架构与特性文档](meta/architecture.md)**

## 安装

详细安装步骤请参考 **[安装指南](start/installation.md)**。

## 快速开始

### 1. 使用 `run_backtest` 快速回测 (推荐)

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
        position = self.get_position(bar.symbol)
        if position == 0:
            self.buy(symbol=bar.symbol, quantity=100)
        elif position > 0:
            self.sell(symbol=bar.symbol, quantity=100)

# 3. 运行回测
df = generate_data()
result = run_backtest(
    strategy=MyStrategy,  # 传递类或实例
    data=df,              # 显式传入数据
    symbols="600000",      # 浦发银行
    initial_cash=500_000.0,       # 初始资金
    commission_rate=0.0003     # 万三佣金
)

# 4. 查看结果
print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")

# 5. 获取详细数据 (DataFrame)
# 绩效指标表
print(result.metrics_df)
# 交易记录表
print(result.trades_df)
# 每日持仓表
print(result.positions_df)
```

### 2. 函数式 API

如果你习惯 Zipline 或 Backtrader 的函数式写法，也可以直接使用：

```python
from akquant import run_backtest

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
    symbols="600000"
)
```

### 3. 使用自定义因子 (Custom Factors)

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
run_backtest(strategy=MyStrategy, data=df, symbols="600000")
```

更多示例请参考 `examples/` 目录。

## 参数模型化入口

*   参数优化指南（推荐工作流）：[参数模型驱动优化](guide/optimization.md)
*   示例集合（页面化参数配置）：[页面化参数配置（PARAM_MODEL）](guide/examples.md)
*   API 最佳实践（前后端联动）：[页面化参数输入](reference/api.md)
*   示例脚本入口：[examples/02_parameter_optimization.py](https://github.com/akfamily/akquant/blob/main/examples/02_parameter_optimization.py)

## 快速入口

*   快速查看迁移 FAQ（快速开始）：[快速开始中的阶段 5 迁移 FAQ](start/quickstart.md)
*   查看完整兼容说明（API 参考）：[API 兼容与迁移说明](reference/api.md)
*   理解时间与时区语义（量化基础）：[AKQuant 的时间与时区](guide/quant_basics.md)
*   查看时区 FAQ 与数据准备建议（进阶专题）：[时区处理指南](advanced/timezone.md)
*   多策略清单（进阶专题）：[多策略指南](advanced/multi_strategy_guide.md)
*   动态策略加载说明（进阶专题）：[运行时配置指南](advanced/runtime_config.md)
*   动态策略加载示例（examples）：[44_strategy_source_loader_demo.py](https://github.com/akfamily/akquant/blob/main/examples/44_strategy_source_loader_demo.py)
*   动态策略加载教材章节（examples/textbook）：[ch15_strategy_loader.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch15_strategy_loader.py)
*   热启动状态持久化与恢复：[热启动指南](advanced/warm_start.md)

## 策略实战入口

*   横截面策略实战清单：[横截面策略实战清单](guide/cross_section_checklist.md)
*   自定义指标开发主入口：[自定义指标指南](guide/custom_indicator.md)
*   指标注册与模式切换：[策略开发手册](guide/strategy.md)
