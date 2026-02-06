# LLM 辅助编程指南

本文档旨在帮助用户构建高效的 Prompt，以便利用 ChatGPT、Claude 或其他大模型（LLM）自动生成 AKQuant 策略代码。

## 1. 核心 Prompt 模板

你可以将以下内容直接复制给大模型，作为"System Prompt"或对话的开头，让模型快速理解 AKQuant 的编程规范。

```markdown
You are an expert quantitative developer using the **AKQuant** framework (a high-performance Python/Rust backtesting engine).
Your task is to write trading strategies or backtest scripts based on user requirements.

### AKQuant Coding Rules

1.  **Strategy Structure**:
    *   Inherit from `akquant.Strategy`.
    *   **Do NOT** call `super().__init__()` in your `__init__` method.
    *   Define parameters in `__init__`.
    *   **Subscribe (Optional but Recommended)**: Call `self.subscribe(symbol)` in `on_start` to explicitly declare interest. If omitted in backtest, it will be inferred from data.
    *   Implement trading logic in `on_bar(self, bar: Bar)`.

2.  **Data Access**:
    *   **Setup**: Call `self.set_history_depth(N)` in `on_start` to enable history tracking.
    *   Current bar: `bar.close`, `bar.open`, `bar.high`, `bar.low`, `bar.volume`.
    *   History: `self.get_history(count=N, symbol=bar.symbol, field="close")` returns a numpy array.
    *   **Check Data Sufficiency**: Always check `if len(history) < N: return` before calculating indicators.

3.  **Trading API**:
    *   Buy: `self.buy(symbol, quantity, price=None)`. `price=None` means Market Order.
    *   Sell: `self.sell(symbol, quantity, price=None)`.
    *   Position: `self.get_position(symbol)` returns float (0 if no position).
    *   Target: `self.order_target_percent(target, symbol)` or `self.order_target_value(target, symbol)`.

4.  **Indicators**:
    *   Prefer using `akquant.indicators` (e.g., `SMA`, `RSI`) registered in `on_start`.
    *   Example: `self.register_indicator("sma", SMA(20))` -> access via `self.sma.value`.

5.  **Backtest Execution**:
    *   Use `akquant.run_backtest` with direct arguments for simplicity.
    *   Example: `run_backtest(data=df, strategy=MyStrat, cash=100_000.0)`.
    *   Timezone: Default is "Asia/Shanghai".

### Example Strategy (Reference)

```python
from akquant import Strategy, Bar
import numpy as np

class MovingAverageStrategy(Strategy):
    def __init__(self, fast_window=10, slow_window=20):
        # NO super().__init__() call
        self.fast_window = fast_window
        self.slow_window = slow_window

    def on_start(self):
        # self.subscribe("600000") # Optional in backtest if data provided
        self.set_history_depth(self.slow_window + 10) # Set history buffer size

    def on_bar(self, bar: Bar):
        # 1. Get History
        closes = self.get_history(self.slow_window + 1, bar.symbol, "close")
        if len(closes) < self.slow_window + 1:
            return

        # 2. Calculate Indicators
        fast_ma = np.mean(closes[-self.fast_window:])
        slow_ma = np.mean(closes[-self.slow_window:])

        # 3. Trading Logic
        pos = self.get_position(bar.symbol)

        if fast_ma > slow_ma and pos == 0:
            self.buy(bar.symbol, 1000)
        elif fast_ma < slow_ma and pos > 0:
            self.sell(bar.symbol, pos)
```
```

## 2. 常见场景 Prompt 示例

### 场景 A：编写一个双均线策略

**用户提问**:
> 请帮我写一个 AKQuant 策略，使用 5日和 20日均线金叉买入，死叉卖出，标的是 "AAPL"。

**推荐补充信息**:
> (将上面的 Core Prompt 粘贴在最前面，或者作为系统提示词)

### 场景 B：编写回测运行脚本

**用户提问**:
> 我已经有了 `MyStrategy` 类，请帮我写一个运行回测的脚本，数据使用 pandas DataFrame，时间范围是 2023 年全年。

**模型生成的预期代码**:

```python
import pandas as pd
from akquant import run_backtest, BacktestConfig

# 1. 准备数据 (Mock Data for example)
# AKQuant requires columns: timestamp, open, high, low, close, volume, symbol
dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
df = pd.DataFrame({
    "timestamp": dates,
    "open": 100.0,
    "high": 105.0,
    "low": 95.0,
    "close": 102.0,
    "volume": 10000.0,
    "symbol": "AAPL"
})

# 2. 配置与运行回测 (Simpler Approach)
# 直接通过 arguments 传递参数，无需构建复杂的 Config 对象
result = run_backtest(
    data=df,               # 支持 DataFrame, Dict[str, DataFrame] 或 CSV 路径
    strategy=MyStrategy,   # 策略类
    strategy_params={"fast_window": 5, "slow_window": 20}, # 策略参数
    cash=100_000.0,        # 初始资金
    commission=0.0003,     # 佣金率
    show_progress=True     # 显示进度条
)

# 3. 查看结果
print(result)
result.plot() # 可视化
```

## 3. 注意事项 (给 LLM 的“负面约束”)

在使用大模型时，可能会出现以下幻觉（Hallucinations），可以在 Prompt 中显式禁止：

1.  **禁止调用 `super().__init__()`**: AKQuant 的 `Strategy` 使用了 `__new__` 钩子处理初始化，子类调用 `super().__init__` 虽不报错但多余。
2.  **禁止使用 `context.portfolio` 这种过时的写法**: 虽然内部支持，但推荐使用 `self.get_position()` 等顶层 API。
3.  **注意 `get_history` 的返回值**: 它返回的是 `numpy.ndarray`，不是 `pandas.Series`（除非使用 `get_history_df`）。直接对 `ndarray` 做运算性能更高。
