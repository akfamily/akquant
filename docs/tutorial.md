# 手把手教程：编写你的第一个量化策略

本教程将带你一步步编写一个经典的**双均线策略 (Dual Moving Average)**。这通常是量化交易员的"Hello World"。

我们将涵盖从数据准备、策略编写到回测分析的全过程。

## 1. 策略思路

双均线策略的逻辑非常直观：
*   **金叉 (Golden Cross)**: 当短期均线（如 10日线）由下向上穿过长期均线（如 30日线）时，认为趋势向上，**买入**。
*   **死叉 (Death Cross)**: 当短期均线由上向下穿过长期均线时，认为趋势向下，**卖出**。

## 2. 完整代码

为了方便你直接运行，我们将所有代码放在一个文件中。你可以复制以下代码到 `my_first_strategy.py` 并运行。

```python
import pandas as pd
import numpy as np
from akquant import Strategy, run_backtest

# --------------------------
# 第一步：准备数据
# --------------------------
# 在真实场景中，你会使用 pd.read_csv("stock_data.csv")
# 这里为了演示方便，我们生成一些模拟数据
def generate_mock_data():
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    n = len(dates)

    # 生成一条随机走势
    np.random.seed(42) # 固定随机种子，保证每次运行结果一致
    returns = np.random.normal(0.0005, 0.02, n)
    price = 100 * np.cumprod(1 + returns)

    # 构造 DataFrame
    df = pd.DataFrame({
        "date": dates,
        "open": price,
        "high": price * 1.01,
        "low": price * 0.99,
        "close": price,
        "volume": 10000,
        "symbol": "AAPL" # 假设这是苹果公司的股票
    })
    return df

# --------------------------
# 第二步：编写策略
# --------------------------
class DualMAStrategy(Strategy):
    """
    继承 akquant.Strategy 类，这是所有策略的基类。
    """

    def __init__(self, fast_window=10, slow_window=30):
        # 定义策略参数：快线周期和慢线周期
        self.fast_window = fast_window
        self.slow_window = slow_window

    def on_start(self):
        """
        策略启动时执行一次。
        在这里告诉系统我们要关注哪些股票。
        """
        print("策略启动...")
        self.subscribe("AAPL")

        # 关键：设置历史数据缓存长度
        # 1. 性能优化：AKQuant 默认不缓存历史数据，需显式开启
        # 2. 安全冗余：计算 30日均线理论上只需 30 个数据，但 +10 是为了留出安全余量 (Buffer)，
        #    防止边界计算时数组越界，或用于某些指标的预热。
        self.set_history_depth(self.slow_window + 10)

    def on_bar(self, bar):
        """
        核心逻辑：每一根 K 线走完时，都会触发一次这个函数。
        bar 参数包含了当前的行情数据 (bar.close, bar.high 等)。
        """

        # 1. 获取历史收盘价
        # get_history 返回的是一个 numpy 数组，包含最近 N 天的数据
        closes = self.get_history(count=self.slow_window, symbol=bar.symbol, field="close")

        # 如果数据还不够计算长均线（比如刚开始回测的前几天），就直接返回，不操作
        if len(closes) < self.slow_window:
            return

        # 2. 计算均线
        # 使用 numpy 计算平均值
        fast_ma = np.mean(closes[-self.fast_window:]) # 取最后 fast_window 个数据求平均
        slow_ma = np.mean(closes[-self.slow_window:]) # 取最后 slow_window 个数据求平均

        # 3. 获取当前持仓
        # 如果没持仓返回 0，持有 1000 股返回 1000
        position = self.get_position(bar.symbol)

        # 4. 交易信号判断
        # 金叉：短线 > 长线，且当前空仓 -> 买入
        if fast_ma > slow_ma and position == 0:
            print(f"[{bar.timestamp_str}] 金叉买入! 价格: {bar.close:.2f}")
            self.buy(symbol=bar.symbol, quantity=1000)

        # 死叉：短线 < 长线，且当前持仓 -> 卖出
        elif fast_ma < slow_ma and position > 0:
            print(f"[{bar.timestamp_str}] 死叉卖出! 价格: {bar.close:.2f}")
            self.sell(symbol=bar.symbol, quantity=position) # 卖出所有持仓

# --------------------------
# 第三步：运行回测
# --------------------------
if __name__ == "__main__":
    # 1. 获取数据
    df = generate_mock_data()

    # 2. 运行回测
    print("开始回测...")
    result = run_backtest(
        data=df,
        strategy=DualMAStrategy, # 传入我们的策略类
        strategy_params={"fast_window": 10, "slow_window": 30}, # 调整参数
        cash=100_000.0,    # 初始资金 10万
        commission=0.0003  # 佣金万分之三
    )

    # 3. 打印结果
    print("\n" + "="*30)
    print(result) # 打印详细的绩效报表
    print("="*30)
```

## 3. 代码详细解析

### 3.1 策略结构
每一个 AKQuant 策略都是一个 Python 类，继承自 `Strategy`。你需要关注三个主要方法：

*   **`__init__`**: 设置策略的参数（如均线周期）。
*   **`on_start`**: 初始化工作。**最重要的是调用 `self.set_history_depth(N)`**。
    *   **性能考量**：AKQuant 为了极致性能，默认设计是"不缓存历史数据"的（只处理当前 bar）。
    *   **数据预热**：如果你需要"回头看"（如计算过去 30 天均线），必须显式告诉系统保留多少历史数据。
    *   **安全冗余**：建议设置的值比实际计算周期稍大（例如 +10），以防止数组越界并满足某些指标的预热需求。
*   **`on_bar`**: 这是一个死循环，系统会按时间顺序把每一根 K 线传给你。你在这里做决策：买还是卖？

### 3.2 获取数据
*   `bar.close`: 当前这根 K 线的收盘价。
*   `self.get_history(N, symbol, "close")`: 获取过去 N 天的收盘价数组。这是最高效的数据访问方式。

### 3.3 下单交易
*   `self.buy(symbol, quantity)`: 发出买单。默认是市价单（按当前价格成交）。
*   `self.sell(symbol, quantity)`: 发出卖单。

## 4. 常见问题 (FAQ)

**Q: 为什么程序报错 `RuntimeError: History tracking is not enabled`？**
A: 这是因为你调用了 `get_history` 但忘记在 `on_start` 中调用 `self.set_history_depth()` 了。系统默认为了性能不开启历史缓存，必须显式开启。

**Q: 为什么 `get_history` 返回了很多 `NaN`？**
A: 这通常发生在回测刚开始阶段。比如你设置了 depth=30，但在第 5 天就调用了 `get_history(30)`。此时数据不足，系统会自动在前面填充 `NaN` 以保证返回数组长度一致。建议在 `on_bar` 开头判断 `if len(closes) < N: return`。

**Q: 回测结果里的 `sharpe_ratio` 是什么？**
A: 夏普比率，衡量策略性价比的指标。大于 1 通常被认为是还可以的策略，大于 2 是非常优秀的策略。

**Q: 如何换成我自己的数据？**
A: 只要将 `data=df` 替换为你自己的 DataFrame 即可。确保你的 DataFrame 包含 `date`, `open`, `high`, `low`, `close`, `volume`, `symbol` 这几列。
