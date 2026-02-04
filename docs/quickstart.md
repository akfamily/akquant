# 快速开始

本指南将带您快速上手 `AKQuant`。我们将从一个最简单的 "Hello World" 策略开始，展示如何定义策略、生成模拟数据并运行回测。

## 1. 安装

确保您已经安装了 `akquant` (及 `maturin` 编译环境，如果是从源码安装)。

```bash
pip install akquant
```

## 2. Hello World: 双均线策略

创建一个名为 `hello_akquant.py` 的文件，并粘贴以下代码。这个示例包含了数据生成、策略定义和回测运行的所有步骤。

```python
import pandas as pd
import numpy as np
from akquant import Strategy, run_backtest
from akquant.config import BacktestConfig

# 1. 准备数据 (模拟生成)
# 在实际应用中，您可以使用 pd.read_csv() 加载您的数据
def generate_data():
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    n = len(dates)

    # 随机漫步生成价格
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n)
    price_path = 100 * np.cumprod(1 + returns)

    data = {
        "date": dates,
        "open": price_path,
        "high": price_path * 1.01,
        "low": price_path * 0.99,
        "close": price_path, # 简化起见，收盘价等于路径值
        "volume": np.random.randint(1000, 5000, n),
        "symbol": ["000001"] * n # 标的代码
    }
    return pd.DataFrame(data)

# 2. 定义策略
class DualMovingAverage(Strategy):
    """
    双均线策略:
    - 短周期均线 (Fast MA) 上穿 长周期均线 (Slow MA) -> 买入
    - 短周期均线 (Fast MA) 下穿 长周期均线 (Slow MA) -> 卖出
    """
    def __init__(self, fast_window=10, slow_window=30):
        # 策略参数
        self.fast_window = fast_window
        self.slow_window = slow_window

    def on_start(self):
        # 订阅行情 (虽然 run_backtest 会自动处理，但显式订阅是个好习惯)
        self.subscribe("000001")
        print("策略启动，初始化指标...")

    def on_bar(self, bar):
        # 获取历史收盘价
        # 注意: get_history 返回的是 numpy array，性能极高
        close_prices = self.get_history(count=self.slow_window + 5, symbol=bar.symbol)

        # 数据不足时返回
        if len(close_prices) < self.slow_window:
            return

        # 计算均线 (使用 numpy)
        # 在实际高频场景中，建议使用 akquant 内置的增量计算指标 (SMA, EMA 等)
        fast_ma = np.mean(close_prices[-self.fast_window:])
        slow_ma = np.mean(close_prices[-self.slow_window:])

        # 获取当前持仓
        position = self.get_position(bar.symbol)

        # 交易逻辑
        if fast_ma > slow_ma and position == 0:
            # 金叉买入
            print(f"[{bar.timestamp_str}] 金叉 (Fast={fast_ma:.2f}, Slow={slow_ma:.2f}) -> 买入")
            self.buy(symbol=bar.symbol, quantity=1000)

        elif fast_ma < slow_ma and position > 0:
            # 死叉卖出
            print(f"[{bar.timestamp_str}] 死叉 (Fast={fast_ma:.2f}, Slow={slow_ma:.2f}) -> 卖出")
            self.sell(symbol=bar.symbol, quantity=position) # 卖出所有持仓

# 3. 运行回测
if __name__ == "__main__":
    df = generate_data()
    print(f"生成了 {len(df)} 条 K 线数据")

    # 使用便捷函数运行回测
    result = run_backtest(
        strategy=DualMovingAverage(fast_window=20, slow_window=60), # 实例化策略并传参
        data=df,
        symbol="000001",

        # 回测配置
        cash=100_000.0,           # 初始资金
        commission=0.0003,        # 佣金万三
        execution_mode="current_close" # 收盘价成交模式
    )

    # 4. 查看结果
    print("\n" + "="*30)
    print("回测结果摘要")
    print("="*30)
    print(f"总收益率: {result.metrics_df['total_return_pct'].iloc[0]:.2f}%")
    print(f"夏普比率: {result.metrics_df['sharpe_ratio'].iloc[0]:.2f}")
    print(f"最大回撤: {result.metrics_df['max_drawdown_pct'].iloc[0]:.2f}%")

    # 打印部分交易记录
    if result.trades:
        print(f"\n共发生 {len(result.trades)} 笔交易")
```

## 3. 关键点解析

1.  **数据准备**: `AKQuant` 接受标准的 `pandas.DataFrame`，要求至少包含 `date`, `open`, `high`, `low`, `close`, `volume`, `symbol` 列。
2.  **策略类**: 继承自 `akquant.Strategy`。
    *   `on_bar(self, bar)`: 核心回调函数，每个 K 线周期触发一次。
    *   `self.get_history(...)`: 高效获取历史数据 (Zero-Copy)。
    *   `self.buy(...)` / `self.sell(...)`: 发送交易指令。
3.  **运行回测**: `run_backtest` 是一个高层封装，自动处理了 `Engine` 初始化、数据加载和回测循环。

## 4. 下一步

*   **[策略编写指南](strategy_guide.md)**: 深入了解订单生命周期、更多订单类型（止损/限价）以及内置高性能指标。
*   **[API 参考](api.md)**: 查阅完整的类和方法说明。
*   **[机器学习指南](ml_guide.md)**: 学习如何集成 Scikit-learn/PyTorch 模型。
