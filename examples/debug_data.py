import numpy as np
import pandas as pd
from akquant import Bar, Strategy, run_backtest


# --------------------------
# 第一步：准备数据
# --------------------------
# 在真实场景中，你会使用 pd.read_csv("stock_data.csv")
# 这里为了演示方便，我们生成一些模拟数据
def generate_mock_data() -> pd.DataFrame:
    """Generate mock data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    n = len(dates)

    # 生成一条随机走势
    np.random.seed(42)  # 固定随机种子，保证每次运行结果一致
    returns = np.random.normal(0.0005, 0.02, n)
    price = 100 * np.cumprod(1 + returns)

    # 构造 DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 10000,
            "symbol": "AAPL",  # 假设这是苹果公司的股票
        }
    )
    return df


# --------------------------
# 第二步：编写策略
# --------------------------
class DualMAStrategy(Strategy):
    """继承 akquant.Strategy 类，这是所有策略的基类."""

    # 新增：声明式设置历史数据预热期 (AKQuant 自动感知机制)
    # 框架会自动识别此属性，或通过 AST 静态分析推断指标周期。
    # 由于本策略使用了动态参数 (slow_window)，建议显式定义以确保安全。
    warmup_period = 40  # 30 (slow_window) + 10 (buffer)

    def __init__(self, fast_window: int = 10, slow_window: int = 30) -> None:
        """Initialize the strategy."""
        # 定义策略参数：快线周期和慢线周期
        self.fast_window = fast_window
        self.slow_window = slow_window

        # 动态更新 warmup_period (覆盖类属性)
        # 这样即使外部修改了 slow_window 参数，历史数据也能自动适配
        self.warmup_period = slow_window + 10

    def on_start(self) -> None:
        """
        策略启动时执行一次.

        在这里告诉系统我们要关注哪些股票。
        """
        print("策略启动...")
        self.subscribe("AAPL")

        # 历史数据长度已通过 warmup_period 自动设置，无需手动调用 set_history_depth

    def on_bar(self, bar: Bar) -> None:
        """
        核心逻辑：每一根 K 线走完时，都会触发一次这个函数.

        bar 参数包含了当前的行情数据 (bar.close, bar.high 等)。
        """
        # 1. 获取历史收盘价
        # get_history 返回的是一个 numpy 数组，包含最近 N 天的数据
        closes = self.get_history(
            count=self.slow_window, symbol=bar.symbol, field="close"
        )

        # 如果数据还不够计算长均线（比如刚开始回测的前几天），就直接返回，不操作
        if len(closes) < self.slow_window:
            return

        # 2. 计算均线
        # 使用 numpy 计算平均值
        fast_ma = np.mean(
            closes[-self.fast_window :]
        )  # 取最后 fast_window 个数据求平均
        slow_ma = np.mean(
            closes[-self.slow_window :]
        )  # 取最后 slow_window 个数据求平均

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
            self.sell(symbol=bar.symbol, quantity=position)  # 卖出所有持仓


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
        strategy=DualMAStrategy,  # 传入我们的策略类
        strategy_params={"fast_window": 10, "slow_window": 40},  # 调整参数
        cash=100_000.0,  # 初始资金 10万
        commission=0.0003,  # 佣金万分之三
    )

    # 3. 打印结果
    print("\n" + "=" * 30)
    print(result)  # 打印详细的绩效报表
    print("=" * 30)
