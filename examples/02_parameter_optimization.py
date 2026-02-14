"""
参数优化示例 (Optimization Demo).

演示如何使用 run_optimization 进行网格搜索.
"""

from typing import Any

import numpy as np
import pandas as pd
from akquant import Indicator, Strategy, run_optimization


class SMACrossStrategy(Strategy):
    """
    双均线交叉策略 (Double SMA Crossover).

    参数:
        fast_period (int): 快线周期
        slow_period (int): 慢线周期
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        """初始化策略."""
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

        # 定义指标
        self.sma_fast = Indicator(
            "sma_fast", lambda df: df["close"].rolling(fast_period).mean()
        )
        self.sma_slow = Indicator(
            "sma_slow", lambda df: df["close"].rolling(slow_period).mean()
        )

        # 订阅指标 (自动计算)
        self._indicators = [self.sma_fast, self.sma_slow]

    def on_bar(self, bar: Any) -> None:
        """K线闭合回调."""
        # 获取当前值
        fast = self.sma_fast.get_value(bar.symbol, bar.timestamp)
        slow = self.sma_slow.get_value(bar.symbol, bar.timestamp)

        # 简单的交叉逻辑
        # 获取当前持仓
        qty = self.get_position(bar.symbol)

        if fast > slow and qty <= 0:
            # 金叉买入
            self.buy(symbol=bar.symbol, quantity=100)
        elif fast < slow and qty > 0:
            # 死叉卖出
            self.sell(symbol=bar.symbol, quantity=100)


if __name__ == "__main__":
    # 1. 生成模拟数据
    print("Generating synthetic data...")
    dates = pd.date_range(start="2023-01-01", periods=500)
    # 随机漫步
    close_prices = 100 + np.cumsum(np.random.randn(500))
    df = pd.DataFrame(
        {
            "open": close_prices,
            "high": close_prices + 1,
            "low": close_prices - 1,
            "close": close_prices,
            "volume": 1000,
        },
        index=dates,
    )

    # 2. 运行优化
    print("Starting optimization...")

    # 定义参数网格
    param_grid = {
        "fast_period": [5, 10, 15],
        "slow_period": [20, 30, 40],
    }

    # 运行
    # max_workers=2 for demo safety
    results = run_optimization(
        strategy=SMACrossStrategy,
        param_grid=param_grid,
        data=df,
        symbol="TEST",
        sort_by="total_return",  # 按总收益排序
        max_workers=2,
    )

    if isinstance(results, pd.DataFrame):
        if "error" in results.columns:
            print("\nErrors found:")
            print(results[results["error"].notna()]["error"].head())

        print("\nOptimization Results (Top 5):")
        print(results.head())
    else:
        print(results)

    # 获取最佳参数
    if isinstance(results, pd.DataFrame):
        if not results.empty:
            best_params = results.iloc[0].to_dict()
            print(
                f"\nBest Parameters: fast={best_params['fast_period']}, "
                f"slow={best_params['slow_period']}"
            )
            print(f"Best Return: {best_params['total_return']:.2f}")
    elif results:
        # handle list of OptimizationResult
        best_result = results[0]
        print(
            f"\nBest Parameters: fast={best_result.params['fast_period']}, "
            f"slow={best_result.params['slow_period']}"
        )
        print(f"Best Return: {best_result.metrics.get('total_return', 0.0):.2f}")
