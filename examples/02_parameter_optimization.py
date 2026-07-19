"""
参数优化示例 (Optimization Demo).

演示如何使用 run_optimization 进行网格搜索.
"""

from typing import Any

import numpy as np
import pandas as pd
from akquant import (
    Indicator,
    IntParam,
    Strategy,
    get_strategy_param_schema,
    run_grid_search,
    validate_strategy_params,
)


class SMACrossStrategy(Strategy):
    """
    双均线交叉策略 (Double SMA Crossover, 内联参数声明).

    参数:
        fast_period (int): 快线周期
        slow_period (int): 慢线周期
    """

    fast_period = IntParam(10, ge=2, le=200, title="快线周期")
    slow_period = IntParam(30, ge=3, le=500, title="慢线周期")

    def on_start(self) -> None:
        """策略启动：基于 self.params 派生指标."""
        # 定义指标
        self.sma_fast = Indicator(
            "sma_fast",
            lambda df: df["close"].rolling(self.params.fast_period).mean(),
        )
        self.sma_slow = Indicator(
            "sma_slow",
            lambda df: df["close"].rolling(self.params.slow_period).mean(),
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
    symbols = ["OPT_A", "OPT_B"]
    data_map: dict[str, pd.DataFrame] = {}
    for index, symbol in enumerate(symbols):
        close_prices = 100 + index * 5 + np.cumsum(np.random.randn(500))
        data_map[symbol] = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices + 1,
                "low": close_prices - 1,
                "close": close_prices,
                "volume": 1000,
                "symbol": symbol,
            },
            index=dates,
        )

    # 2. 运行优化
    print("Starting optimization...")
    schema = get_strategy_param_schema(SMACrossStrategy)
    print(f"Schema fields: {sorted(schema.get('properties', {}).keys())}")
    validated = validate_strategy_params(
        SMACrossStrategy,
        {"fast_period": 12, "slow_period": 36},
    )
    print(f"Validated params demo: {validated}")

    # 定义参数网格
    param_grid = {
        "fast_period": [5, 10, 15],
        "slow_period": [20, 30, 40],
    }

    # 运行
    # max_workers=1：策略类定义于 __main__ 脚本内，多进程池要求策略类可从
    # 模块导入以便 pickle；直接以脚本方式运行示例时用单进程演示网格搜索。
    results = run_grid_search(
        strategy=SMACrossStrategy,
        param_grid=param_grid,
        data=data_map,
        symbols=symbols,
        sort_by="total_return",  # 按总收益排序
        max_workers=1,
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
            print(f"Best Return: {best_params['total_return_pct']:.2f}")
    elif results:
        # handle list of OptimizationResult
        best_result = results[0]
        print(
            f"\nBest Parameters: fast={best_result.params['fast_period']}, "
            f"slow={best_result.params['slow_period']}"
        )
        print(f"Best Return: {best_result.metrics.get('total_return_pct', 0.0):.2f}")
