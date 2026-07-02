"""
Parameter optimization example.

Demonstrates how to use grid search for parameter optimization.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from akquant import Bar, Strategy


class DualMovingAverageStrategy(Strategy):
    """Dual Moving Average Strategy for parameter optimization."""

    def __init__(self, short_window: int, long_window: int):
        """Initialize the strategy.

        Args:
            short_window: The short moving average window.
            long_window: The long moving average window.
        """
        super().__init__()
        # 定义策略参数：短期窗口5，长期窗口20
        self.short_window = short_window
        self.long_window = long_window
        self.warmup_period = long_window + 1

    def on_bar(self, bar: Bar) -> None:
        """Handle new bar data.

        Args:
            bar: The new bar data.
        """
        # 获取历史收盘价数据
        # history_data 返回的是一个 DataFrame
        hist = self.get_history(count=self.long_window + 1, field="close")

        # 如果数据不足，无法计算均线，直接返回
        if len(hist) < self.long_window + 1:
            return

        # 计算短期和长期均线
        closes = hist
        ma_short = np.mean(closes[-self.short_window :])
        ma_long = np.mean(closes[-self.long_window :])

        # 获取上一时刻的均线值（用于判断交叉）
        prev_ma_short = np.mean(closes[-self.short_window - 1 : -1])
        prev_ma_long = np.mean(closes[-self.long_window - 1 : -1])

        # 获取当前持仓
        position = self.get_position(bar.symbol)

        # 交易逻辑
        # 1. 金叉：短期均线上穿长期均线，且当前无持仓 -> 买入
        if prev_ma_short <= prev_ma_long and ma_short > ma_long:
            if position == 0:
                self.buy(bar.symbol, 100)  # 买入100股
                print(f"[{bar.timestamp_iso}] 金叉买入 {bar.symbol} @ {bar.close:.2f}")

        # 2. 死叉：短期均线下穿长期均线，且持有仓位 -> 卖出
        elif prev_ma_short >= prev_ma_long and ma_short < ma_long:
            if position > 0:
                self.sell(bar.symbol, 100)  # 卖出100股
                print(f"[{bar.timestamp_iso}] 死叉卖出 {bar.symbol} @ {bar.close:.2f}")


# ------------------------------
# 准备测试数据并运行
# ------------------------------


def warmup_calc(params: Dict[str, Any]) -> int:
    """Calculate warmup period based on parameters.

    Args:
        params: Strategy parameters.

    Returns:
        The warmup period.
    """
    return int(params["long_window"] + 1)


def param_constraint(params: Dict[str, Any]) -> bool:
    """Check parameter constraints.

    Args:
        params: Strategy parameters.

    Returns:
        True if parameters are valid, False otherwise.
    """
    # 确保短期窗口小于长期窗口，避免无效计算
    return bool(params["short_window"] < params["long_window"])


def result_filter(metrics: Dict[str, Any]) -> bool:
    """Filter optimization results.

    Args:
        metrics: Performance metrics.

    Returns:
        True if the result should be kept, False otherwise.
    """
    # 筛选条件：
    # 1. 交易次数至少 2 次 (放宽条件，模拟数据交易较少)
    # 2. 夏普比率 > -99 (更放宽条件)
    # 3. 总收益 > -99
    return bool(metrics.get("closed_trade_count", 0) >= 2)


if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(1024)

    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    price = 100 + np.cumsum(np.random.randn(len(dates)))  # 随机游走价格

    df = pd.DataFrame(
        {
            "date": dates,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 10000,
            "symbol": "DEMO",
        }
    )
    # run_walk_forward 需要 DatetimeIndex
    df.set_index("date", inplace=True)

    # 运行回测
    print("开始回测...")

    from akquant import run_grid_search

    # Define parameter grid
    param_grid = {
        "short_window": list(range(5, 30, 5)),  # [5, 10, 15, 20, 25]
        "long_window": list(range(20, 60, 10)),  # [20, 30, 40, 50]
    }

    # Run optimization
    results_df = run_grid_search(
        strategy=DualMovingAverageStrategy,
        param_grid=param_grid,
        data=df,
        max_workers=1,
        initial_cash=100_000.0,
        sort_by=["sharpe_ratio", "total_return"],  # 多字段排序：先按夏普，再按总收益
        ascending=[False, False],  # 都是降序
        warmup_calc=warmup_calc,  # 动态计算预热期
        constraint=param_constraint,  # 参数约束
        result_filter=result_filter,  # 结果筛选
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    # DataFrame 返回显示的行内的所有内容，不要缩略

    # Print top 5
    print(results_df)

    # ------------------------------
    # 演示 Walk-Forward Optimization
    # ------------------------------
    print("\n\n开始 Walk-Forward Optimization...")
    from akquant import run_walk_forward

    # 确保数据足够长
    # train=100, test=50 -> need 150+
    # current len is 365

    try:
        wfo_result = run_walk_forward(
            strategy=DualMovingAverageStrategy,
            param_grid=param_grid,
            data=df,
            train_period=100,
            test_period=50,
            metric=["sharpe_ratio", "total_return"],  # 多目标排序
            ascending=[False, False],
            initial_cash=100_000.0,
            warmup_calc=warmup_calc,
            constraint=param_constraint,
            result_filter=result_filter,  # 同样可以使用结果过滤
        )

        print("\nWFO Result Head:")
        print(wfo_result.head())
    except TypeError as exc:
        print(f"\nWFO 示例跳过: {exc}")
