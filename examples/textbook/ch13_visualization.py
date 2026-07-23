"""
第 13 章：可视化与报告 (Visualization).

本示例展示如何使用 AKQuant 的 Plotly 报告能力，并加入基准对比分析。

演示内容：
1. 运行一个简单的策略。
2. 构造一个简单基准收益序列。
3. 使用 `result.viz.report(..., benchmark=...)` 生成交互式 HTML 报告。
"""

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, Strategy


# 模拟数据生成
def generate_mock_data(length: int = 500) -> pd.DataFrame:
    """生成模拟数据."""
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=length, freq="D")

    # 构造一个有趋势的数据，让曲线好看一些
    trend = np.linspace(100, 150, length)
    noise = np.cumsum(np.random.randn(length))
    prices = trend + noise

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 100000,
            "symbol": "MOCK_PLOT",
        }
    )
    return df


class PlotStrategy(Strategy):
    """可视化演示策略."""

    def __init__(self) -> None:
        """初始化策略."""
        super().__init__()
        self.ma_window = 20
        self.warmup_period = 20

    def on_bar(self, bar: Bar) -> None:
        """收到 Bar 事件的回调."""
        symbol = bar.symbol
        closes = self.get_history(
            count=self.ma_window + 1, symbol=symbol, field="close"
        )
        if len(closes) < self.ma_window + 1:
            return

        ma = closes[:-1][-self.ma_window :].mean()
        pos = self.get_position(symbol)

        # 简单的均线突破
        if bar.close > ma and pos == 0:
            self.order_target_percent(symbol=symbol, target_percent=0.95)
        elif bar.close < ma and pos > 0:
            self.close_position(symbol)


if __name__ == "__main__":
    df = generate_mock_data()

    print("开始运行第 13 章可视化示例...")
    result = aq.run_backtest(
        strategy=PlotStrategy, data=df, initial_cash=100_000, commission_rate=0.0003
    )

    print("回测完成，正在生成带基准对比的报告...")

    benchmark_returns = (
        df.set_index("date")["close"].pct_change().fillna(0.0).rename("MOCK_BENCH")
    )
    result.viz.report(
        title="AKQuant Chapter 13 - Visualization with Benchmark",
        filename="ch13_report_with_benchmark.html",
        show=False,
        market_data=df,
        plot_symbol="MOCK_PLOT",
        include_trade_kline=True,
        benchmark=benchmark_returns,
    )
    print("报告已保存至: ch13_report_with_benchmark.html")
