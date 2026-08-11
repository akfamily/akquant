"""
第 13 章：可视化与报告 (Visualization)（函数式写法）.

本示例展示如何使用 AKQuant 的 Plotly 报告能力，并加入基准对比分析。

演示内容：
1. 运行一个简单的策略。
2. 构造一个简单基准收益序列。
3. 使用 `result.viz.report(..., benchmark=...)` 生成交互式 HTML 报告。

本示例与 ch13_visualization.py 的交易逻辑、参数、数据源完全一致，
只把策略入口从类风格换成函数式：

1. __init__ 里的参数 (ma_window / warmup_period) -> ctx 属性（在 initialize 中赋值）
2. Strategy.on_bar 方法 -> 模块级 on_bar(ctx, bar) 函数
3. strategy=PlotStrategy -> strategy=on_bar + initialize=initialize

本章的重点不在策略，而在回测之后的绘图：**`result.viz` 的绘图 API 与策略入口风格
无关**。下面 `result.viz.report(...)` 的调用与类风格版逐字相同——报告读的是
`BacktestResult` 里的曲线与成交记录，跟策略是类还是函数没有关系。
这正是"脚手架章"孪生示例真正要说明的事：换写法只影响策略怎么写，
不影响结果怎么画。

两份示例生成的报告产物一致，这说明函数式与类风格在引擎层等价。
"""

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar


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


# ---------------------------------------------------------------------------
# 策略部分：本章只需要一条"能画出来"的权益曲线，因此保持最简。
# 函数式在这里的好处很直接：没有类体、没有 super().__init__()，
# 两个函数就是一个完整策略。
# ---------------------------------------------------------------------------
def initialize(ctx: Any) -> None:
    """
    初始化均线参数；本章重点是绘图输出，策略保持最简.

    关键差异：函数式没有类体，warmup_period 必须在这里挂到 ctx 上。
    这里沿用类风格版写死的 20（而非由 ma_window 推导），保证两版跳过的
    Bar 数量完全相同，输出才可比。

    注意：引擎还有一路“按指标调用自动推断 warmup”的机制，但它靠 AST 解析策略
    类体实现；函数式下被解析的是引擎内部的 FunctionalStrategy 而非本文件的
    on_bar，推断结果恒为 0。因此函数式必须像这样显式赋值，不能依赖自动推断。
    """
    ctx.ma_window = 20
    ctx.warmup_period = 20


def on_bar(ctx: Any, bar: Bar) -> None:
    """收到 Bar 事件的回调，等价于类风格的 on_bar；self 全部换成 ctx."""
    symbol = bar.symbol
    closes = ctx.get_history(count=ctx.ma_window + 1, symbol=symbol, field="close")
    if len(closes) < ctx.ma_window + 1:
        return

    ma = closes[:-1][-ctx.ma_window :].mean()
    pos = ctx.get_position(symbol)

    # 简单的均线突破
    if bar.close > ma and pos == 0:
        ctx.order_target_percent(symbol=symbol, target_percent=0.95)
    elif bar.close < ma and pos > 0:
        ctx.close_position(symbol)


if __name__ == "__main__":
    df = generate_mock_data()

    print("开始运行第 13 章可视化示例...")
    result = aq.run_backtest(
        strategy=on_bar,
        initialize=initialize,
        data=df,
        initial_cash=100_000,
        commission_rate=0.0003,
    )

    print("回测完成，正在生成带基准对比的报告...")

    # ---------------------------------------------------------------------
    # 以下绘图部分与类风格版完全相同：基准序列由行情数据算出，
    # report() 的参数、图层开关（含 include_trade_kline 的成交 K 线）都不变。
    # ---------------------------------------------------------------------
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
