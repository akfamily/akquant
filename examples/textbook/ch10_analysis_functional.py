"""
第 10 章：策略评价体系 (Strategy Analysis)（函数式写法）.

本示例展示了如何深入分析回测结果，通过关键指标评估策略的优劣：
1. **夏普比率 (Sharpe Ratio)**：收益与风险的比值。
2. **最大回撤 (Max Drawdown)**：历史上可能遭受的最大亏损幅度。
3. **胜率 (Win Rate)**：盈利交易的比例。
4. **盈亏比 (Profit/Loss Ratio)**：平均盈利与平均亏损的比值。

示例策略：
- 使用第 5 章的均线策略作为基准。
- 演示如何访问 `result.metrics_df` (总体指标) 和 `result.trades_df` (逐笔交易)。

本示例与 ch10_analysis.py 的交易逻辑、参数、数据源完全一致，
只把策略入口从类风格换成函数式：

1. __init__ 里的参数 (short_window / long_window / warmup_period)
   -> ctx 属性（在 initialize 中赋值）
2. Strategy.on_bar 方法 -> 模块级 on_bar(ctx, bar) 函数
3. strategy=AnalysisStrategy -> strategy=on_bar + initialize=initialize

本章的重点不在策略，而在回测之后的分析：**`result` 的分析 API 与策略入口风格无关**。
下面 `analyze_results` 整段与类风格版逐字相同——无论策略是类还是函数，
`run_backtest` 返回的都是同一种 `BacktestResult`，`metrics_df` / `trades_df` /
各条曲线的读法完全一致。这正是"脚手架章"孪生示例真正要说明的事：
换写法只影响策略怎么写，不影响结果怎么读。

两份示例的分析输出应完全一致，这说明函数式与类风格在引擎层等价。
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
    prices = 100 + np.cumsum(np.random.randn(length))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 100000,
            "symbol": "MOCK",
        }
    )
    return df


# ---------------------------------------------------------------------------
# 策略部分：本章只是需要一段"有交易记录"的回测作为分析素材，因此保持最简。
# 函数式在这里的好处很直接：没有类体、没有 super().__init__()，
# 两个函数就是一个完整策略。
# ---------------------------------------------------------------------------
def initialize(ctx: Any) -> None:
    """
    初始化双均线参数；本章重点是回测后的分析，策略保持最简.

    关键差异：函数式没有类体，warmup_period 必须在这里挂到 ctx 上。
    本示例会请求 long_window + 1 根数据，并用 [:-1] 排除当前 Bar，
    因此预热期同步加 1（= 21），与类风格版取值一致。

    注意：引擎还有一路“按指标调用自动推断 warmup”的机制，但它靠 AST 解析策略
    类体实现；函数式下被解析的是引擎内部的 FunctionalStrategy 而非本文件的
    on_bar，推断结果恒为 0。因此函数式必须像这样显式赋值，不能依赖自动推断。
    """
    ctx.short_window = 5
    ctx.long_window = 20
    ctx.warmup_period = ctx.long_window + 1


def on_bar(ctx: Any, bar: Bar) -> None:
    """收到 Bar 事件的回调，等价于类风格的 on_bar；self 全部换成 ctx."""
    symbol = bar.symbol
    closes = ctx.get_history(count=ctx.long_window + 1, symbol=symbol, field="close")
    if len(closes) < ctx.long_window + 1:
        return

    history_closes = closes[:-1]
    ma_short = history_closes[-ctx.short_window :].mean()
    ma_long = history_closes[-ctx.long_window :].mean()

    pos = ctx.get_position(symbol)

    if ma_short > ma_long and pos == 0:
        ctx.order_target_percent(symbol=symbol, target_percent=0.95)
    elif ma_short < ma_long and pos > 0:
        ctx.close_position(symbol)


# ---------------------------------------------------------------------------
# 本章真正的教学内容：结果分析。以下代码与类风格版完全相同，
# 因为 result 不知道也不关心策略是怎么写的。
# ---------------------------------------------------------------------------
def analyze_results(result: Any) -> None:
    """详细分析回测结果."""
    print("\n" + "=" * 40)
    print("1. 核心指标概览 (Key Metrics)")
    print("=" * 40)

    # 从 result.metrics_df 中提取关键指标
    metrics = result.metrics_df

    # 辅助函数：安全获取指标值
    def get_metric(name: str, default: float = 0.0) -> float:
        if name in metrics.index:
            val = metrics.loc[name, "value"]
            return float(val)
        return default

    total_return = get_metric("total_return_pct")
    annual_return = get_metric("annualized_return")
    max_dd = get_metric("max_drawdown_pct")
    sharpe = get_metric("sharpe_ratio")

    print(f"累计收益率: {total_return:.2f}%")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"最大回撤  : {max_dd:.2f}%")
    print(f"夏普比率  : {sharpe:.2f}")

    print("\n" + "=" * 40)
    print("2. 交易行为分析 (Trade Analysis)")
    print("=" * 40)

    trades_df = result.trades_df
    if not trades_df.empty:
        closed_trade_count = get_metric("closed_trade_count", float(len(trades_df)))
        win_rate = len(trades_df[trades_df["pnl"] > 0]) / len(trades_df)
        avg_pnl = trades_df["pnl"].mean()

        print(f"已完成交易数: {closed_trade_count:.0f}")
        print(f"胜率      : {win_rate:.2%}")
        print(f"平均每笔盈亏: {avg_pnl:.2f}")

        # 打印前 5 笔交易详情
        print("\n交易详情 (前5笔):")
        print(
            trades_df[
                ["entry_time", "exit_time", "symbol", "side", "pnl", "return_pct"]
            ].head()
        )
    else:
        print("无交易记录")

    print("\n" + "=" * 40)
    print("3. 曲线与报告频率 (Curves & Report Frequency)")
    print("=" * 40)
    print(f"权益曲线点数: {len(result.equity_curve)}")
    print(f"现金曲线点数: {len(result.cash_curve)}")
    print(f"保证金曲线点数: {len(result.margin_curve)}")
    print(f"日频权益点数: {len(result.equity_curve_daily)}")
    print(f"日频现金点数: {len(result.cash_curve_daily)}")
    print(f"日频保证金点数: {len(result.margin_curve_daily)}")


if __name__ == "__main__":
    df = generate_mock_data()

    print("开始运行第 10 章分析示例...")
    result = aq.run_backtest(
        strategy=on_bar,
        initialize=initialize,
        data=df,
        initial_cash=100_000,
        commission_rate=0.0003,
    )

    # 执行分析函数
    analyze_results(result)
    result.viz.report(
        filename="ch10_analysis_report_daily.html", show=False, curve_freq="D"
    )
