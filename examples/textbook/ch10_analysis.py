"""
第 10 章：策略评价体系 (Strategy Analysis).

本示例展示了如何深入分析回测结果，通过关键指标评估策略的优劣：
1. **夏普比率 (Sharpe Ratio)**：收益与风险的比值。
2. **最大回撤 (Max Drawdown)**：历史上可能遭受的最大亏损幅度。
3. **胜率 (Win Rate)**：盈利交易的比例。
4. **盈亏比 (Profit/Loss Ratio)**：平均盈利与平均亏损的比值。

示例策略：
- 使用第 5 章的均线策略作为基准。
- 演示如何访问 `result.metrics_df` (总体指标) 和 `result.trades_df` (逐笔交易)。
"""

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, Strategy


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


class AnalysisStrategy(Strategy):
    """分析演示策略."""

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        """初始化策略."""
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        # 本示例会请求 long_window + 1 根数据，并用 [:-1] 排除当前 Bar。
        self.warmup_period = long_window + 1

    def on_bar(self, bar: Bar) -> None:
        """收到 Bar 事件的回调."""
        symbol = bar.symbol
        closes = self.get_history(
            count=self.long_window + 1, symbol=symbol, field="close"
        )
        if len(closes) < self.long_window + 1:
            return

        history_closes = closes[:-1]
        ma_short = history_closes[-self.short_window :].mean()
        ma_long = history_closes[-self.long_window :].mean()

        pos = self.get_position(symbol)

        if ma_short > ma_long and pos == 0:
            self.order_target_percent(symbol=symbol, target_percent=0.95)
        elif ma_short < ma_long and pos > 0:
            self.close_position(symbol)


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
        strategy=AnalysisStrategy, data=df, initial_cash=100_000, commission_rate=0.0003
    )

    # 执行分析函数
    analyze_results(result)
    result.viz.report(
        filename="ch10_analysis_report_daily.html", show=False, curve_freq="D"
    )
