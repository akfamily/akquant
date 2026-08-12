"""
A股趋势跟踪策略示例 (Stock Trend Following Strategy).

================================================

本示例展示如何使用 AKShare 获取 A 股历史数据，并运行一个简单的双均线策略。

教学目标 (Learning Objectives):
1.  使用 `ak.stock_zh_a_daily` 获取特定股票的历史数据。
2.  使用 AKQuant 内置的 `self.get_history` 接口获取历史数据。
3.  实现双均线策略逻辑。

前置条件 (Prerequisites):
- 安装 akshare: `pip install akshare`
"""

import akquant as aq
import akshare as ak
import numpy as np
from akquant import Bar, IntParam, Strategy


class DualMovingAverageStrategy(Strategy):
    """
    双均线策略 (Dual Moving Average Strategy).

    逻辑:
    - 当 短期均线 > 长期均线 且 无持仓 -> 买入 (金叉)
    - 当 短期均线 < 长期均线 且 有持仓 -> 卖出 (死叉)
    """

    short_window = IntParam(5, ge=2, le=200, title="短期均线窗口")
    long_window = IntParam(20, ge=3, le=500, title="长期均线窗口")

    def on_start(self) -> None:
        """设置预热期, 使 get_history 能取到足够长度."""
        # 预热期至少等于长期均线窗口
        self.warmup_period = self.params.long_window

    def on_bar(self, bar: Bar) -> None:
        """Process each Bar data."""
        symbol = bar.symbol

        # 1. 获取历史收盘价
        # 注意：get_history 返回的数据包含当前 bar
        # 所以我们需要获取 long_window 个数据

        # 获取过去 N 个历史收盘价 (包含当前 bar)
        closes = self.get_history(
            count=self.params.long_window, symbol=symbol, field="close"
        )
        # print(closes)

        # 如果历史数据不足，直接返回
        if len(closes) < self.params.long_window:
            return

        # 2. 计算均线
        # 使用 numpy 计算更高效
        short_ma = np.mean(closes[-self.params.short_window :])
        long_ma = np.mean(closes[-self.params.long_window :])

        # 3. 获取当前持仓
        current_pos = self.get_position(symbol)

        # 4. 交易逻辑
        # 金叉：短均线 > 长均线
        if short_ma > long_ma and current_pos == 0:
            # 全仓买入 (order_target_percent 自动计算数量)
            # 注意：A股买入必须是 100 的倍数，框架会自动处理(向下取整)
            self.order_target_percent(symbol=symbol, target_percent=0.95)
            print(
                f"[{bar.timestamp_iso}] BUY SIGNAL: Short({short_ma:.2f}) > "
                f"Long({long_ma:.2f}), Price={bar.close:.2f}"
            )

        # 死叉：短均线 < 长均线
        elif short_ma < long_ma and current_pos > 0:
            # 清仓
            self.close_position(symbol=symbol)
            print(
                f"[{bar.timestamp_iso}] SELL SIGNAL: Short({short_ma:.2f}) < "
                f"Long({long_ma:.2f}), Price={bar.close:.2f}"
            )


if __name__ == "__main__":
    # 1. 准备数据
    # 以平安银行 (sz000001) 为例
    # 注意：ak.stock_zh_a_daily 需要带前缀 sh 或 sz
    symbol = "sz000001"

    df = ak.stock_zh_a_daily(
        symbol=symbol, start_date="20220101", end_date="20231231", adjust="qfq"
    )

    # 2. 运行回测
    result = aq.run_backtest(
        data=df,
        strategy=DualMovingAverageStrategy,
        symbols=symbol,
        initial_cash=100_000.0,
        commission_rate=0.0003,  # 万三佣金
        min_commission=5.0,  # 最低5元
        stamp_tax_rate=0.001,  # 千一印花税 (仅卖出)
        lot_size=100,  # A股每手100股
    )

    # 3. 输出结果
    print("\n=== Backtest Result ===")
    print(result)
