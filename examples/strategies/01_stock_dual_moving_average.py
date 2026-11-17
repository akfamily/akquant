"""
A股趋势跟踪策略示例 (Stock Trend Following Strategy).

================================================

本示例展示如何使用 AKShare 获取 A 股历史数据，并运行一个简单的双均线策略。

教学目标 (Learning Objectives):
1.  使用 `ak.stock_zh_a_hist_sina` 获取特定股票的历史数据。
2.  使用 AKQuant 内置的 `self.get_history` 接口获取历史数据。
3.  实现双均线策略逻辑。

前置条件 (Prerequisites):
- 安装 akshare: `pip install akshare`
"""

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, IntParam, Strategy


def synthetic_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """生成固定种子的合成日线（几何随机游走），只用于取数失败时让示例跑通."""
    dates = pd.bdate_range(start_date, end_date)
    rng = np.random.default_rng(sum(map(ord, symbol)))
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, len(dates))))
    open_ = np.concatenate([[10.0], close[:-1]])
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
        }
    )


def fetch_daily(
    symbol: str, start_date: str, end_date: str, adjust: str = ""
) -> pd.DataFrame:
    """用 AKShare 取新浪 A 股日线（代码带市场前缀，如 sh600000）.

    取数失败（未安装 AKShare、断网、接口改名等）时改用合成数据，保证示例能跑通；
    合成数据的回测结果没有参考意义。
    """
    try:
        import akshare as ak

        df = ak.stock_zh_a_hist_sina(
            symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust
        )
        if df.empty:
            raise ValueError("返回空表")
        return pd.DataFrame(df)
    except Exception as exc:  # noqa: BLE001 - 示例容错，任何取数问题都回退
        print(f"警告: AKShare 取数失败（{type(exc).__name__}: {exc}），改用合成数据")
        return synthetic_daily(symbol, start_date, end_date)


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
    # 注意：ak.stock_zh_a_hist_sina 需要带前缀 sh 或 sz
    symbol = "sz000001"

    df = fetch_daily(
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
