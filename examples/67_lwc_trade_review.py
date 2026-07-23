"""AKQuant LWC 交互式交易复盘 Demo.

演示 ``result.viz.review()``:基于 TradingView Lightweight Charts 生成
**离线自包含**的单文件 HTML,在 K 线上标注买卖点,面向大数据量 / 日内的
交易复盘。与 ``result.viz.report()`` 的 plotly 分析报告互补——分析图仍由
plotly 负责,本方法只补交互式 K 线复盘。

本示例使用合成数据,无需联网即可运行。
"""

import numpy as np
import pandas as pd
from akquant import Bar, Strategy, run_backtest


class MACrossStrategy(Strategy):
    """5/15 均线交叉策略(演示用)."""

    def __init__(self) -> None:
        """初始化窗口与价格缓存."""
        super().__init__()
        self.fast, self.slow = 5, 15
        self.prices: list[float] = []

    def on_bar(self, bar: Bar) -> None:
        """均线金叉买入、死叉平仓."""
        self.prices.append(bar.close)
        if len(self.prices) < self.slow:
            return
        fast_ma = float(np.mean(self.prices[-self.fast :]))
        slow_ma = float(np.mean(self.prices[-self.slow :]))
        pos = self.get_position(bar.symbol)
        if pos == 0 and fast_ma > slow_ma:
            self.buy(bar.symbol, 100)
        elif pos > 0 and fast_ma < slow_ma:
            self.close_position(bar.symbol)


def _make_synthetic_data(symbol: str, n: int = 250) -> pd.DataFrame:
    """生成一段带趋势的合成日线 OHLCV 数据."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    steps = rng.normal(0.0004, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": symbol,
        },
        index=dates,
    )


if __name__ == "__main__":
    SYMBOL = "DEMO001"
    df = _make_synthetic_data(SYMBOL)

    print("Running backtest...")
    result = run_backtest(
        data=df,
        strategy=MACrossStrategy,
        symbols=SYMBOL,
        initial_cash=100_000.0,
        show_progress=False,
    )
    print(f"  Total trades: {len(result.trades_df)}")

    # LWC 交互式复盘:离线自包含单文件 HTML,K 线上标注买卖点。
    # theme 只决定初始主题,页面顶部有明暗切换按钮可即时切换。
    # show=False 便于在 CI/无头环境运行;本地可改 True 自动打开浏览器。
    out = result.viz.review(
        market_data=df,
        title=f"AKQuant 交易复盘 - {SYMBOL}",
        theme="dark",
        filename="akquant_lwc_review.html",
        show=False,
    )
    print(f"Review saved to: {out}")
