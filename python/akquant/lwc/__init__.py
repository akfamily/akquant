"""LWC(TradingView Lightweight Charts)交互式交易复盘.

生成**离线自包含**的单文件 HTML:内联 vendored 的
lightweight-charts standalone 构建 + 回测数据,用于大数据量 / 日内 K 线的
买卖点复盘。这是对 plotly 报告的补充(分析图仍由 plotly 负责),而非替代。

公开入口::func:`plot_kline_review`,通常经 ``result.viz.review()`` 调用。
"""

from .review import plot_kline_review

__all__ = ["plot_kline_review"]
