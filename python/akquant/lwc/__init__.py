"""LWC(TradingView Lightweight Charts)交互式交易复盘.

生成**离线自包含**的单文件 HTML:内联 vendored 的
lightweight-charts standalone 构建 + 回测数据,用于大数据量 / 日内 K 线的
买卖点复盘。这是对 plotly 报告的补充(分析图仍由 plotly 负责),而非替代。

公开入口::func:`plot_kline_review`,通常经 ``result.viz.review()`` 调用。

实时路径的消费端适配器: :func:`to_lwc_update` 把 ``akquant.to_indicator_message``
的 point 消息转成 ``series.update()`` 可直接吃的增量; :func:`load_lwc_js` 返回
vendored 的 LWC 源码文本, 供自建页面内联(见 ``examples/73``)。
"""

from ._indicators import (
    LWC_SERIES_TYPE,
    lwc_pane_index,
    to_lwc_indicator_series,
    to_lwc_update,
)
from ._template import _load_lwc_js as load_lwc_js
from .review import plot_kline_review

__all__ = [
    "LWC_SERIES_TYPE",
    "load_lwc_js",
    "lwc_pane_index",
    "plot_kline_review",
    "to_lwc_indicator_series",
    "to_lwc_update",
]
