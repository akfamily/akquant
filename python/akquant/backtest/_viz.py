"""可视化命名空间.

将 ``BacktestResult`` 的可视化能力聚合到 ``result.viz.*`` 一个入口,方法按
"用户要做的事"(job)命名而非按介质/引擎命名。各方法内部仍 **惰性导入**
绘图后端,保持 ``plotly`` / ``quantstats`` / ``lwc`` 为可选依赖;``viz`` 属性
本身不触发任何绘图库导入.

设计见 ``docs/zh/meta/viz-namespace-and-lwc-review-rfc.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    from .result import BacktestResult


class VizNamespace:
    """``result.viz`` 下聚合的可视化方法族.

    - :meth:`dashboard` —— 交互式总览(权益/回撤),返回 Plotly Figure.
    - :meth:`indicators` —— 指标序列预览,返回 Plotly Figure.
    - :meth:`report` —— akquant 原生全量静态 HTML 报告.
    - :meth:`quantstats` —— QuantStats 版式报告.
    - :meth:`review` —— LWC 交互式 K 线交易复盘(P1 落地).
    """

    def __init__(self, result: "BacktestResult") -> None:
        """绑定到一个 ``BacktestResult`` 实例.

        :param result: 回测结果对象.
        """
        self._result = result

    def dashboard(
        self,
        show: bool = True,
        title: str = "Backtest Result",
    ) -> Any:
        """交互式总览仪表盘(权益曲线 / 回撤等).

        :param show: 是否立即在浏览器/Notebook 中显示.
        :param title: 图标题.
        :return: Plotly ``Figure``;未安装 plotly 时返回 ``None``.
        """
        try:
            from ..plot import plot_dashboard
        except ImportError:
            print(
                "Plotly is not installed. Please install it using "
                "`pip install plotly` or `pip install akquant[plot]`."
            )
            return None
        return plot_dashboard(result=self._result, show=show, title=title)

    def indicators(
        self,
        name: Optional[str] = None,
        symbol: Optional[str] = None,
        include_warmup: bool = True,
        show: bool = True,
        title: str = "Indicator History",
        theme: str = "light",
        filename: Optional[str] = None,
    ) -> Any:
        """指标历史多面板预览图.

        :param name: 可选指标键过滤.
        :param symbol: 可选标的过滤.
        :param include_warmup: 是否保留预热点.
        :param show: 是否立即显示.
        :param title: 图标题.
        :param theme: 主题键(``light`` / ``dark``).
        :param filename: 可选 HTML 输出路径.
        :return: Plotly ``Figure``;未安装 plotly 时返回 ``None``.
        """
        try:
            from ..plot import plot_indicators
        except ImportError:
            print(
                "Plotly is not installed. Please install it using "
                "`pip install plotly` or `pip install akquant[plot]`."
            )
            return None
        return plot_indicators(
            result=self._result,
            name=name,
            symbol=symbol,
            include_warmup=include_warmup,
            show=show,
            title=title,
            theme=theme,
            filename=filename,
        )

    def report(
        self,
        title: str = "AKQuant 策略回测报告",
        filename: str = "akquant_report.html",
        show: bool = False,
        compact_currency: bool = True,
        market_data: Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]] = None,
        plot_symbol: Optional[str] = None,
        include_trade_kline: bool = True,
        include_indicators: bool = False,
        indicator_name: Optional[str] = None,
        indicator_symbol: Optional[str] = None,
        indicator_include_warmup: bool = True,
        benchmark: Optional[Union[str, pd.Series]] = None,
        curve_freq: str = "D",
    ) -> None:
        """生成 akquant 原生全量 HTML 报告(便捷入口).

        与 :meth:`quantstats` 的区别:本方法为 akquant 原生、A股/中文优化,
        且与引擎打通(含风控拒单 / 强平 / 归因审计);:meth:`quantstats`
        输出行业标准 QuantStats 版式.

        :param title: 报告标题.
        :param filename: 保存文件名.
        :param show: 是否在浏览器中自动打开.
        :param compact_currency: 金额是否按 K/M/B 紧凑显示.
        :param market_data: 可选行情数据,用于 K 线买卖点图.
        :param plot_symbol: 指定 K 线复盘标的.
        :param include_trade_kline: 是否包含 K 线复盘图.
        :param include_indicators: 是否包含自定义指标预览区块.
        :param indicator_name: 可选指标键过滤.
        :param indicator_symbol: 可选标的过滤.
        :param indicator_include_warmup: 指标区块是否保留预热点.
        :param benchmark: 基准收益序列或标识字符串.
        :param curve_freq: 曲线频率,``D`` 为日频末值,``raw`` 为原始频率.
        """
        try:
            from ..plot.report import plot_report
        except ImportError:
            print("Plot module not found. Please install akquant[plot] or plotly.")
            return
        return plot_report(
            result=self._result,
            title=title,
            filename=filename,
            show=show,
            compact_currency=compact_currency,
            market_data=market_data,
            plot_symbol=plot_symbol,
            include_trade_kline=include_trade_kline,
            include_indicators=include_indicators,
            indicator_name=indicator_name,
            indicator_symbol=indicator_symbol,
            indicator_include_warmup=indicator_include_warmup,
            benchmark=benchmark,
            curve_freq=curve_freq,
        )

    def quantstats(
        self,
        benchmark: Optional[Union[str, pd.Series]] = None,
        title: str = "Strategy Report",
        filename: str = "quantstats-report.html",
        **kwargs: Any,
    ) -> None:
        """生成 QuantStats 版式 HTML 报告.

        与 :meth:`report` 的区别见后者文档:本方法输出行业标准 QuantStats
        布局,面向已熟悉 QuantStats 的用户.

        :param benchmark: 基准标识(如 ``"SPY"``)或 ``pd.Series``.
        :param title: 报告标题.
        :param filename: 输出文件名.
        :param kwargs: 透传给 ``qs.reports.html`` 的额外参数.
        """
        try:
            import quantstats as qs
        except ImportError:
            print(
                "QuantStats is not installed. Please install it using "
                "`pip install quantstats` or `pip install akquant[quantstats]`."
            )
            return
        qs.extend_pandas()
        returns = self._result.to_quantstats()
        if returns.empty:
            print("No returns data available to generate report.")
            return
        print(f"Generating QuantStats report to {filename}...")
        qs.reports.html(
            returns, benchmark=benchmark, title=title, output=filename, **kwargs
        )
        print("Done.")

    def review(
        self,
        market_data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        *,
        symbols: Optional[list[str]] = None,
        title: str = "AKQuant 交易复盘",
        filename: str = "akquant_review.html",
        theme: str = "light",
        initial_symbol: Optional[str] = None,
        show: bool = False,
    ) -> str:
        """LWC 交互式 K 线交易复盘(离线自包含 HTML).

        面向大数据量 / 日内 K 线的买卖点复盘,是对 :meth:`report` 中 plotly
        分析图的补充而非替代:分析图仍由 plotly 负责,本方法只补交互式 K 线。
        生成的 HTML 内联 vendored 的 lightweight-charts,无 CDN 依赖。

        :param market_data: 单个 DataFrame 或 ``{symbol: df}`` 行情;必填.
        :param symbols: 可选,限定并排序复盘的标的;默认全部可用标的.
        :param title: 报告标题(将被 HTML 转义).
        :param filename: 输出 HTML 路径.
        :param theme: ``"light"`` 或 ``"dark"``.
        :param initial_symbol: 初始展示标的;缺省为首个.
        :param show: 是否在浏览器中打开.
        :return: 写出的 HTML 文件绝对路径.
        :raises ValueError: ``market_data`` 为空或无有效行情可复盘.
        """
        from ..lwc import plot_kline_review

        return plot_kline_review(
            self._result,
            market_data,
            symbols=symbols,
            title=title,
            filename=filename,
            theme=theme,
            initial_symbol=initial_symbol,
            show=show,
        )
