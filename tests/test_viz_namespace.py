"""``result.viz.*`` 可视化命名空间的结构性测试.

覆盖 RFC(``docs/zh/meta/viz-namespace-and-lwc-review-rfc.md``)P0 阶段的
契约:旧方法已破坏性删除、``viz`` 属性就位且惰性、五个 viz 方法可达、
``to_quantstats`` 保留在顶层。
"""

from __future__ import annotations

import importlib.util
import sys

import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest
from akquant.backtest._viz import VizNamespace
from akquant.backtest.result import BacktestResult


class _RoundTrip(Strategy):
    """产生一笔完整往返交易,便于构造非空 result."""

    def __init__(self) -> None:
        """初始化步进计数器."""
        super().__init__()
        self.step = 0

    def on_bar(self, bar: Bar) -> None:
        """第 1 根买入,第 3 根卖出."""
        self.step += 1
        if self.step == 1:
            self.buy(symbol=bar.symbol, quantity=100)
        elif self.step == 3:
            self.sell(symbol=bar.symbol, quantity=100)


def _make_result() -> BacktestResult:
    """跑一个极小回测,返回 BacktestResult 实例."""
    bars = [
        Bar(
            timestamp=pd.Timestamp(f"2024-01-{i + 1:02d} 10:00:00").value,
            open=10.0 + i,
            high=10.2 + i,
            low=9.8 + i,
            close=10.0 + i,
            volume=1000.0,
            symbol="TEST",
        )
        for i in range(5)
    ]
    return run_backtest(
        data=bars,
        strategy=_RoundTrip,
        symbols="TEST",
        initial_cash=100000.0,
    )


_OLD_METHODS = ("plot", "plot_indicators", "report", "report_quantstats")
_VIZ_METHODS = ("dashboard", "indicators", "report", "quantstats", "review")


@pytest.mark.parametrize("old", _OLD_METHODS)
def test_old_methods_removed(old: str) -> None:
    """破坏性变更:旧的顶层可视化方法应已删除."""
    assert not hasattr(BacktestResult, old)


def test_to_quantstats_retained() -> None:
    """``to_quantstats`` 是数据导出方法,应保留在顶层."""
    assert hasattr(BacktestResult, "to_quantstats")


def test_viz_is_property() -> None:
    """``viz`` 应为属性而非普通方法."""
    assert isinstance(BacktestResult.viz, property)


@pytest.mark.parametrize("method", _VIZ_METHODS)
def test_viz_namespace_has_methods(method: str) -> None:
    """五个 job 命名的方法应全部可达."""
    assert callable(getattr(VizNamespace, method))


def test_viz_returns_namespace_bound_to_result() -> None:
    """``result.viz`` 返回绑定到该 result 的命名空间实例."""
    result = _make_result()
    ns = result.viz
    assert isinstance(ns, VizNamespace)
    assert ns._result is result


def test_viz_property_does_not_import_plotly() -> None:
    """访问 ``viz`` 属性本身不应触发 plotly 导入(惰性依赖)."""
    sys.modules.pop("plotly", None)
    result = _make_result()
    _ = result.viz  # 仅访问属性,不调用任何绘图方法
    assert "plotly" not in sys.modules


def test_review_requires_market_data() -> None:
    """P1 后 ``review`` 已落地;缺 market_data 应抛 ValueError."""
    result = _make_result()
    with pytest.raises(ValueError):
        result.viz.review(None)  # type: ignore[arg-type]


def test_dashboard_smoke() -> None:
    """``viz.dashboard`` 应能产出 Figure(需 plotly)."""
    if importlib.util.find_spec("plotly") is None:
        pytest.skip("plotly 未安装")
    result = _make_result()
    fig = result.viz.dashboard(show=False)
    assert fig is not None
