"""``import akquant`` 的导入范围: 只在画图时才用的重依赖不应在导入时加载."""

import subprocess
import sys

import pytest


def _run_snippet(code: str) -> str:
    """在全新解释器里执行代码片段并返回标准输出(本进程里相关模块早已加载)."""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_import_akquant_does_not_import_plotly() -> None:
    """只做回测的进程不该为画图库付导入时间."""
    output = _run_snippet(
        "import sys, akquant\n"
        "print(any(name.split('.')[0] == 'plotly' for name in sys.modules))"
    )

    assert output == "False"


def test_plotting_names_load_plotly_on_first_use() -> None:
    """绘图模块里的 go / make_subplots 第一次用到时才导入 plotly, 用法不变."""
    pytest.importorskip("plotly")
    output = _run_snippet(
        "import sys, akquant\n"
        "from akquant.plot.utils import check_plotly, go, make_subplots\n"
        "before = 'plotly' in sys.modules\n"
        "figure = make_subplots(rows=2, cols=1)\n"
        "figure.add_trace(go.Scatter(x=[1, 2], y=[3, 4]), row=1, col=1)\n"
        "print(before, check_plotly(), type(figure).__module__, len(figure.data))"
    )

    assert output == "False True plotly.graph_objs._figure 1"
