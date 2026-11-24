"""Utility functions for plotting."""

import importlib
import importlib.util
from typing import Any


class _LazyModule:
    """第一次取属性时才导入的模块代理.

    plotly 只在画图时用到, 导入要几十毫秒; ``import akquant`` 会连带导入本包,
    所以不在模块顶层导入它。绘图代码照常写 ``go.Figure(...)``。
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._module: Any = None

    def __getattr__(self, attr: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return getattr(self._module, attr)


go: Any = _LazyModule("plotly.graph_objects")


def make_subplots(*args: Any, **kwargs: Any) -> Any:
    """Forward to ``plotly.subplots.make_subplots``; plotly 在调用时才导入."""
    from plotly.subplots import make_subplots as _make_subplots  # type: ignore

    return _make_subplots(*args, **kwargs)


def check_plotly() -> bool:
    """Check if plotly is installed."""
    if importlib.util.find_spec("plotly") is None:
        print(
            "Plotly is not installed. Please install it using `pip install plotly` "
            "or `pip install akquant[plot]`."
        )
        return False
    return True


THEMES = {
    "light": {
        "up_color": "#d32f2f",  # China Red (Rise)
        "down_color": "#2e7d32",  # China Green (Fall)
        "bg_color": "#ffffff",
        "grid_color": "#f0f0f0",
        "text_color": "#333333",
    },
    "dark": {
        "up_color": "#ff5252",  # Bright Red
        "down_color": "#69f0ae",  # Bright Green
        "bg_color": "#1e1e1e",
        "grid_color": "#2b2b2b",
        "text_color": "#e0e0e0",
    },
}


def get_color(theme: str, key: str) -> str:
    """Get color from theme."""
    return THEMES.get(theme, THEMES["light"]).get(key, "#000000")
