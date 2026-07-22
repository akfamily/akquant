"""TradingView Lightweight Charts based reporting and trade review.

This package is the plotly-free visualization path of AKQuant:

- :func:`plot_report` renders a static, self-contained HTML backtest report
  (metrics, equity/drawdown, K-line trade review) with all reviewed symbols
  embedded; the reviewed stock can be hot-switched inside the page.
- :func:`serve_review` starts a local web server where the reviewed stock is
  hot-switched by typing any resolvable code in the page; symbols missing
  from the preloaded data are resolved on demand from a data provider.
"""

from .report import build_app_data, plot_report, render_html
from .server import serve_review

__all__ = [
    "build_app_data",
    "plot_report",
    "render_html",
    "serve_review",
]
