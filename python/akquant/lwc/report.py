"""Static HTML report generation based on TradingView Lightweight Charts.

Produces a single self-contained HTML file (the Lightweight Charts bundle is
vendored into the package, no CDN required) containing:

- a metric card grid (from ``result.metrics`` / ``result.trade_metrics``),
- an equity curve with a synchronized drawdown pane,
- a trade-review section (candlesticks + volume + buy/sell markers) where the
  reviewed symbol can be hot-switched from the page via an interactive input.

Typical usage::

    from akquant.lwc import plot_report

    plot_report(result, market_data={"600000": df1, "600004": df2},
                filename="report.html")
"""

from __future__ import annotations

import html as _html
import json
import warnings
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ._normalize import (
    build_symbol_payload,
    coerce_market_data,
    extract_equity,
    extract_metrics,
    extract_trades_by_symbol,
    pick_initial_symbol,
)
from ._template import HTML_TEMPLATE

_ASSETS_DIR = Path(__file__).parent / "assets"
_LWC_BUNDLE = "lightweight-charts.standalone.production.js"


def load_standalone_js() -> str:
    """Read the vendored Lightweight Charts standalone bundle.

    :return: JavaScript source of the bundle.
    :raises FileNotFoundError: If the bundle is missing from the package.
    """
    path = _ASSETS_DIR / _LWC_BUNDLE
    if not path.exists():
        raise FileNotFoundError(
            "Lightweight Charts bundle not found at %s; the akquant.lwc "
            "package data may be incomplete." % path
        )
    return path.read_text(encoding="utf-8")


def build_app_data(
    result: Any,
    market_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    symbols: Optional[List[str]] = None,
    title: str = "",
    server_mode: bool = False,
    plot_symbol: Optional[str] = None,
    extra_symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the JSON payload consumed by the browser application.

    :param result: ``BacktestResult``-like object.
    :param market_data: OHLCV data used for the trade-review chart. Accepts a
        ``{symbol: frame}`` dict, a single frame, or a long frame carrying a
        symbol column.
    :param symbols: Optional subset of symbols to embed in the page. Only
        meaningful in static mode; the server loads symbols on demand.
    :param title: Report title.
    :param server_mode: When true, the page fetches unknown symbols from the
        review server instead of failing.
    :param plot_symbol: Symbol displayed initially.
    :param extra_symbols: Extra codes listed in the page's autocomplete even
        when their data is not embedded (used by the server mode).
    :return: JSON-serializable application payload.
    """
    trades_by_symbol = extract_trades_by_symbol(result)
    frames = coerce_market_data(market_data, list(trades_by_symbol))
    if symbols is not None:
        wanted = {str(s) for s in symbols}
        frames = {k: v for k, v in frames.items() if k in wanted}

    payloads: Dict[str, Any] = {}
    for symbol, frame in frames.items():
        try:
            payloads[symbol] = build_symbol_payload(
                symbol, frame, trades_by_symbol.get(symbol, [])
            )
        except ValueError as exc:
            warnings.warn("skipping symbol %r: %s" % (symbol, exc))

    equity, drawdown = extract_equity(result)
    symbol_list = sorted(
        set(payloads) | set(trades_by_symbol) | set(extra_symbols or [])
    )
    return {
        "title": title,
        "serverMode": server_mode,
        "metrics": extract_metrics(result),
        "equity": equity,
        "drawdown": drawdown,
        "payloads": payloads,
        "symbols": symbol_list,
        "initialSymbol": pick_initial_symbol(
            payloads, trades_by_symbol, preferred=plot_symbol
        ),
    }


def render_html(title: str, app_data: Dict[str, Any]) -> str:
    """Render the final self-contained HTML document.

    :param title: Report title (HTML-escaped during rendering).
    :param app_data: Payload from :func:`build_app_data`.
    :return: Complete HTML source.
    """
    app_json = json.dumps(app_data, ensure_ascii=False, separators=(",", ":"))
    # Guard against an accidental "</script>" terminating the data island.
    app_json = app_json.replace("<", "\\u003c")
    escaped_title = _html.escape(title)
    page = HTML_TEMPLATE.replace("__TITLE__", escaped_title)
    page = page.replace("__LWC_JS__", load_standalone_js(), 1)
    page = page.replace("__APP_JSON__", app_json, 1)
    return page


def plot_report(
    result: Any,
    market_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    title: str = "AKQuant 策略回测报告 (Lightweight Charts)",
    filename: str = "akquant_lwc_report.html",
    symbols: Optional[List[str]] = None,
    plot_symbol: Optional[str] = None,
    show: bool = False,
) -> str:
    """Generate a static single-file backtest report.

    :param result: ``BacktestResult``-like object.
    :param market_data: OHLCV data for the trade-review chart; pass a
        ``{symbol: frame}`` dict to enable in-page hot switching between
        symbols without a server.
    :param title: Report title.
    :param filename: Output HTML path.
    :param symbols: Optional subset of symbols to embed.
    :param plot_symbol: Symbol displayed initially.
    :param show: Open the report in the default browser when done.
    :return: Absolute path of the generated HTML file.
    """
    app_data = build_app_data(
        result,
        market_data=market_data,
        symbols=symbols,
        title=title,
        server_mode=False,
        plot_symbol=plot_symbol,
    )
    page = render_html(title, app_data)
    out = Path(filename).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    if show:
        webbrowser.open(out.as_uri())
    return str(out)
