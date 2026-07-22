"""Interactive trade-review server with on-demand symbol loading.

Serves the same Lightweight Charts page as :mod:`akquant.lwc.report`, but the
page runs in *server mode*: when the user types a symbol code that is not
already cached in the browser, it is fetched from ``/api/symbol``. This lets
the user hot-switch the reviewed stock through the web UI without regenerating
any file, and lets new codes be resolved live from a user-supplied data
provider.

Typical usage::

    from akquant.lwc import serve_review

    serve_review(
        result,
        market_data={"600000": df1, "600004": df2},
        data_provider=my_loader,  # optional, callable(code) -> DataFrame
    )
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import parse_qs, urlparse

import pandas as pd

from ._normalize import (
    build_symbol_payload,
    coerce_market_data,
    extract_trades_by_symbol,
)
from .report import build_app_data, render_html

DataProvider = Callable[[str], pd.DataFrame]


class _ReviewState:
    """Shared state behind the review HTTP handler."""

    def __init__(
        self,
        result: Any,
        market_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
        data_provider: Optional[DataProvider],
    ) -> None:
        """Initialize caches and the trade index.

        :param result: ``BacktestResult``-like object.
        :param market_data: Pre-loaded OHLCV frames (dict or single frame).
        :param data_provider: Optional callable resolving a symbol code into
            an OHLCV frame for symbols missing from ``market_data``.
        """
        self.trades_by_symbol = extract_trades_by_symbol(result)
        self.frames = coerce_market_data(market_data, list(self.trades_by_symbol))
        self.data_provider = data_provider
        self.cache: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def known_symbols(self) -> list:
        """Return codes offered in the page autocomplete."""
        return sorted(set(self.trades_by_symbol) | set(self.frames))

    def get_payload(self, code: str) -> Dict[str, Any]:
        """Return (and cache) the chart payload for one symbol code.

        :param code: Symbol code requested from the web page.
        :return: Payload from
            :func:`akquant.lwc._normalize.build_symbol_payload`.
        :raises LookupError: If no data source can provide the symbol.
        """
        with self.lock:
            if code in self.cache:
                return self.cache[code]
        frame = self.frames.get(code)
        if frame is None and self.data_provider is not None:
            frame = self.data_provider(code)
        if frame is None:
            raise LookupError(
                "无可用行情数据（不在 market_data 中，且 data_provider "
                "未提供该代码或返回 None）"
            )
        payload = build_symbol_payload(code, frame, self.trades_by_symbol.get(code, []))
        with self.lock:
            self.cache[code] = payload
        return payload


def _make_handler(state: _ReviewState, page_html: str) -> type:
    """Create a request handler class bound to the review state."""

    class ReviewHandler(BaseHTTPRequestHandler):
        """Serves the review page and its JSON data API."""

        def _send(self, status: int, body: str, content_type: str) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            """Route GET requests to the page or the JSON API."""
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, page_html, "text/html")
            elif parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif parsed.path == "/api/symbols":
                body = json.dumps(state.known_symbols(), ensure_ascii=False)
                self._send(200, body, "application/json")
            elif parsed.path == "/api/symbol":
                code = (parse_qs(parsed.query).get("code") or [""])[0].strip()
                if not code:
                    self._send(400, '{"error": "missing code"}', "application/json")
                    return
                try:
                    payload = state.get_payload(code)
                except Exception as exc:  # provider errors included
                    body = json.dumps(
                        {"error": "未能加载标的 %s: %s" % (code, exc)},
                        ensure_ascii=False,
                    )
                    self._send(404, body, "application/json")
                    return
                self._send(
                    200,
                    json.dumps(payload, ensure_ascii=False),
                    "application/json",
                )
            else:
                self._send(404, "not found", "text/plain")

        def log_message(self, format: str, *args: Any) -> None:
            """Silence default stderr logging."""

    return ReviewHandler


def serve_review(
    result: Any,
    market_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    data_provider: Optional[DataProvider] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    title: str = "AKQuant 交易复盘 (Lightweight Charts)",
    open_browser: bool = True,
) -> None:
    """Start the interactive trade-review web server (blocking).

    The served page embeds only portfolio-level data (metrics, equity,
    drawdown) plus the trade index; per-symbol K-line payloads are fetched by
    the page on demand, so typing any resolvable stock code in the input box
    hot-switches the review chart.

    :param result: ``BacktestResult``-like object.
    :param market_data: Pre-loaded OHLCV frames (dict, single frame or long
        frame with a symbol column).
    :param data_provider: Optional callable ``(code) -> DataFrame`` used to
        resolve codes not present in ``market_data``.
    :param host: Bind host.
    :param port: Bind port; 0 picks a free port.
    :param title: Page title.
    :param open_browser: Open the page in the default browser on start.
    """
    state = _ReviewState(result, market_data, data_provider)
    app_data = build_app_data(
        result,
        market_data=None,
        title=title,
        server_mode=True,
        extra_symbols=state.known_symbols(),
    )
    page_html = render_html(title, app_data)
    server = ThreadingHTTPServer((host, port), _make_handler(state, page_html))
    url = "http://%s:%d/" % (host, server.server_address[1])
    print("AKQuant 交易复盘服务已启动: %s (Ctrl+C 停止)" % url)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("交易复盘服务已停止。")
