#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant live indicators on a TradingView Lightweight Charts page.

The closed loop this example demonstrates: **one `self.I()` declaration in the
strategy -> one more live line on the browser K-line chart**. Nothing else to
wire up.

1. `run_live(broker="replay")` drives the strategy on a deterministic bar feed
   in a background thread - no broker or market connection needed.
2. Indicators declared with plot metadata are reported automatically; the
   `on_event` callback bridges them with `aq.to_indicator_message()` and then
   `akquant.lwc.to_lwc_update()` into `series.update()`-ready increments.
3. A tiny stdlib `http.server` exposes `/state?since_seq=N`; the page (LWC
   inlined via `akquant.lwc.load_lwc_js()`, no CDN) polls it and creates each
   indicator series lazily the first time its key shows up.

This is deliberately an *example*, not a core module: the HTTP/WS transport
belongs to the application layer (see the viz RFC). The reusable pieces are
`to_lwc_update()` and `load_lwc_js()`.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy, run_live
from akquant.akquant import AssetType, Instrument
from akquant.lwc import load_lwc_js, to_lwc_update

SYMBOL = "600000"


class LiveDeclarativeStrategy(Strategy):
    """Declare indicators once; the framework updates and reports them."""

    #: Set by ``main()`` before ``run_live``: where closed bars go. A class
    #: attribute because ``run_live`` instantiates the class itself.
    page_state: "PageState | None" = None

    def on_start(self) -> None:
        """Two overlays on the main pane, one oscillator in a sub-pane."""
        self.ema_fast = self.I(
            aq.EMA(5), name="ema_fast", pane=0, color="#e91e63", label="EMA5"
        )
        self.ema_slow = self.I(
            aq.EMA(12), name="ema_slow", pane=0, color="#3f51b5", label="EMA12"
        )
        self.rsi = self.I(
            aq.RSI(6),
            name="rsi",
            pane=1,
            reference_lines=[
                {"value": 70.0, "label": "OB"},
                {"value": 30.0, "label": "OS"},
            ],
        )

    def on_bar(self, bar: Bar) -> None:
        """Push the closed bar into the page state (the stream has no bar event)."""
        if self.page_state is not None:
            self.page_state.add_candle(bar)


class PageState:
    """Shared, lock-guarded state polled by the browser."""

    def __init__(self) -> None:
        """Start empty with a monotonically increasing cursor."""
        self.lock = threading.Lock()
        self.seq = 0
        self.candles: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.finished = False

    def add_candle(self, bar: Bar) -> None:
        """Append one closed bar as an LWC candlestick point."""
        with self.lock:
            self.seq += 1
            self.candles.append(
                {
                    "seq": self.seq,
                    "time": int(bar.timestamp) // 1_000_000_000,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                }
            )

    def add_update(self, update: dict[str, Any]) -> None:
        """Append one indicator increment produced by ``to_lwc_update``."""
        with self.lock:
            self.seq += 1
            self.updates.append({"seq": self.seq, **update})

    def snapshot(self, since_seq: int | None) -> dict[str, Any]:
        """Return everything newer than ``since_seq`` (all when None)."""
        with self.lock:
            floor = -1 if since_seq is None else since_seq
            return {
                "finished": self.finished,
                "latest_seq": self.seq,
                "candles": [c for c in self.candles if c["seq"] > floor],
                "updates": [u for u in self.updates if u["seq"] > floor],
            }


def make_bars(count: int = 60) -> list[Bar]:
    """Build deterministic intraday bars with a visible swing."""
    start = pd.Timestamp("2024-03-01 09:30:00", tz="Asia/Shanghai")
    bars: list[Bar] = []
    close = 10.0
    for i in range(count):
        direction = 1 if (i // 8) % 2 == 0 else -1
        close = close + 0.05 * direction + 0.02 * ((i % 3) - 1)
        bars.append(
            Bar(
                timestamp=int((start + pd.Timedelta(minutes=i)).value),
                open=close - 0.02,
                high=close + 0.04,
                low=close - 0.05,
                close=close,
                volume=1000.0 + 10.0 * i,
                symbol=SYMBOL,
            )
        )
    return bars


_PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>AKQuant live indicators (LWC)</title>
<style>
  html,body{margin:0;background:#fff;color:#333;
    font-family:-apple-system,"Segoe UI",Roboto,sans-serif;}
  #bar{display:flex;gap:16px;align-items:center;padding:8px 14px;
    border-bottom:1px solid #eee;font-size:13px;}
  #chart{width:100%;height:calc(100vh - 40px);}
</style></head><body>
<div id="bar"><b>AKQuant live indicators</b>
  <span id="status">waiting</span><span id="count">0 updates</span></div>
<div id="chart"></div>
<script>%%LWC_JS%%</script>
<script>
(function(){
  var LWC = window.LightweightCharts;
  var chart = LWC.createChart(document.getElementById('chart'), {
    autoSize:true, timeScale:{timeVisible:true, secondsVisible:false}});
  var candle = chart.addSeries(LWC.CandlestickSeries, {}, 0);
  var series = {};
  // This page has no volume pane, so sub-panes start at LWC pane 1.
  var paneMap = {}, nextPane = 1;
  var lastSeq = null, n = 0;
  function ensureSeries(u){
    if(series[u.indicator_key]) return series[u.indicator_key];
    var ctor = LWC[u.series_type + 'Series'] || LWC.LineSeries;
    var paneIdx = 0;
    if(u.pane >= 2){
      if(!(u.pane in paneMap)){ paneMap[u.pane] = nextPane++; }
      paneIdx = paneMap[u.pane];
    }
    var s = chart.addSeries(ctor, {color: u.color || '#1976d2', lineWidth:2,
      priceLineVisible:false, title: u.display_name}, paneIdx);
    if(paneIdx >= 1 && chart.panes && chart.panes()[paneIdx]){
      chart.panes()[paneIdx].setHeight(120);
    }
    series[u.indicator_key] = s;
    return s;
  }
  async function tick(){
    try{
      var qs = lastSeq === null ? '' : ('since_seq=' + lastSeq + '&');
      var r = await fetch('/state?' + qs + '_=' + Date.now());
      var st = await r.json();
      (st.candles||[]).forEach(function(c){
        candle.update({time:c.time, open:c.open, high:c.high,
          low:c.low, close:c.close});
      });
      (st.updates||[]).forEach(function(u){ ensureSeries(u).update(u.point); n++; });
      if(typeof st.latest_seq === 'number') lastSeq = st.latest_seq;
      document.getElementById('status').textContent =
        st.finished ? 'finished' : 'running';
      document.getElementById('count').textContent = n + ' updates';
    }catch(e){}
  }
  setInterval(tick, 250); tick();
})();
</script></body></html>"""


def make_handler(state: PageState) -> type[BaseHTTPRequestHandler]:
    """Bind a request handler to the shared state."""
    page = _PAGE.replace("%%LWC_JS%%", load_lwc_js())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 http.server 约定的方法名
            parts = urlsplit(self.path)
            if parts.path == "/state":
                raw = parse_qs(parts.query).get("since_seq", [None])[0]
                since = int(raw) if isinstance(raw, str) and raw.isdigit() else None
                body = json.dumps(state.snapshot(since)).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
                return
            if parts.path != "/":
                self.send_response(404)
                self.end_headers()
                return
            self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return Handler


def run_live_thread(state: PageState, done: threading.Event, sleep_ms: int) -> None:
    """Drive the strategy on the replay feed and feed the page state."""

    def on_event(event: aq.BacktestStreamEvent) -> None:
        message = aq.to_indicator_message(event)
        if message is None:
            return
        update = to_lwc_update(message)
        if update is not None:
            state.add_update(update)
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    LiveDeclarativeStrategy.page_state = state

    instruments = [
        Instrument(
            symbol=SYMBOL,
            asset_type=AssetType.Stock,
            multiplier=1.0,
            margin_ratio=1.0,
            tick_size=0.01,
            lot_size=100,
            option_type=None,
            strike_price=None,
            expiry_date=None,
        )
    ]
    run_live(
        strategy_cls=LiveDeclarativeStrategy,
        instruments=instruments,
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": make_bars()},
        cash=1_000_000,
        show_progress=False,
        on_event=on_event,
        duration="60s",  # safety net; replay ends on its own after the last bar
    )
    with state.lock:
        state.finished = True
    done.set()


def main() -> None:
    """Serve the page, run the live session, exit after the session ends."""
    parser = argparse.ArgumentParser(description="Live indicators on an LWC page.")
    parser.add_argument("--port", type=int, default=8898)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--sleep-ms", type=int, default=20)
    parser.add_argument("--keep-seconds", type=float, default=2.0)
    args = parser.parse_args()

    state = PageState()
    done = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    server.timeout = 0.2
    url = f"http://127.0.0.1:{args.port}/"
    thread = threading.Thread(
        target=run_live_thread, args=(state, done, args.sleep_ms), daemon=True
    )
    thread.start()
    if args.open:
        webbrowser.open(url)
    print(f"lwc_live_url={url}")

    end_after: float | None = None
    try:
        while True:
            server.handle_request()
            if done.is_set() and end_after is None:
                end_after = time.time() + args.keep_seconds
            if end_after is not None and time.time() >= end_after:
                break
    finally:
        server.server_close()
    thread.join(timeout=2.0)

    snap = state.snapshot(None)
    keys = sorted({u["indicator_key"] for u in snap["updates"]})
    print(f"lwc_live_candles={len(snap['candles'])}")
    print(f"lwc_live_updates={len(snap['updates'])}")
    print(f"lwc_live_indicator_keys={keys}")
    print("done_lwc_live_indicators")


if __name__ == "__main__":
    main()
