#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant live indicator web demo.

This example shows a zero-extra-dependency browser demo for enterprise integration:
1. Run a backtest with indicator stream events enabled.
2. Bridge raw events into frontend-friendly indicator messages.
3. Serve a tiny polling web page that renders live indicator values.
"""

import argparse
import json
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def build_data(n: int = 90) -> pd.DataFrame:
    """Build synthetic bars that generate visible indicator movement."""
    ts = pd.date_range("2024-01-01", periods=n, freq="D")
    rows = []
    for i, dt in enumerate(ts):
        close = 100.0 + 0.12 * i + 2.6 * __import__("math").sin(i / 6.0)
        rows.append(
            {
                "date": dt,
                "open": close - 0.2,
                "high": close + 0.4,
                "low": close - 0.5,
                "close": close,
                "volume": 1000.0 + float(i % 20) * 50.0,
                "symbol": "LIVE_IND",
            }
        )
    return pd.DataFrame(rows)


class LiveIndicatorStrategy(Strategy):
    """Emit two simple indicators for live browser preview."""

    def on_bar(self, bar: Bar) -> None:
        """Record a main-pane line and a signal-pane bar metric."""
        self.record_indicator(
            name="close_echo",
            value=bar.close,
            display_name="Close Echo",
            pane=0,
            render_type="line",
            meta={"source": "close"},
        )
        self.record_indicator(
            name="intrabar_range",
            value=bar.high - bar.low,
            display_name="Intrabar Range",
            pane=1,
            render_type="bar",
            meta={"source": ["high", "low"]},
        )


@dataclass
class IndicatorLiveState:
    """Shared state polled by the browser page."""

    started: bool = False
    finished: bool = False
    run_id: str = ""
    point_messages: list[dict[str, Any]] | None = None
    snapshot_messages: list[dict[str, Any]] | None = None
    latest_indicator_values: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize mutable defaults."""
        self.point_messages = []
        self.snapshot_messages = []
        self.latest_indicator_values = {}


def _message_seq(message: dict[str, Any]) -> int:
    """Extract comparable seq from one bridged message."""
    try:
        return int(message.get("seq", 0))
    except (TypeError, ValueError):
        return 0


def _build_state_payload(
    state: IndicatorLiveState,
    *,
    since_seq: int | None = None,
) -> dict[str, Any]:
    """Build full or incremental browser state payload."""
    point_messages = list(state.point_messages or [])
    snapshot_messages = list(state.snapshot_messages or [])
    latest_seq = 0
    if point_messages:
        latest_seq = max(
            latest_seq, max(_message_seq(message) for message in point_messages)
        )
    if snapshot_messages:
        latest_seq = max(
            latest_seq, max(_message_seq(message) for message in snapshot_messages)
        )

    payload = {
        "started": state.started,
        "finished": state.finished,
        "run_id": state.run_id,
        "latest_indicator_values": dict(state.latest_indicator_values or {}),
        "counts": {
            "point_messages": len(point_messages),
            "snapshot_messages": len(snapshot_messages),
        },
        "cursor": {
            "latest_seq": latest_seq,
            "since_seq": since_seq,
            "incremental": since_seq is not None,
        },
    }
    if since_seq is None:
        payload["window"] = {
            "point_messages": point_messages,
            "snapshot_messages": snapshot_messages,
        }
        return payload

    payload["delta"] = {
        "point_messages": [
            message for message in point_messages if _message_seq(message) > since_seq
        ],
        "snapshot_messages": [
            message
            for message in snapshot_messages
            if _message_seq(message) > since_seq
        ],
    }
    return payload


def make_handler(
    state: IndicatorLiveState, lock: threading.Lock
) -> type[BaseHTTPRequestHandler]:
    """Create one request handler bound to shared indicator state."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            split_result = urlsplit(self.path)
            req_path = split_result.path
            if req_path == "/state":
                query = parse_qs(split_result.query)
                since_seq_raw = query.get("since_seq", [None])[0]
                since_seq: int | None = None
                if isinstance(since_seq_raw, str) and since_seq_raw != "":
                    try:
                        since_seq = int(since_seq_raw)
                    except (TypeError, ValueError):
                        since_seq = None
                with lock:
                    payload = _build_state_payload(state, since_seq=since_seq)
                body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path != "/":
                self.send_response(404)
                self.end_headers()
                return

            html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AKQuant Indicator Live Demo</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 20px;
      background: #fafafa;
      color: #222;
    }
    .row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    .card {
      background: #fff;
      border: 1px solid #e6e6e6;
      border-radius: 8px;
      padding: 14px;
      min-width: 220px;
    }
    .title { font-size: 13px; color: #666; margin-bottom: 6px; }
    .value { font-size: 22px; font-weight: 700; }
    .section {
      background: #fff;
      border: 1px solid #e6e6e6;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 12px;
    }
    canvas {
      width: 100%;
      height: 280px;
      border: 1px solid #ddd;
      border-radius: 6px;
      background: #fff;
    }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
  </style>
</head>
<body>
  <h2>AKQuant Indicator Live Browser Demo</h2>
  <div class="row">
    <div class="card">
      <div class="title">run_id</div>
      <div class="value" id="run_id">-</div>
    </div>
    <div class="card">
      <div class="title">status</div>
      <div class="value" id="status">waiting</div>
    </div>
    <div class="card">
      <div class="title">point_messages</div>
      <div class="value" id="point_count">0</div>
    </div>
    <div class="card">
      <div class="title">snapshot_messages</div>
      <div class="value" id="snapshot_count">0</div>
    </div>
  </div>
  <div class="section">
    <h3>Close Echo</h3>
    <canvas id="close_chart" width="1200" height="280"></canvas>
  </div>
  <div class="section">
    <h3>Latest Indicator Values</h3>
    <pre id="latest_values">{}</pre>
  </div>
  <div class="section">
    <h3>Latest Snapshot</h3>
    <pre id="latest_snapshot">-</pre>
  </div>
  <script>
    const closeCtx = document.getElementById("close_chart").getContext("2d");
    const runIdEl = document.getElementById("run_id");
    const statusEl = document.getElementById("status");
    const pointCountEl = document.getElementById("point_count");
    const snapshotCountEl = document.getElementById("snapshot_count");
    const latestValuesEl = document.getElementById("latest_values");
    const latestSnapshotEl = document.getElementById("latest_snapshot");
    let livePointMessages = [];
    let liveSnapshotMessages = [];
    let lastSeq = null;

    function drawLine(ctx, values) {
      const w = ctx.canvas.width;
      const h = ctx.canvas.height;
      ctx.clearRect(0, 0, w, h);
      if (!values || values.length < 2) return;
      let minV = Math.min(...values);
      let maxV = Math.max(...values);
      if (maxV <= minV) maxV = minV + 1;
      ctx.strokeStyle = "#d32f2f";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < values.length; i++) {
        const x = (i / (values.length - 1)) * (w - 20) + 10;
        const y = h - ((values[i] - minV) / (maxV - minV)) * (h - 20) - 10;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    async function tickState() {
      try {
        const qs = lastSeq === null
          ? ""
          : ("since_seq=" + encodeURIComponent(String(lastSeq)) + "&");
        const res = await fetch("/state?" + qs + "_=" + Date.now());
        const s = await res.json();
        const cursor = s.cursor || {};
        const counts = s.counts || {};
        if (cursor.incremental) {
          const delta = s.delta || {};
          livePointMessages = livePointMessages
            .concat(delta.point_messages || [])
            .slice(-240);
          liveSnapshotMessages = liveSnapshotMessages
            .concat(delta.snapshot_messages || [])
            .slice(-60);
        } else {
          const windowData = s.window || {};
          livePointMessages = (windowData.point_messages || []).slice(-240);
          liveSnapshotMessages = (windowData.snapshot_messages || []).slice(-60);
        }
        if (typeof cursor.latest_seq === "number") {
          lastSeq = cursor.latest_seq;
        }
        runIdEl.textContent = s.run_id || "-";
        statusEl.textContent = s.finished
          ? "finished"
          : (s.started ? "running" : "waiting");
        pointCountEl.textContent = String(
          counts.point_messages || livePointMessages.length
        );
        snapshotCountEl.textContent = String(
          counts.snapshot_messages || liveSnapshotMessages.length
        );
        latestValuesEl.textContent = JSON.stringify(
          s.latest_indicator_values || {},
          null,
          2
        );
        latestSnapshotEl.textContent = liveSnapshotMessages.length
          ? JSON.stringify(
              liveSnapshotMessages[liveSnapshotMessages.length - 1],
              null,
              2
            )
          : "-";
        const closeValues = livePointMessages
          .filter(
            m =>
              m.type === "point"
              && m.indicator
              && m.indicator.indicator_key === "close_echo"
          )
          .map(m => Number(m.indicator.value))
          .filter(v => !Number.isNaN(v));
        drawLine(closeCtx, closeValues);
      } catch (_e) {}
    }
    setInterval(tickState, 250);
    tickState();
  </script>
</body>
</html>
            """.strip()
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def run_backtest_thread(
    state: IndicatorLiveState,
    lock: threading.Lock,
    sleep_ms: int,
    done_event: threading.Event,
) -> None:
    """Run the stream-enabled backtest and update browser state."""

    def on_event(event: aq.BacktestStreamEvent) -> None:
        message = aq.to_indicator_message(event)
        with lock:
            if str(event.get("event_type", "")) == "started":
                state.started = True
                state.run_id = str(event.get("run_id", ""))
            elif str(event.get("event_type", "")) == "finished":
                state.finished = True
            if message is not None:
                if message["type"] == "point":
                    points = state.point_messages or []
                    points.append(message)
                    state.point_messages = points[-240:]
                    indicator = message.get("indicator", {})
                    if isinstance(indicator, dict):
                        key = str(indicator.get("indicator_key", ""))
                        latest_values = state.latest_indicator_values or {}
                        latest_values[key] = indicator.get("value")
                        state.latest_indicator_values = latest_values
                elif message["type"] == "snapshot":
                    snapshots = state.snapshot_messages or []
                    snapshots.append(message)
                    state.snapshot_messages = snapshots[-60:]
        if sleep_ms > 0:
            time.sleep(float(sleep_ms) / 1000.0)

    aq.run_backtest(
        data=build_data(),
        strategy=LiveIndicatorStrategy,
        symbols="LIVE_IND",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
        on_event=on_event,
        stream_progress_interval=6,
        stream_equity_interval=12,
        stream_batch_size=1,
        stream_max_buffer=128,
    )
    with lock:
        state.finished = True
    done_event.set()


def main() -> None:
    """Serve a lightweight browser page that polls live indicator state."""
    parser = argparse.ArgumentParser(description="Live indicator browser demo.")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--sleep-ms", type=int, default=30)
    parser.add_argument("--keep-seconds", type=int, default=20)
    args = parser.parse_args()

    state = IndicatorLiveState()
    lock = threading.Lock()
    done_event = threading.Event()
    handler_cls = make_handler(state, lock)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_cls)
    server.timeout = 0.2
    url = f"http://127.0.0.1:{args.port}/"

    thread = threading.Thread(
        target=run_backtest_thread,
        args=(state, lock, args.sleep_ms, done_event),
        daemon=True,
    )
    thread.start()

    if args.open:
        webbrowser.open(url)
    print(f"indicator_live_web_url={url}")
    print(f"sleep_ms={args.sleep_ms}")
    print(f"keep_seconds={args.keep_seconds}")

    end_after = None
    try:
        while True:
            server.handle_request()
            if done_event.is_set() and end_after is None:
                end_after = time.time() + float(args.keep_seconds)
            if end_after is not None and time.time() >= end_after:
                break
    finally:
        server.server_close()
    thread.join(timeout=1.0)
    with lock:
        point_count = len(state.point_messages or [])
        snapshot_count = len(state.snapshot_messages or [])
        latest_keys = sorted((state.latest_indicator_values or {}).keys())
    print(f"indicator_live_point_messages={point_count}")
    print(f"indicator_live_snapshot_messages={snapshot_count}")
    print(f"indicator_live_latest_keys={latest_keys}")
    print("done_indicator_live_web")


if __name__ == "__main__":
    main()
