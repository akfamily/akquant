import argparse
import json
import math
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def build_data(n: int = 520) -> pd.DataFrame:
    """Build synthetic data for visible live web animation."""
    ts = pd.date_range("2021-01-01", periods=n, freq="D")
    close = [
        100.0 + 0.03 * i + 3.2 * math.sin(i / 10.0) + 1.8 * math.sin(i / 3.4)
        for i in range(n)
    ]
    return pd.DataFrame(
        {
            "date": ts,
            "open": [c * 0.998 for c in close],
            "high": [c * 1.005 for c in close],
            "low": [c * 0.995 for c in close],
            "close": close,
            "volume": [900.0 + float(i % 35) * 26.0 for i in range(n)],
            "symbol": "LIVE_WEB",
        }
    )


class LiveWebStrategy(Strategy):
    """Simple MA crossover strategy for realtime web demo."""

    def __init__(self) -> None:
        """Initialize strategy warmup period."""
        super().__init__()
        self.warmup_period = 20

    def on_bar(self, bar: Bar) -> None:
        """Trade by short and long moving average relation."""
        closes = self.get_history(20, bar.symbol, "close")
        if len(closes) < 20:
            return
        short = float(sum(closes[-5:])) / 5.0
        long = float(sum(closes)) / 20.0
        pos = self.get_position(bar.symbol)
        if short > long and pos == 0:
            self.order_target_percent(symbol=bar.symbol, target_percent=0.9)
        elif short < long and pos > 0:
            self.close_position(symbol=bar.symbol)


@dataclass
class LiveState:
    """Mutable stream state for browser polling."""

    started: bool = False
    finished: bool = False
    run_id: str = ""
    progress_events: int = 0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    total_return: float | None = None
    equity_points: list[float] | None = None
    alerts: list[str] | None = None
    console_equity_ticks: int = 0

    def __post_init__(self) -> None:
        """Initialize list fields."""
        self.equity_points = []
        self.alerts = []


def make_handler(
    state: LiveState, lock: threading.Lock
) -> type[BaseHTTPRequestHandler]:
    """Create request handler bound to shared state."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            req_path = urlsplit(self.path).path
            if req_path == "/state":
                with lock:
                    payload = {
                        "started": state.started,
                        "finished": state.finished,
                        "run_id": state.run_id,
                        "progress_events": state.progress_events,
                        "max_drawdown": state.max_drawdown,
                        "total_return": state.total_return,
                        "alerts": list(state.alerts or []),
                        "equity_points": list(state.equity_points or []),
                    }
                body = json.dumps(payload).encode("utf-8")
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
  <title>AKQuant Live Streaming Demo</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 20px;
      background: #fafafa;
      color: #222;
    }
    .card {
      background: #fff;
      border: 1px solid #e6e6e6;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 12px;
    }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .metric { min-width: 220px; }
    #chart {
      width: 100%;
      height: 320px;
      border: 1px solid #ddd;
      border-radius: 6px;
      background: #fff;
    }
    #alerts { color: #b00020; white-space: pre-wrap; line-height: 1.4; }
    .ok { color: #2e7d32; }
  </style>
</head>
<body>
  <h2>Streaming Live Backtest (肉眼可见实时变化)</h2>
  <div class="row">
    <div class="card metric"><div>run_id</div><div id="run_id">-</div></div>
    <div class="card metric">
      <div>progress_events</div>
      <div id="progress_events">0</div>
    </div>
    <div class="card metric">
      <div>max_drawdown</div>
      <div id="max_drawdown">0.00%</div>
    </div>
    <div class="card metric">
      <div>total_return</div>
      <div id="total_return">-</div>
    </div>
    <div class="card metric"><div>status</div><div id="status">waiting</div></div>
    <div class="card metric"><div>drawn_points</div><div id="drawn_points">0</div></div>
  </div>
  <div class="card"><canvas id="chart" width="1200" height="320"></canvas></div>
  <div class="card">
    <div>alerts</div>
    <div id="alerts">none</div>
  </div>
  <script>
    const ctx = document.getElementById("chart").getContext("2d");
    const runIdEl = document.getElementById("run_id");
    const progressEl = document.getElementById("progress_events");
    const ddEl = document.getElementById("max_drawdown");
    const retEl = document.getElementById("total_return");
    const statusEl = document.getElementById("status");
    const alertsEl = document.getElementById("alerts");
    const drawnEl = document.getElementById("drawn_points");
    let pointsCache = [];
    let drawHead = 0;

    function drawLine(values) {
      const w = ctx.canvas.width;
      const h = ctx.canvas.height;
      ctx.clearRect(0, 0, w, h);
      if (!values || values.length < 2) return;
      let minV = Math.min(...values);
      let maxV = Math.max(...values);
      if (maxV <= minV) maxV = minV + 1;
      const n = values.length;
      ctx.strokeStyle = "#d32f2f";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = (i / (n - 1)) * (w - 20) + 10;
        const y = h - ((values[i] - minV) / (maxV - minV)) * (h - 20) - 10;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    function renderFrame() {
      if (drawHead < pointsCache.length) {
        drawHead = Math.min(drawHead + 2, pointsCache.length);
      }
      drawnEl.textContent = String(drawHead);
      drawLine(pointsCache.slice(0, drawHead));
    }

    async function tickState() {
      try {
        const res = await fetch("/state?_=" + Date.now());
        const s = await res.json();
        runIdEl.textContent = s.run_id || "-";
        progressEl.textContent = String(s.progress_events || 0);
        ddEl.textContent = ((s.max_drawdown || 0) * 100).toFixed(2) + "%";
        retEl.textContent = s.total_return == null
          ? "-"
          : ((s.total_return || 0) * 100).toFixed(2) + "%";
        alertsEl.textContent = (s.alerts && s.alerts.length)
          ? s.alerts.join("\\n")
          : "none";
        statusEl.textContent = s.finished
          ? "finished"
          : (s.started ? "running" : "waiting");
        statusEl.className = s.finished ? "ok" : "";
        const latest = s.equity_points || [];
        if (latest.length < pointsCache.length) {
          pointsCache = latest;
          drawHead = 0;
        } else if (latest.length > pointsCache.length) {
          pointsCache = latest;
        }
      } catch (_e) {}
    }
    setInterval(tickState, 260);
    setInterval(renderFrame, 90);
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
    state: LiveState,
    lock: threading.Lock,
    sleep_ms: int,
    done_event: threading.Event,
) -> None:
    """Run backtest and update shared state from stream events."""

    def sparkline(values: list[float], width: int = 32) -> str:
        if not values:
            return ""
        window = values[-width:]
        lo = min(window)
        hi = max(window)
        blocks = "▁▂▃▄▅▆▇█"
        if hi <= lo:
            return blocks[0] * len(window)
        span = hi - lo
        out: list[str] = []
        for v in window:
            idx = int((v - lo) / span * (len(blocks) - 1))
            out.append(blocks[idx])
        return "".join(out)

    def on_event(event: aq.BacktestStreamEvent) -> None:
        event_type = str(event.get("event_type", "unknown"))
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        console_line = ""
        console_note = ""
        with lock:
            if event_type == "started":
                state.started = True
                state.run_id = str(event.get("run_id", ""))
                console_note = f"\n[started] run_id={state.run_id}"
            elif event_type == "progress":
                state.progress_events += 1
                if state.progress_events % 10 == 0:
                    console_note = (
                        f"\n[progress] progress_events={state.progress_events}"
                    )
            elif event_type == "equity":
                raw = payload.get("equity")
                try:
                    equity = float(str(raw))
                except Exception:
                    equity = 0.0
                if equity > 0:
                    points = state.equity_points or []
                    points.append(equity)
                    if len(points) > 1200:
                        del points[: len(points) - 1200]
                    state.equity_points = points
                    state.peak_equity = max(state.peak_equity, equity)
                    if state.peak_equity > 0:
                        dd = (equity - state.peak_equity) / state.peak_equity
                        state.max_drawdown = min(state.max_drawdown, dd)
                        if dd <= -0.03:
                            msg = f"drawdown alert: {dd:.2%}"
                            alerts = state.alerts or []
                            if not alerts or alerts[-1] != msg:
                                alerts.append(msg)
                            state.alerts = alerts[-8:]
                            console_note = f"\n[alert] {msg}"
                    state.console_equity_ticks += 1
                    if state.console_equity_ticks % 2 == 0:
                        line = sparkline(points, width=32)
                        console_line = (
                            f"\r[live] equity={equity:,.2f} "
                            f"dd={state.max_drawdown:.2%} {line}"
                        )
            elif event_type == "finished":
                state.finished = True
                msg = (
                    f"finished status={payload.get('status')} "
                    f"callback_error_count={payload.get('callback_error_count', '0')}"
                )
                alerts = state.alerts or []
                alerts.append(msg)
                state.alerts = alerts[-8:]
                console_note = f"\n[finished] {msg}"
        if console_line:
            print(console_line, end="", flush=True)
        if console_note:
            print(console_note, flush=True)
        if sleep_ms > 0:
            time.sleep(float(sleep_ms) / 1000.0)

    result = aq.run_backtest(
        data=build_data(),
        strategy=LiveWebStrategy,
        symbols="LIVE_WEB",
        show_progress=False,
        initial_cash=500000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        on_event=on_event,
        stream_progress_interval=3,
        stream_equity_interval=1,
        stream_batch_size=16,
        stream_max_buffer=128,
        stream_error_mode="continue",
    )
    with lock:
        state.total_return = float(result.metrics.total_return)
        state.finished = True
    done_event.set()


def main() -> None:
    """Start live web dashboard and run streaming backtest in parallel."""
    parser = argparse.ArgumentParser(description="Live streaming web demo.")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--sleep-ms", type=int, default=20)
    parser.add_argument("--keep-seconds", type=int, default=30)
    args = parser.parse_args()

    state = LiveState()
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
    print(f"live_web_url={url}")
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
        total_return = state.total_return
        max_dd = state.max_drawdown
        progress_events = state.progress_events
    print(f"progress_events={progress_events}")
    print(f"total_return={0.0 if total_return is None else total_return:.6f}")
    print(f"max_drawdown_live_web={max_dd:.2%}")
    print("done_streaming_live_web")


if __name__ == "__main__":
    main()
