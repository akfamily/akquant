import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def build_data(n: int = 360) -> pd.DataFrame:
    """Build synthetic daily bars with trend and oscillation."""
    ts = pd.date_range("2022-01-01", periods=n, freq="D")
    close = [
        100.0 + 0.04 * i + 2.8 * math.sin(i / 7.5) + 1.6 * math.sin(i / 2.7)
        for i in range(n)
    ]
    return pd.DataFrame(
        {
            "date": ts,
            "open": [c * 0.998 for c in close],
            "high": [c * 1.005 for c in close],
            "low": [c * 0.995 for c in close],
            "close": close,
            "volume": [1000.0 + float(i % 40) * 25.0 for i in range(n)],
            "symbol": "STREAM_ALERT",
        }
    )


class MomentumSwitchStrategy(Strategy):
    """Simple momentum/revert switch strategy for event-rich stream output."""

    def __init__(self) -> None:
        """Initialize warmup length for moving average windows."""
        super().__init__()
        self.warmup_period = 20

    def on_bar(self, bar: Bar) -> None:
        """Place or close positions based on short/long moving average relation."""
        closes = self.get_history(count=20, symbol=bar.symbol, field="close")
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
class AlertState:
    """Holds state for drawdown alerts and compact event logs."""

    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    drawdown_alert_triggered: bool = False
    progress_counter: int = 0


def run_stream_case() -> Path:
    """Run one stream-enabled backtest and persist compact event snapshots."""
    state = AlertState()
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "stream_alert_events.csv"
    rows: list[dict[str, Any]] = []

    def on_event(event: aq.BacktestStreamEvent) -> None:
        event_type = str(event.get("event_type", "unknown"))
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {"raw_payload": payload}
        seq = int(event.get("seq", 0))
        rows.append(
            {
                "seq": seq,
                "event_type": event_type,
                "ts": int(event.get("ts", 0)),
                "run_id": str(event.get("run_id", "")),
                "value": "",
            }
        )

        if event_type == "started":
            print(f"[started] run_id={event.get('run_id')}")

        elif event_type == "progress":
            state.progress_counter += 1
            if state.progress_counter % 8 == 0:
                print(f"[progress] seq={seq} progress_events={state.progress_counter}")

        elif event_type == "equity":
            equity_raw = payload.get("equity")
            try:
                equity = float(str(equity_raw))
            except Exception:
                return
            if equity > state.peak_equity:
                state.peak_equity = equity
            if state.peak_equity <= 0:
                return
            dd = (equity - state.peak_equity) / state.peak_equity
            if dd < state.max_drawdown:
                state.max_drawdown = dd
            if dd <= -0.03 and not state.drawdown_alert_triggered:
                state.drawdown_alert_triggered = True
                print(f"[alert] drawdown={dd:.2%} seq={seq}")
                rows.append(
                    {
                        "seq": seq,
                        "event_type": "alert_drawdown",
                        "ts": int(event.get("ts", 0)),
                        "run_id": str(event.get("run_id", "")),
                        "value": f"{dd:.6f}",
                    }
                )

        elif event_type == "finished":
            print(
                "[finished] "
                f"status={payload.get('status')} "
                f"callback_error_count={payload.get('callback_error_count', '0')}"
            )

    result = aq.run_backtest(
        data=build_data(),
        strategy=MomentumSwitchStrategy,
        symbols="STREAM_ALERT",
        show_progress=False,
        initial_cash=500000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        on_event=on_event,
        stream_progress_interval=4,
        stream_equity_interval=4,
        stream_batch_size=16,
        stream_max_buffer=128,
        stream_error_mode="continue",
    )

    with out_file.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["seq", "event_type", "ts", "run_id", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"max_drawdown_seen={state.max_drawdown:.2%}")
    print(f"total_return={result.metrics.total_return:.6f}")
    print(f"saved_events={len(rows)}")
    print(f"event_csv={out_file}")
    print("done_streaming_alerts_and_persist")
    return out_file


def main() -> None:
    """Entry point for running the stream alert and persistence demo."""
    run_stream_case()


if __name__ == "__main__":
    main()
