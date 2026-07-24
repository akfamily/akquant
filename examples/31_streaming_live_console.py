import math
from dataclasses import dataclass, field

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def build_data(n: int = 320) -> pd.DataFrame:
    """Build synthetic daily bars for live console rendering."""
    ts = pd.date_range("2022-01-01", periods=n, freq="D")
    close = [
        100.0 + 0.05 * i + 2.5 * math.sin(i / 8.2) + 1.3 * math.sin(i / 2.9)
        for i in range(n)
    ]
    return pd.DataFrame(
        {
            "date": ts,
            "open": [c * 0.998 for c in close],
            "high": [c * 1.004 for c in close],
            "low": [c * 0.996 for c in close],
            "close": close,
            "volume": [1200.0 + float(i % 30) * 18.0 for i in range(n)],
            "symbol": "LIVE_STREAM",
        }
    )


class LiveDemoStrategy(Strategy):
    """Simple MA crossover strategy for stream event demo."""

    def __init__(self) -> None:
        """Initialize warmup setting."""
        super().__init__()
        self.warmup_period = 20

    def on_bar(self, bar: Bar) -> None:
        """Trade by short/long moving average relation."""
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


def sparkline(values: list[float], width: int = 42) -> str:
    """Convert a sequence of values to unicode sparkline."""
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


@dataclass
class LiveState:
    """Mutable stream state for console chart and alerts."""

    equities: list[float] = field(default_factory=list)
    peak: float = 0.0
    max_drawdown: float = 0.0
    progress_events: int = 0
    alert_emitted: bool = False


def run_live_console_demo() -> None:
    """Run backtest and render live chart plus warning messages."""
    state = LiveState()

    def on_event(event: aq.BacktestStreamEvent) -> None:
        event_type = str(event.get("event_type", "unknown"))
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if event_type == "started":
            print(f"[started] run_id={event.get('run_id')}")
            return

        if event_type == "progress":
            state.progress_events += 1
            if state.progress_events % 10 == 0:
                print(f"[progress] progress_events={state.progress_events}")
            return

        if event_type == "equity":
            raw = payload.get("equity")
            try:
                equity = float(str(raw))
            except Exception:
                return
            state.equities.append(equity)
            state.peak = max(state.peak, equity)
            if state.peak > 0:
                dd = (equity - state.peak) / state.peak
                state.max_drawdown = min(state.max_drawdown, dd)
                if dd <= -0.03 and not state.alert_emitted:
                    state.alert_emitted = True
                    print(f"\n[alert] drawdown={dd:.2%}")
            line = sparkline(state.equities, width=42)
            print(
                f"\r[live] equity={equity:,.2f} dd={state.max_drawdown:.2%} {line}",
                end="",
                flush=True,
            )
            return

        if event_type == "finished":
            print()
            print(
                f"[finished] status={payload.get('status')} "
                f"callback_error_count={payload.get('callback_error_count', '0')}"
            )

    result = aq.run_backtest(
        data=build_data(),
        strategy=LiveDemoStrategy,
        symbols="LIVE_STREAM",
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
        stream_equity_interval=2,
        stream_batch_size=16,
        stream_max_buffer=128,
        stream_error_mode="continue",
    )
    print(f"total_return={result.metrics.total_return:.6f}")
    print(f"max_drawdown_live={state.max_drawdown:.2%}")
    print("done_streaming_live_console")


def main() -> None:
    """Entry point for live streaming console demo."""
    run_live_console_demo()


if __name__ == "__main__":
    main()
