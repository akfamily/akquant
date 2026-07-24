import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Type

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def build_demo_data(n: int = 320) -> pd.DataFrame:
    """Build deterministic synthetic bars for stream-monitor demonstrations."""
    ts = pd.date_range("2023-01-01", periods=n, freq="D")
    base = 100.0
    close = [
        base + 0.05 * i + 3.0 * math.sin(i / 8.0) + 1.2 * math.sin(i / 3.0)
        for i in range(n)
    ]
    df = pd.DataFrame(
        {
            "date": ts,
            "open": [c * 0.998 for c in close],
            "high": [c * 1.004 for c in close],
            "low": [c * 0.996 for c in close],
            "close": close,
            "volume": [1000.0 + float(i % 50) * 20.0 for i in range(n)],
            "symbol": "STREAM_MONITOR",
        }
    )
    return df


class _MovingAverageStrategy(Strategy):
    def __init__(self, short_window: int, long_window: int) -> None:
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.warmup_period = long_window

    def on_bar(self, bar: Bar) -> None:
        closes = self.get_history(
            count=self.long_window, symbol=bar.symbol, field="close"
        )
        if len(closes) < self.long_window:
            return
        short_ma = float(sum(closes[-self.short_window :])) / float(self.short_window)
        long_ma = float(sum(closes[-self.long_window :])) / float(self.long_window)
        pos = self.get_position(bar.symbol)
        if short_ma > long_ma and pos == 0:
            self.order_target_percent(symbol=bar.symbol, target_percent=0.9)
        elif short_ma < long_ma and pos > 0:
            self.close_position(symbol=bar.symbol)


def strategy_factory(short_window: int, long_window: int) -> Type[Strategy]:
    """Create a strategy class bound to one parameter configuration."""

    class StrategyImpl(_MovingAverageStrategy):
        def __init__(self) -> None:
            super().__init__(short_window=short_window, long_window=long_window)

    return StrategyImpl


@dataclass
class StreamMonitor:
    """Mutable stream state captured by callback."""

    event_counts: Counter[str]
    seq_last: int = 0
    progress_seen: int = 0
    finished_payload: dict[str, Any] | None = None


def run_one_config(short_window: int, long_window: int) -> Any:
    """Run one parameter configuration with realtime stream monitoring output."""
    monitor = StreamMonitor(event_counts=Counter())

    def on_event(event: aq.BacktestStreamEvent) -> None:
        event_type = str(event.get("event_type", "unknown"))
        monitor.event_counts[event_type] += 1
        raw_seq = event.get("seq", monitor.seq_last)
        try:
            monitor.seq_last = int(raw_seq)
        except Exception:
            pass
        if event_type == "started":
            print(
                f"[started] run_id={event.get('run_id')} "
                f"cfg={short_window}/{long_window}"
            )
        elif event_type == "progress":
            monitor.progress_seen += 1
            if monitor.progress_seen % 6 == 0:
                print(
                    f"[progress] cfg={short_window}/{long_window} "
                    f"seq={monitor.seq_last} progress_events={monitor.progress_seen}"
                )
        elif (
            event_type in {"order", "trade"}
            and monitor.event_counts[event_type] % 8 == 0
        ):
            print(
                f"[monitor] cfg={short_window}/{long_window} "
                f"{event_type}_events={monitor.event_counts[event_type]} "
                f"seq={monitor.seq_last}"
            )
        elif event_type == "finished":
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                monitor.finished_payload = payload
            else:
                monitor.finished_payload = {"payload": payload}
            print(
                f"[finished] cfg={short_window}/{long_window} "
                f"status={monitor.finished_payload.get('status')}"
            )

    result = aq.run_backtest(
        data=build_demo_data(),
        strategy=strategy_factory(short_window, long_window),
        symbols="STREAM_MONITOR",
        show_progress=False,
        initial_cash=500000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        on_event=on_event,
        stream_progress_interval=5,
        stream_equity_interval=10,
        stream_batch_size=16,
        stream_max_buffer=128,
        stream_error_mode="continue",
    )

    ordered = sorted(monitor.event_counts.items(), key=lambda item: item[0])
    print(f"stream_event_counts_cfg_{short_window}_{long_window}={ordered}")
    print(
        f"stream_finished_callback_error_count_cfg_{short_window}_{long_window}="
        f"{(monitor.finished_payload or {}).get('callback_error_count', '0')}"
    )
    print(
        f"stream_result_total_return_cfg_{short_window}_{long_window}="
        f"{result.metrics.total_return:.6f}"
    )
    return result


def main() -> None:
    """Run multiple configs and print end marker for smoke checks."""
    cfgs = [(5, 20), (8, 30)]
    for short_window, long_window in cfgs:
        run_one_config(short_window=short_window, long_window=long_window)
    print("done_streaming_monitoring_console")


if __name__ == "__main__":
    main()
