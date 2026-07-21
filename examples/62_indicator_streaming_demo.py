#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant indicator streaming demo.

This example shows the minimal runtime workflow for indicator stream events:
1. Run a strategy that records custom indicators.
2. Consume `indicator_point` and `indicator_snapshot` from `on_event`.
3. Demonstrate stream sampling controls for indicator events.
"""

import json

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def make_demo_bars() -> list[Bar]:
    """Build deterministic bars for indicator stream consumption."""
    closes = [10.0, 10.4, 10.9, 10.7, 11.2]
    bars: list[Bar] = []
    for i, close in enumerate(closes):
        bars.append(
            Bar(
                timestamp=pd.Timestamp(f"2024-03-0{i + 1} 10:00:00").value,
                open=close - 0.1,
                high=close + 0.2,
                low=close - 0.3,
                close=close,
                volume=1000.0 + float(i) * 10.0,
                symbol="STREAM_IND",
            )
        )
    return bars


class IndicatorStreamingStrategy(Strategy):
    """Record two indicators so stream events can show point and snapshot payloads."""

    def on_bar(self, bar: Bar) -> None:
        """Emit a price echo and a simple range metric on every bar."""
        self.record_indicator(
            name="close_echo",
            value=bar.close,
            display_name="Close Echo",
            pane=0,
            render_type="line",
            precision=2,
        )
        self.record_indicator(
            name="intrabar_range",
            value=bar.high - bar.low,
            display_name="Intrabar Range",
            pane=1,
            render_type="bar",
            precision=2,
        )


def run_full_stream(data: list[Bar]) -> list[aq.BacktestStreamEvent]:
    """Collect all stream events without indicator sampling."""
    events: list[aq.BacktestStreamEvent] = []
    aq.run_backtest(
        data=data,
        strategy=IndicatorStreamingStrategy,
        symbols="STREAM_IND",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
        on_event=events.append,
        stream_progress_interval=16,
        stream_equity_interval=16,
        stream_batch_size=1,
        stream_max_buffer=128,
    )
    return events


def run_sampled_stream(data: list[Bar]) -> list[aq.BacktestStreamEvent]:
    """Collect stream events with indicator sampling enabled."""
    events: list[aq.BacktestStreamEvent] = []
    aq.run_backtest(
        data=data,
        strategy=IndicatorStreamingStrategy,
        symbols="STREAM_IND",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
        on_event=events.append,
        indicator_stream_point_interval=2,
        indicator_stream_snapshot_interval=2,
        stream_progress_interval=16,
        stream_equity_interval=16,
        stream_batch_size=1,
        stream_max_buffer=128,
    )
    return events


def main() -> None:
    """Run full and sampled indicator streams, then print compact summaries."""
    data = make_demo_bars()
    full_events = run_full_stream(data)
    sampled_events = run_sampled_stream(data)

    full_point_events = [
        event for event in full_events if event["event_type"] == "indicator_point"
    ]
    full_snapshot_events = [
        event for event in full_events if event["event_type"] == "indicator_snapshot"
    ]
    sampled_point_events = [
        event for event in sampled_events if event["event_type"] == "indicator_point"
    ]
    sampled_snapshot_events = [
        event for event in sampled_events if event["event_type"] == "indicator_snapshot"
    ]

    first_point_payload = full_point_events[0]["payload"] if full_point_events else {}
    first_snapshot_payload = (
        full_snapshot_events[0]["payload"] if full_snapshot_events else {}
    )
    first_snapshot_items = json.loads(first_snapshot_payload.get("items_json", "[]"))

    full_seq_values = [int(event["seq"]) for event in full_events]
    print(f"full_indicator_point_events={len(full_point_events)}")
    print(f"full_indicator_snapshot_events={len(full_snapshot_events)}")
    print(f"sampled_indicator_point_events={len(sampled_point_events)}")
    print(f"sampled_indicator_snapshot_events={len(sampled_snapshot_events)}")
    print(f"stream_seq_monotonic={full_seq_values == sorted(full_seq_values)}")
    print(
        "first_indicator_point="
        f"{first_point_payload.get('indicator_key')}:{first_point_payload.get('value')}"
    )
    print(
        "first_indicator_snapshot_count="
        f"{first_snapshot_payload.get('indicator_count', '0')}"
    )
    print(f"first_indicator_snapshot_items={len(first_snapshot_items)}")
    print("done_indicator_streaming_demo")


if __name__ == "__main__":
    main()
