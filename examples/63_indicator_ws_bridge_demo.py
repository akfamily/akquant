#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant indicator websocket bridge demo.

This example shows one minimal enterprise-facing workflow:
1. Run a strategy that records custom indicators.
2. Receive indicator events from `on_event`.
3. Convert those events into frontend-friendly websocket messages.
"""

import json

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def make_demo_bars() -> list[Bar]:
    """Build deterministic bars for websocket bridge demo."""
    closes = [10.0, 10.3, 10.8, 10.6]
    bars: list[Bar] = []
    for i, close in enumerate(closes):
        bars.append(
            Bar(
                timestamp=pd.Timestamp(f"2024-04-0{i + 1} 10:00:00").value,
                open=close - 0.1,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=1000.0 + float(i) * 20.0,
                symbol="WS_IND",
            )
        )
    return bars


class IndicatorBridgeStrategy(Strategy):
    """Record indicators that are later bridged to websocket-style messages."""

    def on_bar(self, bar: Bar) -> None:
        """Emit one line-series point and one bar-style signal."""
        self.record_indicator(
            name="close_echo",
            value=bar.close,
            display_name="Close Echo",
            pane=0,
            render_type="line",
            meta={"source": "close"},
        )
        self.record_indicator(
            name="daily_range",
            value=bar.high - bar.low,
            display_name="Daily Range",
            pane=1,
            render_type="bar",
            meta={"source": ["high", "low"]},
        )


def main() -> None:
    """Run the bridge demo and print example websocket-ready messages."""
    events: list[aq.BacktestStreamEvent] = []
    aq.run_backtest(
        data=make_demo_bars(),
        strategy=IndicatorBridgeStrategy,
        symbols="WS_IND",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
        on_event=events.append,
        stream_batch_size=1,
        stream_max_buffer=128,
    )

    messages = aq.to_indicator_messages(events)
    point_messages = [message for message in messages if message["type"] == "point"]
    snapshot_messages = [
        message for message in messages if message["type"] == "snapshot"
    ]

    print(f"bridge_message_count={len(messages)}")
    print(f"bridge_point_message_count={len(point_messages)}")
    print(f"bridge_snapshot_message_count={len(snapshot_messages)}")
    if point_messages:
        print(
            "bridge_first_point_message="
            + json.dumps(point_messages[0], ensure_ascii=True)
        )
    if snapshot_messages:
        print(
            "bridge_first_snapshot_message="
            + json.dumps(snapshot_messages[0], ensure_ascii=True)
        )
    print("done_indicator_ws_bridge_demo")


if __name__ == "__main__":
    main()
