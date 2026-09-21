#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant intrabar indicators demo (Pine ``barstate.isrealtime`` / ``isconfirmed``).

A window-period indicator declared with ``intrabar=True`` gets a **provisional**
value on every base bar while its window is still forming, computed on a copy
of the indicator state from the partial window snapshot - the real state is
never touched. When the window closes the real ``update()`` runs and the value
is **confirmed**.

Index semantics follow Pine: while a provisional value exists, ``ind[0]`` is it
and ``ind[1]`` is the last confirmed window; ``ind.confirmed`` tells which case
you are in. Stream events carry ``confirmed=false`` for provisional points and
share the window's timestamp with the confirmed point that follows, so a chart
consumer overwrites in place (see ``examples/73`` for the LWC side).

Runs on ``run_live(broker="replay")`` with the gateway declaring ``freq="1min"``
so the 5-minute windows close on time and carry stable labels.
"""

from __future__ import annotations

from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy, run_live
from akquant.akquant import AssetType, Instrument

SYMBOL = "600000"


def make_bars(count: int = 22) -> list[Bar]:
    """One-minute bars with a rising drift and a small wiggle."""
    idx = pd.date_range(
        "2024-03-01 09:31:00", periods=count, freq="1min", tz="Asia/Shanghai"
    )
    bars: list[Bar] = []
    for i, ts in enumerate(idx):
        close = 10.0 + 0.1 * i + (0.05 if i % 3 == 0 else -0.02)
        bars.append(
            Bar(
                int(ts.value),
                close - 0.02,
                close + 0.04,
                close - 0.05,
                close,
                1000.0,
                SYMBOL,
            )
        )
    return bars


class IntrabarStrategy(Strategy):
    """Declare a 5-minute SMA and watch it breathe between window closes."""

    def __init__(self) -> None:
        """Window subscriptions must be declared in ``__init__``."""
        super().__init__()
        self.subscribe_bars("5min")

    def on_start(self) -> None:
        """``intrabar=True`` needs ``freq=``; plot metadata turns on streaming."""
        self.sma5 = self.I(aq.SMA(2), name="sma5", freq="5min", intrabar=True, pane=0)

    def on_bar(self, bar: Bar) -> None:
        """Every base bar: provisional [0], last confirmed [1]."""
        state = "confirmed" if self.sma5.confirmed else "provisional"
        v0, v1 = self.sma5[0], self.sma5[1]
        print(
            f"[1min] {self.format_time(bar.timestamp)} close={bar.close:.2f} "
            f"sma5[0]={_fmt(v0)} ({state})  sma5[1]={_fmt(v1)}"
        )

    def on_window_bar(self, bar: Bar) -> None:
        """Window closed: [0] is now the confirmed value."""
        when = self.format_time(bar.timestamp)
        print(f"[5min] {when} CLOSED  sma5={_fmt(self.sma5[0])}")


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main() -> None:
    """Run the replay session and summarize the stream."""
    points: list[dict[str, Any]] = []

    def on_event(event: aq.BacktestStreamEvent) -> None:
        message = aq.to_indicator_message(event)
        if message is not None and message["type"] == "point":
            points.append(message["indicator"])

    run_live(
        strategy_cls=IntrabarStrategy,
        instruments=[
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
        ],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": make_bars(), "freq": "1min"},
        cash=1_000_000,
        show_progress=False,
        on_event=on_event,
        duration="60s",
    )

    provisional = [p for p in points if not p["confirmed"]]
    confirmed = [p for p in points if p["confirmed"]]
    confirmed_ts = {p["timestamp"] for p in confirmed}
    print(f"stream_provisional_points={len(provisional)}")
    print(f"stream_confirmed_points={len(confirmed)}")
    all_covered = all(p["timestamp"] in confirmed_ts for p in provisional)
    print(f"every_provisional_has_confirmed_successor={all_covered}")
    print("done_intrabar_indicators")


if __name__ == "__main__":
    main()
