#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant tick-driven intrabar indicators (base period, Pine realtime-bar style).

``examples/75`` shows intrabar values for a *window* indicator (5-minute SMA
refreshed on every 1-minute bar). This example goes one level down: a
*base-period* indicator refreshed on every **tick** while its 1-minute bar is
still forming.

With ``intrabar=True`` on a base-period indicator:

- ticks no longer ``update()`` the indicator; they only compute a provisional
  value on a copy, fed with the bar that is forming from those ticks (real
  running open/high/low/close - so ATR-style H/L indicators work too);
- the bar close is the only thing that commits state (``confirmed=True``);
- ``ind[0]`` is the provisional value, ``ind[1]`` the last confirmed bar;
- streamed provisional points carry the timestamp the forming bar will close
  with (``(ts // interval + 1) * interval - 1ns``, the aggregator's own
  end-of-interval stamp), so a chart overwrites them in place.

Default ``intrabar=False`` keeps the legacy behaviour where ticks drive base
indicators directly (pure-tick strategies rely on that).
"""

from __future__ import annotations

from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy, run_backtest
from akquant.akquant import Tick

SYMBOL = "TICK_DEMO"
_BASE_NS = int(pd.Timestamp("2024-03-01 09:31:00", tz="Asia/Shanghai").value)
_STEP_NS = 20_000_000_000  # one tick every 20 seconds -> three ticks per 1-minute bar


def make_ticks(count: int = 30) -> list[Tick]:
    """Ticks every 20 seconds with a gentle drift and wiggle."""
    ticks: list[Tick] = []
    for i in range(count):
        price = 20.0 + 0.05 * i + (0.08 if i % 4 == 0 else -0.03)
        ticks.append(
            Tick(
                timestamp=_BASE_NS + i * _STEP_NS,
                price=price,
                volume=100.0,
                symbol=SYMBOL,
            )
        )
    return ticks


class TickIntrabarStrategy(Strategy):
    """SMA(3) on 1-minute bars, refreshed on every tick; ATR(3) on real H/L."""

    def on_start(self) -> None:
        """No ``freq=``: these are base-period indicators."""
        self.sma3 = self.I(aq.SMA(3), name="sma3", intrabar=True, pane=0)
        self.atr3 = self.I(
            aq.ATR(3), name="atr3", input_mode="hlc", intrabar=True, pane=1
        )

    def on_tick(self, tick: Tick) -> None:
        """Provisional values while the 1-minute bar is forming."""
        state = "confirmed" if self.sma3.confirmed else "provisional"
        print(
            f"[tick] {self.format_time(tick.timestamp)} px={tick.price:.2f} "
            f"sma3[0]={_fmt(self.sma3[0])} ({state}) atr3[0]={_fmt(self.atr3[0])}"
        )

    def on_bar(self, bar: Bar) -> None:
        """Bar closed: state committed, [0] confirmed."""
        print(
            f"[1min] {self.format_time(bar.timestamp)} close={bar.close:.2f} CLOSED "
            f"sma3={_fmt(self.sma3[0])} sma3[1]={_fmt(self.sma3[1])} "
            f"atr3={_fmt(self.atr3[0])}"
        )


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main() -> None:
    """Run the tick backtest and summarize the stream."""
    points: list[dict[str, Any]] = []

    def on_event(event: aq.BacktestStreamEvent) -> None:
        message = aq.to_indicator_message(event)
        if message is not None and message["type"] == "point":
            points.append(message["indicator"])

    result = run_backtest(
        strategy=TickIntrabarStrategy,
        data=make_ticks(),
        freq="1min",  # aggregates ticks into 1-minute bars; also sets self.freq
        symbols=[SYMBOL],
        initial_cash=100000.0,
        show_progress=False,
        on_event=on_event,
        stream_batch_size=1,
    )

    provisional = [p for p in points if not p["confirmed"]]
    confirmed_ts = {p["timestamp"] for p in points if p["confirmed"]}
    covered = [p for p in provisional if p["timestamp"] <= max(confirmed_ts)]
    print(f"stream_provisional_points={len(provisional)}")
    print(f"stream_confirmed_points={len(confirmed_ts)}")
    matched = all(p["timestamp"] in confirmed_ts for p in covered)
    print(f"provisional_labels_match_bar_close={matched}")
    print(f"dataframe_rows_confirmed_only={len(result.indicator_df(name='sma3'))}")
    print("done_tick_intrabar_indicators")


if __name__ == "__main__":
    main()
