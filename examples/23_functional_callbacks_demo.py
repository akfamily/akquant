#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Function-style callbacks demo."""

from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar


def make_bars() -> list[Bar]:
    """Build deterministic bar data for demo run."""
    idx = pd.date_range(start="2023-01-01", periods=4, freq="D")
    bars: list[Bar] = []
    for i, ts in enumerate(idx):
        price = 100.0 + i
        bars.append(
            Bar(
                timestamp=int(ts.value),
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.5,
                volume=1000.0 + i,
                symbol="TEST",
            )
        )
    return bars


def initialize(ctx: Any) -> None:
    """Initialize function-style strategy state."""
    ctx.events = []
    ctx.timer_registered = False


def on_bar(ctx: Any, bar: Bar) -> None:
    """Handle bar events and submit simple alternating orders."""
    if not ctx.timer_registered:
        ctx.schedule_daily("00:00:00", "daily_probe")
        ctx.timer_registered = True
    ctx.events.append(f"bar:{bar.symbol}")
    pos = ctx.get_position(bar.symbol)
    if pos == 0:
        ctx.buy(bar.symbol, 1)
    else:
        ctx.sell(bar.symbol, 1)


def on_tick(ctx: Any, tick: Any) -> None:
    """Handle tick events."""
    ctx.events.append("tick")


def on_order(ctx: Any, order: Any) -> None:
    """Handle order state updates."""
    ctx.events.append(f"order:{order.id}")


def on_trade(ctx: Any, trade: Any) -> None:
    """Handle trade reports."""
    ctx.events.append(f"trade:{trade.order_id}")


def on_timer(ctx: Any, payload: str) -> None:
    """Handle timer callbacks."""
    ctx.events.append(f"timer:{payload}")


def main() -> None:
    """Run demo and print callback counters."""
    result = aq.run_backtest(
        data=make_bars(),
        strategy=on_bar,
        initialize=initialize,
        on_tick=on_tick,
        on_order=on_order,
        on_trade=on_trade,
        on_timer=on_timer,
        symbols="TEST",
        show_progress=False,
        initial_cash=100000.0,
    )
    strategy = result.strategy
    if strategy is None:
        raise RuntimeError("Strategy should not be None")
    events = getattr(strategy, "events", [])
    print(f"events_total={len(events)}")
    print(f"orders={sum(1 for x in events if x.startswith('order:'))}")
    print(f"trades={sum(1 for x in events if x.startswith('trade:'))}")
    print(f"timers={sum(1 for x in events if x.startswith('timer:'))}")
    print(f"bars={sum(1 for x in events if x.startswith('bar:'))}")
    print(f"ticks={sum(1 for x in events if x == 'tick')}")
    print("done_functional_callbacks_demo")


if __name__ == "__main__":
    main()
