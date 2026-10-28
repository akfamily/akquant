#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Function-style pre-open demo."""

from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar


def build_data() -> list[Bar]:
    """Build two daily bars for the pre-open callback demo."""
    rows = [
        ("2023-01-03", 10.0, 10.4),
        ("2023-01-04", 10.8, 11.0),
    ]
    bars: list[Bar] = []
    for dt_str, open_price, close_price in rows:
        # 只有日期(本地 0 点)的日线: 引擎改写到交易日收盘 15:00, 区间覆盖整个交易日
        ts = pd.Timestamp(dt_str, tz="Asia/Shanghai").value
        bars.append(
            Bar(
                timestamp=ts,
                open=open_price,
                high=max(open_price, close_price) + 0.2,
                low=min(open_price, close_price) - 0.2,
                close=close_price,
                volume=1000.0,
                symbol="PREOPEN_FUNC",
            )
        )
    return bars


def initialize(ctx: Any) -> None:
    """Initialize function-style strategy state."""
    ctx.events = []
    ctx.submitted_dates = set()


def on_start(ctx: Any) -> None:
    """Subscribe the demo symbol."""
    ctx.subscribe("PREOPEN_FUNC")


def on_pre_open(ctx: Any, event: dict[str, object]) -> None:
    """Place one default market order before the session open."""
    trading_date = event["trading_date"]
    ctx.events.append(f"pre_open:{trading_date}")
    # expected_open_at 是本回调的触发时刻（首个时段 09:15 开始前 1ns），不是成交时刻
    ctx.log(
        "on_pre_open "
        f"trading_date={trading_date} expected_open_at={event['expected_open_at']}"
    )
    if trading_date in ctx.submitted_dates:
        return
    ctx.submitted_dates.add(trading_date)
    ctx.buy("PREOPEN_FUNC", quantity=1)


def on_bar(ctx: Any, bar: Bar) -> None:
    """Log bar arrival after the pre-open hook."""
    ctx.events.append(f"bar:{ctx.format_time(bar.timestamp)}:{bar.open}")
    ctx.log(
        f"on_bar ts={ctx.format_time(bar.timestamp)} open={bar.open} close={bar.close}"
    )


def on_order(ctx: Any, order: Any) -> None:
    """Log order transitions."""
    ctx.log(
        "on_order "
        f"id={getattr(order, 'id', '<unknown>')} "
        f"status={getattr(order, 'status', '<unknown>')}"
    )


def on_trade(ctx: Any, trade: Any) -> None:
    """Log fills to show the open-price execution."""
    ctx.log(
        "on_trade "
        f"price={getattr(trade, 'price', '<unknown>')} "
        f"qty={getattr(trade, 'quantity', '<unknown>')}"
    )


def main() -> None:
    """Run the functional pre-open callback demo."""
    result = aq.run_backtest(
        data=build_data(),
        strategy=on_bar,
        initialize=initialize,
        on_start=on_start,
        on_order=on_order,
        on_trade=on_trade,
        on_pre_open=on_pre_open,
        symbols="PREOPEN_FUNC",
        initial_cash=10000.0,
        lot_size=1,
        show_progress=False,
    )
    strategy = result.strategy
    if strategy is None:
        raise RuntimeError("Strategy should not be None")
    events = getattr(strategy, "events", [])
    print("\n=== functional pre_open summary ===")
    for event in events:
        print(event)
    print("done_functional_pre_open_demo")


if __name__ == "__main__":
    main()
