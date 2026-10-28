#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streaming backtest demo for continue and fail-fast callback modes."""

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def make_bars() -> list[Bar]:
    """Build deterministic daily bars on business days for the streaming demo.

    ``Bar.timestamp`` 是 UTC 纳秒, 须由带时区的时间换算: 本地 0 点(只有日期)
    的日线由引擎改写到交易日收盘(15:00), 订单在下一交易日开盘集合竞价成交。
    """
    idx = pd.bdate_range(start="2023-01-03", periods=20, tz="Asia/Shanghai")
    bars: list[Bar] = []
    for i, ts in enumerate(idx):
        price = 100.0 + float(i)
        bars.append(
            Bar(
                timestamp=int(ts.value),
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.3,
                volume=1000.0 + i,
                symbol="STREAM",
            )
        )
    return bars


class StreamDemoStrategy(Strategy):
    """Simple strategy that alternates between open and close."""

    def on_bar(self, bar: Bar) -> None:
        """Submit alternating orders on each bar."""
        pos = self.get_position(bar.symbol)
        if pos == 0:
            self.buy(bar.symbol, 1)
        else:
            self.close_position(bar.symbol)


def run_continue_mode(data: list[Bar]) -> None:
    """Run stream demo with callback errors tolerated."""
    events: list[aq.BacktestStreamEvent] = []
    counter = {"n": 0}

    def on_event(event: aq.BacktestStreamEvent) -> None:
        counter["n"] += 1
        if counter["n"] <= 2:
            raise RuntimeError("demo callback error")
        events.append(event)

    aq.run_backtest(
        data=data,
        strategy=StreamDemoStrategy,
        symbols="STREAM",
        show_progress=False,
        initial_cash=100000.0,
        on_event=on_event,
        stream_progress_interval=5,
        stream_equity_interval=5,
        stream_batch_size=8,
        stream_max_buffer=64,
        stream_error_mode="continue",
    )
    finished: aq.BacktestStreamEvent | None = events[-1] if events else None
    payload = finished["payload"] if finished is not None else {}
    callback_error_count = payload.get("callback_error_count", "0")
    print(f"continue_events={len(events)}")
    print(f"continue_callback_error_count={callback_error_count}")


def run_fail_fast_mode(data: list[Bar]) -> None:
    """Run stream demo with callback errors failing immediately."""

    def on_event(_event: aq.BacktestStreamEvent) -> None:
        raise RuntimeError("demo callback error")

    try:
        aq.run_backtest(
            data=data,
            strategy=StreamDemoStrategy,
            symbols="STREAM",
            show_progress=False,
            initial_cash=100000.0,
            on_event=on_event,
            stream_error_mode="fail_fast",
        )
    except RuntimeError as exc:
        print(f"fail_fast_exception={exc}")
        return
    print("fail_fast_exception=none")


def main() -> None:
    """Execute both stream error modes and print summary markers."""
    data = make_bars()
    run_continue_mode(data)
    run_fail_fast_mode(data)
    print("done_streaming_backtest_demo")


if __name__ == "__main__":
    main()
