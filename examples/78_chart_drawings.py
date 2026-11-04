#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant strategy drawings: ``self.draw`` objects and ``self.stats`` tables.

A dual moving-average strategy draws on the price chart, Pine-style:

- ``draw.label``: "golden cross" / "death cross" at every crossover
  (id ``x-<timestamp>``, so each signal keeps its own label);
- ``draw.box``: the high/low range of the last 20 bars (fixed id, overwritten
  on every bar);
- ``draw.line(extend="right")``: trend line from the latest golden cross to the
  current bar;
- ``draw.fib``: Fibonacci retracement between the high and low of the last 60 bars;
- ``draw.polyline``: the last 5 closes joined together;
- ``draw.table``: a global performance table fed by ``self.stats.as_rows()``
  and a per-symbol table with coloured cells ("trend: up / down").

Drawing objects are not time series like ``self.I()`` indicators: they have an id
and can be overwritten or deleted, and only the final state survives. Calling
``draw.*`` with unchanged content emits no event, so per-bar calls are cheap.

运行方式::

    python examples/78_chart_drawings.py

输出：控制台打印胜率与绘图对象数量；复盘页面写到临时目录下的
``chart_drawings_review.html``（路径会打印出来），用浏览器打开即可。
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from akquant import Bar, Strategy, run_backtest

SYMBOL = "DRAW_DEMO"
RED = "#e03131"
GREEN = "#2f9e44"
BLUE = "#1971c2"


def make_demo_data(periods: int = 200) -> pd.DataFrame:
    """Build a synthetic daily series with several swings (fixed, offline)."""
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    rows = []
    for i, ts in enumerate(timestamps):
        close = 100.0 + 0.05 * i + 8.0 * math.sin(i / 14.0) + 2.0 * math.sin(i / 3.5)
        rows.append(
            {
                "timestamp": ts,
                "symbol": SYMBOL,
                "open": close - 0.5 * math.cos(i / 2.0),
                "high": close + 1.2,
                "low": close - 1.3,
                "close": close,
                "volume": 10000.0 + float(i % 9) * 300.0,
            }
        )
    return pd.DataFrame(rows)


class DrawingMA(Strategy):
    """Dual moving average that annotates itself on the chart."""

    fast = 5
    slow = 20

    def __init__(self) -> None:
        """Initialise the signal state."""
        self.last_cross_ts: Optional[int] = None
        self.last_cross_price: float = 0.0
        self.times: list[int] = []  # bar.timestamp of every bar seen so far

    def on_start(self) -> None:
        """Keep enough history for the 60-bar fib and the slow average."""
        self.set_history_depth(80)

    def on_bar(self, bar: Bar) -> None:
        """Trade the crossovers and draw the story on the chart."""
        symbol = bar.symbol
        self.times.append(int(bar.timestamp))
        if len(self.times) < 60:  # wait until the 60-bar fib window is full
            return
        closes = self.get_history(self.slow + 1, symbol, "close")
        highs = self.get_history(60, symbol, "high")
        lows = self.get_history(60, symbol, "low")

        fast_now = float(closes[-self.fast :].mean())
        slow_now = float(closes[-self.slow :].mean())
        fast_prev = float(closes[-self.fast - 1 : -1].mean())
        slow_prev = float(closes[-self.slow - 1 : -1].mean())

        pos = self.get_position(symbol)
        if fast_prev <= slow_prev and fast_now > slow_now:
            # id carries the bar timestamp: every signal keeps its own label
            self.draw.label(
                f"x-{bar.timestamp}",
                (bar.timestamp, bar.low),
                "金叉",
                style="up",
                color=RED,
            )
            self.last_cross_ts = int(bar.timestamp)
            self.last_cross_price = float(bar.close)
            if pos == 0:
                self.buy(symbol, 100)
        elif fast_prev >= slow_prev and fast_now < slow_now:
            self.draw.label(
                f"x-{bar.timestamp}",
                (bar.timestamp, bar.high),
                "死叉",
                style="down",
                color=GREEN,
            )
            if pos > 0:
                self.close_position(symbol)

        # Range of the last 20 bars: fixed id, overwritten on every bar.
        recent_hi = float(highs[-20:].max())
        recent_lo = float(lows[-20:].min())
        first_ts = self.times[-20]
        self.draw.box(
            "range20",
            (first_ts, recent_hi),
            (bar.timestamp, recent_lo),
            border_color=BLUE,
            fill_color="#1971c222",
            text="近20根区间",
        )

        # Trend line from the latest golden cross to the current bar.
        if self.last_cross_ts is not None and self.last_cross_ts != bar.timestamp:
            self.draw.line(
                "trend",
                (self.last_cross_ts, self.last_cross_price),
                (bar.timestamp, float(bar.close)),
                extend="right",
                color=RED,
                width=2,
                style="dashed",
            )

        # Fibonacci retracement between the 60-bar high and low.
        self.draw.fib(
            "fib60",
            (self.times[-60], float(highs.max())),
            (bar.timestamp, float(lows.min())),
            color="#868e96",
        )

        # The last 5 closes joined together.
        self.draw.polyline(
            "last5",
            [(t, float(c)) for t, c in zip(self.times[-5:], closes[-5:])],
            color="#f08c00",
            width=2,
        )

        # Global performance table (symbol="*": shown on every symbol's chart).
        self.draw.table(
            "perf",
            self.stats.as_rows(),
            header=["指标", "数值"],
            symbol="*",
            position="top_right",
        )
        # Per-symbol table with coloured cells.
        up = fast_now > slow_now
        self.draw.table(
            "trend_tbl",
            [
                [
                    "趋势",
                    {
                        "value": "多头" if up else "空头",
                        "bg_color": "#ffc9c9" if up else "#b2f2bb",
                    },
                ]
            ],
            position="bottom_left",
        )


def main() -> None:
    """Run the backtest, print a few numbers and write the review page."""
    data = make_demo_data()
    result = run_backtest(
        strategy=DrawingMA,
        data=data,
        symbols=[SYMBOL],
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
    )

    win_rate = float(result.metrics.win_rate)
    print(f"win_rate={win_rate:.2f}%")
    print(f"drawings={len(result.drawings)}")
    kinds = sorted({d["kind"] for d in result.drawings})
    print(f"drawing_kinds={kinds}")

    out_dir = Path(tempfile.mkdtemp(prefix="akq_chart_drawings_"))
    html_path = result.viz.review(
        data,
        filename=str(out_dir / "chart_drawings_review.html"),
        include_drawings=True,
    )
    print(f"review_html={html_path}")
    print("done_chart_drawings")


if __name__ == "__main__":
    main()
