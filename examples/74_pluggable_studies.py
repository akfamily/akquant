#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant pluggable studies demo (TradingView "indicator script" style).

A *study* declares indicators and never trades - the counterpart of a
TradingView indicator script next to a strategy script. Plug any number of them
into a run with ``studies=[...]``; each one becomes its own slot, so its points
carry their own ``owner_strategy_id`` and never mix with the strategy's.

Shown here:
1. Two studies alongside one trading strategy in a single backtest.
2. ``indicator_df(owner=...)`` to pull one study's points.
3. The hard guarantee: a study that calls ``self.buy`` raises
   ``StudyCannotTradeError`` instead of silently placing an order.
4. The same ``studies=`` argument works for ``run_live`` (not run here).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy, Study, StudyCannotTradeError, run_backtest

SYMBOL = "STUDY_DEMO"


def make_demo_data(periods: int = 120) -> pd.DataFrame:
    """Build a synthetic daily series with a couple of swings."""
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    rows = []
    for i, ts in enumerate(timestamps):
        close = 100.0 + 0.1 * i + 5.0 * math.sin(i / 10.0)
        rows.append(
            {
                "timestamp": ts,
                "symbol": SYMBOL,
                "open": close - 0.3,
                "high": close + 0.6,
                "low": close - 0.7,
                "close": close,
                "volume": 1000.0 + float(i % 13) * 30.0,
            }
        )
    return pd.DataFrame(rows)


class RsiStudy(Study):
    """Only draws RSI in sub-pane 1. Never trades."""

    def on_start(self) -> None:
        """Declare once; the framework updates and reports it."""
        self.rsi = self.I(
            aq.RSI(14),
            name="rsi",
            pane=1,
            reference_lines=[
                {"value": 70.0, "label": "OB"},
                {"value": 30.0, "label": "OS"},
            ],
        )


class MacdStudy(Study):
    """MACD split into three lines in sub-pane 2; explicit study_id."""

    study_id = "macd_view"

    def on_start(self) -> None:
        """Multi-output indicator -> three independent lines."""
        self.macd = self.I(
            aq.MACD(12, 26, 9), name="macd", pane=2, outputs=("dif", "dea", "hist")
        )


class TrendStrategy(Strategy):
    """The one unit that actually trades: EMA crossover."""

    def on_start(self) -> None:
        """Overlay both EMAs on the main pane."""
        self.fast = self.I(aq.EMA(10), name="ema_fast", pane=0, color="#e91e63")
        self.slow = self.I(aq.EMA(30), name="ema_slow", pane=0, color="#3f51b5")

    def on_bar(self, bar: Bar) -> None:
        """Enter on golden cross, exit on death cross."""
        f0, f1, s0, s1 = self.fast[0], self.fast[1], self.slow[0], self.slow[1]
        if None in (f0, f1, s0, s1):
            return
        position = self.get_position(bar.symbol)
        if f1 <= s1 and f0 > s0 and position == 0:
            self.buy(bar.symbol, 100)
        elif f1 >= s1 and f0 < s0 and position > 0:
            self.sell(bar.symbol, position)


class SneakyStudy(Study):
    """A study that tries to trade - used to show the guard."""

    def on_bar(self, bar: Bar) -> None:
        """Raise on the first bar - the guard fires inside the engine loop."""
        self.buy(bar.symbol, 100)


def main() -> None:
    """Run the strategy with two studies attached, then show the guard."""
    data = make_demo_data()
    result = run_backtest(
        strategy=TrendStrategy,
        data=data,
        symbols=[SYMBOL],
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
        strategy_id="trend",
        studies=[RsiStudy, MacdStudy],
    )

    frame = result.indicator_df()
    owners = sorted(set(frame["owner_strategy_id"]))
    print(f"indicator_owners={owners}")
    for owner in owners:
        keys = sorted(set(result.indicator_df(owner=owner)["indicator_key"]))
        print(f"  {owner}: {keys}")
    print(f"strategy_orders={len(result.orders_df)}")
    print(
        f"study_orders={int((result.orders_df['owner_strategy_id'] != 'trend').sum())}"
    )

    # Studies land on the LWC review chart like any other indicator.
    out_dir = Path(tempfile.mkdtemp(prefix="akq_studies_"))
    html_path = result.viz.review(data, filename=str(out_dir / "review.html"))
    print(f"review_html_written={Path(html_path).exists()}")

    # The guard: a trading call inside a study fails fast, in the engine loop.
    try:
        run_backtest(
            strategy=TrendStrategy,
            data=data,
            symbols=[SYMBOL],
            initial_cash=100000.0,
            show_progress=False,
            timezone="UTC",
            strategy_id="trend",
            studies=[SneakyStudy],
        )
        print("sneaky_study_blocked=False")
    except StudyCannotTradeError as exc:
        print(f"sneaky_study_blocked=True ({type(exc).__name__})")

    print("done_pluggable_studies")


if __name__ == "__main__":
    main()
