#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant indicator scripts demo (``aq.Indicator``, the Pine ``indicator()`` counterpart).

An *indicator script* declares indicators and never trades - the counterpart of a
TradingView ``indicator()`` script next to a ``strategy()`` script. The class line
carries the script metadata: ``class X(aq.Indicator, title="RSI", overlay=False,
id="rsi_indicator")``; ``overlay`` defaults to ``False`` (sub-pane), ``title`` is
display-only (defaults to the class name) and ``id`` defaults to the snake_case
class name. Mount any number of them with ``indicators=[...]``; each one becomes
its own slot, so its points carry their own ``owner_strategy_id`` and never mix
with the strategy's.

Shown here:
1. Two indicator scripts alongside one trading strategy in a single backtest.
2. ``indicator_df(owner=...)`` to pull one script's points.
3. The hard guarantee: an indicator script that calls ``self.buy`` raises
   ``IndicatorCannotTradeError`` instead of silently placing an order.
4. ``aq.inspect_strategy(__file__)``: a static (ast-only) look at this very file,
   telling indicators from strategies and main pane from sub-panes without
   running any code.
5. The same ``indicators=`` argument works for ``run_live`` (not run here).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import akquant as aq
import pandas as pd
from akquant import (
    Bar,
    Indicator,
    IndicatorCannotTradeError,
    Strategy,
    run_backtest,
)

SYMBOL = "INDICATOR_DEMO"


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


class RsiIndicator(Indicator, title="RSI"):
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


class MacdIndicator(Indicator, title="MACD", id="macd_view"):
    """MACD split into three lines in sub-pane 2; explicit id."""

    def on_start(self) -> None:
        """Multi-output indicator -> three independent lines."""
        self.macd = self.I(
            aq.MACD(12, 26, 9), name="macd", pane=2, outputs=("dif", "dea", "hist")
        )


class TrendStrategy(Strategy):
    """The one unit that actually trades: EMA crossover."""

    def on_start(self) -> None:
        """Overlay both EMAs on the main pane."""
        self.fast = self.I(
            aq.EMA(10), name="ema_fast", pane=0, color="#e91e63", plot=True
        )
        self.slow = self.I(
            aq.EMA(30), name="ema_slow", pane=0, color="#3f51b5", plot=True
        )

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


class SneakyIndicator(Indicator):
    """An indicator script that tries to trade - used to show the guard."""

    def on_bar(self, bar: Bar) -> None:
        """Raise on the first bar - the guard fires inside the engine loop."""
        self.buy(bar.symbol, 100)


def main() -> None:
    """Run the strategy with two indicator scripts, show the guard, inspect."""
    data = make_demo_data()
    result = run_backtest(
        strategy=TrendStrategy,
        data=data,
        symbols=[SYMBOL],
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
        strategy_id="trend",
        indicators=[RsiIndicator, MacdIndicator],
    )

    frame = result.indicator_df()
    owners = sorted(set(frame["owner_strategy_id"]))
    print(f"indicator_owners={owners}")
    for owner in owners:
        keys = sorted(set(result.indicator_df(owner=owner)["indicator_key"]))
        print(f"  {owner}: {keys}")
    print(f"strategy_orders={len(result.orders_df)}")
    non_trend = result.orders_df["owner_strategy_id"] != "trend"
    print(f"indicator_orders={int(non_trend.sum())}")

    # Indicator scripts land on the LWC review chart like any other indicator.
    out_dir = Path(tempfile.mkdtemp(prefix="akq_indicator_scripts_"))
    html_path = result.viz.review(data, filename=str(out_dir / "review.html"))
    print(f"review_html_written={Path(html_path).exists()}")

    # The guard: a trading call inside an indicator script fails fast.
    try:
        run_backtest(
            strategy=TrendStrategy,
            data=data,
            symbols=[SYMBOL],
            initial_cash=100000.0,
            show_progress=False,
            timezone="UTC",
            strategy_id="trend",
            indicators=[SneakyIndicator],
        )
        print("sneaky_indicator_blocked=False")
    except IndicatorCannotTradeError as exc:
        print(f"sneaky_indicator_blocked=True ({type(exc).__name__})")

    # Static inspection: pure ast, this file is parsed but never executed.
    report = aq.inspect_strategy(__file__)
    for cls in report.classes:
        panes = {item.name: item.pane for item in cls.indicators}
        print(
            f"inspect {cls.name}: kind={cls.kind} title={cls.title} "
            f"overlay={cls.overlay} panes={panes}"
        )

    print("done_indicator_scripts")


if __name__ == "__main__":
    main()
