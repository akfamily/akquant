#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant candle-shaped indicators: Heikin-Ashi with ``render_type="candle"``.

Pine has ``plotcandle``; AKQuant's counterpart is a four-output indicator
declared with ``render_type="candle"``::

    self.I(aq.HeikinAshi(), name="ha", input_mode="ohlc",
           outputs=("open", "high", "low", "close"), render_type="candle")

The four components are still reported as ordinary single-value points
(``ha.open`` / ``ha.high`` / ``ha.low`` / ``ha.close``); their definitions carry
``group="ha"`` and ``role="open"|...``, so every exporter can put them back
together:

- LWC review (``result.viz.review``): one Candlestick series per group, in a
  single colour (solid up / hollow down) so it never looks like the price candles;
- d3kline (``akquant.chart.to_d3kline_options``): no candle series type there,
  so only the ``close`` component is drawn as a line named after the group;
- raw panes / the live stream: four components with ``group``/``role``.

The class line ``replaces_price=True`` (requires ``overlay=True``) is only a
hint for frontends that the indicator replaces the price candles; hiding the
original candles is the frontend's job. ``aq.inspect_strategy`` reports it
statically.

运行方式::

    python examples/77_heikin_ashi_indicator.py
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import akquant as aq
import pandas as pd
from akquant import Bar, Indicator, Strategy, run_backtest
from akquant.chart import group_points_by_identity, to_d3kline_options

SYMBOL = "HA_DEMO"
OHLC = ("open", "high", "low", "close")


def make_demo_data(periods: int = 60) -> pd.DataFrame:
    """Build a synthetic daily series with a couple of swings (fixed, offline)."""
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    rows = []
    for i, ts in enumerate(timestamps):
        close = 50.0 + 0.05 * i + 3.0 * math.sin(i / 6.0)
        rows.append(
            {
                "timestamp": ts,
                "symbol": SYMBOL,
                "open": close - 0.4 * math.cos(i / 3.0),
                "high": close + 0.8,
                "low": close - 0.9,
                "close": close,
                "volume": 1000.0 + float(i % 7) * 50.0,
            }
        )
    return pd.DataFrame(rows)


class HeikinAshiIndicator(
    Indicator, title="平均K线", overlay=True, replaces_price=True
):
    """Heikin-Ashi candles on the main pane; never trades."""

    def on_start(self) -> None:
        """Declare the four Heikin-Ashi components as one candle group."""
        self.ha = self.I(
            aq.HeikinAshi(),
            name="ha",
            input_mode="ohlc",
            outputs=OHLC,
            render_type="candle",
            label="Heikin-Ashi",
        )


class HoldStrategy(Strategy):
    """Buy once and hold - the strategy itself is not the point here."""

    def on_bar(self, bar: Bar) -> None:
        """Enter on the first bar."""
        if self.get_position(bar.symbol) == 0:
            self.buy(bar.symbol, 100)


def main() -> None:
    """Run the backtest, export the candle group three ways, inspect the file."""
    data = make_demo_data()
    result = run_backtest(
        strategy=HoldStrategy,
        data=data,
        symbols=[SYMBOL],
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
        indicators=[HeikinAshiIndicator],
    )

    defs = result.indicator_definitions
    candle = defs.loc[defs["render_type"] == "candle"]
    print(f"candle_components={sorted(candle['indicator_key'])}")
    print(f"candle_groups={sorted(set(candle['group']))}")

    # Hand check of the first bar: ha_close = (O + H + L + C) / 4.
    first = data.iloc[0]
    expected_close = (first["open"] + first["high"] + first["low"] + first["close"]) / 4
    ha_close = result.indicator_df(name="ha.close")["value"].iloc[0]
    print(f"first_ha_close_matches={math.isclose(ha_close, expected_close)}")

    # LWC review: one Candlestick series for the group.
    out_dir = Path(tempfile.mkdtemp(prefix="akq_heikin_ashi_"))
    html_path = result.viz.review(data, filename=str(out_dir / "review.html"))
    html = Path(html_path).read_text(encoding="utf-8")
    print(f"review_html_written={Path(html_path).exists()}")
    has_candlestick = '"series_type":"Candlestick"' in html
    print(f"review_has_candlestick={has_candlestick}")

    # d3kline degrades the group to its close line, named after the group.
    outputs = result.indicator_outputs
    options = to_d3kline_options(
        outputs["definitions"], group_points_by_identity(outputs["points"])
    )
    names = [s["name"] for o in options for s in o["dataList"]]
    print(f"d3kline_series={names}")

    # Static inspection: replaces_price is read from the class line via ast.
    report = aq.inspect_strategy(__file__)
    for cls in report.classes:
        print(
            f"inspect {cls.name}: kind={cls.kind} overlay={cls.overlay} "
            f"replaces_price={cls.replaces_price}"
        )
    print(f"resolve_replaces_price={HeikinAshiIndicator.resolve_replaces_price()}")

    print("done_heikin_ashi")


if __name__ == "__main__":
    main()
