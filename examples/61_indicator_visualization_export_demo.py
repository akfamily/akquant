#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant indicator visualization export demo.

This example shows the minimal workflow for frontend-facing indicator output:
1. Run a strategy that records custom indicator points.
2. Read indicator series from `BacktestResult.indicator_df(...)`.
3. Render a lightweight local preview with `plot_indicators(...)`.
4. Export normalized indicator outputs to JSON.
"""

from pathlib import Path

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def make_demo_bars() -> list[Bar]:
    """Build deterministic data for indicator export demo."""
    closes = [10.0, 10.4, 10.9, 10.7, 11.2]
    bars: list[Bar] = []
    for i, close in enumerate(closes):
        bars.append(
            Bar(
                timestamp=pd.Timestamp(f"2024-02-0{i + 1} 10:00:00").value,
                open=close - 0.1,
                high=close + 0.2,
                low=close - 0.3,
                close=close,
                volume=1000.0 + float(i) * 50.0,
                symbol="VIS",
            )
        )
    return bars


class IndicatorVisualizationStrategy(Strategy):
    """Record one custom visualization-oriented indicator on every bar."""

    def on_bar(self, bar: Bar) -> None:
        """Record one spread-style indicator for export and preview."""
        intrabar_spread = bar.high - bar.low
        self.record_indicator(
            name="intrabar_spread",
            value=intrabar_spread,
            display_name="Intra Bar Spread",
            pane=1,
            render_type="line",
            precision=4,
            meta={"source": ["high", "low"]},
        )


def main() -> None:
    """Run backtest, preview indicator history, and export indicator data."""
    result = aq.run_backtest(
        data=make_demo_bars(),
        strategy=IndicatorVisualizationStrategy,
        symbols="VIS",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    indicator_df = result.indicator_df(name="intrabar_spread", symbol="VIS")
    html_path = Path(__file__).with_name("indicator_visualization_preview.html")
    export_path = Path(__file__).with_name("indicator_visualization_outputs.json")
    result.plot_indicators(
        name="intrabar_spread",
        symbol="VIS",
        show=False,
        filename=str(html_path),
        title="AKQuant Indicator Preview",
    )
    result.export_indicators(str(export_path), format="json")

    print(indicator_df[["datetime", "indicator_key", "symbol", "value"]])
    print(f"indicator_rows={len(indicator_df)}")
    print(f"indicator_plot_html={html_path.resolve()}")
    print(f"indicator_export_json={export_path.resolve()}")
    print("done_indicator_visualization_export_demo")


if __name__ == "__main__":
    main()
