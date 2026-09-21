#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AKQuant declarative indicators demo (TradingView / Pine Script style).

`self.I()` declares an indicator once and the framework takes care of the rest:
1. Incremental update before every bar callback - no manual `update()` calls.
2. Series lookback `ind[0]` / `ind[1]`, matching Pine's `sma[0]` / `sma[1]`.
3. Automatic chart reporting when plot metadata is supplied - no per-bar
   `record_indicator()` boilerplate.
4. Multi-output indicators (MACD) split into separate lines via `outputs=`.
"""

import math

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy

SYMBOL = "DECL_DEMO"


def make_demo_data(periods: int = 120) -> pd.DataFrame:
    """Build a synthetic series with visible trend reversals."""
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    records: list[dict[str, object]] = []
    for i, ts in enumerate(timestamps):
        close = 100.0 + 0.15 * i + 6.0 * math.sin(i / 9.0)
        records.append(
            {
                "timestamp": ts,
                "symbol": SYMBOL,
                "open": close - 0.3,
                "high": close + 0.6,
                "low": close - 0.7,
                "close": close,
                "volume": 1000.0 + float(i % 17) * 40.0,
            }
        )
    return pd.DataFrame(records)


class DeclarativeIndicatorStrategy(Strategy):
    """Pine-style declaration: declare once in on_start, read with [0] / [1]."""

    def on_start(self) -> None:
        """Declare every indicator up front.

        Declaring in ``on_start`` (not ``__init__``) is what lets the periods
        come from ``self.params`` - parameters are injected after ``__init__``.
        """
        # Plot metadata present -> reported automatically for charting.
        self.ema_fast = self.I(
            aq.EMA(10), name="ema_fast", pane=0, color="#e91e63", label="EMA10"
        )
        self.ema_slow = self.I(
            aq.EMA(30), name="ema_slow", pane=0, color="#3f51b5", label="EMA30"
        )
        self.rsi = self.I(
            aq.RSI(14),
            name="rsi",
            pane=1,
            reference_lines=[
                {"value": 30.0, "label": "超卖"},
                {"value": 70.0, "label": "超买"},
            ],
        )
        # Multi-output: one declaration, three independent lines.
        self.macd = self.I(
            aq.MACD(12, 26, 9), name="macd", pane=2, outputs=("dif", "dea", "hist")
        )
        # No plot metadata -> computed only, never reported. Same as Pine's
        # bare `ta.atr(14)` without a `plot()` call.
        self.atr = self.I(aq.ATR(14), name="atr", input_mode="hlc")

        self.crosses = 0

    def on_bar(self, bar: Bar) -> None:
        """Golden-cross entry using one-bar lookback, RSI as a filter."""
        fast_now, fast_prev = self.ema_fast[0], self.ema_fast[1]
        slow_now, slow_prev = self.ema_slow[0], self.ema_slow[1]
        rsi_now = self.rsi[0]
        if None in (fast_now, fast_prev, slow_now, slow_prev, rsi_now):
            return

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now
        position = self.get_position(bar.symbol)

        # RSI 只挡极端超买 (>85): 金叉本身通常就伴随 RSI 抬升, 阈值定在 70
        # 会把所有入场都挡掉 —— 演示用的合成数据尤其如此。
        if crossed_up and position == 0 and rsi_now < 85.0:
            self.crosses += 1
            self.buy(bar.symbol, 100)
        elif crossed_down and position > 0:
            self.crosses += 1
            self.sell(bar.symbol, position)


def main() -> None:
    """Run the declarative indicator demo and summarize what got reported."""
    result = aq.run_backtest(
        strategy=DeclarativeIndicatorStrategy,
        data=make_demo_data(),
        symbols=[SYMBOL],
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
    )

    frame = result.indicator_df()
    reported = sorted(set(frame["indicator_key"]))
    print(f"declared_but_unplotted=atr (absent below): {'atr' not in reported}")
    print(f"reported_indicator_keys={reported}")
    print(f"reported_points={len(frame)}")

    definitions = result.indicator_definitions.set_index("indicator_key")
    print(f"ema_fast_display_name={definitions.loc['ema_fast', 'display_name']}")
    print(f"ema_fast_color={definitions.loc['ema_fast', 'color']}")
    print(f"rsi_pane={definitions.loc['rsi', 'pane']}")
    print(f"macd_hist_pane={definitions.loc['macd.hist', 'pane']}")
    print(f"end_market_value={result.metrics.end_market_value:.2f}")
    strategy = result.strategy
    crosses = getattr(strategy, "crosses", 0)
    print(f"crossover_signals={crosses}")
    print(f"trades={len(result.trades)}")
    print("done_declarative_indicators")


if __name__ == "__main__":
    main()
