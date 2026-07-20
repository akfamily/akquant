# -*- coding: utf-8 -*-
"""Regression for issue #329.

on_timer-driven full-investment rotation must NOT freeze permanently when a
co-submitted sell can never fill (a suspended symbol with no bars). Prior to the
fix, the timer event advanced the clock and wiped the deferred-order registry
before the slice was finalized; the deferred phantom buys stayed ``New`` in
``active_orders``, the submission-time margin check projected them into free
margin (``Available: 0``) and rejected every subsequent order — including the
sells that would have unwound the position — trading was frozen for good.
"""

import akquant as aq
import pandas as pd
from akquant import Strategy
from akquant.backtest import BacktestResult

TZ = "Asia/Shanghai"
DAYS = pd.date_range("2024-01-02", periods=8, freq="B", tz=TZ)
SUSPEND_AFTER = DAYS[2]  # CCC has no bars from the 4th trading day on


def _make_bars(symbol: str, base: float) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(DAYS):
        if symbol == "CCC" and d > SUSPEND_AFTER:
            continue  # suspension: no bars at all
        c = base * (1 + 0.001 * i)
        rows.append(
            {
                "date": d,
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": 1_000_000.0,
                "symbol": symbol,
            }
        )
    return pd.DataFrame(rows)


class _TimerFullRotation(Strategy):
    """Every day at 09:00 liquidate everything, then equal-weight buy all picks."""

    PICKS = ["AAA", "BBB", "CCC", "DDD"]

    def on_start(self) -> None:
        self.schedule_daily("09:00", "rebalance")

    def on_timer(self, payload: str) -> None:
        if payload != "rebalance":
            return
        for s in list(self.positions.keys()):
            self.close_position(s)
        w = 0.99 / len(self.PICKS)
        for s in self.PICKS:
            self.order_target_percent(target_percent=w, symbol=s)


def _run() -> BacktestResult:
    data = (
        pd.concat(
            [
                _make_bars(s, b)
                for s, b in (("AAA", 10.0), ("BBB", 20.0), ("CCC", 30.0), ("DDD", 40.0))
            ],
            ignore_index=True,
        )
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    return aq.run_backtest(
        data=data,
        strategy=_TimerFullRotation,
        symbols=["AAA", "BBB", "CCC", "DDD"],
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        start_time=str(DAYS[0].date()),
        end_time=str(DAYS[-1].date()),
        show_progress=False,
    )


def test_on_timer_rotation_survives_suspended_symbol() -> None:
    """Trading must keep entering positions after a co-submitted sell is stuck.

    Regression for #329: the deadlock stops all new entries shortly after CCC's
    suspension day. A healthy run keeps rotating through the whole window, so the
    last entry day must reach at least the penultimate trading day.
    """
    result = _run()
    trades = result.trades_df
    assert len(trades) > 0, "no trades at all — engine produced nothing"

    entry_days = sorted(pd.to_datetime(trades["entry_time"]).dt.date.unique())
    last_entry = entry_days[-1]

    assert last_entry >= DAYS[-2].date(), (
        f"trading froze after {last_entry}; expected entries to continue through "
        f"{DAYS[-2].date()} (deadlock from issue #329)"
    )
