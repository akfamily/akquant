# -*- coding: utf-8 -*-
"""Regression: a next-open Day order must survive to its matchable slice.

In ``on_cross_section`` (post-close rebalance) with the default next-open
execution (``bar_offset == 1``), an order issued after day D's close is meant to
fill at day D+1's open. Daily settlement, however, ran at the *start* of D+1
(before D+1's bar is matched) and unconditionally expired **every** Day order —
so the order died before it ever reached its fill slice and never traded.

Correct behavior: a next-open Day order expires only once its fill day has
passed without a fill (i.e. it gets its D+1 open matching chance first). A
same-cycle (bar_offset == 0) Day order keeps expiring the day after creation.
"""

import akquant as aq
import pandas as pd
from akquant import Strategy, TimeInForce

TZ = "Asia/Shanghai"
DAYS = pd.date_range("2024-01-02", periods=5, freq="B", tz=TZ)


def _make_bars() -> pd.DataFrame:
    rows = []
    for i, d in enumerate(DAYS):
        c = 100.0 * (1 + 0.001 * i)
        rows.append(
            {
                "date": d,
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": 1_000_000.0,
                "symbol": "AAA",
            }
        )
    return pd.DataFrame(rows)


class _CrossSectionDayBuy(Strategy):
    """Issue a single next-open Day buy on the first cross-section slice."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()
        self._done = False

    def on_cross_section(self, trading_date: object, timestamp: int) -> None:
        if not self._done:
            self._done = True
            self.order_target_percent(
                target_percent=0.9,
                symbol="AAA",
                time_in_force=TimeInForce.Day,
            )


def test_cross_section_next_open_day_order_fills_next_day() -> None:
    """A next-open Day order from on_cross_section must reach its fill slice.

    Pre-fix, start-of-next-day settlement expired the order before its D+1 open
    matching chance, so it never filled (trades == 0). It should fill instead.
    """
    result = aq.run_backtest(
        data=_make_bars(),
        strategy=_CrossSectionDayBuy,
        symbols=["AAA"],
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        start_time=str(DAYS[0].date()),
        end_time=str(DAYS[-1].date()),
        show_progress=False,
    )

    buys = result.orders_df[
        result.orders_df["side"].astype(str).str.lower().str.contains("buy")
    ]
    assert len(buys) == 1, f"expected exactly one buy order, got {len(buys)}"

    buy = buys.iloc[0]
    status = str(buy["status"]).lower()
    filled = float(buy.get("filled_quantity", 0) or 0)

    # Pre-fix: start-of-next-day settlement expired the order before its D+1 open
    # matching chance, so status=='expired' and filled_quantity==0. It should
    # instead reach that slice and fill.
    assert "filled" in status, (
        f"next-open Day order should reach its D+1 open and fill, but status was "
        f"{status!r} (expired by start-of-day settlement before its fill slice)"
    )
    assert filled > 0, (
        f"a filled Day order must have a positive filled_quantity, got {filled}"
    )
