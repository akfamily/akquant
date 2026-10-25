"""Futures night-session weekend strategy (rule 2.0.0 golden scenario).

rb2405 with ``sessions={"rb": "CN_FUTURES_NIGHT_23"}``: Friday day session,
Friday night session (belongs to the next trading day, Monday) and Monday day
session.

* open 1 lot on the first bar whose trading day is later than its calendar
  date, i.e. the Friday night session (fills on the next night bar);
* close it on the first Monday day-session bar (fills on the next bar).

Both decisions branch on ``self.trading_day``: if the night session were
attributed to Friday, no position would be opened and the baseline trips.
"""

from typing import Any

import pandas as pd
from akquant import Bar, OrderStatus, Strategy


class FuturesNightWeekendStrategy(Strategy):
    """Open on the Friday night session, close on the Monday day session."""

    def __init__(self) -> None:
        """Initialize."""
        self.opened = False
        self.closed = False

    def on_bar(self, bar: Bar) -> None:
        """Open on the night session, close on the next day session.

        Args:
            bar: Bar data.
        """
        trading_day = self.trading_day
        if trading_day is None:
            return
        calendar_day = pd.Timestamp(bar.timestamp, tz="Asia/Shanghai").date()
        if not self.opened and trading_day > calendar_day:
            self.opened = True
            self.buy(bar.symbol, 1)
        elif (
            self.opened
            and not self.closed
            and trading_day == calendar_day
            and self.get_position(bar.symbol) > 0
        ):
            self.closed = True
            self.sell(bar.symbol, 1)

    def on_order(self, order: Any) -> None:
        """Handle order updates.

        Args:
            order: Order object.
        """
        if order.status == OrderStatus.Rejected:
            print(f"Order Rejected: {order.reject_reason}")
