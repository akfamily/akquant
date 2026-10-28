"""A-share 1-minute auction strategy (rule 2.0.0 golden scenario).

Locks the A 股分钟线集合竞价 semantics on 600000.SH, two trading days:

* day 1 10:00 bar: market buy 100 in continuous trading (fills next bar open);
  plus a far-away limit buy with ``TimeInForce.Day`` that must expire at the
  day-1 close;
* day 2 09:20 timer (``schedule_daily``; day 1 does not fire because the first
  bar is 09:25, later than 09:20): market buy 100 queued before the opening
  auction, filled at the 09:25 auction price on the first continuous bar;
* day 2 14:58 bar (inside the 14:57-15:00 closing auction): market sell 100
  (the day-1 lot, allowed under T+1), filled at the 15:00 auction price.

The data prices are tiered so the fill price identifies the matching source.
"""

from typing import Any, List, Optional

import pandas as pd
from akquant import Bar, OrderStatus, Strategy, TimeInForce

SYMBOL = "600000.SH"
# 远离市价的 Day 限价买单：不会成交，只验证当日收盘过期。
DAY_ORDER_LIMIT = 9.0


def _local_hhmm(timestamp_ns: int) -> str:
    return pd.Timestamp(timestamp_ns, tz="Asia/Shanghai").strftime("%H:%M")


class StockMinuteAuctionStrategy(Strategy):
    """Exercise opening/closing auction fills and Day-order expiry."""

    def __init__(self) -> None:
        """Initialize."""
        self.seen_days: List[Any] = []

    def on_start(self) -> None:
        """Register the pre-open timer."""
        self.schedule_daily("09:20:00", "auction_buy")

    def _day_index(self) -> Optional[int]:
        day = self.trading_day
        if day is None:
            return None
        if day not in self.seen_days:
            self.seen_days.append(day)
        return self.seen_days.index(day)

    def on_timer(self, payload: str) -> None:
        """Queue a market buy before the opening auction.

        Args:
            payload: Timer payload.
        """
        if payload == "auction_buy":
            self.buy(SYMBOL, 100)

    def on_bar(self, bar: Bar) -> None:
        """Place the continuous, Day and closing-auction orders.

        Args:
            bar: Bar data.
        """
        day_index = self._day_index()
        hhmm = _local_hhmm(bar.timestamp)
        if day_index == 0 and hhmm == "10:00":
            self.buy(bar.symbol, 100)
            self.submit_order(
                symbol=bar.symbol,
                side="Buy",
                quantity=100.0,
                price=DAY_ORDER_LIMIT,
                time_in_force=TimeInForce.Day,
            )
        elif day_index == 1 and hhmm == "14:58":
            self.sell(bar.symbol, 100)

    def on_order(self, order: Any) -> None:
        """Handle order updates.

        Args:
            order: Order object.
        """
        if order.status == OrderStatus.Rejected:
            print(f"Order Rejected: {order.reject_reason}")
