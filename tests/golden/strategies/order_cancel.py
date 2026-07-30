"""Order cancel lifecycle strategy.

Locks the撤单 (cancel) semantics that the other golden scenarios never touch:

* a resting limit order can be cancelled;
* the cancelled order stays queryable **across bars**, reporting ``Cancelled``
  (not ``None``) — the engine only exposes ``ctx.canceled_order_ids`` for the
  bar the cancel happened on, so this asserts the strategy-side bookkeeping;
* ``cancel_all_orders`` clears the remaining resting orders;
* a filled order likewise stays queryable after it leaves the live book.

Day 3 deliberately branches on the queried status, so a regression in cancel
bookkeeping changes the order/trade/equity output and trips the baseline even
though the baseline compares aggregates rather than raw statuses.
"""

from typing import Any, Dict, List

from akquant import Bar, OrderStatus, Strategy

# 远离市价的限价, 保证挂单不成交(数据 close 恒为 100)。
RESTING_LIMIT_PRICES = (50.0, 51.0, 52.0)


class OrderCancelStrategy(Strategy):
    """Exercise the cancel/query lifecycle of resting and filled orders."""

    def __init__(self) -> None:
        """Initialize."""
        self.day_count = 0
        self.resting_ids: List[str] = []
        self.filled_id = ""
        self.observations: Dict[str, str] = {}

    def on_bar(self, bar: Bar) -> None:
        """Drive the cancel lifecycle, one step per bar.

        Args:
            bar: Bar data.
        """
        self.day_count += 1

        if self.day_count == 1:
            for price in RESTING_LIMIT_PRICES:
                self.resting_ids.append(
                    str(
                        self.submit_order(
                            symbol=bar.symbol, side="Buy", quantity=100.0, price=price
                        )
                    )
                )
            # 这笔市价单会在下一根 bar 的 open 成交(NextOpen)。
            self.filled_id = str(self.buy(bar.symbol, 100))

        elif self.day_count == 2:
            self.cancel_order(self.resting_ids[0])
            self.observations["same_bar"] = self._status_of(self.resting_ids[0])

        elif self.day_count == 3:
            # 跨拍查询: 撤单记账若失效, 这里读到 None, 下面的分支就不会下单,
            # 订单数/成交数/权益随之改变, 基线即被触发。
            status = self._status_of(self.resting_ids[0])
            self.observations["next_bar"] = status
            if status == "cancelled":
                self.buy(bar.symbol, 100)

            # 已成交的订单离开在途账本后同样应可查。
            self.observations["filled_lookup"] = self._status_of(self.filled_id)

        elif self.day_count == 4:
            self.cancel_all_orders()
            self.observations["after_cancel_all"] = str(len(self.get_open_orders()))

        elif self.day_count == 5:
            position = self.get_position(bar.symbol)
            if position > 0:
                self.sell(bar.symbol, position)

    def _status_of(self, order_id: str) -> str:
        """Return the queried order status as a stable lowercase string."""
        order = self.get_order(order_id)
        if order is None:
            return "missing"
        status = getattr(order, "status", None)
        if status == OrderStatus.Cancelled:
            return "cancelled"
        if status == OrderStatus.Filled:
            return "filled"
        return str(status)

    def on_order(self, order: Any) -> None:
        """Handle order updates.

        Args:
            order: Order object.
        """
        if order.status == OrderStatus.Rejected:
            print(f"Order Rejected: {order.reject_reason}")
