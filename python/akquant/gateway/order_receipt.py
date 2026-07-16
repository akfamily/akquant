"""OrderReceipt: 逻辑委托的返回句柄，携带跨边界拆出的全部腿 id 与语义."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class OrderLeg:
    """单腿委托：语义 + client/broker id."""

    position_effect: str
    quantity: float
    client_order_id: str
    broker_order_id: str


@dataclass(frozen=True)
class OrderReceipt:
    """一次下单调用的结果：一个逻辑委托可拆成多腿分别报单."""

    group_id: str
    order_ids: tuple[str, ...]
    legs: tuple[OrderLeg, ...]

    def __str__(self) -> str:
        """Return group_id as string representation."""
        return self.group_id

    def __iter__(self) -> Iterator[str]:
        """Iterate over order_ids."""
        return iter(self.order_ids)

    def __len__(self) -> int:
        """Return count of order_ids."""
        return len(self.order_ids)

    @property
    def primary(self) -> str:
        """Return first order_id or empty string if no ids."""
        return self.order_ids[0] if self.order_ids else ""

    @classmethod
    def single(
        cls,
        group_id: str,
        broker_order_id: str,
        position_effect: str = "auto",
        quantity: float = 0.0,
        client_order_id: str = "",
    ) -> "OrderReceipt":
        """Create a single-leg receipt for convenience."""
        leg = OrderLeg(
            position_effect=position_effect,
            quantity=quantity,
            client_order_id=client_order_id or group_id,
            broker_order_id=broker_order_id,
        )
        return cls(group_id=group_id, order_ids=(broker_order_id,), legs=(leg,))
