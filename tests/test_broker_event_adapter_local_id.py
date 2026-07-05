"""map_* 的 local_id 覆盖 id/order_id(本地止损 id 连续性)."""

from akquant.gateway.broker_event_adapter import map_order_snapshot, map_trade
from akquant.gateway.broker_models import (
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedTrade,
)


def _snap():
    return UnifiedOrderSnapshot(
        client_order_id="c1",
        broker_order_id="B9",
        symbol="X",
        status=UnifiedOrderStatus.FILLED,
        filled_quantity=100.0,
        avg_fill_price=10.0,
    )


def test_order_id_uses_local_when_given() -> None:
    """给定 local_id 时 id 使用 local_id, broker_order_id 保持不变."""
    assert map_order_snapshot(_snap(), local_id="LSTOP-1").id == "LSTOP-1"
    assert map_order_snapshot(_snap(), local_id="LSTOP-1").broker_order_id == "B9"


def test_order_id_falls_back_to_broker_id() -> None:
    """local_id 为 None 时行为不变(回退到 broker_order_id)."""
    assert map_order_snapshot(_snap()).id == "B9"  # local_id None → unchanged


def test_trade_order_id_uses_local_when_given() -> None:
    """给定 local_id 时 trade.order_id 使用 local_id, 否则回退 broker_order_id."""
    t = UnifiedTrade(
        trade_id="T1",
        broker_order_id="B9",
        client_order_id="c1",
        symbol="X",
        side="Buy",
        quantity=100.0,
        price=10.0,
        timestamp_ns=1,
    )
    assert map_trade(t, local_id="LSTOP-1").order_id == "LSTOP-1"
    assert map_trade(t).order_id == "B9"  # None → unchanged
