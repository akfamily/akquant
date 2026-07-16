"""broker_event_adapter：Unified* → 回测同形状对象（枚举/名对齐/回填）."""

from typing import Any, cast

from akquant.akquant import OrderSide, OrderStatus, PositionEffect
from akquant.gateway.broker_event_adapter import map_order_snapshot, map_trade
from akquant.gateway.broker_models import (
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedTrade,
)


def _snap(**kw: Any) -> UnifiedOrderSnapshot:
    client_order_id = cast(str, kw.get("client_order_id", "c1"))
    broker_order_id = cast(str, kw.get("broker_order_id", "B1"))
    symbol = cast(str, kw.get("symbol", "600000.SH"))
    status = cast(UnifiedOrderStatus, kw.get("status", UnifiedOrderStatus.FILLED))
    filled_quantity = cast(float, kw.get("filled_quantity", 100.0))
    avg_fill_price = cast(float, kw.get("avg_fill_price", 10.5))
    return UnifiedOrderSnapshot(
        client_order_id=client_order_id,
        broker_order_id=broker_order_id,
        symbol=symbol,
        status=status,
        filled_quantity=filled_quantity,
        avg_fill_price=avg_fill_price,
    )


def test_order_name_and_enum_alignment() -> None:
    """字段名/枚举与回测 Order 对齐."""
    o = map_order_snapshot(_snap())
    assert o.id == "B1"  # id = broker_order_id
    assert o.symbol == "600000.SH"
    assert o.filled_quantity == 100.0
    assert o.average_filled_price == 10.5  # avg_fill_price → average_filled_price
    assert o.status is OrderStatus.Filled  # UnifiedOrderStatus → OrderStatus enum
    assert o.commission == 0.0  # broker snapshot lacks; default


def test_order_backfill_from_request() -> None:
    """side/quantity/price 从 request 回填."""
    req = UnifiedOrderRequest(
        client_order_id="c1",
        symbol="600000.SH",
        side="Buy",
        quantity=200.0,
        price=10.0,
        order_type="Limit",
    )
    o = map_order_snapshot(_snap(), request=req)
    assert o.side is OrderSide.Buy
    assert o.quantity == 200.0
    assert o.price == 10.0


def test_order_without_request_defaults() -> None:
    """无 request 时 side/price 应为 None."""
    o = map_order_snapshot(_snap())
    assert o.side is None
    assert o.price is None


def test_status_map_all_variants() -> None:
    """UnifiedOrderStatus 全量枚举映射到 OrderStatus."""
    pairs = {
        UnifiedOrderStatus.NEW: OrderStatus.New,
        UnifiedOrderStatus.SUBMITTED: OrderStatus.Submitted,
        UnifiedOrderStatus.PARTIALLY_FILLED: OrderStatus.PartiallyFilled,
        UnifiedOrderStatus.FILLED: OrderStatus.Filled,
        UnifiedOrderStatus.CANCELLED: OrderStatus.Cancelled,
        UnifiedOrderStatus.REJECTED: OrderStatus.Rejected,
    }
    for u, expected in pairs.items():
        assert map_order_snapshot(_snap(status=u)).status is expected


def test_trade_mapping() -> None:
    """UnifiedTrade 字段名/枚举与回测 Trade 对齐."""
    t = map_trade(
        UnifiedTrade(
            trade_id="T1",
            broker_order_id="B1",
            client_order_id="c1",
            symbol="600000.SH",
            side="Sell",
            quantity=100.0,
            price=10.5,
            timestamp_ns=123,
        )
    )
    assert t.id == "T1"  # trade_id → id
    assert t.order_id == "B1"  # broker_order_id → order_id
    assert t.side is OrderSide.Sell
    assert t.timestamp == 123  # timestamp_ns → timestamp
    assert t.price == 10.5
    assert t.commission == 0.0  # default
    assert t.position_effect is PositionEffect.Auto


def test_owner_strategy_id_backfill() -> None:
    """owner_strategy_id 回填到 StrategyOrder/StrategyTrade,与回测 Order/Trade 对齐."""
    o = map_order_snapshot(_snap(), owner_strategy_id="s1")
    assert o.owner_strategy_id == "s1"
    t = map_trade(
        UnifiedTrade(
            trade_id="T1",
            broker_order_id="B1",
            client_order_id="c1",
            symbol="600000.SH",
            side="Sell",
            quantity=100.0,
            price=10.5,
            timestamp_ns=123,
        ),
        owner_strategy_id="s1",
    )
    assert t.owner_strategy_id == "s1"


def test_owner_strategy_id_defaults_none() -> None:
    """未传 owner_strategy_id 时默认为 None，不影响既有调用方."""
    o = map_order_snapshot(_snap())
    assert o.owner_strategy_id is None


def test_accepts_dict_payload() -> None:
    """map_order_snapshot 接受 dict payload."""
    o = map_order_snapshot(
        {
            "broker_order_id": "B9",
            "symbol": "X",
            "status": UnifiedOrderStatus.NEW,
            "filled_quantity": 0.0,
            "avg_fill_price": 0.0,
        }
    )
    assert o.id == "B9"
    assert o.status is OrderStatus.New
