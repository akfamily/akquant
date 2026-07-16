"""同一 on_order/on_trade 回调读通用字段, 在回测风格对象与 broker 适配对象都工作."""

from types import SimpleNamespace
from typing import Any

from akquant.akquant import OrderSide, OrderStatus
from akquant.gateway.broker_event_adapter import map_order_snapshot, map_trade
from akquant.gateway.broker_models import (
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedTrade,
)


def _read_order(order: Any) -> tuple[Any, Any, Any, Any, Any]:
    # 一个"策略回调"会读的通用字段（回测/实盘同名）
    return (
        order.symbol,
        order.status,
        order.filled_quantity,
        order.average_filled_price,
        order.side,
    )


def _read_trade(trade: Any) -> tuple[Any, Any, Any, Any]:
    return (trade.symbol, trade.side, trade.quantity, trade.price)


def test_order_reads_same_on_backtest_like_and_broker() -> None:
    """同一读字段逻辑在回测风格对象与 broker 适配对象上结果一致."""
    # 回测风格：一个具备回测 Order 属性名的对象
    bt = SimpleNamespace(
        symbol="X",
        status=OrderStatus.Filled,
        filled_quantity=100.0,
        average_filled_price=10.5,
        side=OrderSide.Buy,
    )
    req = UnifiedOrderRequest(
        client_order_id="c1", symbol="X", side="Buy", quantity=100.0, price=10.0
    )
    live = map_order_snapshot(
        UnifiedOrderSnapshot(
            client_order_id="c1",
            broker_order_id="B1",
            symbol="X",
            status=UnifiedOrderStatus.FILLED,
            filled_quantity=100.0,
            avg_fill_price=10.5,
        ),
        request=req,
    )
    assert _read_order(bt) == _read_order(live)


def test_trade_reads_same_on_backtest_like_and_broker() -> None:
    """同一读字段逻辑在回测风格对象与 broker 适配对象上结果一致."""
    bt = SimpleNamespace(symbol="X", side=OrderSide.Sell, quantity=100.0, price=10.5)
    live = map_trade(
        UnifiedTrade(
            trade_id="T1",
            broker_order_id="B1",
            client_order_id="c1",
            symbol="X",
            side="Sell",
            quantity=100.0,
            price=10.5,
            timestamp_ns=1,
        )
    )
    assert _read_trade(bt) == _read_trade(live)
