"""broker_live 事件对象 → 与回测 Order/Trade 同形状（duck-typed）适配层.

回测 on_order/on_trade 收原生 Rust Order/Trade;broker_live 经此层把
Unified* 映射为同属性名 + 同枚举类型的 dataclass, 使策略回调两模式一致.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..akquant import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionEffect,
    TimeInForce,
)
from .broker_models import UnifiedOrderStatus


@dataclass
class StrategyOrder:
    """与回测 Order 同形状的委托事件对象(broker_live 用)."""

    id: str
    symbol: str
    status: Any  # OrderStatus
    filled_quantity: float
    average_filled_price: Optional[float]
    reject_reason: str
    updated_at: int
    position_effect: Any  # PositionEffect
    side: Any = None  # OrderSide | None
    order_type: Any = None  # OrderType | None
    time_in_force: Any = None  # TimeInForce | None
    quantity: Optional[float] = None
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    reduce_only: bool = False
    tag: str = ""
    commission: float = 0.0
    created_at: Optional[int] = None
    client_order_id: str = ""
    broker_order_id: str = ""
    owner_strategy_id: Optional[str] = None


@dataclass
class StrategyTrade:
    """与回测 Trade 同形状的成交事件对象(broker_live 用)."""

    id: str
    order_id: str
    symbol: str
    side: Any  # OrderSide
    timestamp: int
    quantity: float
    price: float
    position_effect: Any  # PositionEffect
    commission: float = 0.0
    client_order_id: str = ""
    broker_order_id: str = ""
    owner_strategy_id: Optional[str] = None


_STATUS_MAP = {
    UnifiedOrderStatus.NEW: OrderStatus.New,
    UnifiedOrderStatus.SUBMITTED: OrderStatus.Submitted,
    UnifiedOrderStatus.PARTIALLY_FILLED: OrderStatus.PartiallyFilled,
    UnifiedOrderStatus.FILLED: OrderStatus.Filled,
    UnifiedOrderStatus.CANCELLED: OrderStatus.Cancelled,
    UnifiedOrderStatus.REJECTED: OrderStatus.Rejected,
}


def _get(payload: Any, name: str, default: Any = None) -> Any:
    """兼容 dataclass 与 dict 两种输入取字段."""
    if isinstance(payload, dict):
        return payload.get(name, default)
    return getattr(payload, name, default)


def _to_status(value: Any) -> Any:
    try:
        return _STATUS_MAP.get(UnifiedOrderStatus(value), OrderStatus.New)
    except Exception:
        return _STATUS_MAP.get(value, OrderStatus.New)


def _to_side(value: Any) -> Optional[Any]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "buy":
        return OrderSide.Buy
    if text == "sell":
        return OrderSide.Sell
    return None


def _to_position_effect(value: Any) -> Any:
    text = str(value or "auto").strip().lower()
    mapping = {
        "auto": PositionEffect.Auto,
        "open": PositionEffect.Open,
        "close": PositionEffect.Close,
        "close_today": PositionEffect.CloseToday,
        "closetoday": PositionEffect.CloseToday,
        "close_yesterday": PositionEffect.CloseYesterday,
        "closeyesterday": PositionEffect.CloseYesterday,
    }
    return mapping.get(text, PositionEffect.Auto)


def _to_order_type(value: Any) -> Optional[Any]:
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        "market": OrderType.Market,
        "limit": OrderType.Limit,
        "stopmarket": OrderType.StopMarket,
        "stop": OrderType.StopMarket,
        "stoplimit": OrderType.StopLimit,
        "stop_limit": OrderType.StopLimit,
        "stoptrail": OrderType.StopTrail,
        "stoptraillimit": OrderType.StopTrailLimit,
    }
    return mapping.get(text)


def _to_tif(value: Any) -> Optional[Any]:
    if value is None:
        return None
    text = str(value).strip().upper()
    mapping = {
        "GTC": TimeInForce.GTC,
        "IOC": TimeInForce.IOC,
        "FOK": TimeInForce.FOK,
        "GTD": TimeInForce.Day,
        "DAY": TimeInForce.Day,
    }
    return mapping.get(text)


def map_order_snapshot(
    snapshot: Any,
    request: Any = None,
    owner_strategy_id: Optional[str] = None,
    local_id: Optional[str] = None,
) -> StrategyOrder:
    """把 UnifiedOrderSnapshot 映射成与回测 Order 同形状的 StrategyOrder."""
    broker_order_id = str(_get(snapshot, "broker_order_id", "") or "")
    return StrategyOrder(
        id=local_id or broker_order_id,
        symbol=str(_get(snapshot, "symbol", "") or ""),
        status=_to_status(_get(snapshot, "status")),
        filled_quantity=float(_get(snapshot, "filled_quantity", 0.0) or 0.0),
        average_filled_price=(
            float(_get(snapshot, "avg_fill_price"))
            if _get(snapshot, "avg_fill_price") is not None
            else None
        ),
        reject_reason=str(_get(snapshot, "reject_reason", "") or ""),
        updated_at=int(_get(snapshot, "timestamp_ns", 0) or 0),
        position_effect=_to_position_effect(_get(snapshot, "position_effect", "auto")),
        side=_to_side(_get(request, "side")),
        order_type=_to_order_type(_get(request, "order_type")),
        time_in_force=_to_tif(_get(request, "time_in_force")),
        quantity=(
            float(_get(request, "quantity"))
            if _get(request, "quantity") is not None
            else None
        ),
        price=(
            float(_get(request, "price"))
            if _get(request, "price") is not None
            else None
        ),
        trigger_price=None,
        reduce_only=bool(_get(request, "reduce_only", False)),
        tag="",
        commission=0.0,
        created_at=None,
        client_order_id=str(_get(snapshot, "client_order_id", "") or ""),
        broker_order_id=broker_order_id,
        owner_strategy_id=owner_strategy_id,
    )


def map_trade(
    trade: Any,
    request: Any = None,
    owner_strategy_id: Optional[str] = None,
    local_id: Optional[str] = None,
) -> StrategyTrade:
    """把 UnifiedTrade 映射成与回测 Trade 同形状的 StrategyTrade."""
    return StrategyTrade(
        id=str(_get(trade, "trade_id", "") or ""),
        order_id=local_id or str(_get(trade, "broker_order_id", "") or ""),
        symbol=str(_get(trade, "symbol", "") or ""),
        side=_to_side(_get(trade, "side")) or OrderSide.Buy,
        timestamp=int(_get(trade, "timestamp_ns", 0) or 0),
        quantity=float(_get(trade, "quantity", 0.0) or 0.0),
        price=float(_get(trade, "price", 0.0) or 0.0),
        position_effect=_to_position_effect(_get(trade, "position_effect", "auto")),
        commission=0.0,
        client_order_id=str(_get(trade, "client_order_id", "") or ""),
        broker_order_id=str(_get(trade, "broker_order_id", "") or ""),
        owner_strategy_id=owner_strategy_id,
    )
