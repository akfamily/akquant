from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class UnifiedOrderStatus(str, Enum):
    """Canonical order status used by broker adapters."""

    NEW = "New"
    SUBMITTED = "Submitted"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"


class UnifiedErrorType(str, Enum):
    """Canonical error category used by broker adapters."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    RISK_REJECTED = "risk_rejected"


@dataclass(frozen=True)
class BrokerCapability:
    """Broker execution capability matrix."""

    broker_name: str
    broker_live: bool = True
    client_order_id: bool = True
    order_type: bool = True
    time_in_force_str: bool = True
    position_effect: bool = False
    reduce_only: bool = False
    position_details: bool = False
    supports_short_sell: bool = False
    broker_extra_fields: tuple[str, ...] = field(default_factory=tuple)
    supported_position_effects: tuple[str, ...] = ("auto",)
    features: frozenset[str] = frozenset()

    def as_execution_capabilities(self) -> dict[str, Any]:
        """Return broker capabilities in the strategy-facing dict shape."""
        return {
            "broker_live": self.broker_live,
            "client_order_id": self.client_order_id,
            "order_type": self.order_type,
            "time_in_force_str": self.time_in_force_str,
            "position_effect": self.position_effect,
            "reduce_only": self.reduce_only,
            "position_details": self.position_details,
            "supports_short_sell": self.supports_short_sell,
            "broker_extra_fields": list(self.broker_extra_fields),
            "supported_position_effects": list(self.supported_position_effects),
            "features": sorted(self.features),
            "broker_name": self.broker_name,
        }

    @classmethod
    def from_value(cls, value: Any, broker_name: str = "") -> "BrokerCapability":
        """Build a capability object from an existing instance or raw dict."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            raw_supported = value.get("supported_position_effects", ("auto",))
            if isinstance(raw_supported, str):
                raw_supported = (raw_supported,)
            return cls(
                broker_name=str(value.get("broker_name", broker_name or "broker")),
                broker_live=bool(value.get("broker_live", True)),
                client_order_id=bool(value.get("client_order_id", True)),
                order_type=bool(value.get("order_type", True)),
                time_in_force_str=bool(value.get("time_in_force_str", True)),
                position_effect=bool(value.get("position_effect", False)),
                reduce_only=bool(value.get("reduce_only", False)),
                position_details=bool(value.get("position_details", False)),
                supports_short_sell=bool(value.get("supports_short_sell", False)),
                broker_extra_fields=tuple(
                    str(item) for item in value.get("broker_extra_fields", ())
                ),
                supported_position_effects=tuple(
                    normalize_position_effect(item) for item in raw_supported
                )
                or ("auto",),
                features=frozenset(str(item) for item in value.get("features", ())),
            )
        raise TypeError("broker capability must be a BrokerCapability or dict")


@dataclass
class UnifiedOrderRequest:
    """Normalized order request."""

    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float | None = None
    order_type: str = "Market"
    time_in_force: str = "GTC"
    position_effect: str = "auto"
    reduce_only: bool = False
    asset_type: str = "stock"
    extra: dict[str, Any] = field(default_factory=dict)


def normalize_position_effect(position_effect: str | None) -> str:
    """Normalize broker position effect text into the supported canonical form."""
    normalized = str(position_effect or "auto").strip().lower()
    if normalized not in {"auto", "open", "close", "close_today", "close_yesterday"}:
        raise RuntimeError(
            "position_effect must be one of: auto, open, close, "
            "close_today, close_yesterday"
        )
    return normalized


# 规范集(文档参考): stock/option/future/fund/bond/index/fx/crypto。
# 别名映射到规范名；规范集外的值原样放行(保持对新品种可扩展)。
_ASSET_TYPE_ALIASES = {
    "opt": "option",
    "stk": "stock",
    "fut": "future",
    "futures": "future",
    "etf": "fund",
}


def normalize_asset_type(value: str | None) -> str:
    """Normalize an asset type to its canonical name; pass unknowns through.

    Canonical names: stock, option, future, fund, bond, index, fx, crypto.
    Unknown values are returned lowercased so new products need no core change.
    """
    text = str(value or "stock").strip().lower()
    if not text:
        return "stock"
    return _ASSET_TYPE_ALIASES.get(text, text)


def validate_broker_extra(
    capability: "BrokerCapability", extra: dict[str, Any] | None
) -> None:
    """Ensure every extra key is declared in capability.broker_extra_fields."""
    if not extra:
        return
    declared = set(capability.broker_extra_fields)
    unknown = sorted(k for k in extra if k not in declared)
    if unknown:
        raise RuntimeError(
            f"broker '{capability.broker_name}' 未声明的 extra 字段: {unknown} "
            f"(已声明: {sorted(declared)})"
        )


def validate_execution_semantics(
    capability: BrokerCapability,
    position_effect: str | None,
    reduce_only: bool = False,
) -> str:
    """Validate order semantics against the declared broker capability matrix."""
    normalized_effect = normalize_position_effect(position_effect)
    supported = tuple(
        normalize_position_effect(item)
        for item in capability.supported_position_effects
    ) or ("auto",)
    if reduce_only and not capability.reduce_only:
        raise RuntimeError(
            f"broker '{capability.broker_name}' does not support reduce_only orders"
        )
    if normalized_effect != "auto" and not capability.position_effect:
        raise RuntimeError(
            f"broker '{capability.broker_name}' does not support "
            "explicit position_effect"
        )
    if normalized_effect not in supported:
        supported_text = ", ".join(supported)
        raise RuntimeError(
            f"broker '{capability.broker_name}' does not support "
            f"position_effect='{normalized_effect}' (supported: {supported_text})"
        )
    return normalized_effect


@dataclass
class UnifiedOrderSnapshot:
    """Normalized order state snapshot."""

    client_order_id: str
    broker_order_id: str
    symbol: str
    status: UnifiedOrderStatus
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    reject_reason: str = ""
    timestamp_ns: int = 0
    position_effect: str = "auto"
    group_id: str = ""


@dataclass
class UnifiedTrade:
    """Normalized trade fill record."""

    trade_id: str
    broker_order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp_ns: int
    position_effect: str = "auto"
    group_id: str = ""


@dataclass
class UnifiedExecutionReport:
    """Normalized execution report."""

    broker_order_id: str
    client_order_id: str
    status: UnifiedOrderStatus
    symbol: str
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    reject_reason: str = ""
    timestamp_ns: int = 0
    position_effect: str = "auto"
    group_id: str = ""


@dataclass
class UnifiedAccount:
    """Normalized account snapshot."""

    account_id: str
    equity: float
    cash: float
    available_cash: float
    timestamp_ns: int = 0


@dataclass
class UnifiedPosition:
    """Normalized position snapshot."""

    symbol: str
    quantity: float
    available_quantity: float
    direction: str = ""
    today_quantity: float = 0.0
    yesterday_quantity: float = 0.0
    available_today_quantity: float = 0.0
    available_yesterday_quantity: float = 0.0
    avg_price: float = 0.0
    timestamp_ns: int = 0
