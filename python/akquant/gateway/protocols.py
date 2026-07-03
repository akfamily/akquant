from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from .broker_models import (
    BrokerCapability,
    UnifiedAccount,
    UnifiedExecutionReport,
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedPosition,
    UnifiedTrade,
)


class MarketGateway(Protocol):
    """Market data gateway protocol."""

    def connect(self) -> None:
        """Connect market data channel."""

    def disconnect(self) -> None:
        """Disconnect market data channel."""

    def subscribe(self, symbols: Sequence[str]) -> None:
        """Subscribe symbols."""

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        """Unsubscribe symbols."""

    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register tick callback."""

    def on_bar(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register bar callback."""

    def start(self) -> None:
        """Start gateway event loop."""


class TraderGateway(Protocol):
    """Trader gateway protocol."""

    def connect(self) -> None:
        """Connect trader channel."""

    def disconnect(self) -> None:
        """Disconnect trader channel."""

    def place_order(self, req: UnifiedOrderRequest) -> str:
        """Place order."""

    def get_capabilities(self) -> BrokerCapability:
        """Return trader capability matrix."""

    def cancel_order(self, broker_order_id: str) -> None:
        """Cancel order."""

    def query_order(self, broker_order_id: str) -> UnifiedOrderSnapshot | None:
        """Query order."""

    def query_trades(self, since: int | None = None) -> list[UnifiedTrade]:
        """Query trades."""

    def query_account(self) -> UnifiedAccount | None:
        """Query account."""

    def query_positions(self) -> list[UnifiedPosition]:
        """Query positions."""

    def on_order(self, callback: Callable[[UnifiedOrderSnapshot], None]) -> None:
        """Register order callback."""

    def on_trade(self, callback: Callable[[UnifiedTrade], None]) -> None:
        """Register trade callback."""

    def on_execution_report(
        self, callback: Callable[[UnifiedExecutionReport], None]
    ) -> None:
        """Register execution report callback."""

    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        """Sync open orders."""

    def sync_today_trades(self) -> list[UnifiedTrade]:
        """Sync today's trades."""

    def heartbeat(self) -> bool:
        """Heartbeat check."""

    def start(self) -> None:
        """Start gateway event loop."""


@dataclass
class GatewayBundle:
    """Gateway instances and optional metadata."""

    market_gateway: MarketGateway | None = None
    trader_gateway: TraderGateway | None = None
    trader_capabilities: BrokerCapability | None = None
    metadata: dict[str, Any] | None = None
