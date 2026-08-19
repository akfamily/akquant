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
    """Trader gateway protocol.

    可选方法 ``classify_order_error(exc) -> UnifiedErrorType``:
        下单/撤单调用抛异常时, 核心用它区分「柜台明确回绝」(``RISK_REJECTED`` /
        ``NON_RETRYABLE`` → 回吐 Rejected 事件)与「订单状态不可知」(``RETRYABLE``
        → 只发 on_error, 不谎报拒单)。**不实现则一律按状态未知处理**, 这是安全
        缺省: 超时往往发生在报文已发出之后, 谎报拒单会诱导策略重下单。
        见 ``gateway/order_errors.py``。
    """

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
        """Sync open orders — **only** ones still working at the broker.

        Implementations must filter to ``NEW`` / ``SUBMITTED`` /
        ``PARTIALLY_FILLED``. Many venues answer "query orders" with the day's
        *full* order book (cancelled and filled rows included); returning that
        as-is breaks both consumers: ``BrokerExecution.cancel_all_orders``
        cancels every entry one by one (re-cancelling a cancelled order is a
        broker-side reject), and ``BrokerRecovery`` replays the whole history as
        if it were still working.
        """

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
