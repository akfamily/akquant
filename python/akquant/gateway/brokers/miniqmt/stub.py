import time
from typing import Any, Callable, Sequence

from ....akquant import DataFeed
from ...broker_event_mapper import BrokerEventMapper, create_default_mapper
from ...broker_models import (
    BrokerCapability,
    UnifiedAccount,
    UnifiedExecutionReport,
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedPosition,
    UnifiedTrade,
    validate_execution_semantics,
)


class MiniQMTMarketGateway:
    """MiniQMT market gateway placeholder."""

    def __init__(
        self,
        feed: DataFeed,
        symbols: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """Initialize market gateway parameters."""
        self.feed = feed
        self.symbols = list(symbols)
        self.kwargs = kwargs
        self.connected = False
        self.tick_callback: Callable[[dict[str, Any]], None] | None = None
        self.bar_callback: Callable[[dict[str, Any]], None] | None = None

    def connect(self) -> None:
        """Connect market data channel."""
        self.connected = True

    def disconnect(self) -> None:
        """Disconnect market data channel."""
        self.connected = False

    def subscribe(self, symbols: Sequence[str]) -> None:
        """Subscribe symbols."""
        self.symbols = list(symbols)

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        """Unsubscribe symbols."""
        removed = set(symbols)
        self.symbols = [s for s in self.symbols if s not in removed]

    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register tick callback."""
        self.tick_callback = callback

    def on_bar(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register bar callback."""
        self.bar_callback = callback

    def start(self) -> None:
        """Start market gateway event loop."""
        self.connect()


class MiniQMTTraderGateway:
    """MiniQMT trader gateway placeholder."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize trader gateway parameters."""
        self.kwargs = kwargs
        self.connected = False
        self.orders: dict[str, UnifiedOrderSnapshot] = {}
        self.trades: list[UnifiedTrade] = []
        self.client_to_broker_order_ids: dict[str, str] = {}
        self.broker_to_client_order_ids: dict[str, str] = {}
        self._order_seq = 0
        self.enforce_client_order_id_uniqueness = bool(
            kwargs.get("enforce_client_order_id_uniqueness", True)
        )
        self.mapper: BrokerEventMapper = kwargs.get(
            "event_mapper", create_default_mapper()
        )
        self.order_callback: Callable[[UnifiedOrderSnapshot], None] | None = None
        self.trade_callback: Callable[[UnifiedTrade], None] | None = None
        self.execution_callback: Callable[[UnifiedExecutionReport], None] | None = None

    def connect(self) -> None:
        """Connect trader channel."""
        self.connected = True

    def disconnect(self) -> None:
        """Disconnect trader channel."""
        self.connected = False

    def place_order(self, req: UnifiedOrderRequest) -> str:
        """Place order."""
        req.position_effect = validate_execution_semantics(
            self.get_capabilities(),
            req.position_effect,
            req.reduce_only,
        )
        existing_broker_order_id = self.client_to_broker_order_ids.get(
            req.client_order_id
        )
        if (
            self.enforce_client_order_id_uniqueness
            and existing_broker_order_id is not None
        ):
            existing = self.orders.get(existing_broker_order_id)
            if existing is not None and existing.status in (
                UnifiedOrderStatus.NEW,
                UnifiedOrderStatus.SUBMITTED,
                UnifiedOrderStatus.PARTIALLY_FILLED,
            ):
                return existing_broker_order_id
            if existing is not None and self._is_terminal_status(existing.status):
                self._unlink_order_mapping(
                    client_order_id=req.client_order_id,
                    broker_order_id=existing_broker_order_id,
                )
        now_ns = time.time_ns()
        self._order_seq += 1
        broker_order_id = f"miniqmt-{req.client_order_id}-{self._order_seq}"
        snapshot = UnifiedOrderSnapshot(
            client_order_id=req.client_order_id,
            broker_order_id=broker_order_id,
            symbol=req.symbol,
            status=UnifiedOrderStatus.SUBMITTED,
            timestamp_ns=now_ns,
            position_effect=req.position_effect,
        )
        self.orders[broker_order_id] = snapshot
        self.client_to_broker_order_ids[req.client_order_id] = broker_order_id
        self.broker_to_client_order_ids[broker_order_id] = req.client_order_id
        self._emit_order(snapshot)
        report = UnifiedExecutionReport(
            broker_order_id=broker_order_id,
            client_order_id=req.client_order_id,
            status=UnifiedOrderStatus.SUBMITTED,
            symbol=req.symbol,
            timestamp_ns=now_ns,
            position_effect=req.position_effect,
        )
        self._emit_execution_report(report)
        return broker_order_id

    def cancel_order(self, broker_order_id: str) -> None:
        """Cancel order."""
        order = self.orders.get(broker_order_id)
        if order is not None:
            order.status = UnifiedOrderStatus.CANCELLED
            order.timestamp_ns = time.time_ns()
            self._emit_order(order)
            report = UnifiedExecutionReport(
                broker_order_id=order.broker_order_id,
                client_order_id=order.client_order_id,
                status=order.status,
                symbol=order.symbol,
                filled_quantity=order.filled_quantity,
                avg_fill_price=order.avg_fill_price,
                reject_reason=order.reject_reason,
                timestamp_ns=order.timestamp_ns,
                position_effect=order.position_effect,
            )
            self._emit_execution_report(report)
            self._cleanup_terminal_order_mapping(order)

    def query_order(self, broker_order_id: str) -> UnifiedOrderSnapshot | None:
        """Query order."""
        return self.orders.get(broker_order_id)

    def query_trades(self, since: int | None = None) -> list[UnifiedTrade]:
        """Query trades."""
        if since is None:
            return list(self.trades)
        return [t for t in self.trades if t.timestamp_ns >= since]

    def query_account(self) -> UnifiedAccount | None:
        """Query account."""
        return UnifiedAccount(
            account_id=self.kwargs.get("account_id", "miniqmt"),
            equity=float(self.kwargs.get("equity", 0.0)),
            cash=float(self.kwargs.get("cash", 0.0)),
            available_cash=float(self.kwargs.get("available_cash", 0.0)),
            timestamp_ns=time.time_ns(),
        )

    def query_positions(self) -> list[UnifiedPosition]:
        """Query positions."""
        return []

    def on_order(self, callback: Callable[[UnifiedOrderSnapshot], None]) -> None:
        """Register order callback."""
        self.order_callback = callback

    def on_trade(self, callback: Callable[[UnifiedTrade], None]) -> None:
        """Register trade callback."""
        self.trade_callback = callback

    def on_execution_report(
        self, callback: Callable[[UnifiedExecutionReport], None]
    ) -> None:
        """Register execution report callback."""
        self.execution_callback = callback

    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        """Sync open orders."""
        open_statuses = (
            UnifiedOrderStatus.NEW,
            UnifiedOrderStatus.SUBMITTED,
            UnifiedOrderStatus.PARTIALLY_FILLED,
        )
        return [
            order for order in self.orders.values() if order.status in open_statuses
        ]

    def sync_today_trades(self) -> list[UnifiedTrade]:
        """Sync today's trades."""
        return list(self.trades)

    def heartbeat(self) -> bool:
        """Heartbeat check."""
        return self.connected

    def get_capabilities(self) -> BrokerCapability:
        """Return broker capability matrix."""
        return BrokerCapability(
            broker_name="miniqmt",
            broker_live=True,
            client_order_id=True,
            order_type=True,
            time_in_force_str=True,
            position_effect=False,
            reduce_only=False,
            position_details=False,
            supported_position_effects=("auto",),
        )

    def ingest_order_event(self, payload: dict[str, Any]) -> UnifiedOrderSnapshot:
        """Map and consume broker order event."""
        snapshot = self.mapper.map_order_event(payload)
        self.orders[snapshot.broker_order_id] = snapshot
        self._sync_order_mapping(
            client_order_id=snapshot.client_order_id,
            broker_order_id=snapshot.broker_order_id,
        )
        self._emit_order(snapshot)
        report = self.mapper.map_execution_report(payload)
        self._emit_execution_report(report)
        self._cleanup_terminal_order_mapping(snapshot)
        return snapshot

    def ingest_trade_event(self, payload: dict[str, Any]) -> UnifiedTrade:
        """Map and consume broker trade event."""
        trade = self.mapper.map_trade_event(payload)
        self._sync_order_mapping(
            client_order_id=trade.client_order_id,
            broker_order_id=trade.broker_order_id,
        )
        self.trades.append(trade)
        self._emit_trade(trade)
        return trade

    def start(self) -> None:
        """Start trader gateway event loop."""
        self.connect()

    def _emit_order(self, order: UnifiedOrderSnapshot) -> None:
        if self.order_callback is not None:
            self.order_callback(order)

    def _emit_trade(self, trade: UnifiedTrade) -> None:
        if self.trade_callback is not None:
            self.trade_callback(trade)

    def _emit_execution_report(self, report: UnifiedExecutionReport) -> None:
        if self.execution_callback is not None:
            self.execution_callback(report)

    def _sync_order_mapping(self, client_order_id: str, broker_order_id: str) -> None:
        if client_order_id and broker_order_id:
            self.client_to_broker_order_ids[client_order_id] = broker_order_id
            self.broker_to_client_order_ids[broker_order_id] = client_order_id

    def _unlink_order_mapping(self, client_order_id: str, broker_order_id: str) -> None:
        if client_order_id:
            self.client_to_broker_order_ids.pop(client_order_id, None)
        if broker_order_id:
            self.broker_to_client_order_ids.pop(broker_order_id, None)

    def _cleanup_terminal_order_mapping(self, snapshot: UnifiedOrderSnapshot) -> None:
        if self._is_terminal_status(snapshot.status):
            self._unlink_order_mapping(
                client_order_id=snapshot.client_order_id,
                broker_order_id=snapshot.broker_order_id,
            )

    def _is_terminal_status(self, status: UnifiedOrderStatus) -> bool:
        return status in (
            UnifiedOrderStatus.FILLED,
            UnifiedOrderStatus.CANCELLED,
            UnifiedOrderStatus.REJECTED,
            UnifiedOrderStatus.EXPIRED,
        )
