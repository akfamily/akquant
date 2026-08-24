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
from .native import CTPMarketGateway, CTPTraderGateway


class CTPMarketAdapter:
    """CTP market adapter implementing unified market gateway protocol."""

    def __init__(
        self,
        feed: DataFeed,
        front_url: str,
        symbols: Sequence[str],
        use_aggregator: bool = True,
        emit_ticks: bool | None = None,
        emit_bars: bool | None = None,
    ) -> None:
        """Initialize CTP market adapter.

        ``emit_ticks``/``emit_bars`` are thin pass-throughs to
        :class:`CTPMarketGateway` — this adapter does no fallback logic of its
        own, the per-parameter ``use_aggregator`` alias resolution and the
        both-False guard live entirely in the gateway (see
        ``ctp/native.py::CTPMarketGateway.__init__``). Keeping the adapter a
        dumb pass-through avoids having two places that could disagree on the
        fallback semantics.
        """
        self.front_url = front_url
        self.symbols = list(symbols)
        self.gateway = CTPMarketGateway(
            feed=feed,
            front_url=front_url,
            symbols=list(symbols),
            use_aggregator=use_aggregator,
            emit_ticks=emit_ticks,
            emit_bars=emit_bars,
        )
        self.tick_callback: Callable[[dict[str, Any]], None] | None = None
        self.bar_callback: Callable[[dict[str, Any]], None] | None = None

    def connect(self) -> None:
        """Connect and start market stream."""
        self.gateway.start()

    def disconnect(self) -> None:
        """Disconnect market stream."""
        if hasattr(self.gateway, "api") and self.gateway.api:
            self.gateway.api.Release()

    def subscribe(self, symbols: Sequence[str]) -> None:
        """Update subscribed symbols."""
        self.symbols = list(symbols)
        self.gateway.symbols = list(symbols)

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        """Remove symbols from subscription list."""
        removed = set(symbols)
        self.symbols = [s for s in self.symbols if s not in removed]
        self.gateway.symbols = list(self.symbols)

    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register tick callback."""
        self.tick_callback = callback

    def on_bar(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register bar callback."""
        self.bar_callback = callback

    def start(self) -> None:
        """Start market adapter."""
        self.connect()


class CTPTraderAdapter:
    """CTP trader adapter implementing unified trader gateway protocol."""

    def __init__(
        self,
        front_url: str,
        broker_id: str = "9999",
        user_id: str = "",
        password: str = "",
        auth_code: str = "0000000000000000",
        app_id: str = "simnow_client_test",
        execution_semantics_mode: str = "strict",
    ) -> None:
        """Initialize CTP trader adapter."""
        normalized_mode = str(execution_semantics_mode).strip().lower()
        if normalized_mode not in {"strict", "compatible"}:
            raise ValueError(
                "execution_semantics_mode must be 'strict' or 'compatible'"
            )
        self.execution_semantics_mode = normalized_mode
        self.mapper: BrokerEventMapper = create_default_mapper()
        self.order_callback: Callable[[UnifiedOrderSnapshot], None] | None = None
        self.trade_callback: Callable[[UnifiedTrade], None] | None = None
        self.execution_callback: Callable[[UnifiedExecutionReport], None] | None = None
        self.orders: dict[str, UnifiedOrderSnapshot] = {}
        self.trades: list[UnifiedTrade] = []
        self.trade_ids: set[str] = set()
        self.positions: dict[tuple[str, str], UnifiedPosition] = {}
        self.account: UnifiedAccount | None = None
        self.client_to_broker_order_ids: dict[str, str] = {}
        self.broker_to_client_order_ids: dict[str, str] = {}
        self.order_ref_to_client_order_ids: dict[str, str] = {}
        self.pending_reject_reasons: dict[str, str] = {}
        self._order_seq = 0
        self.gateway = CTPTraderGateway(
            front_url=front_url,
            broker_id=broker_id,
            user_id=user_id,
            password=password,
            auth_code=auth_code,
            app_id=app_id,
        )
        if hasattr(self.gateway, "set_order_handler"):
            self.gateway.set_order_handler(self._handle_native_order_event)
        if hasattr(self.gateway, "set_trade_handler"):
            self.gateway.set_trade_handler(self._handle_native_trade_event)
        if hasattr(self.gateway, "set_error_handler"):
            self.gateway.set_error_handler(self._handle_native_error_event)

    def connect(self) -> None:
        """Connect and start trader stream."""
        self.gateway.start()

    def disconnect(self) -> None:
        """Disconnect trader stream."""
        if hasattr(self.gateway, "api") and self.gateway.api:
            self.gateway.api.Release()

    def place_order(self, req: UnifiedOrderRequest) -> str:
        """Place order through CTP trader channel."""
        req.position_effect = validate_execution_semantics(
            self.get_capabilities(),
            req.position_effect,
            req.reduce_only,
        )
        existing_broker_order_id = self.client_to_broker_order_ids.get(
            req.client_order_id
        )
        if existing_broker_order_id is not None:
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
        if not self.heartbeat():
            raise RuntimeError("CTP trader is not connected")
        native_result = self.gateway.insert_order(
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            quantity=req.quantity,
            price=req.price,
            order_type=req.order_type,
            time_in_force=req.time_in_force,
            position_effect=req.position_effect,
        )
        self._order_seq += 1
        now_ns = int(native_result.get("timestamp_ns", time.time_ns()))
        broker_order_id = str(
            native_result.get(
                "broker_order_id", f"ctp-{req.client_order_id}-{self._order_seq}"
            )
        )
        order_ref = str(native_result.get("order_ref", "")).strip()
        if order_ref:
            self.order_ref_to_client_order_ids[order_ref] = req.client_order_id
        snapshot = UnifiedOrderSnapshot(
            client_order_id=req.client_order_id,
            broker_order_id=broker_order_id,
            symbol=req.symbol,
            status=UnifiedOrderStatus.SUBMITTED,
            timestamp_ns=now_ns,
            position_effect=req.position_effect,
        )
        self.orders[broker_order_id] = snapshot
        self._sync_order_mapping(req.client_order_id, broker_order_id)
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
        """Cancel order through CTP trader channel."""
        self.gateway.cancel_order(broker_order_id)
        if self.execution_semantics_mode != "compatible":
            return
        order = self.orders.get(broker_order_id)
        if order is None:
            return
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
        """Query order status from broker."""
        return self.orders.get(broker_order_id)

    def query_trades(self, since: int | None = None) -> list[UnifiedTrade]:
        """Query trade fills from broker."""
        if since is None:
            return list(self.trades)
        return [t for t in self.trades if t.timestamp_ns >= since]

    def query_account(self) -> UnifiedAccount | None:
        """Query account snapshot from broker."""
        query_account = getattr(self.gateway, "query_account", None)
        if not callable(query_account):
            return self.account
        raw_account = query_account()
        if raw_account is None:
            return self.account
        self.account = self._map_native_account_payload(raw_account)
        return self.account

    def query_positions(self) -> list[UnifiedPosition]:
        """Query position snapshots from broker."""
        query_positions = getattr(self.gateway, "query_positions", None)
        if not callable(query_positions):
            return list(self.positions.values())
        raw_positions = query_positions()
        mapped_positions = [
            self._map_native_position_payload(payload) for payload in raw_positions
        ]
        self.positions = {
            self._position_cache_key(position.symbol, position.direction): position
            for position in mapped_positions
        }
        return mapped_positions

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
        """Sync open order snapshots from broker."""
        open_statuses = (
            UnifiedOrderStatus.NEW,
            UnifiedOrderStatus.SUBMITTED,
            UnifiedOrderStatus.PARTIALLY_FILLED,
        )
        return [
            order for order in self.orders.values() if order.status in open_statuses
        ]

    def sync_today_trades(self) -> list[UnifiedTrade]:
        """Sync today's trade fills from broker."""
        query_trades_today = getattr(self.gateway, "query_trades_today", None)
        if not callable(query_trades_today):
            return list(self.trades)
        raw_trades = query_trades_today()
        mapped_trades = [
            self.mapper.map_trade_event(dict(payload)) for payload in raw_trades
        ]
        for trade in mapped_trades:
            self._cache_trade(trade)
        return mapped_trades

    def heartbeat(self) -> bool:
        """Return whether trader connection is alive."""
        can_trade = getattr(self.gateway, "can_trade", None)
        if callable(can_trade):
            return bool(can_trade())
        return bool(getattr(self.gateway, "connected", False))

    def get_capabilities(self) -> BrokerCapability:
        """Return broker capability matrix."""
        return BrokerCapability(
            broker_name="ctp",
            broker_live=True,
            client_order_id=True,
            order_type=True,
            time_in_force_str=True,
            position_effect=True,
            reduce_only=False,
            position_details=True,
            supports_short_sell=True,
            supported_position_effects=(
                "auto",
                "open",
                "close",
                "close_today",
                "close_yesterday",
            ),
        )

    def start(self) -> None:
        """Start trader adapter."""
        self.connect()

    def ingest_order_event(self, payload: dict[str, Any]) -> UnifiedOrderSnapshot:
        """Map and consume broker order event."""
        snapshot = self.mapper.map_order_event(payload)
        self.orders[snapshot.broker_order_id] = snapshot
        self._sync_order_mapping(
            client_order_id=snapshot.client_order_id,
            broker_order_id=snapshot.broker_order_id,
        )
        if self.order_callback is not None:
            self.order_callback(snapshot)
        report = self.mapper.map_execution_report(payload)
        if self.execution_callback is not None:
            self.execution_callback(report)
        self._cleanup_terminal_order_mapping(snapshot)
        return snapshot

    def ingest_trade_event(self, payload: dict[str, Any]) -> UnifiedTrade:
        """Map and consume broker trade event."""
        trade = self.mapper.map_trade_event(payload)
        self._sync_order_mapping(
            client_order_id=trade.client_order_id,
            broker_order_id=trade.broker_order_id,
        )
        self._cache_trade(trade)
        if self.trade_callback is not None:
            self.trade_callback(trade)
        return trade

    def _emit_order(self, order: UnifiedOrderSnapshot) -> None:
        if self.order_callback is not None:
            self.order_callback(order)

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

    def _map_native_position_payload(self, payload: Any) -> UnifiedPosition:
        if isinstance(payload, UnifiedPosition):
            return payload

        def _field(name: str, default: Any = "") -> Any:
            if isinstance(payload, dict):
                return payload.get(name, default)
            return getattr(payload, name, default)

        direction = str(
            _field("direction", _field("side", _field("posi_direction", "")))
        ).strip()
        quantity = float(_field("quantity", _field("position", 0.0)) or 0.0)
        available_quantity = float(
            _field("available_quantity", _field("available", abs(quantity))) or 0.0
        )
        today_quantity = float(
            _field("today_quantity", _field("today_position", 0.0)) or 0.0
        )
        yesterday_quantity = float(
            _field("yesterday_quantity", _field("yd_position", 0.0)) or 0.0
        )
        available_today_quantity = float(
            _field(
                "available_today_quantity", _field("today_available", today_quantity)
            )
            or 0.0
        )
        available_yesterday_quantity = float(
            _field(
                "available_yesterday_quantity",
                _field("yesterday_available", yesterday_quantity),
            )
            or 0.0
        )
        if direction.lower() in {"sell", "short", "1"} and quantity > 0:
            quantity = -quantity
        normalized_direction = self._normalize_position_direction(direction, quantity)
        return UnifiedPosition(
            symbol=str(
                _field("symbol", _field("instrument_id", _field("InstrumentID", "")))
            ),
            quantity=quantity,
            available_quantity=available_quantity,
            direction=normalized_direction,
            today_quantity=today_quantity,
            yesterday_quantity=yesterday_quantity,
            available_today_quantity=available_today_quantity,
            available_yesterday_quantity=available_yesterday_quantity,
            avg_price=float(_field("avg_price", _field("price", 0.0)) or 0.0),
            timestamp_ns=int(_field("timestamp_ns", 0) or 0),
        )

    def _map_native_account_payload(self, payload: Any) -> UnifiedAccount:
        if isinstance(payload, UnifiedAccount):
            return payload

        def _field(name: str, default: Any = "") -> Any:
            if isinstance(payload, dict):
                return payload.get(name, default)
            return getattr(payload, name, default)

        gateway_user_id = str(getattr(self.gateway, "user_id", ""))
        return UnifiedAccount(
            account_id=str(_field("account_id", _field("AccountID", gateway_user_id))),
            equity=float(_field("equity", _field("Balance", 0.0)) or 0.0),
            cash=float(_field("cash", _field("Balance", 0.0)) or 0.0),
            available_cash=float(
                _field("available_cash", _field("Available", 0.0)) or 0.0
            ),
            timestamp_ns=int(_field("timestamp_ns", 0) or 0),
        )

    def _position_cache_key(self, symbol: str, direction: str) -> tuple[str, str]:
        return (str(symbol).strip(), self._normalize_position_direction(direction))

    def _normalize_position_direction(
        self, direction: str, quantity: float = 0.0
    ) -> str:
        normalized = str(direction).strip().lower()
        if normalized in {"buy", "long", "2"}:
            return "Buy"
        if normalized in {"sell", "short", "1", "3"}:
            return "Sell"
        if quantity > 0:
            return "Buy"
        if quantity < 0:
            return "Sell"
        return ""

    def _cache_trade(self, trade: UnifiedTrade) -> None:
        trade_id = str(trade.trade_id).strip()
        if trade_id and trade_id in self.trade_ids:
            return
        if trade_id:
            self.trade_ids.add(trade_id)
        self.trades.append(trade)

    def _handle_native_order_event(self, payload: dict[str, Any]) -> None:
        data = dict(payload)
        order_ref = str(data.get("order_ref", "")).strip()
        client_order_id = str(data.get("client_order_id", "")).strip()
        if not client_order_id and order_ref:
            client_order_id = self.order_ref_to_client_order_ids.get(order_ref, "")
            if client_order_id:
                data["client_order_id"] = client_order_id
        broker_order_id = str(data.get("broker_order_id", "")).strip()
        if not broker_order_id:
            return
        pending_reject_reason = self.pending_reject_reasons.pop(
            broker_order_id, ""
        ).strip()
        if pending_reject_reason and not str(data.get("reject_reason", "")).strip():
            data["reject_reason"] = pending_reject_reason
        existing_order = self.orders.get(broker_order_id)
        if (
            existing_order is not None
            and not str(data.get("position_effect", "")).strip()
        ):
            data["position_effect"] = existing_order.position_effect
        self.ingest_order_event(data)

    def _handle_native_trade_event(self, payload: dict[str, Any]) -> None:
        data = dict(payload)
        order_ref = str(data.get("order_ref", "")).strip()
        client_order_id = str(data.get("client_order_id", "")).strip()
        if not client_order_id and order_ref:
            client_order_id = self.order_ref_to_client_order_ids.get(order_ref, "")
            if client_order_id:
                data["client_order_id"] = client_order_id
        broker_order_id = str(data.get("broker_order_id", "")).strip()
        if not broker_order_id:
            return
        existing_order = self.orders.get(broker_order_id)
        if (
            existing_order is not None
            and not str(data.get("position_effect", "")).strip()
        ):
            data["position_effect"] = existing_order.position_effect
        self.ingest_trade_event(data)

    def _handle_native_error_event(self, payload: dict[str, Any]) -> None:
        data = dict(payload)
        broker_order_id = str(data.get("broker_order_id", "")).strip()
        if not broker_order_id:
            return
        existing_order = self.orders.get(broker_order_id)
        if (
            existing_order is not None
            and not str(data.get("position_effect", "")).strip()
        ):
            data["position_effect"] = existing_order.position_effect
        if self.execution_semantics_mode == "compatible":
            self.ingest_order_event(
                {
                    "client_order_id": str(data.get("client_order_id", "")).strip(),
                    "broker_order_id": broker_order_id,
                    "symbol": str(data.get("symbol", "")).strip(),
                    "status": "rejected",
                    "filled_quantity": 0.0,
                    "avg_fill_price": 0.0,
                    "reject_reason": str(data.get("error_message", "")).strip(),
                    "timestamp_ns": int(data.get("timestamp_ns", time.time_ns())),
                    "position_effect": str(data.get("position_effect", "auto")),
                }
            )
            return
        reject_reason = str(data.get("error_message", "")).strip()
        if reject_reason:
            self.pending_reject_reasons[broker_order_id] = reject_reason
