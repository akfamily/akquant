"""所有 broker 可复用的 TraderGateway 基类：回调、emit、id 反查、默认实现."""

from __future__ import annotations

from typing import Callable

from .broker_models import (
    UnifiedExecutionReport,
    UnifiedOrderSnapshot,
    UnifiedTrade,
)


class TraderGatewayBase:
    """共享管件基类；子类实现 connect/place_order/query_* 等具体方法."""

    def __init__(self) -> None:
        """Initialize callback slots and the broker->client id map."""
        self._on_order: Callable[[UnifiedOrderSnapshot], None] | None = None
        self._on_trade: Callable[[UnifiedTrade], None] | None = None
        self._on_exec: Callable[[UnifiedExecutionReport], None] | None = None
        self._client_id_by_broker: dict[str, str] = {}

    # --- 回调注册 ---
    def on_order(self, callback: Callable[[UnifiedOrderSnapshot], None]) -> None:
        """Register the order-update callback."""
        self._on_order = callback

    def on_trade(self, callback: Callable[[UnifiedTrade], None]) -> None:
        """Register the trade callback."""
        self._on_trade = callback

    def on_execution_report(
        self, callback: Callable[[UnifiedExecutionReport], None]
    ) -> None:
        """Register the execution-report callback."""
        self._on_exec = callback

    # --- None 安全分发 ---
    def _emit_order(self, snapshot: UnifiedOrderSnapshot) -> None:
        """Dispatch an order snapshot to the registered callback, if any."""
        if self._on_order is not None:
            self._on_order(snapshot)

    def _emit_trade(self, trade: UnifiedTrade) -> None:
        """Dispatch a trade to the registered callback, if any."""
        if self._on_trade is not None:
            self._on_trade(trade)

    def _emit_exec(self, report: UnifiedExecutionReport) -> None:
        """Dispatch an execution report to the registered callback, if any."""
        if self._on_exec is not None:
            self._on_exec(report)

    def _emit_exec_from_order(self, snapshot: UnifiedOrderSnapshot) -> None:
        """Derive and emit an execution report from an order snapshot."""
        self._emit_exec(
            UnifiedExecutionReport(
                broker_order_id=snapshot.broker_order_id,
                client_order_id=snapshot.client_order_id,
                status=snapshot.status,
                symbol=snapshot.symbol,
                filled_quantity=snapshot.filled_quantity,
                avg_fill_price=snapshot.avg_fill_price,
                reject_reason=snapshot.reject_reason,
            )
        )

    # --- id 反查表 ---
    def record_broker_order(self, broker_order_id: str, client_order_id: str) -> None:
        """Record the broker_order_id -> client_order_id mapping."""
        if broker_order_id:
            self._client_id_by_broker[str(broker_order_id)] = client_order_id

    def client_order_id_for(self, broker_order_id: str) -> str:
        """Return the client_order_id for a broker id (empty if unknown)."""
        return self._client_id_by_broker.get(str(broker_order_id), "")

    # --- 默认实现（子类可覆盖）---
    def heartbeat(self) -> bool:
        """Default heartbeat: always alive."""
        return True

    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        """Default: nothing to re-sync."""
        return []

    def sync_today_trades(self) -> list[UnifiedTrade]:
        """Default: nothing to re-sync."""
        return []
