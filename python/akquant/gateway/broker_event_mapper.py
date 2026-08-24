from dataclasses import dataclass
from typing import Any

from .broker_models import (
    UnifiedErrorType,
    UnifiedExecutionReport,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedTrade,
)


@dataclass
class BrokerEventMapper:
    """Map broker-specific status and errors to unified enums."""

    status_map: dict[str, UnifiedOrderStatus]
    retryable_error_codes: set[str]
    risk_error_codes: set[str]

    def map_order_status(self, raw_status: Any) -> UnifiedOrderStatus:
        """Map broker raw status to unified status."""
        key = str(raw_status).strip().lower()
        return self.status_map.get(key, UnifiedOrderStatus.SUBMITTED)

    def classify_error(self, code: Any, message: str = "") -> UnifiedErrorType:
        """Classify broker errors by retry and risk semantics."""
        code_key = str(code).strip().lower()
        if code_key in self.risk_error_codes:
            return UnifiedErrorType.RISK_REJECTED
        if code_key in self.retryable_error_codes:
            return UnifiedErrorType.RETRYABLE

        msg = message.lower()
        if "risk" in msg or "风控" in msg:
            return UnifiedErrorType.RISK_REJECTED
        if "timeout" in msg or "network" in msg or "连接" in msg:
            return UnifiedErrorType.RETRYABLE
        return UnifiedErrorType.NON_RETRYABLE

    def map_order_event(self, payload: dict[str, Any]) -> UnifiedOrderSnapshot:
        """Map broker order payload to unified order snapshot."""
        status = self.map_order_status(payload.get("status", "submitted"))
        return UnifiedOrderSnapshot(
            client_order_id=str(payload.get("client_order_id", "")),
            broker_order_id=str(payload.get("broker_order_id", "")),
            symbol=str(payload.get("symbol", "")),
            status=status,
            filled_quantity=float(payload.get("filled_quantity", 0.0)),
            avg_fill_price=float(payload.get("avg_fill_price", 0.0)),
            reject_reason=str(payload.get("reject_reason", "")),
            timestamp_ns=int(payload.get("timestamp_ns", 0)),
            position_effect=str(payload.get("position_effect", "auto")),
        )

    def map_trade_event(self, payload: dict[str, Any]) -> UnifiedTrade:
        """Map broker trade payload to unified trade record."""
        return UnifiedTrade(
            trade_id=str(payload.get("trade_id", "")),
            broker_order_id=str(payload.get("broker_order_id", "")),
            client_order_id=str(payload.get("client_order_id", "")),
            symbol=str(payload.get("symbol", "")),
            side=str(payload.get("side", "")),
            quantity=float(payload.get("quantity", 0.0)),
            price=float(payload.get("price", 0.0)),
            timestamp_ns=int(payload.get("timestamp_ns", 0)),
            position_effect=str(payload.get("position_effect", "auto")),
        )

    def map_execution_report(self, payload: dict[str, Any]) -> UnifiedExecutionReport:
        """Map broker payload to unified execution report."""
        status = self.map_order_status(payload.get("status", "submitted"))
        return UnifiedExecutionReport(
            broker_order_id=str(payload.get("broker_order_id", "")),
            client_order_id=str(payload.get("client_order_id", "")),
            status=status,
            symbol=str(payload.get("symbol", "")),
            filled_quantity=float(payload.get("filled_quantity", 0.0)),
            avg_fill_price=float(payload.get("avg_fill_price", 0.0)),
            reject_reason=str(payload.get("reject_reason", "")),
            timestamp_ns=int(payload.get("timestamp_ns", 0)),
            position_effect=str(payload.get("position_effect", "auto")),
        )


DEFAULT_STATUS_MAP = {
    "new": UnifiedOrderStatus.NEW,
    "submitted": UnifiedOrderStatus.SUBMITTED,
    "accepted": UnifiedOrderStatus.SUBMITTED,
    "partially_filled": UnifiedOrderStatus.PARTIALLY_FILLED,
    "partfilled": UnifiedOrderStatus.PARTIALLY_FILLED,
    "partial": UnifiedOrderStatus.PARTIALLY_FILLED,
    "filled": UnifiedOrderStatus.FILLED,
    "cancelled": UnifiedOrderStatus.CANCELLED,
    "canceled": UnifiedOrderStatus.CANCELLED,
    "rejected": UnifiedOrderStatus.REJECTED,
    "expired": UnifiedOrderStatus.EXPIRED,
    "alltraded": UnifiedOrderStatus.FILLED,
    "parttradedqueueing": UnifiedOrderStatus.PARTIALLY_FILLED,
    "parttradednotqueueing": UnifiedOrderStatus.PARTIALLY_FILLED,
    "notradequeueing": UnifiedOrderStatus.SUBMITTED,
    "notradenotqueueing": UnifiedOrderStatus.SUBMITTED,
    "unknown": UnifiedOrderStatus.SUBMITTED,
    # 以下单字符码值是 **CTP 口径**(THOST_FTDC_OST_*): 0=全部成交、1/2=部分成交、
    # 3/4=未成交、5=已撤单、a=未知、b/c=未触发/已触发。
    # 注意与恒生「委托状态」(数据词典 1203)的数字码**完全冲突**——那边 0=未报、
    # 5=部撤、8=已成。接恒生柜台的 broker 必须自带映射表(中间件插件即如此),
    # 不要复用这张表, 否则「未报」会被读成「已成交」。
    "0": UnifiedOrderStatus.FILLED,
    "1": UnifiedOrderStatus.PARTIALLY_FILLED,
    "2": UnifiedOrderStatus.PARTIALLY_FILLED,
    "3": UnifiedOrderStatus.SUBMITTED,
    "4": UnifiedOrderStatus.SUBMITTED,
    "5": UnifiedOrderStatus.CANCELLED,
    "a": UnifiedOrderStatus.SUBMITTED,
    "b": UnifiedOrderStatus.SUBMITTED,
    "c": UnifiedOrderStatus.SUBMITTED,
}


def create_default_mapper() -> BrokerEventMapper:
    """Create default status and error mapper."""
    return BrokerEventMapper(
        status_map=DEFAULT_STATUS_MAP,
        retryable_error_codes={"1001", "1002", "timeout", "network"},
        risk_error_codes={"2001", "2002", "risk"},
    )
