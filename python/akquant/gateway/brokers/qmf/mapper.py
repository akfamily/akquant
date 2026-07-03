"""QMF 网关字段映射（纯函数，无 IO）."""

from __future__ import annotations

from ...broker_models import (
    UnifiedErrorType,
    UnifiedOrderRequest,
    UnifiedOrderStatus,
)

EXCHANGE_BY_SUFFIX = {"SH": "1", "SZ": "2"}
SUFFIX_BY_EXCHANGE = {v: k for k, v in EXCHANGE_BY_SUFFIX.items()}
SIDE_TO_ENTRUST_BS = {"buy": "1", "sell": "2"}
ORDER_TYPE_TO_ENTRUST_PROP = {"limit": "0"}
ENTRUST_STATUS_MAP = {
    "0": UnifiedOrderStatus.NEW,
    "1": UnifiedOrderStatus.SUBMITTED,
    "2": UnifiedOrderStatus.PARTIALLY_FILLED,
    "6": UnifiedOrderStatus.CANCELLED,
    "8": UnifiedOrderStatus.FILLED,
}
_RISK_KEYWORDS = ("风控", "风险", "限制", "禁止")
_RETRYABLE_KEYWORDS = ("连接", "超时", "网络", "繁忙", "重试")


def split_symbol(symbol: str) -> tuple[str, str]:
    """'600000.SH' -> ('1', '600000')."""
    text = str(symbol).strip()
    if "." not in text:
        raise ValueError(f"symbol 需形如 CODE.SH/CODE.SZ，收到: {symbol!r}")
    code, suffix = text.rsplit(".", 1)
    exchange_type = EXCHANGE_BY_SUFFIX.get(suffix.upper())
    if exchange_type is None:
        raise ValueError(f"不支持的交易所后缀: {suffix!r}")
    return exchange_type, code


def join_symbol(exchange_type: str, stock_code: str) -> str:
    """('1', '600000') -> '600000.SH'."""
    suffix = SUFFIX_BY_EXCHANGE.get(str(exchange_type).strip())
    if suffix is None:
        raise ValueError(f"不支持的 exchange_type: {exchange_type!r}")
    return f"{str(stock_code).strip()}.{suffix}"


def _format_number(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"


def build_order_payload(req: UnifiedOrderRequest) -> dict[str, str]:
    """UnifiedOrderRequest -> chibi_quant OrderRequest 字段（不含 fund_account）."""
    exchange_type, stock_code = split_symbol(req.symbol)
    side_key = str(req.side).strip().lower()
    entrust_bs = SIDE_TO_ENTRUST_BS.get(side_key)
    if entrust_bs is None:
        raise ValueError(f"不支持的 side: {req.side!r}")
    order_type_key = str(req.order_type).strip().lower()
    entrust_prop = ORDER_TYPE_TO_ENTRUST_PROP.get(order_type_key)
    if entrust_prop is None:
        raise ValueError(
            f"Phase 1 仅支持 Limit 委托，收到 order_type={req.order_type!r}"
        )
    if req.price is None:
        raise ValueError("Limit 委托必须提供 price")
    return {
        "exchange_type": exchange_type,
        "stock_code": stock_code,
        "entrust_bs": entrust_bs,
        "entrust_prop": entrust_prop,
        "entrust_price": _format_number(req.price),
        "entrust_amount": _format_number(req.quantity),
    }


def map_order_status(entrust_status: str, error_no: str = "0") -> UnifiedOrderStatus:
    """柜台 entrust_status/error_no -> UnifiedOrderStatus."""
    if str(error_no).strip() not in ("", "0"):
        return UnifiedOrderStatus.REJECTED
    return ENTRUST_STATUS_MAP.get(
        str(entrust_status).strip(), UnifiedOrderStatus.SUBMITTED
    )


def classify_error(error_no: str, error_info: str) -> UnifiedErrorType:
    """error_no/error_info -> UnifiedErrorType."""
    text = str(error_info or "")
    if any(k in text for k in _RISK_KEYWORDS):
        return UnifiedErrorType.RISK_REJECTED
    if str(error_no).strip() in ("", "0"):
        return UnifiedErrorType.RETRYABLE
    if any(k in text for k in _RETRYABLE_KEYWORDS):
        return UnifiedErrorType.RETRYABLE
    return UnifiedErrorType.NON_RETRYABLE
