"""中间件标准字段 <-> akquant Unified 模型的映射（纯函数，无 IO）.

instrument_id 格式与 status 取值集是与中间件的对齐点，集中在本文件，
待中间件团队确认后只改这里。
"""

from __future__ import annotations

from typing import Any

from ...broker_models import (
    UnifiedAccount,
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedPosition,
    UnifiedTrade,
)

# akquant symbol 后缀 <-> 中间件 market 段
_MARKET_BY_SUFFIX = {"SH": "SSE", "SZ": "SZSE"}
_SUFFIX_BY_MARKET = {market: suffix for suffix, market in _MARKET_BY_SUFFIX.items()}

# 中间件 side（小写）<-> akquant 规范（首字母大写）
_SIDE_IN = {"buy": "Buy", "sell": "Sell"}

# 中间件 order status -> UnifiedOrderStatus（未知安全落到 SUBMITTED）
_STATUS_MAP = {
    "pending": UnifiedOrderStatus.NEW,
    "submitted": UnifiedOrderStatus.SUBMITTED,
    "partially_filled": UnifiedOrderStatus.PARTIALLY_FILLED,
    "filled": UnifiedOrderStatus.FILLED,
    "cancelled": UnifiedOrderStatus.CANCELLED,
    "partially_cancelled": UnifiedOrderStatus.CANCELLED,
    "rejected": UnifiedOrderStatus.REJECTED,
}


def symbol_to_instrument(symbol: str, asset_type: str = "stock") -> str:
    """'600000.SH'+stock -> 'SSE:600000'；option -> 'SSE_OPT:600000'."""
    text = str(symbol).strip()
    if "." not in text:
        raise ValueError(f"symbol 需形如 CODE.SH/CODE.SZ，收到: {symbol!r}")
    code, suffix = text.rsplit(".", 1)
    market = _MARKET_BY_SUFFIX.get(suffix.upper())
    if market is None:
        raise ValueError(f"不支持的交易所后缀: {suffix!r}")
    if str(asset_type).strip().lower() == "option":
        market = f"{market}_OPT"
    return f"{market}:{code}"


def instrument_to_symbol(instrument_id: str) -> str:
    """'SSE:600000' / 'SSE_OPT:10003456' -> '600000.SH' / '10003456.SH'."""
    text = str(instrument_id).strip()
    if ":" not in text:
        return text
    market, code = text.split(":", 1)
    market = market.upper()
    if market.endswith("_OPT"):
        market = market[: -len("_OPT")]
    suffix = _SUFFIX_BY_MARKET.get(market)
    if suffix is None:
        raise ValueError(f"不支持的 market: {market!r}")
    return f"{code}.{suffix}"


def map_status(value: Any) -> UnifiedOrderStatus:
    """中间件 order status -> UnifiedOrderStatus（未知落 SUBMITTED）."""
    key = str(value or "").strip().lower()
    return _STATUS_MAP.get(key, UnifiedOrderStatus.SUBMITTED)


def _offset(position_effect: str) -> str:
    """position_effect -> 中间件 offset（auto 视为 open）."""
    value = str(position_effect or "").strip().lower()
    return "open" if value in ("", "auto") else value


def build_order_body(req: UnifiedOrderRequest) -> dict[str, Any]:
    """UnifiedOrderRequest -> POST /accounts/{id}/orders body."""
    body: dict[str, Any] = {
        "client_order_id": req.client_order_id,
        "instrument_id": symbol_to_instrument(req.symbol, req.asset_type),
        "side": str(req.side).strip().lower(),
        "offset": _offset(req.position_effect),
        "order_type": str(req.order_type).strip().lower(),
        "quantity": req.quantity,
        "time_in_force": req.time_in_force,
        "legs": [],
    }
    if req.price is not None:
        body["price"] = req.price
    if req.extra:
        body["extra"] = dict(req.extra)
    return body


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_order(row: dict[str, Any], client_order_id: str = "") -> UnifiedOrderSnapshot:
    """中间件 Order -> UnifiedOrderSnapshot."""
    return UnifiedOrderSnapshot(
        client_order_id=client_order_id or str(row.get("client_order_id", "")),
        broker_order_id=str(row.get("broker_order_id") or row.get("order_id", "")),
        symbol=instrument_to_symbol(row.get("instrument_id", "")),
        status=map_status(row.get("status")),
        filled_quantity=_to_float(row.get("filled_quantity")),
        avg_fill_price=_to_float(row.get("avg_price")),
        reject_reason=str(row.get("status_msg") or row.get("reject_reason", "")),
    )


def parse_trade(row: dict[str, Any], client_order_id: str = "") -> UnifiedTrade:
    """中间件 trade -> UnifiedTrade（side 归一为 Buy/Sell）."""
    return UnifiedTrade(
        trade_id=str(row.get("trade_id", "")),
        broker_order_id=str(row.get("broker_order_id") or row.get("order_id", "")),
        client_order_id=client_order_id or str(row.get("client_order_id", "")),
        symbol=instrument_to_symbol(row.get("instrument_id", "")),
        side=_SIDE_IN.get(str(row.get("side", "")).strip().lower(), ""),
        quantity=_to_float(row.get("quantity")),
        price=_to_float(row.get("price")),
        timestamp_ns=0,
    )


def parse_position(row: dict[str, Any]) -> UnifiedPosition:
    """中间件 position -> UnifiedPosition."""
    return UnifiedPosition(
        symbol=instrument_to_symbol(row.get("instrument_id", "")),
        quantity=_to_float(row.get("quantity")),
        available_quantity=_to_float(row.get("available_quantity")),
    )


def parse_account(summary: dict[str, Any]) -> UnifiedAccount:
    """中间件 /summary -> UnifiedAccount."""
    return UnifiedAccount(
        account_id=str(summary.get("account_id", "")),
        equity=_to_float(summary.get("net_asset") or summary.get("total_asset")),
        cash=_to_float(summary.get("cash_balance") or summary.get("cash")),
        available_cash=_to_float(summary.get("available")),
    )
