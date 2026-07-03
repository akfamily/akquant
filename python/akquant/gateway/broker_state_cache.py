"""broker_live 状态缓存：查柜台 + 事件失效，柜台为唯一真相."""

from __future__ import annotations

from typing import Any

from ..log import get_logger

logger = get_logger("gateway.live")


class BrokerStateCache:
    """缓存 trader_gateway 的持仓/资金/委托查询；on_trade/on_order 时 invalidate."""

    def __init__(self, trader_gateway: Any) -> None:
        """Bind the trader gateway; start with empty caches."""
        self._gw = trader_gateway
        self._positions: dict[str, float] = {}
        self._available: dict[str, float] = {}
        self._positions_loaded = False
        self._account: Any = None
        self._account_loaded = False
        self._open_orders: list[Any] = []
        self._open_orders_loaded = False

    def invalidate(self) -> None:
        """Mark all cached snapshots stale (next read re-queries the broker)."""
        self._positions_loaded = False
        self._account_loaded = False
        self._open_orders_loaded = False

    def _load_positions(self) -> None:
        try:
            rows = list(self._gw.query_positions())
        except Exception:  # noqa: BLE001 柜台查询失败 → 保留上次缓存, 不抛
            logger.exception("broker_state_cache.query_positions failed")
            return
        self._positions = {str(p.symbol): float(p.quantity) for p in rows}
        self._available = {str(p.symbol): float(p.available_quantity) for p in rows}
        self._positions_loaded = True

    def positions(self) -> dict[str, float]:
        """Return {symbol: quantity} from the broker (cached until invalidate)."""
        if not self._positions_loaded:
            self._load_positions()
        return dict(self._positions)

    def available_positions(self) -> dict[str, float]:
        """Return {symbol: available_quantity} from the broker (cached)."""
        if not self._positions_loaded:
            self._load_positions()
        return dict(self._available)

    def account(self) -> Any:
        """Return the broker UnifiedAccount (cached; last value kept on error)."""
        if not self._account_loaded:
            try:
                self._account = self._gw.query_account()
                self._account_loaded = True
            except Exception:  # noqa: BLE001 保留上次缓存, 不抛
                logger.exception("broker_state_cache.query_account failed")
        return self._account

    def open_orders(self) -> list[Any]:
        """Return open order snapshots from the broker (cached)."""
        if not self._open_orders_loaded:
            try:
                self._open_orders = list(self._gw.sync_open_orders())
                self._open_orders_loaded = True
            except Exception:  # noqa: BLE001 保留上次缓存, 不抛
                logger.exception("broker_state_cache.sync_open_orders failed")
        return list(self._open_orders)
