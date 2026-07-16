"""broker_live 状态缓存：总持仓事件溯源(成交叠 delta)+可用/资金/委托查柜台失效.

总持仓: 启动/恢复整柜台 seed, 会话中成交叠 delta(不重查), invalidate() 全量对账。
可用: 走柜台查询, 成交后失效重查(T+1/T+0 归柜台)。总/可用 loaded 拆开互不覆盖。
"""

from __future__ import annotations

from typing import Any

from ..log import get_logger

logger = get_logger("gateway.live")


class BrokerStateCache:
    """缓存 trader_gateway 状态；总持仓事件溯源, 其余查柜台失效."""

    def __init__(self, trader_gateway: Any) -> None:
        """Bind the trader gateway; start with empty caches."""
        self._gw = trader_gateway
        self._positions: dict[str, float] = {}
        self._available: dict[str, float] = {}
        self._total_loaded = False
        self._available_loaded = False
        self._account: Any = None
        self._account_loaded = False
        self._open_orders: list[Any] = []
        self._open_orders_loaded = False

    def invalidate(self) -> None:
        """Mark ALL snapshots stale (启动/恢复全量重 seed 对账)."""
        self._total_loaded = False
        self._available_loaded = False
        self._account_loaded = False
        self._open_orders_loaded = False

    def invalidate_available(self) -> None:
        """Mark available-position snapshot stale (下次重查柜台)."""
        self._available_loaded = False

    def invalidate_account(self) -> None:
        """Mark account snapshot stale."""
        self._account_loaded = False

    def invalidate_open_orders(self) -> None:
        """Mark open-orders snapshot stale."""
        self._open_orders_loaded = False

    def apply_fill(self, symbol: str, signed_qty: float) -> None:
        """成交同步叠总持仓 delta(Buy +, Sell -).

        仅在总持仓已 seed 时叠; 未 seed 则 no-op——下次 positions() 从柜台整快照
        seed(已含该笔), 避免"柜台已含 + 又叠 delta"双计。
        """
        if self._total_loaded:
            self._positions[symbol] = self._positions.get(symbol, 0.0) + float(
                signed_qty
            )

    def _load_positions(self) -> None:
        try:
            rows = list(self._gw.query_positions())
        except Exception:  # noqa: BLE001 柜台查询失败 → 保留上次缓存, 不抛
            logger.exception("broker_state_cache.query_positions failed")
            return
        self._positions = {str(p.symbol): float(p.quantity) for p in rows}
        self._available = {str(p.symbol): float(p.available_quantity) for p in rows}
        self._total_loaded = True
        self._available_loaded = True

    def _load_available(self) -> None:
        try:
            rows = list(self._gw.query_positions())
        except Exception:  # noqa: BLE001 保留上次缓存, 不抛
            logger.exception("broker_state_cache.query_positions failed")
            return
        # 仅刷新可用, 不动事件溯源的总持仓
        self._available = {str(p.symbol): float(p.available_quantity) for p in rows}
        self._available_loaded = True

    def positions(self) -> dict[str, float]:
        """Return {symbol: quantity}. 事件溯源总持仓(seed 后成交叠 delta)."""
        if not self._total_loaded:
            self._load_positions()
        return dict(self._positions)

    def available_positions(self) -> dict[str, float]:
        """Return {symbol: available_quantity} from the broker (查柜台失效)."""
        if not self._available_loaded:
            self._load_available()
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
