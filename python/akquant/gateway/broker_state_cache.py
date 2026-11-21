"""broker_live 状态缓存：总持仓事件溯源(成交叠 delta)+可用/资金/委托查柜台失效.

总持仓: 启动/恢复整柜台 seed, 会话中成交叠 delta(不重查), invalidate() 全量对账。
可用: 走柜台查询, 成交后失效重查(T+1/T+0 归柜台)。总/可用 loaded 拆开互不覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..log import get_logger

logger = get_logger("gateway.live")


_SHORT_DIRECTIONS = frozenset({"sell", "short"})


_CLOSE_EFFECTS = frozenset({"close", "close_today", "close_yesterday"})
_TERMINAL_STATUSES = frozenset({"filled", "cancelled", "rejected", "expired"})


@dataclass
class PendingOrder:
    """本策略报出、尚未完结的一条委托腿(在途量投影用)."""

    symbol: str
    side: str  # "buy" / "sell"
    remaining: float
    reducing: bool  # 显式平仓语义或 reduce_only


def is_short_row(row: Any) -> bool:
    """持仓行是否为空头: 方向写明时以方向为准, 否则看数量正负."""
    direction = str(getattr(row, "direction", "") or "").strip().lower()
    if direction in _SHORT_DIRECTIONS:
        return True
    if direction in {"buy", "long"}:
        return False
    return float(getattr(row, "quantity", 0.0) or 0.0) < 0


def _net_by_symbol(rows: list[Any], field: str) -> dict[str, float]:
    """把持仓行按标的合并成带符号净数量(多头正, 空头负).

    柜台对空头的写法不统一: 有的给负数, 有的数量恒正、方向写在 ``direction`` 里;
    同一合约锁仓时还会多空各给一行。这里统一成回测的口径: 空头行计负, 同标的
    的行相加。不在这里做的话, 空头会被读成多头, 锁仓的两行会后者覆盖前者。
    """
    net: dict[str, float] = {}
    for row in rows:
        value = float(getattr(row, field, 0.0) or 0.0)
        direction = str(getattr(row, "direction", "") or "").strip().lower()
        if direction in _SHORT_DIRECTIONS:
            value = -abs(value)
        symbol = str(row.symbol)
        net[symbol] = net.get(symbol, 0.0) + value
    return net


class BrokerStateCache:
    """缓存 trader_gateway 状态；总持仓事件溯源, 其余查柜台失效."""

    def __init__(self, trader_gateway: Any) -> None:
        """Bind the trader gateway; start with empty caches."""
        self._gw = trader_gateway
        self._positions: dict[str, float] = {}
        self._available: dict[str, float] = {}
        # 最近一次查到的持仓行(与 _available 同时刷新), 供持仓明细按方向读取
        self._rows: list[Any] = []
        self._total_loaded = False
        self._available_loaded = False
        self._account: Any = None
        self._account_loaded = False
        self._open_orders: list[Any] = []
        # 本策略报出的未完结委托 {broker_order_id: PendingOrder}。只记自己报的:
        # 柜台挂单是账户级的, 含手工单和其他策略的单, 不能拿来投影本策略的仓位。
        self._pending: dict[str, PendingOrder] = {}
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

    def set_account(self, account: Any) -> None:
        """用已查到的账户快照替换缓存(恢复轮询送来的 ``account`` 事件).

        ``None`` 不覆盖。之后的委托/成交回报照旧经 ``invalidate_account`` 让它过期。
        """
        if account is None:
            return
        self._account = account
        self._account_loaded = True

    def invalidate_open_orders(self) -> None:
        """Mark open-orders snapshot stale."""
        self._open_orders_loaded = False

    def track_pending_order(
        self,
        broker_order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        position_effect: str = "auto",
        reduce_only: bool = False,
    ) -> None:
        """登记一条本策略刚报出的委托腿."""
        effect = str(position_effect or "auto").strip().lower()
        self._pending[str(broker_order_id)] = PendingOrder(
            symbol=str(symbol),
            side=str(side).strip().lower(),
            remaining=float(quantity),
            reducing=bool(reduce_only) or effect in _CLOSE_EFFECTS,
        )

    def apply_pending_fill(self, broker_order_id: str, quantity: float) -> None:
        """一笔成交到达: 对应委托的在途量减去成交量, 减完即移除.

        在途量只由成交扣减, 不看委托回报里的已成交量: 成交同时把数量加进持仓,
        两者在同一次回调里完成, 投影持仓不会出现少算的瞬间(少算会导致重复下单)。
        """
        order = self._pending.get(str(broker_order_id))
        if order is None:
            return
        order.remaining -= abs(float(quantity))
        if order.remaining <= 1e-9:
            del self._pending[str(broker_order_id)]

    def release_pending_order(self, broker_order_id: str, status: Any) -> None:
        """委托进入终态(成交完/撤单/拒单/过期)时移除其剩余在途量."""
        text = str(getattr(status, "name", None) or getattr(status, "value", status))
        if text.split(".")[-1].strip().lower() in _TERMINAL_STATUSES:
            self._pending.pop(str(broker_order_id), None)

    def projected_position(self, symbol: str, reducing_only: bool) -> float:
        """持仓叠加本策略在途单的预期效果; 口径同回测 ``project_pending_position``.

        ``reducing_only=True`` 只计平仓/减仓单(可平持仓); 否则全部计入(投影持仓)。
        平仓单最多平到 0, 不会穿过 0 变成反向仓。
        """
        projected = self.positions().get(symbol, 0.0)
        for order in self._pending.values():
            if order.symbol != symbol or order.remaining <= 0:
                continue
            if reducing_only and not order.reducing:
                continue
            if not order.reducing:
                projected += (
                    order.remaining if order.side == "buy" else -order.remaining
                )
            elif order.side == "buy" and projected < 0:
                projected += min(order.remaining, -projected)
            elif order.side == "sell" and projected > 0:
                projected -= min(order.remaining, projected)
        return projected

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
        self._positions = _net_by_symbol(rows, "quantity")
        self._available = _net_by_symbol(rows, "available_quantity")
        self._rows = rows
        self._total_loaded = True
        self._available_loaded = True

    def _load_available(self) -> None:
        try:
            rows = list(self._gw.query_positions())
        except Exception:  # noqa: BLE001 保留上次缓存, 不抛
            logger.exception("broker_state_cache.query_positions failed")
            return
        # 仅刷新可用, 不动事件溯源的总持仓
        self._available = _net_by_symbol(rows, "available_quantity")
        self._rows = rows
        self._available_loaded = True

    def position_rows(self, symbol: str) -> list[Any]:
        """返回某标的最近一次查到的持仓行(每个方向一行), 过期时重查柜台.

        与可用持仓同一份查询: 成交后失效, 下次读取时重查。
        """
        if not self._available_loaded:
            self._load_available()
        return [row for row in self._rows if str(row.symbol) == symbol]

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

    def account_if_loaded(self) -> Any:
        """不查柜台, 返回最近一次查询成功的账户快照; 从未成功过时返回 None.

        成交/委托回报会把账户标记为过期(``invalidate_account``), 但这里仍返回
        上次的值——调用方(每 bar 权益采样)要的是"不触发查询", 不是"最新"。
        柜台断线时它不会每 bar 重查、也不会每 bar 打异常日志。
        """
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
