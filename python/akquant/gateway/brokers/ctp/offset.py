"""CTP 本地持仓账: 按今仓 / 昨仓与挂单冻结量拆分平仓委托.

持仓查询给出基准, 委托回报维护挂单冻结量, 成交回报增减今昨仓; 平仓时按本地账
拆单, 下单路径上不查柜台。

- 上期所、能源中心区分今昨仓: 平仓拆成平今、平昨, 先平今再平昨。普通平仓标志
  ('1') 在这两家只平昨仓(SimNow 实测今仓用 '1' 平报 "平昨仓位不足")。
- 其他交易所不区分今昨仓: 整笔用普通平仓(SimNow 实测大商所、郑商所、广期所的今仓
  用平今 '3'、平昨 '4' 也都能平, 但没有必要拆)。
- 可平数量(扣除挂单冻结)不足时拒绝, 写明今昨仓与冻结量; 持仓账尚未由持仓查询
  同步时拒绝, 不猜测。

挂单冻结只统计本账户的委托回报(私有流推送本投资者全部会话的委托, 含终端手工单)。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

#: 区分今昨仓的交易所
CLOSE_TODAY_EXCHANGES = frozenset({"SHFE", "INE"})


@dataclass
class _Side:
    """一个合约一个方向(多或空)的今昨仓."""

    today: float = 0.0
    yesterday: float = 0.0

    @property
    def total(self) -> float:
        return self.today + self.yesterday


@dataclass
class _Holding:
    exchange: str = ""
    long: _Side = field(default_factory=_Side)
    short: _Side = field(default_factory=_Side)


@dataclass(frozen=True)
class _ActiveOrder:
    symbol: str
    exchange: str
    side: str
    position_effect: str
    remaining: float


def _closed_side(holding: _Holding, side: str) -> _Side:
    """卖出平多头, 买入平空头."""
    return holding.long if side == "sell" else holding.short


class OffsetBook:
    """本地持仓账, 线程安全(回报来自 CTP 线程, 拆单来自策略线程)."""

    def __init__(self) -> None:
        """初始化为未同步状态."""
        self._lock = threading.Lock()
        self._holdings: dict[str, _Holding] = {}
        self._orders: dict[str, _ActiveOrder] = {}
        self._synced = False

    def apply_snapshot(
        self, rows: Iterable[dict[str, Any]], symbol: str | None = None
    ) -> None:
        """用持仓查询结果重置基准; ``symbol`` 非空时只重置该合约.

        ``rows`` 为 ``CTPTraderGateway.query_positions`` 的行: symbol、direction
        (Buy/Sell)、today_quantity、yesterday_quantity、exchange_id。
        """
        with self._lock:
            if symbol is None:
                self._holdings = {}
            else:
                self._holdings.pop(symbol, None)
            for row in rows:
                holding = self._holdings.setdefault(str(row["symbol"]), _Holding())
                holding.exchange = str(row.get("exchange_id") or holding.exchange)
                side = holding.long if row["direction"] == "Buy" else holding.short
                side.today = float(row.get("today_quantity") or 0.0)
                side.yesterday = float(row.get("yesterday_quantity") or 0.0)
            if symbol is None:
                self._synced = True

    def update_order(
        self,
        key: str,
        symbol: str,
        exchange: str,
        side: str,
        position_effect: str,
        remaining: float,
        active: bool,
    ) -> None:
        """委托回报: 未完成的平仓委托计入冻结, 完成(成交、撤单、拒单)后移除."""
        with self._lock:
            if active and position_effect != "open" and remaining > 0:
                self._orders[key] = _ActiveOrder(
                    symbol, exchange, side, position_effect, remaining
                )
            else:
                self._orders.pop(key, None)
            if exchange:
                self._holdings.setdefault(symbol, _Holding()).exchange = exchange

    def update_trade(
        self,
        symbol: str,
        exchange: str,
        side: str,
        position_effect: str,
        volume: float,
    ) -> None:
        """成交回报: 开仓增加今仓; 平仓按平今 / 平昨 / 普通平仓减少对应部分."""
        with self._lock:
            holding = self._holdings.setdefault(symbol, _Holding())
            if exchange:
                holding.exchange = exchange
            if position_effect == "open":
                opened = holding.long if side == "buy" else holding.short
                opened.today += volume
                return
            closed = _closed_side(holding, side)
            if position_effect == "close_today":
                closed.today = max(closed.today - volume, 0.0)
            elif position_effect == "close_yesterday" or (
                holding.exchange in CLOSE_TODAY_EXCHANGES
            ):
                closed.yesterday = max(closed.yesterday - volume, 0.0)
            else:
                # 不区分今昨仓的交易所只看合计, 先减今仓再减昨仓
                from_today = min(closed.today, volume)
                closed.today -= from_today
                closed.yesterday = max(closed.yesterday - (volume - from_today), 0.0)

    def _frozen(self, symbol: str, side: str, cfe: bool) -> tuple[float, float]:
        """该合约被 ``side`` 方向平仓挂单冻结的 (今仓, 昨仓) 数量."""
        holding = self._holdings[symbol]
        closed = _closed_side(holding, side)
        today = yesterday = 0.0
        for order in self._orders.values():
            if order.symbol != symbol or order.side != side:
                continue
            if order.position_effect == "close_today":
                today += order.remaining
            elif order.position_effect == "close_yesterday" or cfe:
                yesterday += order.remaining
            else:
                today += order.remaining
        # 不区分今昨仓时普通平仓的冻结先占今仓, 溢出部分占昨仓
        overflow = max(today - closed.today, 0.0)
        return today - overflow, yesterday + overflow

    def _available(self, symbol: str, side: str) -> tuple[_Side, bool, float, float]:
        """(持仓, 是否区分今昨仓, 冻结今仓, 冻结昨仓); 未同步时抛 ``ValueError``."""
        if not self._synced:
            raise ValueError(
                f"{symbol} 平仓无法拆分今昨仓: 本地持仓账尚未从柜台持仓查询同步"
            )
        holding = self._holdings.get(symbol)
        if holding is None:
            return _Side(), False, 0.0, 0.0
        cfe = holding.exchange in CLOSE_TODAY_EXCHANGES
        frozen_today, frozen_yd = self._frozen(symbol, side, cfe)
        return _closed_side(holding, side), cfe, frozen_today, frozen_yd

    def closable(self, symbol: str, side: str) -> float:
        """``side`` 方向可平的数量(今仓 + 昨仓, 扣除平仓挂单冻结)."""
        with self._lock:
            closed, _, frozen_today, frozen_yd = self._available(symbol, side)
            return max(closed.today - frozen_today, 0.0) + max(
                closed.yesterday - frozen_yd, 0.0
            )

    def split_close(
        self, symbol: str, side: str, quantity: float
    ) -> list[tuple[str, float]]:
        """把平仓量拆成 ``[(position_effect, 数量), ...]``.

        上期所、能源中心先平今再平昨; 其他交易所整笔普通平仓。可平数量不足或持仓账
        尚未同步时抛 ``ValueError``。
        """
        with self._lock:
            closed, cfe, frozen_today, frozen_yd = self._available(symbol, side)
            today = max(closed.today - frozen_today, 0.0)
            yesterday = max(closed.yesterday - frozen_yd, 0.0)
            if quantity > today + yesterday:
                label = "多头" if side == "sell" else "空头"
                raise ValueError(
                    f"{symbol} 平{label} {quantity:g} 手, 超过可平 "
                    f"{today + yesterday:g} 手(今仓 {closed.today:g}、昨仓 "
                    f"{closed.yesterday:g}, 平仓挂单冻结今仓 {frozen_today:g}、"
                    f"昨仓 {frozen_yd:g})"
                )
            if not cfe:
                return [("close", quantity)]
            legs = []
            from_today = min(quantity, today)
            if from_today > 0:
                legs.append(("close_today", from_today))
            if quantity - from_today > 0:
                legs.append(("close_yesterday", quantity - from_today))
            return legs
