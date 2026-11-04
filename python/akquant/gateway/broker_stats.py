"""broker_live 会话级交易统计（spec 5.5）：柜台成交驱动 ``LiveTradeStats``.

规则：

- 只计入本会话的单（``is_own_trade``），前端手动单、其他会话的单不计入。
- 会话启动时用柜台持仓播种（同一标的多空两行先按净持仓合并）。
- 平仓数量超过统计器已有持仓的部分直接截掉，不当成反向开仓：股票、基金等现货类
  品种的卖出一律视为平仓；期货、期权按 ``position_effect`` 判断。这一步在喂给 Rust
  之前完成，不改 ``TradeTracker`` 的撮合逻辑。
- 手续费恒为 0（柜台成交不带手续费）。
- 柜台送来的脏数据（缺字段、非有限数）只记 warning 并跳过：统计出错不能影响柜台
  状态同步。

成交回调在柜台派发线程上执行，权益采样与快照在行情线程上执行，故内部状态由一把锁
保护。
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Mapping, Optional

from ..akquant import LiveTradeStats
from ..log import get_logger
from .broker_strategy_api import _trade_field
from .symbol_match import normalize_symbol_for_match

logger = get_logger("gateway.stats")

#: 按 ``position_effect`` 判断开平的品种；其余（股票、基金、债券……）卖出一律视为平仓。
_DERIVATIVE_ASSET_TYPES = frozenset({"futures", "option"})


def _enum_text(value: Any) -> str:
    """枚举取 ``name``，其余转字符串，统一小写去空白."""
    return str(getattr(value, "name", value) or "").strip().lower()


def _finite_positive(value: Any) -> Optional[float]:
    """转成有限正数；无法转换、非有限或不大于 0 时返回 None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


class BrokerTradeStats:
    """会话级统计器：每个 broker_live 会话一个，由 ``BrokerRuntime`` 持有."""

    def __init__(
        self,
        *,
        is_own_trade: Callable[[Any], bool],
        multiplier_of: Callable[[str], Optional[float]],
        asset_type_of: Callable[[str], str],
    ) -> None:
        """注入成交归属判定、合约乘数与资产类型的查询函数."""
        self._is_own = is_own_trade
        self._multiplier_of = multiplier_of
        self._asset_type_of = asset_type_of
        self._stats: Optional[LiveTradeStats] = None
        self._lock = threading.RLock()
        self._warned: set[str] = set()

    @property
    def initialized(self) -> bool:
        """是否已用柜台权益与持仓初始化（未初始化时 ``snapshot`` 返回 None）."""
        return self._stats is not None

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message, *args)

    def initialize(
        self, *, equity: float, positions: list[Any], timestamp_ns: int
    ) -> None:
        """用会话启动时的柜台权益与持仓快照初始化；重复调用无效.

        :param equity: 柜台账户权益，必须是有限正数，否则保持未初始化。
        :param positions: 柜台持仓快照（``UnifiedPosition`` 或同形对象）。
        :param timestamp_ns: 会话启动时刻，用作初始权益点与播种持仓的开仓时间。
        """
        start_equity = _finite_positive(equity)
        if start_equity is None:
            logger.warning("统计: 柜台权益无效（%r），交易统计不可用", equity)
            return
        ts = int(timestamp_ns)
        with self._lock:
            if self._stats is not None:
                return
            stats = LiveTradeStats(start_equity)
            stats.on_equity(ts, start_equity)
            for symbol, (qty, price) in self._net_positions(positions).items():
                side = "buy" if qty > 0 else "sell"
                try:
                    stats.on_fill(
                        symbol, side, abs(qty), price, ts, self._multiplier(symbol)
                    )
                except ValueError as exc:
                    logger.warning("统计: 持仓 %s 播种失败（%s），已跳过", symbol, exc)
            self._stats = stats

    def _net_positions(self, positions: list[Any]) -> dict[str, tuple[float, float]]:
        """把持仓快照合并成 {标的: (带符号净持仓, 均价)}.

        CTP 同一合约多空分两行（锁仓）；逐行喂给 tracker 会在启动时撮合出一笔并不存在
        的平仓，所以先按净持仓合并，均价取较大一侧。
        """
        sides: dict[str, dict[str, tuple[float, float]]] = {}
        for pos in positions or []:
            symbol = normalize_symbol_for_match(getattr(pos, "symbol", ""))
            try:
                raw_qty = float(getattr(pos, "quantity", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if not symbol or not math.isfinite(raw_qty) or raw_qty == 0:
                continue
            price = _finite_positive(getattr(pos, "avg_price", None))
            if price is None:
                self._warn_once(
                    f"seed:{symbol}",
                    "统计: 持仓 %s 没有均价，无法作为会话前持仓播种；"
                    "平掉这部分仓位的成交不会计入统计",
                    symbol,
                )
                continue
            direction = _enum_text(getattr(pos, "direction", ""))
            short = raw_qty < 0 or direction in {"sell", "short"}
            key = "short" if short else "long"
            old_qty, old_price = sides.setdefault(symbol, {}).get(key, (0.0, 0.0))
            qty = old_qty + abs(raw_qty)
            avg = (old_qty * old_price + abs(raw_qty) * price) / qty
            sides[symbol][key] = (qty, avg)
        result: dict[str, tuple[float, float]] = {}
        for symbol, legs in sides.items():
            long_qty, long_price = legs.get("long", (0.0, 0.0))
            short_qty, short_price = legs.get("short", (0.0, 0.0))
            if long_qty and short_qty:
                self._warn_once(
                    f"hedged:{symbol}",
                    "统计: %s 同时持有多空仓位，按净持仓播种",
                    symbol,
                )
            net = long_qty - short_qty
            if net > 0:
                result[symbol] = (net, long_price)
            elif net < 0:
                result[symbol] = (net, short_price)
        return result

    def _multiplier(self, symbol: str) -> float:
        mult = _finite_positive(self._multiplier_of(symbol))
        if mult is None:
            self._warn_once(
                f"mult:{symbol}",
                "统计: %s 未配置合约乘数，按 1 计算（期货盈亏会偏差）",
                symbol,
            )
            return 1.0
        return mult

    def on_trade(self, payload: Any) -> None:
        """喂入一笔已按 ``trade_id`` 去重的柜台成交（未初始化或非本会话时忽略）."""
        if self._stats is None or not self._is_own(payload):
            return
        symbol = normalize_symbol_for_match(_trade_field(payload, "symbol"))
        side = _enum_text(_trade_field(payload, "side"))
        qty = _finite_positive(_trade_field(payload, "quantity"))
        price = _finite_positive(_trade_field(payload, "price"))
        if not symbol or side not in {"buy", "sell"} or qty is None or price is None:
            self._warn_once(
                f"bad:{symbol}",
                "统计: 成交字段不完整或无效（symbol=%s side=%s quantity=%r price=%r），"
                "已跳过",
                symbol,
                side,
                _trade_field(payload, "quantity"),
                _trade_field(payload, "price"),
            )
            return
        try:
            ts = int(_trade_field(payload, "timestamp_ns") or 0)
        except (TypeError, ValueError):
            ts = 0
        if ts <= 0:
            ts = time.time_ns()
        with self._lock:
            stats = self._stats
            qty = self._truncate_excess_close(stats, payload, symbol, side, qty)
            if qty <= 0:
                return
            try:
                stats.on_fill(symbol, side, qty, price, ts, self._multiplier(symbol))
            except ValueError as exc:
                logger.warning("统计: 成交 %s 无法计入（%s），已跳过", symbol, exc)

    def _truncate_excess_close(
        self,
        stats: LiveTradeStats,
        payload: Any,
        symbol: str,
        side: str,
        qty: float,
    ) -> float:
        """平仓数量超过统计器持仓的部分截掉（返回应喂入的数量，0 表示整笔忽略）."""
        held = stats.position(symbol)
        effect = _enum_text(_trade_field(payload, "position_effect"))
        asset = str(self._asset_type_of(symbol) or "stock").strip().lower()
        if asset not in _DERIVATIVE_ASSET_TYPES:
            # 现货不能卖空：卖出一律是平仓，买入一律是开仓
            closing = side == "sell"
        elif effect.startswith("open"):
            closing = False
        elif effect.startswith("close"):
            closing = True
        else:
            closing = (side == "sell" and held > 0) or (side == "buy" and held < 0)
        if not closing:
            return qty
        # 平仓方向与持仓方向相反才有可平的量（期货 close 但持仓同向时为 0）
        closable = held if side == "sell" else -held
        return min(qty, max(closable, 0.0))

    def on_equity(self, timestamp_ns: int, equity: float) -> None:
        """记录一个权益点；未初始化或权益不是有限正数（柜台未返回）时跳过."""
        value = _finite_positive(equity)
        if self._stats is None or value is None:
            return
        with self._lock:
            self._stats.on_equity(int(timestamp_ns), value)

    def snapshot(
        self, timestamp_ns: int, equity: float, last_prices: Mapping[str, float]
    ) -> Optional[dict[str, float]]:
        """当前统计快照（键与 ``ctx.strategy_stats()`` 相同）；不可用时返回 None."""
        value = _finite_positive(equity)
        if self._stats is None or value is None:
            return None
        prices = {
            normalize_symbol_for_match(symbol): float(price)
            for symbol, price in (last_prices or {}).items()
        }
        with self._lock:
            return dict(self._stats.snapshot(int(timestamp_ns), value, prices))
