"""实盘(broker_live)前置风控: 委托合规规则与账户级盈亏熔断.

在下单器拆完开平腿之后、向柜台报单之前检查, 超限时作为前置拒单回吐给
``on_reject``, 原因写明计数与限额。所有规则默认关闭。

委托合规(对应交易所对报撤单、自成交的监管与柜台流控):

- ``order_flow_limit`` / ``order_flow_window``: 滚动窗口内最多报出的委托笔数
  (按拆出的每条腿计, 即柜台收到的报单数);
- ``active_order_limit``: 未完成委托(本账户)的笔数上限, 含本笔;
- ``cancel_limit``: 单合约本交易日被撤委托(含撤单成功、FAK/FOK 剩余撤销、终端手工
  撤单)达到该次数后, 拒绝该合约的新委托, 撤单本身不受限;
- ``open_count_limit`` / ``open_volume_limit``: 单合约本交易日报出的开仓委托笔数 /
  手数上限;
- ``self_trade_check``: 同一合约本账户有方向相反、价格可成交的未完成委托时拒绝
  (新委托为市价单、或对方是市价单都视为可成交)。

账户级盈亏熔断(按柜台账户快照, 已含手续费与逐日盯市):

- ``max_daily_loss``: 当日亏损 = 当日初始权益(静态权益) − 当前动态权益;
- ``max_drawdown``: 回撤 = 本次运行观察到的权益峰值(含当日初始权益) − 当前权益,
  进程重启后峰值重新计起。
- 超限后锁定: 日亏损锁定到本交易日结束, 回撤锁定到进程结束; 锁定期间含开仓腿的
  委托一律拒绝, 纯平仓委托照常放行, 不自动平仓。

交易日取策略的 ``trading_day``(夜盘属下一交易日); 取不到(如策略回调之外的信号
线程下单)时按本地自然日计数。
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ..log import build_log_extra, get_logger

logger = get_logger("gateway.live_risk")


@dataclass(frozen=True)
class LiveRiskConfig:
    """实盘前置风控配置; 字段为 None / False 表示不启用该规则."""

    order_flow_limit: int | None = None
    order_flow_window: float = 1.0
    active_order_limit: int | None = None
    cancel_limit: int | None = None
    open_count_limit: int | None = None
    open_volume_limit: float | None = None
    self_trade_check: bool = False
    max_daily_loss: float | None = None
    max_drawdown: float | None = None

    def __post_init__(self) -> None:
        """校验各限额为正数."""
        for name in (
            "order_flow_limit",
            "active_order_limit",
            "cancel_limit",
            "open_count_limit",
            "open_volume_limit",
            "max_daily_loss",
            "max_drawdown",
        ):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ValueError(f"LiveRiskConfig.{name} 须为正数, 实际: {value}")
        if not self.order_flow_window > 0:
            raise ValueError(
                "LiveRiskConfig.order_flow_window 须为正数, 实际: "
                f"{self.order_flow_window}"
            )


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _side_name(value: Any) -> str:
    return str(getattr(value, "name", value) or "").strip().lower()


def _crosses(side: str, price: float | None, other_price: float | None) -> bool:
    """``side`` 方向的新委托能否与对方挂单成交; 任一方无价格(市价)视为可成交."""
    if price is None or other_price is None:
        return True
    return price >= other_price if side == "buy" else price <= other_price


class LiveRiskGuard:
    """实盘前置风控的计数与熔断状态; 线程安全."""

    def __init__(
        self,
        config: LiveRiskConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """绑定配置; ``clock`` 为委托流速窗口用的单调时钟."""
        self.config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._flow: deque[float] = deque()
        self._day: dt.date | None = None
        self._cancelled: dict[str, int] = {}
        self._cancelled_ids: set[str] = set()
        self._open_counts: dict[str, int] = {}
        self._open_volumes: dict[str, float] = {}
        self._peak_equity: float | None = None
        self._daily_loss_day: dt.date | None = None
        self._drawdown_locked = False

    def _roll_day(self, trading_day: dt.date | None) -> None:
        day = trading_day or dt.date.today()
        if day != self._day:
            self._day = day
            self._cancelled = {}
            self._cancelled_ids = set()
            self._open_counts = {}
            self._open_volumes = {}

    def on_order_event(self, payload: Any, trading_day: dt.date | None) -> None:
        """委托回报: 统计各合约本交易日被撤的委托(按柜台委托号去重)."""
        if _side_name(_field(payload, "status")) not in {"cancelled", "canceled"}:
            return
        order_id = str(_field(payload, "broker_order_id") or "")
        symbol = str(_field(payload, "symbol") or "")
        with self._lock:
            self._roll_day(trading_day)
            if not symbol or (order_id and order_id in self._cancelled_ids):
                return
            if order_id:
                self._cancelled_ids.add(order_id)
            self._cancelled[symbol] = self._cancelled.get(symbol, 0) + 1

    def observe_equity(self, equity: float | None) -> None:
        """记录观察到的账户权益, 用于回撤峰值."""
        if equity is None:
            return
        with self._lock:
            if self._peak_equity is None or equity > self._peak_equity:
                self._peak_equity = float(equity)

    def check(
        self,
        *,
        symbol: str,
        side: str,
        price: float | None,
        legs: Sequence[tuple[str, float]],
        trading_day: dt.date | None,
        open_orders: Callable[[], Iterable[Any]],
        account: Callable[[], Any],
    ) -> str | None:
        """检查一笔委托(已拆成开平腿); 返回拒单原因, None 表示通过.

        ``open_orders`` / ``account`` 按需调用(只在启用相关规则时取), 分别返回本账户
        未完成委托与柜台账户快照(需有 ``equity``; 日亏损还需 ``day_start_equity``)。
        """
        cfg = self.config
        side = _side_name(side)
        open_qty = sum(q for effect, q in legs if effect == "open")
        open_legs = sum(1 for effect, _ in legs if effect == "open")
        with self._lock:
            self._roll_day(trading_day)
            if cfg.order_flow_limit is not None:
                now = self._clock()
                while self._flow and now - self._flow[0] >= cfg.order_flow_window:
                    self._flow.popleft()
                if len(self._flow) + len(legs) > cfg.order_flow_limit:
                    return (
                        f"委托流速超限: {cfg.order_flow_window:g} 秒内已报 "
                        f"{len(self._flow)} 笔, 本笔 {len(legs)} 笔, 上限 "
                        f"{cfg.order_flow_limit} 笔"
                    )
            cancelled = self._cancelled.get(symbol, 0)
            if cfg.cancel_limit is not None and cancelled >= cfg.cancel_limit:
                return (
                    f"{symbol} 本交易日已撤单 {cancelled} 次, 达到撤单次数上限 "
                    f"{cfg.cancel_limit}, 不再报新委托"
                )
            if open_legs and cfg.open_count_limit is not None:
                done = self._open_counts.get(symbol, 0)
                if done + open_legs > cfg.open_count_limit:
                    return (
                        f"{symbol} 本交易日已报开仓委托 {done} 笔, 本笔 {open_legs} "
                        f"笔, 超过开仓次数上限 {cfg.open_count_limit}"
                    )
            if open_qty and cfg.open_volume_limit is not None:
                done_qty = self._open_volumes.get(symbol, 0.0)
                if done_qty + open_qty > cfg.open_volume_limit:
                    return (
                        f"{symbol} 本交易日已报开仓 {done_qty:g} 手, 本笔 {open_qty:g} "
                        f"手, 超过开仓手数上限 {cfg.open_volume_limit:g}"
                    )
        if cfg.active_order_limit is not None or cfg.self_trade_check:
            working = list(open_orders())
            if cfg.active_order_limit is not None:
                if len(working) + len(legs) > cfg.active_order_limit:
                    return (
                        f"未完成委托 {len(working)} 笔, 本笔 {len(legs)} 笔, 超过上限 "
                        f"{cfg.active_order_limit} 笔"
                    )
            if cfg.self_trade_check:
                for order in working:
                    if str(_field(order, "symbol") or "") != symbol:
                        continue
                    other_side = _side_name(_field(order, "side"))
                    if not other_side or other_side == side:
                        continue
                    other_price = _field(order, "price")
                    if _crosses(side, price, other_price):
                        own = "市价" if price is None else f"{price:g}"
                        other = "市价" if other_price is None else f"{other_price:g}"
                        order_id = _field(order, "broker_order_id")
                        return (
                            f"{symbol} 自成交风险: 本笔 {side} {own} 可与本账户未完成"
                            f"的 {other_side} 委托 {order_id}({other}) 成交"
                        )
        if cfg.max_daily_loss is not None or cfg.max_drawdown is not None:
            reason = self._check_account(account, trading_day, bool(open_legs))
            if reason is not None:
                return reason
        return None

    def _check_account(
        self, account: Callable[[], Any], trading_day: dt.date | None, opening: bool
    ) -> str | None:
        cfg = self.config
        snapshot = account()
        equity = _field(snapshot, "equity")
        day_start = _field(snapshot, "day_start_equity")
        if equity is None:
            return "柜台账户快照没有权益(equity), 无法校验账户级盈亏熔断"
        equity = float(equity)
        with self._lock:
            for value in (day_start, equity):
                if value is not None and (
                    self._peak_equity is None or float(value) > self._peak_equity
                ):
                    self._peak_equity = float(value)
            day = trading_day or dt.date.today()
            if cfg.max_daily_loss is not None:
                if day_start is None:
                    return (
                        "柜台账户快照没有当日初始权益(day_start_equity), "
                        "无法校验 max_daily_loss"
                    )
                loss = float(day_start) - equity
                if loss > cfg.max_daily_loss and self._daily_loss_day != day:
                    self._daily_loss_day = day
                    # 熔断只在触发时记一次, 带提醒标记(见 akquant.live.alerts)
                    logger.warning(
                        "账户当日亏损超限, 当日只允许平仓: 当日初始权益 %.2f, "
                        "当前权益 %.2f, 亏损 %.2f, 上限 %g",
                        float(day_start),
                        equity,
                        loss,
                        cfg.max_daily_loss,
                        extra={**build_log_extra(phase="gateway"), "alert": "风控熔断"},
                    )
                if self._daily_loss_day == day and opening:
                    return (
                        f"账户当日亏损超限后只允许平仓: 当日初始权益 "
                        f"{float(day_start):.2f}, 当前权益 {equity:.2f}, 亏损 "
                        f"{loss:.2f}, 上限 {cfg.max_daily_loss:g}"
                    )
            if cfg.max_drawdown is not None:
                peak = self._peak_equity if self._peak_equity is not None else equity
                drawdown = peak - equity
                if drawdown > cfg.max_drawdown and not self._drawdown_locked:
                    self._drawdown_locked = True
                    logger.warning(
                        "账户回撤超限, 此后只允许平仓: 权益峰值 %.2f, 当前权益 %.2f, "
                        "回撤 %.2f, 上限 %g",
                        peak,
                        equity,
                        drawdown,
                        cfg.max_drawdown,
                        extra={**build_log_extra(phase="gateway"), "alert": "风控熔断"},
                    )
                if self._drawdown_locked and opening:
                    return (
                        f"账户回撤超限后只允许平仓: 权益峰值 {peak:.2f}, 当前权益 "
                        f"{equity:.2f}, 回撤 {drawdown:.2f}, 上限 {cfg.max_drawdown:g}"
                    )
        return None

    def record_sent(
        self,
        symbol: str,
        position_effect: str,
        quantity: float,
        trading_day: dt.date | None,
    ) -> None:
        """一条腿报出后计入流速窗口与开仓计数."""
        with self._lock:
            self._roll_day(trading_day)
            if self.config.order_flow_limit is not None:
                self._flow.append(self._clock())
            if position_effect == "open":
                self._open_counts[symbol] = self._open_counts.get(symbol, 0) + 1
                self._open_volumes[symbol] = (
                    self._open_volumes.get(symbol, 0.0) + quantity
                )
