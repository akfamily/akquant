"""同一订单的同一状态不得重复触发 on_order.

``check_order_events`` 每个 bar/tick 事件都会跑一遍, 而 ``_emit_order_callback``
对 ``on_order`` 是**无条件调用**(``on_reject`` 早有 ``_framework_rejected_order_ids``
去重, ``on_order`` 没有等价物)。于是只要一张单还留在 ``ctx.recent_rejected_orders``
里, 它每个 bar 都会再推一次 —— 这正是测试反馈里 "order、trade 回调没做隔离, 是全量
推送的" 与 "没做时间限制, 非交易时间也在返回" 的一个来源(同一张单跨交易日反复出现,
看起来就像盘后还在推送新回报)。

**去重键必须是状态指纹**(订单标识 + 状态 + 已成交量), **不能含时间戳** ——
含时间戳的键每次重推都变, 去重会完全失效; 而只按订单号去重又会把
``New -> PartiallyFilled -> Filled`` 这些真实的状态推进吞掉。
"""

from typing import Any, List, Tuple

import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar

SYM = "600016.SH"
PRICE = 10.0


def _instrument() -> Instrument:
    return Instrument(
        symbol=SYM,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
    )


def _bars(days: Tuple[int, ...]) -> List[Bar]:
    return [
        Bar(
            timestamp=int(
                pd.Timestamp(f"2023-01-{day:02d} 14:00:00", tz="Asia/Shanghai").value
            ),
            open=PRICE,
            high=PRICE + 0.2,
            low=PRICE - 0.2,
            close=PRICE,
            volume=1_000_000.0,
            symbol=SYM,
        )
        for day in days
    ]


class _OrderEventRecorder(Strategy):
    """记录 on_order 收到的 (订单号, 状态) 序列."""

    def __init__(self, action: str = "reject") -> None:
        self._action = action
        self.n = 0
        self.events: List[Tuple[str, str]] = []

    def on_order(self, order: Any) -> None:
        self.events.append(
            (str(order.id), str(order.status).replace("OrderStatus.", ""))
        )

    def on_bar(self, bar: Bar) -> None:
        self.n += 1
        if self.n != 1:
            return
        if self._action == "reject":
            # 无持仓卖出 -> 被风控拒单(终态), 之后每个 bar 都不该再推一次
            self.sell(SYM, 100, price=PRICE)
        else:
            # 会成交的买单: New -> Filled 两个状态都必须推到
            self.buy(SYM, 100, price=PRICE)


def _run(action: str) -> _OrderEventRecorder:
    strategy = _OrderEventRecorder(action)
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars((3, 4, 5, 6, 9))},
        cash=1_000_000.0,
        show_progress=False,
        duration="60s",
    )
    return strategy


def test_rejected_order_is_not_pushed_repeatedly() -> None:
    """同一张拒单的 (订单号, 状态) 组合只能出现一次."""
    strategy = _run("reject")
    assert strategy.events, "on_order 一次都没触发"
    duplicates = [
        event for event in set(strategy.events) if strategy.events.count(event) > 1
    ]
    assert not duplicates, (
        f"同一 (订单号, 状态) 被重复推送: {duplicates}; 完整序列={strategy.events}"
    )


def test_status_transitions_are_still_delivered() -> None:
    """状态推进不能被去重吞掉: 成交单必须同时收到 New 与 Filled."""
    strategy = _run("fill")
    statuses = [status for _, status in strategy.events]
    assert "New" in statuses, f"缺少 New 状态: {strategy.events}"
    assert "Filled" in statuses, f"缺少 Filled 状态: {strategy.events}"
    duplicates = [
        event for event in set(strategy.events) if strategy.events.count(event) > 1
    ]
    assert not duplicates, f"同一 (订单号, 状态) 被重复推送: {duplicates}"
