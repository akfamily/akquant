# -*- coding: utf-8 -*-
"""量限按 bar 共享, FOK 不部分成交, IOC 成交后撤掉剩余.

修复前 ``volume_limit_pct`` 对每笔委托各算一次“成交量 × 比例”: 同一根 bar 上 8 笔
各 2500 股的委托合计成交 20000 股, 是整根成交量的 2 倍。FOK/IOC 被截断后剩余量继续
挂着, 之后的 bar 接着成交, 与普通委托没有区别。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest
from akquant import Bar, CurrentClose, Strategy, TimeInForce, run_backtest

SYMBOL = "T"
BAR_VOLUME = 10_000.0
LIMIT_PCT = 0.25  # 每根 bar 最多成交 2500 股


def _bars(n: int = 6) -> List[Bar]:
    stamps = pd.date_range("2023-01-03", periods=n, freq="D", tz="Asia/Shanghai")
    return [
        Bar(
            timestamp=(ts + pd.Timedelta(hours=10)).value,
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
            volume=BAR_VOLUME,
            symbol=SYMBOL,
        )
        for ts in stamps
    ]


class _FirstBarOrders(Strategy):
    """第一根 bar 按给定清单下单, 之后每根记录在途订单数."""

    def __init__(self, orders: List[Tuple[str, float, Any]]) -> None:
        super().__init__()
        self.plan = orders
        self.seen = 0
        self.open_counts: List[int] = []
        self.final_status: Dict[str, Tuple[str, float]] = {}

    def on_bar(self, bar: Bar) -> None:
        self.seen += 1
        if self.seen == 1:
            for side, quantity, tif in self.plan:
                kwargs = {} if tif is None else {"time_in_force": tif}
                if side == "buy":
                    self.buy(bar.symbol, quantity, **kwargs)
                else:
                    self.sell(bar.symbol, quantity, **kwargs)
        else:
            self.open_counts.append(len(self.get_open_orders(bar.symbol)))

    def on_order(self, order: Any) -> None:
        self.final_status[str(order.id)] = (
            str(order.status).split(".")[-1].lower(),
            float(order.filled_quantity),
        )


def _run(orders: List[Tuple[str, float, Any]], **kwargs: Any) -> Tuple[Any, Any]:
    strategy = _FirstBarOrders(orders)
    options: Dict[str, Any] = dict(
        data=_bars(),
        strategy=strategy,
        symbols=SYMBOL,
        initial_cash=10_000_000.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        fill_policy=CurrentClose(),
        lot_size=100,
        volume_limit_pct=LIMIT_PCT,
        show_progress=False,
    )
    options.update(kwargs)
    return run_backtest(**options), strategy


def _fills_by_bar(result: Any) -> List[float]:
    """按成交时间汇总的每根 bar 成交量(升序)."""
    totals: Dict[int, float] = defaultdict(float)
    for trade in result.executions:
        totals[int(trade.timestamp)] += float(trade.quantity)
    return [totals[key] for key in sorted(totals)]


@pytest.mark.parametrize("count", [2, 4, 8])
def test_orders_on_one_bar_share_the_volume_budget(count: int) -> None:
    """同一根 bar 上多笔委托合计成交不超过 成交量 × 比例."""
    result, _ = _run([("buy", 2500, None)] * count)
    per_bar = _fills_by_bar(result)
    assert per_bar[0] == 2500.0
    assert max(per_bar) <= 2500.0


def test_budget_left_by_first_order_goes_to_the_next() -> None:
    """先到的委托用掉 1500, 后一笔当根只能成交剩下的 1000, 余量下一根再成交."""
    result, _ = _run([("buy", 1500, None), ("buy", 1500, None)])
    assert _fills_by_bar(result)[:2] == [2500.0, 500.0]
    quantities = [float(t.quantity) for t in result.executions]
    assert quantities == [1500.0, 1000.0, 500.0]


def test_buy_and_sell_share_one_budget() -> None:
    """买卖两个方向共用同一根 bar 的成交量(先卖后买)."""
    result, _ = _run(
        [("buy", 2000, None), ("sell", 1000, None)],
        enable_short_sell=True,
        account_mode="margin",
    )
    assert _fills_by_bar(result)[0] <= 2500.0


def test_single_order_behaviour_is_unchanged() -> None:
    """只有一笔委托时与原先相同: 每根成交 2500, 直到成交完."""
    result, _ = _run([("buy", 6000, None)])
    assert _fills_by_bar(result) == [2500.0, 2500.0, 1000.0]


def test_no_limit_fills_everything_at_once() -> None:
    """不设量限: 多笔委托当根全部成交."""
    result, _ = _run([("buy", 2500, None)] * 4, volume_limit_pct=0.0)
    assert _fills_by_bar(result) == [10000.0]


def test_fok_is_cancelled_when_it_cannot_fill_in_full() -> None:
    """FOK 3000 股、当根最多 2500: 一股不成交, 整笔撤销."""
    result, strategy = _run([("buy", 3000, TimeInForce.FOK)])
    assert list(result.executions) == []
    assert result.orders_df["status"].tolist() == ["cancelled"]
    assert strategy.open_counts[0] == 0


def test_fok_fills_when_volume_is_enough() -> None:
    """FOK 2000 股、当根最多 2500: 全部成交."""
    result, _ = _run([("buy", 2000, TimeInForce.FOK)])
    assert _fills_by_bar(result) == [2000.0]
    assert result.orders_df["status"].tolist() == ["filled"]


def test_fok_sees_budget_already_used_by_earlier_orders() -> None:
    """前一笔已用掉 1500, FOK 2000 股当根成交不完: 整笔撤销."""
    result, _ = _run([("buy", 1500, None), ("buy", 2000, TimeInForce.FOK)])
    assert _fills_by_bar(result) == [1500.0]
    assert result.orders_df.sort_values("created_at")["status"].tolist() == [
        "filled",
        "cancelled",
    ]


def test_ioc_fills_what_it_can_and_cancels_the_rest() -> None:
    """IOC 3000 股、当根最多 2500: 成交 2500, 剩余 500 立即撤销, 之后不再成交."""
    result, strategy = _run([("buy", 3000, TimeInForce.IOC)])
    assert _fills_by_bar(result) == [2500.0]
    assert strategy.open_counts[0] == 0
    orders = result.orders_df
    assert orders["status"].tolist() == ["cancelled"]
    assert float(orders["filled_quantity"].iloc[0]) == 2500.0
    # 策略先后收到成交与撤销两次状态, 终态是已撤销且带成交量
    assert list(strategy.final_status.values()) == [("cancelled", 2500.0)]


def test_ioc_that_fills_in_full_is_filled() -> None:
    """IOC 2000 股、当根最多 2500: 全部成交, 状态为已成交."""
    result, _ = _run([("buy", 2000, TimeInForce.IOC)])
    assert _fills_by_bar(result) == [2000.0]
    assert result.orders_df["status"].tolist() == ["filled"]


def _run_strategy(strategy: Strategy) -> Any:
    return run_backtest(
        data=_bars(),
        strategy=strategy,
        symbols=SYMBOL,
        initial_cash=10_000_000.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        fill_policy=CurrentClose(),
        lot_size=100,
        volume_limit_pct=LIMIT_PCT,
        show_progress=False,
    )


def test_partially_filled_ioc_cancels_its_oco_peer() -> None:
    """IOC 成交一部分后撤掉剩余, 按成交处理: OCO 组里的另一笔被撤."""

    class _Oco(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.done = False

        def on_bar(self, bar: Bar) -> None:
            if self.done:
                return
            self.done = True
            peer = self.buy(bar.symbol, 100, price=5.0)  # 远离市价, 自己不会成交
            ioc = self.buy(bar.symbol, 3000, time_in_force=TimeInForce.IOC)
            self.place_oco(ioc, peer)

    orders = _run_strategy(_Oco()).orders_df.sort_values("created_at")
    assert orders["filled_quantity"].tolist() == [0.0, 2500.0]
    assert orders["status"].tolist() == ["cancelled", "cancelled"]


def test_partially_filled_ioc_bracket_exits_use_filled_quantity() -> None:
    """括号单的进场单是 IOC 且只成交一部分: 止盈止损按已成交量 2500 挂出."""

    class _Bracket(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.done = False

        def on_bar(self, bar: Bar) -> None:
            if self.done:
                return
            self.done = True
            self.place_bracket(
                bar.symbol,
                3000,
                stop_trigger_price=8.0,
                take_profit_price=12.0,
                time_in_force=TimeInForce.IOC,
            )

    orders = _run_strategy(_Bracket()).orders_df.sort_values("created_at")
    sides = [str(side).split(".")[-1].lower() for side in orders["side"]]
    assert sides == ["buy", "sell", "sell"]
    assert orders["quantity"].tolist() == [3000.0, 2500.0, 2500.0]
    assert float(orders["filled_quantity"].iloc[0]) == 2500.0
