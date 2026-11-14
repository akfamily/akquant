# -*- coding: utf-8 -*-
"""cancel_order / cancel_group 的入参: 订单号、下单回执、订单对象; 其余报错.

此前订单对象会被 ``str()`` 成一串描述文字当订单号去撤, 引擎找不到这个号也不报错,
订单一直挂着(``for o in self.get_open_orders(): self.cancel_order(o)`` 静默失效)。
"""

from typing import Any, Dict, List

import akquant as aq
import pandas as pd
import pytest
from akquant import Strategy
from akquant.gateway.order_receipt import OrderReceipt
from akquant.strategy_trading_api import cancel_group, cancel_order


class _Execution:
    def __init__(self) -> None:
        self.cancelled: List[str] = []
        self.cancelled_groups: List[str] = []

    def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(order_id)

    def cancel_group(self, group_id: str) -> None:
        self.cancelled_groups.append(group_id)


class _Strat:
    def __init__(self) -> None:
        self.execution = _Execution()


def _order(order_id: str = "o-1") -> Any:
    return aq.Order(
        id=order_id,
        symbol="AAA",
        side=aq.OrderSide.Buy,
        order_type=aq.OrderType.Limit,
        quantity=100.0,
        price=1.0,
    )


def test_cancel_order_accepts_id_string() -> None:
    """订单号原样下发."""
    strat = _Strat()
    cancel_order(strat, "o-1")
    assert strat.execution.cancelled == ["o-1"]


def test_cancel_order_accepts_order_object() -> None:
    """订单对象取其 id."""
    strat = _Strat()
    cancel_order(strat, _order("o-7"))
    assert strat.execution.cancelled == ["o-7"]


def test_cancel_order_accepts_receipt() -> None:
    """下单回执取首腿订单号(原有行为)."""
    strat = _Strat()
    cancel_order(strat, OrderReceipt.single(group_id="g1", broker_order_id="b1"))
    assert strat.execution.cancelled == ["b1"]


@pytest.mark.parametrize("bad", [None, 123, 1.5, ["o-1"], {"id": "o-1"}, object()])
def test_cancel_order_rejects_other_types(bad: Any) -> None:
    """认不出的入参报错, 不再悄悄转成字符串去撤."""
    strat = _Strat()
    with pytest.raises(TypeError, match="cancel_order"):
        cancel_order(strat, bad)
    assert strat.execution.cancelled == []


def test_cancel_group_accepts_string_and_receipt() -> None:
    """组号与下单回执(原有行为)."""
    strat = _Strat()
    cancel_group(strat, "g1")
    cancel_group(strat, OrderReceipt.single(group_id="g2", broker_order_id="b1"))
    assert strat.execution.cancelled_groups == ["g1", "g2"]


@pytest.mark.parametrize("bad", [None, 123, ["g1"], object()])
def test_cancel_group_rejects_other_types(bad: Any) -> None:
    """认不出的入参报错."""
    strat = _Strat()
    with pytest.raises(TypeError, match="cancel_group"):
        cancel_group(strat, bad)
    assert strat.execution.cancelled_groups == []


def test_cancel_group_rejects_order_object_with_hint() -> None:
    """订单对象没有组号: 报错并提示改用 cancel_order."""
    strat = _Strat()
    with pytest.raises(TypeError, match="cancel_order"):
        cancel_group(strat, _order())
    assert strat.execution.cancelled_groups == []


def test_backtest_cancel_by_order_object() -> None:
    """回测里把 get_open_orders() 返回的订单对象直接传给 cancel_order, 订单被撤."""
    closes = [10.0, 11.0, 12.0, 13.0]
    days = pd.date_range("2024-01-02", periods=len(closes), freq="B")
    data = pd.DataFrame(
        {
            "date": days.tz_localize("Asia/Shanghai"),
            "symbol": "AAA",
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": 1e6,
        }
    )
    seen: Dict[int, int] = {}

    class S(Strategy):
        def on_start(self) -> None:
            self.n = 0

        def on_bar(self, bar: Any) -> None:
            self.n += 1
            open_orders = self.get_open_orders()
            seen[self.n] = len(open_orders)
            if self.n == 1:
                self.buy(symbol=bar.symbol, quantity=100, price=1.0)  # 不会成交
            elif self.n == 2:
                for order in open_orders:
                    self.cancel_order(order)

    result = aq.run_backtest(
        strategy=S, data=data, initial_cash=1e6, show_progress=False
    )
    assert seen == {1: 0, 2: 1, 3: 0, 4: 0}
    assert result.orders_df["status"].tolist() == ["cancelled"]
