"""回归护栏: broker_live 读路径(get_open_orders/get_order)与回测对象同形状.

修复前 get_open_orders/get_order 返回原生 UnifiedOrderSnapshot(有 broker_order_id、
无 `.id`; status 为 UnifiedOrderStatus 字符串枚举), 与回测 Rust `Order`(`.id` /
`OrderStatus` 枚举)不一致 → 策略 `o.id` 触发 AttributeError、
`o.status == OrderStatus.Filled` 恒 False。本文件断言实盘读路径经
broker_event_adapter 适配为与回测同形状的 StrategyOrder。
"""

from __future__ import annotations

from typing import Any

from akquant.akquant import OrderStatus
from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_models import (
    UnifiedAccount,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedPosition,
)
from akquant.gateway.broker_state_cache import BrokerStateCache


class _Gw:
    """假柜台: 一笔未完成委托 + 可回查的已完成委托表."""

    def __init__(self) -> None:
        self.completed: dict[str, UnifiedOrderSnapshot] = {}
        self.canceled: list[str] = []

    def query_positions(self) -> list[UnifiedPosition]:
        return [
            UnifiedPosition(symbol="600000.SH", quantity=1000, available_quantity=800)
        ]

    def query_account(self) -> UnifiedAccount:
        return UnifiedAccount(
            account_id="a", equity=1500.0, cash=1000.0, available_cash=850.0
        )

    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
        return [
            UnifiedOrderSnapshot(
                client_order_id="c1",
                broker_order_id="b1",
                symbol="600000.SH",
                status=UnifiedOrderStatus.NEW,
                filled_quantity=0.0,
            )
        ]

    def query_order(self, broker_order_id: str) -> UnifiedOrderSnapshot | None:
        return self.completed.get(str(broker_order_id))

    def cancel_order(self, bid: str) -> None:
        self.canceled.append(str(bid))


class _Submitter:
    def submit_order(self, **kwargs: Any) -> str:
        return "BID-1"

    def _get_execution_capabilities(self) -> dict[str, bool]:
        return {"broker_live": True}


class _S:
    ctx = None
    current_bar = None
    current_tick = None


def _exec(gw: _Gw | None = None) -> BrokerExecution:
    gw = gw or _Gw()
    return BrokerExecution(_S(), gw, BrokerStateCache(gw), _Submitter())


def test_get_open_orders_returns_backtest_shaped_orders() -> None:
    """柜台未完成委托应适配为 StrategyOrder: 有 `.id`, status 为 OrderStatus 枚举."""
    ex = _exec()
    orders = ex.get_open_orders()
    assert len(orders) == 1
    o = orders[0]
    assert o.id == "b1"  # 回测用 .id, 不再是 broker_order_id
    assert o.broker_order_id == "b1"
    assert o.symbol == "600000.SH"
    assert isinstance(o.status, OrderStatus)
    assert o.status == OrderStatus.New


def test_iterating_open_orders_and_using_id_no_attributeerror() -> None:
    """头号 footgun: `for o in get_open_orders(): cancel_order(o.id)` 实盘不再报错."""
    ex = _exec()
    ids = [o.id for o in ex.get_open_orders()]  # 修复前 AttributeError
    assert ids == ["b1"]


def test_get_open_orders_includes_local_stops_with_id() -> None:
    """本地止损单也应适配为 StrategyOrder(有 `.id` = LSTOP-*, OrderStatus 枚举)."""
    ex = _exec()
    stop_id = ex.submit_order(
        symbol="600000.SH",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    assert stop_id.startswith("LSTOP-")
    orders = ex.get_open_orders()
    by_id = {o.id: o for o in orders}
    assert "b1" in by_id  # 柜台单
    assert stop_id in by_id  # 本地止损单
    stop = by_id[stop_id]
    assert isinstance(stop.status, OrderStatus)
    assert stop.symbol == "600000.SH"
    assert stop.quantity == 100
    assert stop.trigger_price == 9.5


def test_get_order_by_broker_id_returns_adapted() -> None:
    """get_order(broker_order_id) 返回适配后的 StrategyOrder."""
    ex = _exec()
    o = ex.get_order("b1")
    assert o is not None
    assert o.id == "b1"
    assert isinstance(o.status, OrderStatus)


def test_get_order_finds_local_stop_by_local_id() -> None:
    """get_order 应能按 LSTOP-* id 找到本地止损单(修复前恒 None)."""
    ex = _exec()
    stop_id = ex.submit_order(
        symbol="600000.SH",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    o = ex.get_order(stop_id)
    assert o is not None
    assert o.id == stop_id


def test_get_order_falls_back_to_query_order_for_completed() -> None:
    """已成交/已撤委托: 经柜台 query_order 回查(对齐回测能取到完成单)."""
    gw = _Gw()
    gw.completed["bDONE"] = UnifiedOrderSnapshot(
        client_order_id="cDONE",
        broker_order_id="bDONE",
        symbol="600000.SH",
        status=UnifiedOrderStatus.FILLED,
        filled_quantity=100.0,
    )
    ex = _exec(gw)
    o = ex.get_order("bDONE")
    assert o is not None
    assert o.id == "bDONE"
    assert o.status == OrderStatus.Filled
    assert o.filled_quantity == 100.0


def test_get_order_unknown_returns_none() -> None:
    """未知 order_id 返回 None."""
    ex = _exec()
    assert ex.get_order("nope") is None


def test_hold_bar_returns_zero_in_broker_live() -> None:
    """hold_bar 实盘恒 0(保留行为, 仅内部一次性告警)."""
    ex = _exec()
    assert ex.hold_bar("600000.SH") == 0
    assert ex.hold_bar("600000.SH") == 0
