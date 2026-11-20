"""回归护栏: broker_live 读路径(get_open_orders/get_order)与回测对象同形状.

修复前 get_open_orders/get_order 返回原生 UnifiedOrderSnapshot(有 broker_order_id、
无 `.id`; status 为 UnifiedOrderStatus 字符串枚举), 与回测 Rust `Order`(`.id` /
`OrderStatus` 枚举)不一致 → 策略 `o.id` 触发 AttributeError、
`o.status == OrderStatus.Filled` 恒 False。本文件断言实盘读路径经
broker_event_adapter 适配为与回测同形状的 StrategyOrder。
"""

from __future__ import annotations

from typing import Any

import pytest
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
    ).primary
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
    ).primary
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


def test_hold_bar_is_zero_before_any_bar_in_broker_live() -> None:
    """还没收到过 bar 时 hold_bar 为 0."""
    ex = _exec()
    assert ex.hold_bar("600000.SH") == 0
    assert ex.hold_bar("600000.SH") == 0


@pytest.mark.parametrize(
    ("reported", "expected"),
    [(769.6, 769.6), (0.0, 0.0), (None, 1800.0)],
)
def test_account_market_value_prefers_broker_reported_value(
    reported: float | None, expected: float
) -> None:
    """柜台给了持仓市值就用它(含 0), 没给才退回 权益 − 现金."""
    from akquant.gateway.broker_strategy_api import _account_to_dict

    account = UnifiedAccount(
        "a", equity=10000.0, cash=8200.0, available_cash=7500.0, market_value=reported
    )
    view = _account_to_dict(account)
    assert view["market_value"] == expected
    assert (view["equity"], view["cash"], view["available_cash"]) == (
        10000.0,
        8200.0,
        7500.0,
    )


def test_account_market_value_defaults_to_none_and_keeps_positional_order() -> None:
    """不填市值的网关不受影响: 字段缺省为 None, 第五个位置参数仍是 timestamp_ns."""
    account = UnifiedAccount("a", 10000.0, 8200.0, 7500.0, 123)
    assert account.timestamp_ns == 123
    assert account.market_value is None


def test_account_view_uses_broker_reported_frozen_margin_and_pnl() -> None:
    """柜台给了冻结资金、保证金、浮动盈亏就用; 可用保证金 = 权益 − 保证金."""
    from akquant.gateway.broker_strategy_api import _account_to_dict

    account = UnifiedAccount(
        "a",
        equity=10000.0,
        cash=8200.0,
        available_cash=7500.0,
        frozen_cash=300.0,
        margin=1200.0,
        unrealized_pnl=-45.5,
    )
    view = _account_to_dict(account)
    assert view["frozen_cash"] == 300.0
    assert view["margin"] == view["used_margin"] == 1200.0
    assert view["free_margin"] == 8800.0
    assert view["unrealized_pnl"] == -45.5


def test_account_view_keeps_placeholders_when_broker_reports_nothing() -> None:
    """柜台没给这三项时照旧: 冻结与保证金为 0、可用保证金等于权益、浮动盈亏为 0."""
    from akquant.gateway.broker_strategy_api import _account_to_dict

    account = UnifiedAccount("a", equity=10000.0, cash=8200.0, available_cash=7500.0)
    assert (account.frozen_cash, account.margin, account.unrealized_pnl) == (
        None,
        None,
        None,
    )
    view = _account_to_dict(account)
    assert (view["frozen_cash"], view["margin"], view["used_margin"]) == (0.0, 0.0, 0.0)
    assert view["free_margin"] == 10000.0
    assert view["unrealized_pnl"] == 0.0


def _live_strategy(gw: Any) -> Any:
    """挂了柜台执行后端的真实策略; 引擎上下文里的持仓恒为 0(柜台模式下引擎不记成交)."""
    from akquant import Strategy

    strategy = Strategy()
    strategy.execution = BrokerExecution(
        strategy, gw, BrokerStateCache(gw), _Submitter()
    )
    return strategy


def _bar(symbol: str, minute: int) -> Any:
    import pandas as pd
    from akquant.akquant import Bar

    ts = pd.Timestamp(f"2026-01-05 09:{30 + minute}:00", tz="Asia/Shanghai").value
    return Bar(
        timestamp=ts,
        open=10.0,
        high=10.0,
        low=10.0,
        close=10.0,
        volume=1.0,
        symbol=symbol,
    )


def _shadow_ctx() -> Any:
    from _mock_ctx import mock_strategy_context

    ctx = mock_strategy_context()
    ctx.get_position.return_value = 0.0
    ctx.get_available_position.return_value = 0.0
    return ctx


def test_holding_bars_count_from_broker_position_in_broker_live() -> None:
    """持有 bar 数按柜台持仓计, 不按引擎里恒为 0 的持仓计.

    此前实盘 get_holding_bars() 恒为 0, 按持仓时长离场的逻辑在实盘永远不触发。
    """
    strategy = _live_strategy(_Gw())
    ctx = _shadow_ctx()
    for minute in range(3):
        strategy._on_bar_event(_bar("600000.SH", minute), ctx)
    assert strategy.get_holding_bars("600000.SH") == 3


def test_holding_bars_stay_zero_for_symbol_without_broker_position() -> None:
    """柜台没有持仓的标的, 持有 bar 数为 0."""
    strategy = _live_strategy(_Gw())
    ctx = _shadow_ctx()
    strategy._on_bar_event(_bar("000001.SZ", 0), ctx)
    assert strategy.get_holding_bars("000001.SZ") == 0


def test_position_helper_reads_broker_position_in_broker_live() -> None:
    """self.position 与 get_position() 同源: 柜台模式下读柜台持仓."""
    strategy = _live_strategy(_Gw())
    ctx = _shadow_ctx()
    strategy._on_bar_event(_bar("600000.SH", 0), ctx)
    strategy.current_bar = _bar("600000.SH", 0)
    assert strategy.position.size == 1000.0
    assert strategy.position.available == 800.0
    assert strategy.position.size == strategy.get_position()


def test_new_optional_model_fields_default_to_none() -> None:
    """新增字段都排在末尾且默认 None: 既有的构造写法不受影响, None 表示柜台没给."""
    from akquant.gateway.broker_models import UnifiedTrade

    pos = UnifiedPosition("X", 1.0, 1.0)
    assert (
        pos.frozen_quantity,
        pos.market_value,
        pos.unrealized_pnl,
        pos.margin,
        pos.last_price,
    ) == (None, None, None, None, None)
    trade = UnifiedTrade("t", "b", "c", "X", "Buy", 1.0, 2.0, 3)
    assert trade.commission is None
    snap = UnifiedOrderSnapshot("c", "b", "X", UnifiedOrderStatus.NEW)
    assert (snap.side, snap.quantity, snap.price) == (None, None, None)


def test_trade_commission_reaches_strategy_trade() -> None:
    """柜台给了成交手续费就带到策略收到的成交对象上; 没给记 0."""
    from akquant.gateway.broker_event_adapter import map_trade
    from akquant.gateway.broker_models import UnifiedTrade

    base: dict[str, Any] = dict(
        trade_id="t",
        broker_order_id="b",
        client_order_id="c",
        symbol="X",
        side="Buy",
        quantity=100.0,
        price=10.0,
        timestamp_ns=1,
    )
    assert map_trade(UnifiedTrade(**base, commission=5.0)).commission == 5.0
    assert map_trade(UnifiedTrade(**base)).commission == 0.0


def test_open_order_side_quantity_price_come_from_broker_snapshot() -> None:
    """get_open_orders() 不带下单请求, 方向/数量/价格取柜台快照里的值."""
    from akquant.akquant import OrderSide

    class _SnapGw(_Gw):
        def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
            return [
                UnifiedOrderSnapshot(
                    client_order_id="c1",
                    broker_order_id="b1",
                    symbol="600000.SH",
                    status=UnifiedOrderStatus.NEW,
                    side="Sell",
                    quantity=300.0,
                    price=10.5,
                )
            ]

    order = _exec(_SnapGw()).get_open_orders()[0]
    assert order.side == OrderSide.Sell
    assert order.quantity == 300.0
    assert order.price == 10.5


class _DetailGw(_Gw):
    """假柜台: 一个合约多空两行(锁仓), 带成本、今昨仓和柜台给的盈亏/保证金."""

    def query_positions(self) -> list[UnifiedPosition]:
        return [
            UnifiedPosition(
                symbol="rb2601.SHFE",
                quantity=3,
                available_quantity=2,
                direction="long",
                today_quantity=1,
                yesterday_quantity=2,
                avg_price=3500.0,
                frozen_quantity=1,
                market_value=105900.0,
                unrealized_pnl=900.0,
                margin=10590.0,
                last_price=3530.0,
            ),
            UnifiedPosition(
                symbol="rb2601.SHFE",
                quantity=1,
                available_quantity=1,
                direction="short",
                avg_price=3560.0,
                unrealized_pnl=300.0,
                margin=3530.0,
            ),
        ]


def test_position_detail_exposes_both_sides_and_broker_figures() -> None:
    """get_position_detail(): 净数量与 get_position() 一致, 多空两边各自可读."""
    ex = _exec(_DetailGw())
    detail = ex.get_position_detail("rb2601.SHFE")

    assert detail.symbol == "rb2601.SHFE"
    assert detail.quantity == ex.get_position("rb2601.SHFE") == 2.0
    assert detail.available == 1.0
    assert detail.avg_price == 3500.0  # 净持仓在多头一侧, 取多头成本
    assert detail.unrealized_pnl == 1200.0  # 两边相加
    assert detail.margin == 14120.0
    assert detail.market_value is None  # 空头一侧没给, 合计不可知

    assert detail.long.quantity == 3.0
    assert detail.long.available == 2.0
    assert detail.long.today_quantity == 1.0
    assert detail.long.yesterday_quantity == 2.0
    assert detail.long.frozen_quantity == 1.0
    assert detail.long.last_price == 3530.0
    assert detail.short.quantity == 1.0
    assert detail.short.avg_price == 3560.0
    # 空头一侧柜台没给今昨仓拆分(两项都是 0 而总量不为 0) → None, 不是 0
    assert detail.short.today_quantity is None
    assert detail.short.yesterday_quantity is None


def test_position_detail_for_flat_symbol_is_all_zero() -> None:
    """没有持仓的标的: 数量为 0, 成本与柜台数值为 None."""
    detail = _exec(_DetailGw()).get_position_detail("000001.SZ")
    assert (detail.quantity, detail.available) == (0.0, 0.0)
    assert (detail.long.quantity, detail.short.quantity) == (0.0, 0.0)
    assert detail.avg_price is None
    assert detail.unrealized_pnl is None


def test_position_entry_price_reads_broker_cost_in_broker_live() -> None:
    """self.position.entry_price 经柜台交易时读柜台持仓成本."""
    strategy = _live_strategy(_DetailGw())
    strategy.ctx = _shadow_ctx()
    strategy.ctx.get_position_entry_price.return_value = 0.0
    strategy.current_bar = _bar("rb2601.SHFE", 0)
    assert strategy.position.entry_price == 3500.0
    assert strategy.get_position_detail().long.quantity == 3.0
