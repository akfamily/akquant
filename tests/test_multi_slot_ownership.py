"""多槽位：用户定时器与订单/成交回调只派给所属槽位."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, cast

import pandas as pd
from akquant import Bar, OrderStatus, Strategy, run_backtest
from akquant.strategy_order_events import assign_slot_ownership

TZ = "Asia/Shanghai"


def _minute(times: List[str], symbol: str = "600000.SH") -> pd.DataFrame:
    idx = pd.to_datetime(times).tz_localize(TZ)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )


LOG: Dict[str, List[str]] = {}


class _Base(Strategy):
    name = "base"

    def __init__(self) -> None:
        super().__init__()
        LOG.setdefault(self.name, [])
        self.bought = False

    def on_start(self) -> None:
        self.schedule_daily("10:30", f"t-{self.name}")

    def on_timer(self, payload: str) -> None:
        LOG[self.name].append(f"timer:{payload}")

    def on_bar(self, bar: Bar) -> None:
        if self.name == "a" and not self.bought:
            self.bought = True
            self.buy(bar.symbol, 100)

    def on_order(self, order: Any) -> None:
        LOG[self.name].append("order")

    def on_trade(self, trade: Any) -> None:
        LOG[self.name].append("trade")


class _A(_Base):
    name = "a"


class _B(_Base):
    name = "b"


def test_timers_and_order_callbacks_go_to_owning_slot_only() -> None:
    """各槽位只收到自己登记的定时器; 订单/成交回调只派给下单的槽位."""
    LOG.clear()
    run_backtest(
        strategy=_A,
        strategies_by_slot={"b": _B},
        data=_minute(["2024-01-02 10:00", "2024-01-02 11:00"]),
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert LOG["a"].count("timer:t-a") == 1
    assert "timer:t-b" not in LOG["a"]
    assert LOG["b"].count("timer:t-b") == 1
    assert "timer:t-a" not in LOG["b"]
    assert "order" in LOG["a"] and "trade" in LOG["a"]
    assert "order" not in LOG["b"] and "trade" not in LOG["b"]


class _Recorder(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.seen_orders: List[str] = []
        self.seen_trades: List[str] = []

    def on_order(self, order: Any) -> None:
        self.seen_orders.append(order.id)

    def on_trade(self, trade: Any) -> None:
        self.seen_trades.append(trade.order_id)


def _order(oid: str, owner: Any) -> SimpleNamespace:
    return SimpleNamespace(
        id=oid,
        status=OrderStatus.New,
        filled_quantity=0.0,
        average_filled_price=None,
        reject_reason="",
        owner_strategy_id=owner,
    )


def _trade(oid: str, owner: Any) -> SimpleNamespace:
    return SimpleNamespace(
        trade_id=f"t-{oid}",
        id=f"t-{oid}",
        order_id=oid,
        timestamp=1,
        symbol="600000.SH",
        side="Buy",
        quantity=100.0,
        price=10.0,
        owner_strategy_id=owner,
    )


def test_orders_without_owner_go_to_default_slot_only() -> None:
    """owner_strategy_id 为空(如实盘柜台侧产生的订单)或不在任何槽位的只派给默认槽位."""
    main, slot = _Recorder(), _Recorder()
    main._owner_strategy_id = "main"
    slot._owner_strategy_id = "b"
    assign_slot_ownership({"main": main, "b": slot}, "main")

    active = [
        _order("broker", None),
        _order("blank", ""),
        _order("unknown", "zzz"),
        _order("mine-main", "main"),
        _order("mine-b", "b"),
    ]
    trades = [_trade("broker", None), _trade("mine-b", "b")]
    for strategy in (main, slot):
        strategy.ctx = cast(
            Any,
            SimpleNamespace(
                canceled_order_ids=[],
                recent_rejected_orders=[],
                recent_expired_orders=[],
                orders=[],
                active_orders=active,
                recent_trades=trades,
            ),
        )
        strategy._check_order_events()

    assert main.seen_orders == ["broker", "blank", "unknown", "mine-main"]
    assert slot.seen_orders == ["mine-b"]
    assert main.seen_trades == ["broker"]
    assert slot.seen_trades == ["mine-b"]


def test_single_slot_receives_all_orders() -> None:
    """单槽位不按 owner 过滤(柜台侧订单与任意 owner 都送达)."""
    only = _Recorder()
    only._owner_strategy_id = "main"
    assign_slot_ownership({"main": only}, "main")
    only.ctx = cast(
        Any,
        SimpleNamespace(
            canceled_order_ids=[],
            recent_rejected_orders=[],
            recent_expired_orders=[],
            orders=[],
            active_orders=[_order("x", None), _order("y", "other")],
            recent_trades=[],
        ),
    )
    only._check_order_events()
    assert only.seen_orders == ["x", "y"]


def test_foreign_timer_flushes_order_events_without_on_timer() -> None:
    """别的槽位的定时器: 不调本槽位 on_timer, 但本步成交照常回调(否则会丢)."""
    timers: List[str] = []

    class _Slot(_Recorder):
        def on_timer(self, payload: str) -> None:
            timers.append(payload)

    slot = _Slot()
    slot._owner_strategy_id = "b"
    assign_slot_ownership({"main": _Recorder(), "b": slot}, "main")
    ctx: Any = SimpleNamespace(
        canceled_order_ids=[],
        recent_rejected_orders=[],
        recent_expired_orders=[],
        recent_expiry_events=[],
        orders=[],
        active_orders=[],
        recent_trades=[_trade("mine-b", "b")],
        current_time=0,
    )
    slot._on_foreign_timer_event_and_flush("t-main", ctx)
    assert timers == []
    assert slot.seen_trades == ["mine-b"]


def _minute_px(times: List[str], prices: List[float]) -> pd.DataFrame:
    idx = pd.to_datetime(times).tz_localize(TZ)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "600000.SH",
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 1e6,
        }
    )


_CANCEL_TIMES = ["2024-01-02 10:00", "2024-01-02 10:01", "2024-01-02 10:02"]


def _status_log(owner_log: List[str], order: Any) -> None:
    owner_log.append(f"{order.id}:{str(order.status).split('.')[-1]}")


def test_cross_slot_cancel_delivers_cancelled_to_owner_once() -> None:
    """B 撤 A 的挂单: A 收到一次 on_order(Cancelled), B 什么都收不到."""
    logs: Dict[str, List[str]] = {"a": [], "b": []}
    shared: Dict[str, str] = {}

    class _Owner(Strategy):
        def on_bar(self, bar: Bar) -> None:
            if "oid" not in shared:
                receipt = self.buy(bar.symbol, 100, price=5.0)
                shared["oid"] = str(getattr(receipt, "primary", receipt))

        def on_order(self, order: Any) -> None:
            _status_log(logs["a"], order)

    class _Canceller(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def on_bar(self, bar: Bar) -> None:
            self.n += 1
            if self.n == 2:
                self.cancel_order(shared["oid"])

        def on_order(self, order: Any) -> None:
            _status_log(logs["b"], order)

    run_backtest(
        strategy=_Owner,
        strategies_by_slot={"b": _Canceller},
        data=_minute_px(_CANCEL_TIMES, [10.0, 10.0, 10.0]),
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    oid = shared["oid"]
    assert logs["a"].count(f"{oid}:Cancelled") == 1, logs
    assert logs["b"] == []


def test_same_slot_cancel_still_delivers_cancelled_once() -> None:
    """本槽位自己撤单: 仍只回调一次 Cancelled(不因终态快照重复)."""
    logs: Dict[str, List[str]] = {"a": [], "b": []}

    class _Self(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0
            self.oid = ""

        def on_bar(self, bar: Bar) -> None:
            self.n += 1
            if self.n == 1:
                receipt = self.buy(bar.symbol, 100, price=5.0)
                self.oid = str(getattr(receipt, "primary", receipt))
            elif self.n == 2:
                self.cancel_order(self.oid)

        def on_order(self, order: Any) -> None:
            _status_log(logs["a"], order)

    class _Idle(Strategy):
        def on_order(self, order: Any) -> None:
            _status_log(logs["b"], order)

    run_backtest(
        strategy=_Self,
        strategies_by_slot={"b": _Idle},
        data=_minute_px(_CANCEL_TIMES, [10.0, 10.0, 10.0]),
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert sum(1 for x in logs["a"] if x.endswith(":Cancelled")) == 1, logs
    assert logs["b"] == []


def test_bracket_child_callbacks_go_to_parent_slot_only() -> None:
    """B 下的 bracket: 进场单、止损/止盈子单及 OCO 撤单回调都只派给 B."""
    logs: Dict[str, List[str]] = {"a": [], "b": []}
    trades: Dict[str, List[str]] = {"a": [], "b": []}

    class _Idle(Strategy):
        def on_order(self, order: Any) -> None:
            _status_log(logs["a"], order)

        def on_trade(self, trade: Any) -> None:
            trades["a"].append(trade.order_id)

    class _Bracket(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.placed = False

        def on_bar(self, bar: Bar) -> None:
            if not self.placed:
                self.placed = True
                self.place_bracket(
                    bar.symbol, 100, stop_trigger_price=9.5, take_profit_price=11.0
                )

        def on_order(self, order: Any) -> None:
            assert order.owner_strategy_id == "b"
            _status_log(logs["b"], order)

        def on_trade(self, trade: Any) -> None:
            trades["b"].append(trade.order_id)

    run_backtest(
        strategy=_Idle,
        strategies_by_slot={"b": _Bracket},
        data=_minute_px(
            [
                "2024-01-02 10:00",
                "2024-01-02 10:01",
                "2024-01-02 10:02",
                "2024-01-02 10:03",
            ],
            [10.0, 10.0, 10.5, 11.5],
        ),
        symbols=["600000.SH"],
        initial_cash=1e6,
        show_progress=False,
    )
    assert logs["a"] == [] and trades["a"] == []
    ids = {x.split(":")[0] for x in logs["b"]}
    assert len(ids) == 3, logs
    assert sum(1 for x in logs["b"] if x.endswith(":Cancelled")) == 1, logs
    assert len(trades["b"]) == 2, trades
