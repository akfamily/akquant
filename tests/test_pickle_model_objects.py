"""模型对象 pickle 往返，以及带在途订单的策略存档续跑."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy, run_backtest, run_from_checkpoint, save_checkpoint


def _frame(times: List[str]) -> pd.DataFrame:
    idx = pd.to_datetime(times).tz_localize("Asia/Shanghai")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "600000.SH",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e6,
        }
    )


def test_bar_and_tick_roundtrip() -> None:
    """Bar/Tick 可 pickle 往返."""
    bar = Bar(
        timestamp=1,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10.0,
        symbol="A",
        interval_start=0,
    )
    b2 = pickle.loads(pickle.dumps(bar))
    assert (b2.timestamp, b2.close, b2.symbol, b2.interval_start) == (
        bar.timestamp,
        1.5,
        "A",
        0,
    )
    tick = aq.Tick(timestamp=5, price=3.0, volume=2.0, symbol="A")
    t2 = pickle.loads(pickle.dumps(tick))
    assert (t2.timestamp, t2.price, t2.volume, t2.symbol) == (
        tick.timestamp,
        3.0,
        2.0,
        "A",
    )


class _LimitAndHold(Strategy):
    """首根 bar 挂一张够不着的 GTC 限价单并自己持有回执；记录 on_order."""

    def __init__(self) -> None:
        super().__init__()
        self.my_order: Any = None
        self.statuses: List[str] = []

    def on_bar(self, bar: Bar) -> None:
        if self.my_order is None:
            self.my_order = self.buy(
                bar.symbol, 100, price=9.0, time_in_force=aq.TimeInForce.GTC
            )

    def on_order(self, order: Any) -> None:
        self.statuses.append(str(order.status).split(".")[-1])


def test_order_roundtrip_and_checkpoint_with_open_order(tmp_path: Path) -> None:
    """带在途订单的策略可存档并续跑."""
    r1 = run_backtest(
        data=_frame(["2024-01-02 10:00", "2024-01-02 10:01"]),
        strategy=_LimitAndHold,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    strat = r1.strategy
    assert r1.engine is not None
    assert strat is not None and strat._known_orders, "前提：存档时有在途订单"
    order = next(iter(strat._known_orders.values()))
    o2 = pickle.loads(pickle.dumps(order))
    assert (o2.id, o2.symbol, o2.quantity, o2.status) == (
        order.id,
        order.symbol,
        order.quantity,
        order.status,
    )
    ckpt = tmp_path / "c.pkl"
    save_checkpoint(r1.engine, strat, str(ckpt))  # 修复前 TypeError
    frame2 = _frame(["2024-01-02 10:02"])
    frame2["low"] = 8.5  # 续跑段价格触及限价，在途单应成交
    r2 = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=frame2,
        symbols=["600000.SH"],
        show_progress=False,
    )
    filled = r2.orders_df[r2.orders_df["filled_quantity"] > 0]
    assert len(filled) == 1 and float(filled.iloc[0]["avg_price"]) == 9.0


class _DayLimit(Strategy):
    """首根 bar 挂一张够不着的 Day 限价单."""

    def __init__(self) -> None:
        super().__init__()
        self.sent = False

    def on_bar(self, bar: Bar) -> None:
        if not self.sent:
            self.sent = True
            self.buy(bar.symbol, 100, price=9.0, time_in_force=aq.TimeInForce.Day)


def test_day_order_pending_across_checkpoint_expires_next_day(tmp_path: Path) -> None:
    """跨日续跑：前一日的 Day 单不再成交，按日终规则过期."""
    r1 = run_backtest(
        data=_frame(["2024-01-02 10:00", "2024-01-02 10:01"]),
        strategy=_DayLimit,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r1.engine is not None and r1.strategy is not None
    ckpt = tmp_path / "d.pkl"
    save_checkpoint(r1.engine, r1.strategy, str(ckpt))
    frame2 = _frame(["2024-01-03 10:00"])
    frame2["low"] = 8.5
    r2 = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=frame2,
        symbols=["600000.SH"],
        show_progress=False,
    )
    assert (r2.orders_df["filled_quantity"] > 0).sum() == 0
    assert str(r2.orders_df.iloc[0]["status"]).lower() == "expired"


class _DayLimitWithTimer(_DayLimit):
    """同 _DayLimit，但 on_start 登记一个晚于最后一根 bar 的定时器（时钟会越过行情）."""

    def on_start(self) -> None:
        self.schedule("2024-01-02 14:00:00", "t")

    def on_timer(self, payload: str) -> None:
        pass

    def on_bar(self, bar: Bar) -> None:
        if (
            not self.sent
            and len(self.get_open_orders()) == 0
            and self._bars_seen() >= 1
        ):
            self.sent = True
            self.buy(bar.symbol, 100, price=9.0, time_in_force=aq.TimeInForce.Day)
        self._n = getattr(self, "_n", 0) + 1

    def _bars_seen(self) -> int:
        return getattr(self, "_n", 0)


def test_day_order_on_last_bar_with_later_timer_same_as_continuous(
    tmp_path: Path,
) -> None:
    """最后一根 bar 上下的 Day 单 + 之后的定时器：续跑与连续回测结果一致（都成交）."""
    d1 = _frame(["2024-01-02 10:00", "2024-01-02 10:01"])
    d2 = _frame(["2024-01-03 10:00"])
    d2["low"] = 8.5
    full = run_backtest(
        data=pd.concat([d1, d2], ignore_index=True),
        strategy=_DayLimitWithTimer,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    r1 = run_backtest(
        data=d1,
        strategy=_DayLimitWithTimer,
        symbols=["600000.SH"],
        initial_cash=1e5,
        show_progress=False,
    )
    assert r1.engine is not None and r1.strategy is not None
    ckpt = tmp_path / "t.pkl"
    save_checkpoint(r1.engine, r1.strategy, str(ckpt))
    r2 = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=d2,
        symbols=["600000.SH"],
        show_progress=False,
    )
    for r in (full, r2):
        filled = r.orders_df[r.orders_df["filled_quantity"] > 0]
        assert len(filled) == 1 and float(filled.iloc[0]["avg_price"]) == 9.0


def test_restore_functions_not_exported() -> None:
    """还原函数不进 __all__，但 pickle 仍可用."""
    assert not [n for n in aq.__all__ if n.startswith("_restore_")]
    assert hasattr(aq.akquant, "_restore_order")
    bar = Bar(timestamp=1, open=1, high=1, low=1, close=1, volume=1, symbol="A")
    assert pickle.loads(pickle.dumps(bar)).symbol == "A"


def test_order_roundtrip_all_fields() -> None:
    """非默认字段的 Order 往返后逐属性一致."""
    o = aq.Order(
        id="o1",
        symbol="X",
        side=aq.OrderSide.Sell,
        order_type=aq.OrderType.StopLimit,
        quantity=300.0,
        price=11.5,
        time_in_force=aq.TimeInForce.GTC,
        trigger_price=12.25,
        tag="my-tag",
        trail_offset=0.5,
        trail_reference_price=12.0,
        reduce_only=True,
    )
    o.status = aq.OrderStatus.PartiallyFilled
    o.filled_quantity = 100.0
    o.average_filled_price = 11.75
    o2 = pickle.loads(pickle.dumps(o))
    for attr in (
        "id",
        "symbol",
        "side",
        "order_type",
        "quantity",
        "price",
        "time_in_force",
        "trigger_price",
        "tag",
        "trail_offset",
        "trail_reference_price",
        "reduce_only",
        "status",
        "filled_quantity",
        "average_filled_price",
        "created_at",
        "updated_at",
        "position_effect",
        "order_role",
    ):
        assert getattr(o2, attr) == getattr(o, attr), attr
    assert o2.status == aq.OrderStatus.PartiallyFilled
    assert o2.filled_quantity == 100.0 and o2.average_filled_price == 11.75


def test_trade_roundtrip() -> None:
    """Trade 往返后逐属性一致."""
    t = aq.Trade(
        id="t1",
        order_id="o1",
        symbol="X",
        side=aq.OrderSide.Buy,
        quantity=100.0,
        price=9.5,
        commission=1.25,
        timestamp=123456789,
        bar_index=7,
        owner_strategy_id="s1",
    )
    t2 = pickle.loads(pickle.dumps(t))
    for attr in (
        "id",
        "order_id",
        "symbol",
        "side",
        "quantity",
        "price",
        "commission",
        "timestamp",
        "bar_index",
        "owner_strategy_id",
        "position_effect",
    ):
        assert getattr(t2, attr) == getattr(t, attr), attr


def test_bar_roundtrip_with_extra_and_freq() -> None:
    """带 extra 与 freq 的 Bar 往返."""
    bar = Bar(
        timestamp=2,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10.0,
        symbol="A",
        extra={"pe": 12.5},
        freq="5min",
    )
    b2 = pickle.loads(pickle.dumps(bar))
    assert b2.extra == {"pe": 12.5} and b2.freq == "5min"
    assert (b2.open, b2.high, b2.low, b2.volume) == (1.0, 2.0, 0.5, 10.0)
