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
