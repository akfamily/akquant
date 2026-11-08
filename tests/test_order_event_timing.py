"""订单回调的时机约束(订单账本改造前锁定的现状).

用重写 ``_check_order_events`` 的方式在事件记录里插入 "CHECK" 标记, 区分
"同一次检查"与"下一次检查":
- 在撤单回调里补下的单, 同一次检查就收到 on_order(New);
- 在成交回调里下的单, 下一次检查才收到;
- 在回调里对已在引擎里的单发出撤单, 下一次检查才处理。
另有一条: 带在途订单存档续跑, 回调序列与不中断一致(在途单不重复通知)。

记录列表叫 ``events`` 而不是 ``log``: ``Strategy.log`` 是框架方法。
"""

from pathlib import Path
from typing import List

import akquant as aq
import pandas as pd
from akquant import Bar, OrderStatus, Strategy, run_from_checkpoint, save_checkpoint

TZ = "Asia/Shanghai"


def _frame(
    periods: int, start: str = "2024-01-02 09:31", price: float = 10.0
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="min", tz=TZ)
    return pd.DataFrame(
        {
            "date": idx,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1e6,
            "symbol": "X",
        }
    )


def _status(order: object) -> str:
    return str(getattr(order, "status")).split(".")[-1]


def _between(events: List[str], first: str, second: str) -> List[str]:
    i = events.index(first)
    j = events.index(second)
    assert i < j, (first, second, events)
    return events[i + 1 : j]


class _CancelThenResubmit(Strategy):
    def on_start(self) -> None:
        self.events: List[str] = []
        self.k = 0
        self.first = ""
        self.second = ""

    def _check_order_events(self) -> None:
        self.events.append("CHECK")
        super()._check_order_events()

    def on_bar(self, bar: Bar) -> None:
        self.k += 1
        if self.k == 1:
            self.first = self.buy("X", 100, price=1.0).primary
        if self.k == 2:
            self.cancel_order(self.first)

    def on_order(self, order: object) -> None:
        oid = getattr(order, "id")
        name = (
            "first"
            if oid == self.first
            else "second"
            if oid == self.second
            else "other"
        )
        self.events.append(f"{name} {_status(order)}")
        if oid == self.first and getattr(order, "status") == OrderStatus.Cancelled:
            self.second = self.buy("X", 100, price=2.0).primary


def test_resubmit_in_cancel_callback_is_notified_in_same_check() -> None:
    """撤单回调里补下的单, 同一次检查就收到 on_order(New)."""
    r = aq.run_backtest(
        data=_frame(6), strategy=_CancelThenResubmit, symbols="X", show_progress=False
    )
    events: List[str] = r.strategy.events  # type: ignore[union-attr]
    assert "CHECK" not in _between(events, "first Cancelled", "second New"), events


class _OrderFromTrade(Strategy):
    def on_start(self) -> None:
        self.events: List[str] = []
        self.k = 0
        self.child = ""

    def _check_order_events(self) -> None:
        self.events.append("CHECK")
        super()._check_order_events()

    def on_bar(self, bar: Bar) -> None:
        self.k += 1
        if self.k == 1:
            self.buy("X", 100)

    def on_trade(self, trade: object) -> None:
        if not self.child:
            self.events.append("trade")
            # 远离市价的买入限价单: 不成交、不受 T+1 可卖量影响
            self.child = self.buy("X", 100, price=1.0).primary

    def on_order(self, order: object) -> None:
        if getattr(order, "id") == self.child and _status(order) == "New":
            self.events.append("child New")


def test_order_from_trade_callback_is_notified_next_check() -> None:
    """成交回调里下的单, 下一次检查才收到 on_order(New)."""
    r = aq.run_backtest(
        data=_frame(6), strategy=_OrderFromTrade, symbols="X", show_progress=False
    )
    events: List[str] = r.strategy.events  # type: ignore[union-attr]
    assert "CHECK" in _between(events, "trade", "child New"), events


class _CancelInCallback(Strategy):
    """第 1 根挂远离市价的 target.

    第 3 根下 trigger, 在 trigger 的 on_order(New) 里撤 target.

    撤的是已进入引擎的单(第 1 根下的)。注意: 对本步刚提交、尚未进入引擎的单
    在回调里撤单, 现状是撤单请求丢失(另案记录, 不在本改造范围), 所以这里不那样写。
    """

    def on_start(self) -> None:
        self.events: List[str] = []
        self.k = 0
        self.target = ""
        self.trigger = ""

    def _check_order_events(self) -> None:
        self.events.append("CHECK")
        super()._check_order_events()

    def on_bar(self, bar: Bar) -> None:
        self.k += 1
        if self.k == 1:
            self.target = self.buy("X", 100, price=1.0).primary
        if self.k == 3:
            self.trigger = self.buy("X", 100, price=1.0).primary

    def on_order(self, order: object) -> None:
        oid = getattr(order, "id")
        if oid == self.trigger and _status(order) == "New":
            self.events.append("cancel sent")
            self.cancel_order(self.target)
        if oid == self.target:
            self.events.append(f"target {_status(order)}")


def test_cancel_in_callback_is_processed_next_check() -> None:
    """回调里对已在引擎里的单发出撤单, 下一次检查才处理."""
    r = aq.run_backtest(
        data=_frame(6), strategy=_CancelInCallback, symbols="X", show_progress=False
    )
    events: List[str] = r.strategy.events  # type: ignore[union-attr]
    assert "CHECK" in _between(events, "cancel sent", "target Cancelled"), events


class ResumeProbe(Strategy):
    """模块顶层定义以便 pickle; 第 1 根下远离市价的限价单, 之后一直挂着."""

    def on_start(self) -> None:
        """首次启动时初始化记录; 存档续跑时沿用."""
        if not hasattr(self, "events"):
            self.events: List[str] = []
            self.k = 0
            self.first = ""

    def on_bar(self, bar: Bar) -> None:
        """记录 bar 序号, 第 1 根下单."""
        self.k += 1
        self.events.append(f"bar{self.k}")
        if self.k == 1:
            self.first = self.buy("X", 100, price=1.0).primary

    def on_order(self, order: object) -> None:
        """记录订单状态."""
        tag = "first" if getattr(order, "id") == self.first else "other"
        self.events.append(f"{tag} {_status(order)}")


def test_checkpoint_resume_does_not_renotify_open_orders(tmp_path: Path) -> None:
    """带在途订单存档续跑: 在途单不重复通知, 回调序列与不中断一致."""
    full = aq.run_backtest(
        data=_frame(10), strategy=ResumeProbe, symbols="X", show_progress=False
    )

    data = _frame(10)
    first = aq.run_backtest(
        data=data.iloc[:4], strategy=ResumeProbe, symbols="X", show_progress=False
    )
    ckpt = tmp_path / "ckpt.pkl"
    save_checkpoint(first.engine, first.strategy, str(ckpt))  # type: ignore[arg-type]
    resumed = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=data.iloc[4:],
        symbols=["X"],
        show_progress=False,
    )
    assert resumed.strategy.events == full.strategy.events  # type: ignore[union-attr]
    assert resumed.strategy.events.count("first New") == 1  # type: ignore[union-attr]
