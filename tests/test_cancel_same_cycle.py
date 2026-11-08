"""同一次策略回调里下单后又撤掉这张单, 撤单必须生效(issue #440).

回测下单经事件队列送进引擎, 要等策略回调返回后的通道阶段才过风控、登记为挂单;
而回调里的撤单此前在回调一返回就同步处理, 那时订单还在队列里, 引擎不认识它 →
"Ignored cancel request for unknown order"、撤单丢失、订单照常成交, Python 侧
``get_order`` 却一直显示 Cancelled。

现在引擎把这类撤单记下, 等订单通过风控进入引擎时立即撤掉(撮合之前), 策略收到
``on_order(Cancelled)``; 风控拒掉的单静默作罢; 通道阶段结束仍不认识的订单号才按
原样告警。``get_order`` 的"已请求撤单 → 显示 Cancelled"只覆盖未完成的单。
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import akquant as aq
import pandas as pd
import pytest
from akquant import (
    AssetType,
    Bar,
    CurrentClose,
    Instrument,
    OrderStatus,
    Strategy,
    run_live,
)

SYMBOL = "600000"
UNKNOWN_CANCEL = "Ignored cancel request for unknown order"
N_BARS = 12
PRICES = [10.0] * N_BARS
START = "2024-01-02 09:31"


def _index() -> pd.DatetimeIndex:
    return pd.date_range(START, periods=N_BARS, freq="1min", tz="Asia/Shanghai")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": _index(),
            "open": PRICES,
            "high": PRICES,
            "low": PRICES,
            "close": PRICES,
            "volume": 1e6,
            "symbol": SYMBOL,
        }
    )


def _bars() -> List[Bar]:
    return [
        Bar(int(ts.value), p, p, p, p, 1e6, SYMBOL) for ts, p in zip(_index(), PRICES)
    ]


def _instrument() -> Instrument:
    return Instrument(
        symbol=SYMBOL,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _status_name(status: Any) -> str:
    return str(status).split(".")[-1]


class _Recorder(Strategy):
    """记录本单的 on_order 状态序列、每根 bar 的持仓与 get_order 状态."""

    def __init__(self) -> None:
        super().__init__()
        self.k = 0
        self.oid = ""
        self.statuses: List[str] = []
        self.pos_seen: List[float] = []
        self.get_order_statuses: List[Optional[str]] = []

    def on_bar(self, bar: Bar) -> None:
        self.k += 1
        self.act(bar)
        self.pos_seen.append(float(self.get_position(SYMBOL)))
        if self.oid:
            got = self.get_order(self.oid)
            self.get_order_statuses.append(
                None if got is None else _status_name(got.status)
            )

    def act(self, bar: Bar) -> None:
        """子类在这里下单/撤单."""

    def on_order(self, order: Any) -> None:
        if self.oid and order.id == self.oid:
            self.statuses.append(_status_name(order.status))


class _CancelInNewCallback(_Recorder):
    """第 1 根 bar 下市价单, 在它的 on_order(New) 回调里撤单."""

    def act(self, bar: Bar) -> None:
        if self.k == 1:
            self.oid = self.buy(SYMBOL, 100).primary

    def on_order(self, order: Any) -> None:
        super().on_order(order)
        if order.id == self.oid and order.status == OrderStatus.New:
            self.cancel_order(self.oid)


class _CancelSameBar(_Recorder):
    """第 1 根 bar 下市价单后在同一个 on_bar 里立即撤单."""

    def act(self, bar: Bar) -> None:
        if self.k == 1:
            self.oid = self.buy(SYMBOL, 100).primary
            self.cancel_order(self.oid)


def _unknown_cancel_warnings(caplog: pytest.LogCaptureFixture) -> List[str]:
    return [r.getMessage() for r in caplog.records if UNKNOWN_CANCEL in r.getMessage()]


def _run(strategy: Strategy, **kwargs: Any) -> Any:
    return aq.run_backtest(
        data=_frame(),
        strategy=strategy,
        symbols=SYMBOL,
        show_progress=False,
        **kwargs,
    )


def _assert_cancelled_never_filled(strategy: _Recorder, result: Any) -> None:
    assert strategy.statuses == ["New", "Cancelled"]
    assert all(p == 0.0 for p in strategy.pos_seen), strategy.pos_seen
    assert strategy.get_order_statuses[-1] == "Cancelled"
    orders = result.orders_df
    assert len(orders) == 1
    assert orders["status"].iloc[0] == "cancelled"
    assert float(orders["filled_quantity"].iloc[0]) == 0.0


def test_cancel_in_on_order_new_callback(caplog: pytest.LogCaptureFixture) -> None:
    """在该单的 on_order(New) 回调里撤: New → Cancelled, 从不成交、不告警."""
    strategy = _CancelInNewCallback()
    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = _run(strategy)
    _assert_cancelled_never_filled(strategy, result)
    assert _unknown_cancel_warnings(caplog) == []


def test_cancel_in_same_on_bar(caplog: pytest.LogCaptureFixture) -> None:
    """同一个 on_bar 里下单后立即撤: 同上."""
    strategy = _CancelSameBar()
    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = _run(strategy)
    _assert_cancelled_never_filled(strategy, result)
    assert _unknown_cancel_warnings(caplog) == []


def test_cancel_same_bar_with_current_close_fill() -> None:
    """当根收盘就能成交的设置下同回调下单撤单: 也不成交."""
    strategy = _CancelSameBar()
    result = _run(strategy, fill_policy=CurrentClose())
    _assert_cancelled_never_filled(strategy, result)


class _CancelInTimer(_Recorder):
    """定时器回调里下单后立即撤单."""

    def on_start(self) -> None:
        self.schedule(pd.Timestamp("2024-01-02 09:32:30", tz="Asia/Shanghai"), "go")

    def on_timer(self, payload: str) -> None:
        if payload == "go":
            self.oid = self.buy(SYMBOL, 100).primary
            self.cancel_order(self.oid)


def test_cancel_same_on_timer(caplog: pytest.LogCaptureFixture) -> None:
    """on_timer 里下单后立即撤: 不成交、收到 Cancelled."""
    strategy = _CancelInTimer()
    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = _run(strategy)
    _assert_cancelled_never_filled(strategy, result)
    assert _unknown_cancel_warnings(caplog) == []


class _CancelInWindowBar(_Recorder):
    """5 分钟窗口 bar 回调里下单后立即撤单."""

    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars("5min")
        self.windows = 0

    def on_window_bar(self, bar: Bar) -> None:
        self.windows += 1
        if self.windows == 1:
            self.oid = self.buy(SYMBOL, 100).primary
            self.cancel_order(self.oid)


def test_cancel_same_window_bar_callback(caplog: pytest.LogCaptureFixture) -> None:
    """窗口 bar 回调里下单后立即撤: 不成交、收到 Cancelled."""
    strategy = _CancelInWindowBar()
    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = _run(strategy)
    assert strategy.windows >= 1
    _assert_cancelled_never_filled(strategy, result)
    assert _unknown_cancel_warnings(caplog) == []


class _CancelUnknownId(_Recorder):
    def act(self, bar: Bar) -> None:
        if self.k == 1:
            self.cancel_order("no-such-order")


def test_cancel_unknown_order_id_still_warns(caplog: pytest.LogCaptureFixture) -> None:
    """撤一个根本不存在的订单号: 照旧告警 unknown order."""
    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run(_CancelUnknownId())
    assert len(_unknown_cancel_warnings(caplog)) == 1


class _CancelRejected(_Recorder):
    """资金不足的大单被风控拒绝, 同回调里撤它."""

    def act(self, bar: Bar) -> None:
        if self.k == 1:
            self.oid = self.buy(SYMBOL, 10_000_000).primary
            self.cancel_order(self.oid)


def test_cancel_risk_rejected_order_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """风控拒单后同回调里撤: 回调里有 Rejected, 不出 unknown order 告警."""
    strategy = _CancelRejected()
    with caplog.at_level(logging.WARNING, logger="akquant"):
        result = _run(strategy, initial_cash=100_000.0)
    assert "Rejected" in strategy.statuses
    assert "Filled" not in strategy.statuses
    assert all(p == 0.0 for p in strategy.pos_seen)
    assert result.orders_df["status"].iloc[0] == "rejected"
    assert _unknown_cancel_warnings(caplog) == []


def test_cancel_same_bar_replay_paper(caplog: pytest.LogCaptureFixture) -> None:
    """run_live(broker='replay') 下同一个 on_bar 里下单撤单: 同样不成交."""
    strategy = _CancelSameBar()
    with caplog.at_level(logging.WARNING, logger="akquant"):
        run_live(
            strategy_cls=strategy,
            instruments=[_instrument()],
            broker="replay",
            trading_mode="paper",
            gateway_options={"bars": _bars(), "freq": "1min"},
            cash=1_000_000.0,
            show_progress=False,
            duration="60s",
        )
    assert strategy.k == N_BARS
    assert strategy.statuses == ["New", "Cancelled"]
    assert all(p == 0.0 for p in strategy.pos_seen), strategy.pos_seen
    assert strategy.get_order_statuses[-1] == "Cancelled"
    assert _unknown_cancel_warnings(caplog) == []


class _CancelAfterFill(_Recorder):
    """第 1 根 bar 下市价单, 成交后再撤它."""

    def act(self, bar: Bar) -> None:
        if self.k == 1:
            self.oid = self.buy(SYMBOL, 100).primary
        elif self.k == 4:
            self.cancel_order(self.oid)


def test_get_order_keeps_terminal_status_after_cancel_request() -> None:
    """已成交的单再请求撤单: get_order 仍是 Filled, 不被改显示为 Cancelled."""
    strategy = _CancelAfterFill()
    result = _run(strategy)
    assert strategy.statuses == ["New", "Filled"]
    # 第 1 根下单时 get_order 尚无记录; 第 2 根起已成交, 第 4 根请求撤单之后仍是 Filled
    assert strategy.get_order_statuses[1:] == ["Filled"] * (N_BARS - 1)
    assert result.orders_df["status"].iloc[0] == "filled"
