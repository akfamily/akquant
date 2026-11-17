# -*- coding: utf-8 -*-
"""回测下 cancel_group: 按组号撤掉组内全部订单, 而不是把组号当订单号去撤."""

from __future__ import annotations

from typing import Any, List

import pandas as pd
from akquant import Bar, Strategy, run_backtest

SYMBOL = "T"


def _bars(n: int = 5) -> List[Bar]:
    stamps = pd.date_range("2023-01-03", periods=n, freq="D", tz="Asia/Shanghai")
    return [
        Bar(
            timestamp=(ts + pd.Timedelta(hours=10)).value,
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
            volume=10_000.0,
            symbol=SYMBOL,
        )
        for ts in stamps
    ]


class _OcoThenCancel(Strategy):
    """第 1 根挂两笔远离市价的限价买单并组成 OCO, 第 2 根按组号撤, 第 3 根看结果."""

    def __init__(self, cancel_with: str) -> None:
        super().__init__()
        self.cancel_with = cancel_with
        self.seen = 0
        self.open_before = -1
        self.open_after = -1
        self.group: Any = None

    def on_bar(self, bar: Bar) -> None:
        self.seen += 1
        if self.seen == 1:
            first = self.buy(bar.symbol, 100, price=5.0)
            second = self.buy(bar.symbol, 100, price=6.0)
            gid = self.place_oco(first, second, group_id=self.cancel_with or None)
            self.group = gid
        elif self.seen == 2:
            self.open_before = len(self.get_open_orders(bar.symbol))
            self.cancel_group(self.group)
        elif self.seen == 3:
            self.open_after = len(self.get_open_orders(bar.symbol))


def _run(strategy: Strategy) -> Any:
    return run_backtest(
        data=_bars(),
        strategy=strategy,
        symbols=SYMBOL,
        initial_cash=1e6,
        commission_rate=0.0,
        show_progress=False,
    )


def test_cancel_group_cancels_both_oco_legs_with_auto_group_id() -> None:
    """place_oco 自动生成的组号: 两条腿都撤掉."""
    strategy = _OcoThenCancel(cancel_with="")
    result = _run(strategy)
    assert strategy.open_before == 2
    assert strategy.open_after == 0
    assert result.orders_df["status"].tolist() == ["cancelled", "cancelled"]


def test_cancel_group_cancels_both_oco_legs_with_custom_group_id() -> None:
    """调用方自己起的组号同样生效."""
    strategy = _OcoThenCancel(cancel_with="my-exit")
    _run(strategy)
    assert strategy.group == "my-exit"
    assert strategy.open_after == 0


def test_cancel_group_twice_is_harmless() -> None:
    """同一组撤第二次: 组已不在, 不抛错."""

    class _Twice(_OcoThenCancel):
        def on_bar(self, bar: Bar) -> None:
            super().on_bar(bar)
            if self.seen == 4:
                self.cancel_group(self.group)

    strategy = _Twice(cancel_with="")
    _run(strategy)
    assert strategy.open_after == 0


def test_cancel_group_with_single_order_receipt_still_cancels_it() -> None:
    """传单笔下单回执: 撤掉这笔订单(原有行为)."""

    class _Single(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.seen = 0
            self.open_after = -1
            self.receipt: Any = None

        def on_bar(self, bar: Bar) -> None:
            self.seen += 1
            if self.seen == 1:
                self.receipt = self.buy(bar.symbol, 100, price=5.0)
            elif self.seen == 2:
                self.cancel_group(self.receipt)
            elif self.seen == 3:
                self.open_after = len(self.get_open_orders(bar.symbol))

    strategy = _Single()
    _run(strategy)
    assert strategy.open_after == 0
