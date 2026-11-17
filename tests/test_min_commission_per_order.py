# -*- coding: utf-8 -*-
"""最低佣金按委托收, 不按成交收.

量限(``volume_limit_pct``, A 股标准模板默认开)会把一笔委托拆成多笔成交。修复前
每笔成交各收一次最低佣金: 300 股拆 3 笔, 佣金从 5 元变成 15 元。券商的规则是整笔
委托的佣金不足最低佣金时按最低佣金收。
"""

from __future__ import annotations

from typing import Any, List

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy

SYMBOL = "X"
RATE = 0.0003
MIN_COMMISSION = 5.0


def _frame(price: float = 10.0, bars: int = 8) -> pd.DataFrame:
    idx = pd.date_range(
        "2024-01-02 09:31", periods=bars, freq="min", tz="Asia/Shanghai"
    )
    return pd.DataFrame(
        {
            "date": idx,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1000.0,
            "symbol": SYMBOL,
        }
    )


def _one_order_strategy(quantity: float, side: str = "buy") -> type[Strategy]:
    class _One(Strategy):
        def __init__(self) -> None:
            self.done = False

        def on_bar(self, bar: Bar) -> None:
            if self.done:
                return
            self.done = True
            if side == "buy":
                self.buy(bar.symbol, quantity)
            else:
                self.sell(bar.symbol, quantity)

    return _One


def _run(quantity: float, price: float = 10.0, **kwargs: Any) -> Any:
    options: dict[str, Any] = dict(
        data=_frame(price),
        strategy=_one_order_strategy(quantity),
        symbols=[SYMBOL],
        show_progress=False,
        initial_cash=10_000_000.0,
        commission_rate=RATE,
        min_commission=MIN_COMMISSION,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        lot_size=100,
    )
    options.update(kwargs)
    return aq.run_backtest(**options)


def _fills(result: Any) -> List[float]:
    return [float(t.commission) for t in result.executions]


def test_single_fill_charges_minimum_once() -> None:
    """不拆单: 一笔成交, 佣金 5 元(对照)."""
    result = _run(300)
    assert _fills(result) == [MIN_COMMISSION]


def test_split_small_order_charges_minimum_once() -> None:
    """300 股拆成 3 笔成交: 整笔委托佣金 0.9 元不足 5 元, 合计只收 5 元."""
    result = _run(300, volume_limit_pct=0.1)
    fills = _fills(result)
    assert len(fills) == 3
    assert sum(fills) == pytest.approx(MIN_COMMISSION)
    # 最低佣金在首笔成交收齐, 后续成交不再补收
    assert fills == pytest.approx([MIN_COMMISSION, 0.0, 0.0])
    assert float(result.orders_df["commission"].iloc[0]) == pytest.approx(
        MIN_COMMISSION
    )


def test_split_order_crossing_minimum_charges_rate_on_whole_order() -> None:
    """拆单后累计佣金超过最低佣金: 合计等于按整笔委托计的比例佣金.

    500 股 × 100 元 × 万三 = 15 元, 拆 5 笔(每笔 3 元)。首笔收 5 元, 第二笔累计
    6 元补 1 元, 之后每笔 3 元, 合计 15 元; 修复前是 5 笔 × 5 元 = 25 元。
    """
    result = _run(500, price=100.0, volume_limit_pct=0.1)
    fills = _fills(result)
    assert len(fills) == 5
    assert fills == pytest.approx([5.0, 1.0, 3.0, 3.0, 3.0])
    assert sum(fills) == pytest.approx(500 * 100.0 * RATE)


def test_split_order_keeps_per_fill_taxes() -> None:
    """按成交额计的过户费仍逐笔收, 只有佣金部分按委托累计."""
    transfer = 0.001
    result = _run(300, volume_limit_pct=0.1, transfer_fee_rate=transfer)
    fills = _fills(result)
    per_fill_transfer = 100 * 10.0 * transfer
    assert fills == pytest.approx(
        [MIN_COMMISSION + per_fill_transfer, per_fill_transfer, per_fill_transfer]
    )


def test_two_separate_orders_each_charge_minimum() -> None:
    """两笔独立委托各收一次最低佣金(按委托累计不能串到别的委托上)."""

    class _Two(Strategy):
        def __init__(self) -> None:
            self.count = 0

        def on_bar(self, bar: Bar) -> None:
            if self.count < 2:
                self.count += 1
                self.buy(bar.symbol, 100)

    result = _run(100, strategy=_Two)
    assert _fills(result) == [MIN_COMMISSION, MIN_COMMISSION]
