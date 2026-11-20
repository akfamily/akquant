"""回测下 get_position_detail(): 与 get_position() / get_account() 同口径."""

from __future__ import annotations

from typing import Any

import pandas as pd
from akquant import Bar, Strategy, run_backtest


class _HoldOne(Strategy):
    """第一根 bar 买 100 股, 之后每根 bar 记一次持仓明细."""

    def __init__(self) -> None:
        self.seen: list[Any] = []
        self.sent = False

    def on_bar(self, bar: Bar) -> None:
        if not self.sent:
            self.sent = True
            self.buy(quantity=100)
        if self.get_position() != 0:
            self.seen.append((self.get_position_detail(), self.get_account()))


def _bars() -> pd.DataFrame:
    closes = [10.0, 10.0, 11.0, 12.0]
    index = pd.date_range("2026-01-05", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "date": index,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1e6] * len(closes),
            "symbol": "X",
        }
    )


def test_backtest_position_detail_matches_position_and_account() -> None:
    """回测: 净数量、成本、浮动盈亏与引擎口径一致; 今昨仓等回测没有的项为 None."""
    strategy = _HoldOne()
    run_backtest(
        data=_bars(),
        strategy=strategy,
        symbols="X",
        initial_cash=100000.0,
        commission_rate=0.0,
        show_progress=False,
    )

    assert strategy.seen, "策略应当至少有一根持仓中的 bar"
    detail, account = strategy.seen[-1]
    assert detail.symbol == "X"
    assert detail.quantity == 100.0
    assert detail.long.quantity == 100.0
    assert detail.short.quantity == 0.0
    assert detail.avg_price == 10.0
    # (12 - 10) * 100。账户级 unrealized_pnl 只统计期货保证金账户, 股票恒为 0,
    # 所以这里不与 get_account() 比。
    assert detail.unrealized_pnl == 200.0
    assert detail.market_value == account["market_value"] == 1200.0
    assert detail.long.today_quantity is None
    assert detail.margin is None
