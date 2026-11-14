# -*- coding: utf-8 -*-
"""回测中 get_account()["frozen_cash"] 的取值.

冻结资金 = 在途订单成交后多占用的保证金。引擎在回测运行期间不再每根 bar 预先算好,
而是策略读到时才算; 这里锁定读到的数值, 以及"不读也不影响别的"。
"""

from typing import Any, Dict, List

import akquant as aq
import pandas as pd
from akquant import Strategy

CLOSES = [10.0, 11.0, 12.0, 13.0, 14.0]


def _data() -> pd.DataFrame:
    days = pd.date_range("2024-01-02", periods=len(CLOSES), freq="B")
    return pd.DataFrame(
        {
            "date": days.tz_localize("Asia/Shanghai"),
            "symbol": "AAA",
            "open": CLOSES,
            "high": [c + 0.5 for c in CLOSES],
            "low": [c - 0.5 for c in CLOSES],
            "close": CLOSES,
            "volume": 1e6,
        }
    )


class _Base(Strategy):
    seen: List[Dict[str, Any]]

    def on_start(self) -> None:
        self.seen = []
        self.n = 0


def _run(cls: Any) -> Any:
    holder: Dict[str, Any] = {}

    class Wrapped(cls):  # type: ignore[misc, valid-type]
        def on_stop(self) -> None:
            holder["seen"] = self.seen

    result = aq.run_backtest(
        strategy=Wrapped,
        data=_data(),
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        show_progress=False,
    )
    return holder["seen"], result


def test_resting_order_reserves_margin_at_current_price() -> None:
    """挂着一笔不会成交的限价买单: 冻结资金 = 未成交数量 * 当前价, 撤单后归零."""

    class S(_Base):
        def on_bar(self, bar: Any) -> None:
            self.n += 1
            account = self.get_account()
            self.seen.append(
                {
                    "n": self.n,
                    "frozen": account["frozen_cash"],
                    # 同一回调里再读一次, 数值不变
                    "again": self.get_account()["frozen_cash"],
                    "open_orders": len(self.get_open_orders()),
                }
            )
            if self.n == 1:
                self.buy(symbol=bar.symbol, quantity=200, price=1.0)
            elif self.n == 3:
                for order in self.get_open_orders():
                    self.cancel_order(order.id)

    seen, _ = _run(S)
    by_n = {row["n"]: row for row in seen}
    # 第 1 根: 还没有在途订单
    assert by_n[1]["open_orders"] == 0
    assert by_n[1]["frozen"] == 0.0
    # 第 2、3 根: 限价 1.0 的买单挂着, 按当根收盘价计
    for n in (2, 3):
        assert by_n[n]["open_orders"] == 1
        assert by_n[n]["frozen"] == 200 * CLOSES[n - 1]
        assert by_n[n]["again"] == by_n[n]["frozen"]
    # 第 3 根撤单后: 第 4、5 根没有在途订单
    for n in (4, 5):
        assert by_n[n]["open_orders"] == 0
        assert by_n[n]["frozen"] == 0.0


def test_existing_position_is_not_counted() -> None:
    """已有持仓的保证金不算冻结: 只算在途订单新增的部分."""

    class S(_Base):
        def on_bar(self, bar: Any) -> None:
            self.n += 1
            self.seen.append(
                {
                    "n": self.n,
                    "frozen": self.get_account()["frozen_cash"],
                    "position": self.get_position(bar.symbol),
                }
            )
            if self.n == 1:
                self.buy(symbol=bar.symbol, quantity=300)  # 下一根开盘成交
            elif self.n == 2:
                self.buy(symbol=bar.symbol, quantity=100, price=1.0)  # 挂着

    seen, _ = _run(S)
    by_n = {row["n"]: row for row in seen}
    assert by_n[2]["position"] == 300
    assert by_n[2]["frozen"] == 0.0
    for n in (3, 4, 5):
        assert by_n[n]["position"] == 300
        assert by_n[n]["frozen"] == 100 * CLOSES[n - 1]


def test_reading_frozen_cash_does_not_change_results() -> None:
    """读不读冻结资金, 回测结果相同."""

    def make(read: bool) -> Any:
        class S(_Base):
            def on_bar(self, bar: Any) -> None:
                self.n += 1
                if read:
                    self.get_account()
                if self.n == 1:
                    self.buy(symbol=bar.symbol, quantity=300)
                elif self.n == 2:
                    self.buy(symbol=bar.symbol, quantity=100, price=1.0)
                elif self.n == 4:
                    self.sell(symbol=bar.symbol, quantity=300)

        return S

    _, with_read = _run(make(True))
    _, without_read = _run(make(False))
    pd.testing.assert_series_equal(with_read.equity_curve, without_read.equity_curve)
    assert len(with_read.trades_df) == len(without_read.trades_df) == 1
