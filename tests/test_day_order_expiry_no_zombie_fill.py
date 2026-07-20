# -*- coding: utf-8 -*-
"""Regression: a Day order must not fill after expiry (issue #329 hygiene b).

When a ``TimeInForce.Day`` order does not fill on its active day, daily
settlement marks it ``Expired`` in the order manager. But the expiry path pushed
the order back into ``order_manager.orders`` without telling the execution
client to cancel it (``data.rs`` skipped ``execution_model.on_cancel``), so the
simulated execution client kept the order ``New`` in its own book — a zombie.
On a later bar whose price crossed the limit, the zombie matched and filled,
long after the order should have died.
"""

import akquant as aq
import pandas as pd
from akquant import Strategy, TimeInForce

TZ = "Asia/Shanghai"
DAYS = pd.date_range("2024-01-02", periods=5, freq="B", tz=TZ)

# Day-order limit buy @50 submitted on day1.
#   day1=100, day2=100 -> never fills (100 > 50); must expire at day2 settlement.
#   day3=40           -> if the zombie survived, it would wrongly fill here.
_PRICES = {DAYS[0]: 100.0, DAYS[1]: 100.0, DAYS[2]: 40.0, DAYS[3]: 40.0, DAYS[4]: 40.0}


class _OneShotDayLimit(Strategy):
    _submitted = False

    def on_bar(self, bar) -> None:  # type: ignore[no-untyped-def]
        if not self._submitted:
            self._submitted = True
            self.buy(
                symbol="AAA",
                quantity=100,
                price=50.0,
                time_in_force=TimeInForce.Day,
            )


def test_expired_day_order_does_not_fill_on_a_later_bar() -> None:
    """A Day order that expired unfilled must never fill on a subsequent day."""
    data = pd.DataFrame(
        [
            {
                "date": d,
                "open": p,
                "high": p,
                "low": p,
                "close": p,
                "volume": 1_000_000.0,
                "symbol": "AAA",
            }
            for d, p in _PRICES.items()
        ]
    )

    fills: list[float] = []

    class _S(_OneShotDayLimit):
        def on_trade(self, trade) -> None:  # type: ignore[no-untyped-def]
            fills.append(float(getattr(trade, "quantity", 0.0)))

    result = aq.run_backtest(
        data=data,
        strategy=_S,
        symbols=["AAA"],
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        start_time=str(DAYS[0].date()),
        end_time=str(DAYS[-1].date()),
        show_progress=False,
    )

    # The order expired unfilled at day2 settlement; no trade may ever occur.
    assert not fills, (
        f"expired Day order wrongly filled on a later bar (zombie order in the "
        f"execution client): fills={fills}"
    )
    # Cash must be untouched — no phantom position was opened.
    assert len(result.trades_df) == 0
