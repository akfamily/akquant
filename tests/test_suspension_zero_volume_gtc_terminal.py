# -*- coding: utf-8 -*-
"""Regression for issue #334.

A GTC order whose only matchable slice lands on a *suspended* bar — a bar that
is present in the feed but carries ``volume == 0`` — must reach a terminal state
(Rejected) once that slice passes, exactly as it would for a symbol with no bar
at all (issue #329). Prior to the fix the order survived as ``New`` until the
symbol resumed trading: the matcher skips ``volume <= 0`` bars (correct), but the
terminal-state path ``reject_missing_symbol_orders`` keyed off ``seen_symbols``
("did a bar appear?") instead of tradability ("was there a volume>0 bar?"), so a
volume=0 placeholder bar kept the order alive forever.

This is the volume=0 variant of #329 that PR #333 did not cover.
"""

import akquant as aq
import pandas as pd
from akquant import Strategy
from akquant.backtest import BacktestResult

TZ = "Asia/Shanghai"
DAYS = pd.date_range("2024-01-02", periods=7, freq="B", tz=TZ)
# SUSP has volume=0 placeholder bars on the 4th and 5th trading days, then resumes.
SUSPEND_DAYS = {DAYS[3], DAYS[4]}


def _make_bars() -> pd.DataFrame:
    rows = []
    for i, d in enumerate(DAYS):
        if d in SUSPEND_DAYS:
            # Suspension: bar row is present but not tradable (volume == 0).
            rows.append(
                {
                    "date": d,
                    "open": float("nan"),
                    "high": float("nan"),
                    "low": float("nan"),
                    "close": float("nan"),
                    "volume": 0.0,
                    "symbol": "SUSP",
                }
            )
            continue
        c = 100.0 * (1 + 0.001 * i)
        rows.append(
            {
                "date": d,
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": 1_000_000.0,
                "symbol": "SUSP",
            }
        )
    return pd.DataFrame(rows)


class _BuyThenSellIntoSuspension(Strategy):
    """Buy on day 1, then issue a single GTC sell whose fill slice is suspended."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()
        self._day = 0
        self._sold = False

    def on_bar(self, *args: object, **kwargs: object) -> None:
        self._day += 1
        if self._day == 1:
            self.order_target_percent(target_percent=0.95, symbol="SUSP")
        elif self._day == 3 and not self._sold:
            # Issued on a tradable day; its next-open matchable slice (day 4) is
            # suspended (volume=0). Issue exactly once so the order's fate is
            # observable and not masked by re-submission.
            qty = self.positions.get("SUSP", 0)
            if qty:
                self.sell(symbol="SUSP", quantity=qty)
                self._sold = True


def _run() -> BacktestResult:
    data = _make_bars().sort_values(["symbol", "date"]).reset_index(drop=True)
    return aq.run_backtest(
        data=data,
        strategy=_BuyThenSellIntoSuspension,
        symbols=["SUSP"],
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        start_time=str(DAYS[0].date()),
        end_time=str(DAYS[-1].date()),
        show_progress=False,
    )


def test_gtc_order_on_zero_volume_bar_reaches_terminal_state() -> None:
    """The GTC sell must be rejected once its suspended matchable slice passes.

    Regression for #334: pre-fix the sell survived as New through the suspension
    and only filled on resume, so its final status was 'filled'. After the fix it
    reaches a terminal 'rejected' state during suspension and never fills.
    """
    result = _run()
    orders = result.orders_df
    assert not orders.empty, "no orders recorded"

    sells = orders[orders["side"].astype(str).str.lower().str.contains("sell")]
    assert len(sells) == 1, f"expected exactly one sell order, got {len(sells)}"

    sell = sells.iloc[0]
    status = str(sell["status"]).lower()
    filled = float(sell.get("filled_quantity", 0) or 0)

    assert "reject" in status, (
        f"GTC sell on a volume=0 (suspended) matchable slice should reach a "
        f"terminal Rejected state, but status was {status!r} "
        f"(filled_quantity={filled}). Pre-fix it stayed New and filled on resume."
    )
    assert filled == 0.0, (
        f"a rejected sell must not have filled, but filled_quantity={filled}"
    )


def test_zero_volume_reject_reason_names_suspension_not_missing_data() -> None:
    """The reject reason must reflect the true cause: a present-but-untradable bar.

    Regression for #334 log wording: a volume=0 bar is *not* missing market data —
    the bar exists, it is just not tradable (suspension). Reusing the "Missing
    market data" wording (the #329 no-bar case) misleads audit/log analysis. The
    reason for a volume=0 slice must name suspension / zero volume / non-tradable,
    and must not claim data was missing.
    """
    result = _run()
    orders = result.orders_df
    sells = orders[orders["side"].astype(str).str.lower().str.contains("sell")]
    reason = str(sells.iloc[0].get("reject_reason", "") or "").lower()

    assert reason, "rejected sell carried no reject_reason"
    assert "missing market data" not in reason, (
        f"volume=0 suspension is not missing data, but reason was {reason!r}"
    )
    assert any(k in reason for k in ("suspend", "zero volume", "not tradable")), (
        f"reject_reason should name the suspension/zero-volume cause, got {reason!r}"
    )
