from typing import Any

import pandas as pd
from akquant import Bar, CurrentClose, Strategy, run_backtest
from akquant.config import RiskConfig


class HedgedMarginStrategy(Strategy):
    """Open one long and one short leg for liquidation-priority comparison."""

    def __init__(self) -> None:
        """Initialize one-shot submit flags."""
        self.long_submitted = False
        self.short_submitted = False

    def on_bar(self, bar: Bar) -> None:
        """Submit initial hedged legs."""
        if bar.symbol == "LONG" and not self.long_submitted:
            self.buy(symbol=bar.symbol, quantity=100)
            self.long_submitted = True
        if bar.symbol == "SHORT" and not self.short_submitted:
            self.sell(symbol=bar.symbol, quantity=50)
            self.short_submitted = True


def _build_data() -> list[Bar]:
    rows = [
        ("2023-01-01 10:00:00", "LONG", 100.0),
        ("2023-01-01 10:00:00", "SHORT", 100.0),
        ("2023-01-02 10:00:00", "LONG", 100.0),
        ("2023-01-02 10:00:00", "SHORT", 100.0),
    ]
    bars: list[Bar] = []
    for dt_str, symbol, close in rows:
        ts = pd.Timestamp(dt_str, tz="Asia/Shanghai").value
        bars.append(
            Bar(
                timestamp=ts,
                open=close,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=10000.0,
                symbol=symbol,
            )
        )
    return bars


def _run_once(liquidation_priority: str) -> Any:
    return run_backtest(
        data=_build_data(),
        strategy=HedgedMarginStrategy,
        symbols=["LONG", "SHORT"],
        initial_cash=50000.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        risk_config=RiskConfig(
            account_mode="margin",
            enable_short_sell=True,
            initial_margin_ratio=0.5,
            maintenance_margin_ratio=4.0,
            financing_rate_annual=0.0,
            borrow_rate_annual=0.0,
            allow_force_liquidation=True,
            liquidation_priority=liquidation_priority,
        ),
    )


def run_example() -> None:
    """Run and compare short_first vs long_first liquidation results."""
    result_short_first = _run_once("short_first")
    result_long_first = _run_once("long_first")

    audit_short_first = result_short_first.liquidation_audit_df
    audit_long_first = result_long_first.liquidation_audit_df
    if audit_short_first.empty or audit_long_first.empty:
        raise RuntimeError("liquidation audit is empty in at least one run")

    symbol_short_first = str(audit_short_first.iloc[-1]["liquidated_symbols"])
    symbol_long_first = str(audit_long_first.iloc[-1]["liquidated_symbols"])

    print("short_first_liquidation_audit")
    print(audit_short_first)
    print("long_first_liquidation_audit")
    print(audit_long_first)
    print(
        "liquidation_priority_compare",
        {
            "short_first_liquidated_symbol": symbol_short_first,
            "long_first_liquidated_symbol": symbol_long_first,
        },
    )

    short_report = "margin_liquidation_priority_short_first.html"
    long_report = "margin_liquidation_priority_long_first.html"
    result_short_first.viz.report(filename=short_report, show=False)
    result_long_first.viz.report(filename=long_report, show=False)
    print(f"report_saved={short_report}")
    print(f"report_saved={long_report}")


if __name__ == "__main__":
    run_example()
