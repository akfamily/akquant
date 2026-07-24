import pandas as pd
from akquant import Bar, CurrentClose, Strategy, run_backtest
from akquant.config import RiskConfig


class MarginLiquidationAuditStrategy(Strategy):
    """Open leveraged long once and let maintenance breach trigger liquidation."""

    def __init__(self) -> None:
        """Initialize one-shot order flag."""
        self.ordered = False

    def on_bar(self, bar: Bar) -> None:
        """Submit first order and print account snapshot fields."""
        if not self.ordered:
            self.buy(symbol=bar.symbol, quantity=150)
            self.ordered = True
        snap = self.get_account()
        print(
            "account_snapshot",
            {
                "ts": pd.to_datetime(bar.timestamp, unit="ns", utc=True)
                .tz_convert("Asia/Shanghai")
                .isoformat(),
                "account_mode": snap.get("account_mode"),
                "cash": snap.get("cash"),
                "borrowed_cash": snap.get("borrowed_cash"),
                "short_market_value": snap.get("short_market_value"),
                "maintenance_ratio": snap.get("maintenance_ratio"),
                "accrued_interest": snap.get("accrued_interest"),
                "daily_interest": snap.get("daily_interest"),
            },
        )


def _build_data() -> list[Bar]:
    rows = [
        ("2023-01-01 10:00:00", 100.0),
        ("2023-01-01 14:00:00", 20.0),
        ("2023-01-02 10:00:00", 20.0),
    ]
    bars: list[Bar] = []
    for dt_str, close in rows:
        ts = pd.Timestamp(dt_str, tz="Asia/Shanghai").value
        bars.append(
            Bar(
                timestamp=ts,
                open=close,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=10000.0,
                symbol="LIQ",
            )
        )
    return bars


def run_example() -> None:
    """Run margin liquidation audit demo."""
    result = run_backtest(
        data=_build_data(),
        strategy=MarginLiquidationAuditStrategy,
        symbols="LIQ",
        initial_cash=10000.0,
        fill_policy=CurrentClose(),
        lot_size=1,
        show_progress=False,
        risk_config=RiskConfig(
            account_mode="margin",
            enable_short_sell=True,
            initial_margin_ratio=0.5,
            maintenance_margin_ratio=0.5,
            financing_rate_annual=0.0,
            borrow_rate_annual=0.0,
            allow_force_liquidation=True,
            liquidation_priority="short_first",
        ),
    )
    print("liquidation_audit_df")
    print(result.liquidation_audit_df)
    report_file = "margin_liquidation_report.html"
    result.viz.report(filename=report_file, show=False)
    print(f"report_saved={report_file}")


if __name__ == "__main__":
    run_example()
