from pathlib import Path

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy


def make_bars() -> list[Bar]:
    """Build deterministic bars for report and analysis output demo."""
    idx = pd.date_range(start="2024-01-01", periods=24, freq="D")
    bars: list[Bar] = []
    for i, ts in enumerate(idx):
        base = 100.0 + float(i) * 0.5
        bars.append(
            Bar(
                timestamp=int(ts.value),
                open=base,
                high=base + 1.0,
                low=base - 1.0,
                close=base + (0.7 if i % 2 == 0 else -0.4),
                volume=1000.0 + float(i) * 10.0,
                symbol="REPORT",
            )
        )
    return bars


class ReportDemoStrategy(Strategy):
    """A minimal strategy that alternates between open and close."""

    def on_bar(self, bar: Bar) -> None:
        """Submit an order when flat, otherwise close current position."""
        position = self.get_position(bar.symbol)
        if position == 0:
            self.buy(bar.symbol, 10)
        else:
            self.close_position(bar.symbol)


def main() -> None:
    """Run backtest, generate report, and print analysis output summaries."""
    result = aq.run_backtest(
        data=make_bars(),
        strategy=ReportDemoStrategy,
        symbols="REPORT",
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    report_path = Path(__file__).with_name("report_and_analysis_outputs.html").resolve()
    result.viz.report(filename=str(report_path), show=False, compact_currency=True)

    exposure = result.exposure_df()
    attribution_by_symbol = result.attribution_df(by="symbol")
    attribution_by_tag = result.attribution_df(by="tag")
    capacity = result.capacity_df()
    orders_by_strategy = result.orders_by_strategy()
    executions_by_strategy = result.executions_by_strategy()

    print(result)
    print(f"report_html={report_path}")
    print(f"exposure_rows={len(exposure)}")
    print(f"attribution_symbol_rows={len(attribution_by_symbol)}")
    print(f"attribution_tag_rows={len(attribution_by_tag)}")
    print(f"capacity_rows={len(capacity)}")
    print(f"orders_by_strategy_rows={len(orders_by_strategy)}")
    print(f"executions_by_strategy_rows={len(executions_by_strategy)}")
    print("done_report_and_analysis_outputs")


if __name__ == "__main__":
    main()
