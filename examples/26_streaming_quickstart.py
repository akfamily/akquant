from typing import Any

import akquant as aq
import akshare as ak
import pandas as pd
from akquant import Bar, Strategy
from akquant.config import BacktestConfig, RiskConfig, StrategyConfig


def load_data() -> dict[str, pd.DataFrame]:
    """Load multi-symbol daily bars from AKShare."""
    df_1 = ak.stock_zh_a_daily(
        symbol="sh600000", start_date="20000101", end_date="20261231"
    )
    df_1["symbol"] = "600000"
    df_2 = ak.stock_zh_a_daily(
        symbol="sh600004", start_date="20000101", end_date="20261231"
    )
    df_2["symbol"] = "600004"
    df_3 = ak.stock_zh_a_daily(
        symbol="sh600006", start_date="20000101", end_date="20261231"
    )
    df_3["symbol"] = "600006"
    return {"600000": df_1, "600004": df_2, "600006": df_3}


class MyStreamStrategy(Strategy):
    """Quickstart-like strategy used for stream demonstration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize per-symbol state."""
        super().__init__()
        self.bars_held: dict[str, int] = {}
        self.entry_prices: dict[str, float] = {}

    def on_bar(self, bar: Bar) -> None:
        """Handle bar events and execute simple position logic."""
        symbol = bar.symbol
        pos = self.get_position(symbol)
        if pos > 0:
            self.bars_held[symbol] = self.bars_held.get(symbol, 0) + 1
        else:
            self.bars_held.pop(symbol, None)
            self.entry_prices.pop(symbol, None)

        if pos == 0:
            self.order_target_percent(target_percent=0.33, symbol=symbol)
            self.bars_held[symbol] = 0
            self.entry_prices[symbol] = bar.close
            return

        entry_price = self.entry_prices.get(symbol, bar.close)
        current_bars_held = self.bars_held.get(symbol, 0)
        pnl_pct = (bar.close - entry_price) / entry_price
        if pnl_pct >= 0.10:
            self.sell(symbol, pos)
        elif current_bars_held >= 100:
            self.close_position()


def main() -> None:
    """Run a stream backtest and print compact stream summary."""
    data = load_data()
    risk_config = RiskConfig(safety_margin=0.0001)
    strategy_config = StrategyConfig(risk=risk_config)
    backtest_config = BacktestConfig(strategy_config=strategy_config)

    events: list[aq.BacktestStreamEvent] = []
    result = aq.run_backtest(
        strategy=MyStreamStrategy,
        data=data,
        initial_cash=5000000,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=5.0,
        lot_size=1,
        fill_policy=aq.NextAverage(),
        config=backtest_config,
        start_time="20250101",
        end_time="20250105",
        symbols=["600000", "600004", "600006"],
        on_event=events.append,
        show_progress=False,
        stream_progress_interval=1,
        stream_equity_interval=1,
        stream_batch_size=32,
        stream_max_buffer=256,
        stream_error_mode="continue",
        strategy_id="quickstart_alpha",
    )

    print(result)
    print(result.orders_df)

    started = sum(1 for e in events if e.get("event_type") == "started")
    finished = sum(1 for e in events if e.get("event_type") == "finished")
    progress = sum(1 for e in events if e.get("event_type") == "progress")
    equity = sum(1 for e in events if e.get("event_type") == "equity")
    seq_values = [e["seq"] for e in events]
    is_monotonic = seq_values == sorted(seq_values)
    finished_payload = (
        events[-1].get("payload", {})
        if events and events[-1].get("event_type") == "finished"
        else {}
    )
    owner_counter: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type", ""))
        if event_type not in {"order", "trade", "risk"}:
            continue
        payload = event.get("payload", {})
        owner = "_default"
        if isinstance(payload, dict):
            owner = str(payload.get("owner_strategy_id", "_default"))
        owner_counter[owner] = owner_counter.get(owner, 0) + 1
    print(f"stream_started={started}")
    print(f"stream_finished={finished}")
    print(f"stream_progress={progress}")
    print(f"stream_equity={equity}")
    print(f"stream_seq_monotonic={is_monotonic}")
    print(
        "stream_callback_error_count="
        f"{finished_payload.get('callback_error_count', '0')}"
    )
    print(f"stream_owner_event_counts={owner_counter}")
    print("done_streaming_quickstart")


if __name__ == "__main__":
    main()
