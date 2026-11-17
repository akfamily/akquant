from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, Strategy
from akquant.config import BacktestConfig, RiskConfig, StrategyConfig


def synthetic_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """生成固定种子的合成日线（几何随机游走），只用于取数失败时让示例跑通."""
    dates = pd.bdate_range(start_date, end_date)
    rng = np.random.default_rng(sum(map(ord, symbol)))
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, len(dates))))
    open_ = np.concatenate([[10.0], close[:-1]])
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
        }
    )


def fetch_daily(
    symbol: str, start_date: str, end_date: str, adjust: str = ""
) -> pd.DataFrame:
    """用 AKShare 取新浪 A 股日线（代码带市场前缀，如 sh600000）.

    取数失败（未安装 AKShare、断网、接口改名等）时改用合成数据，保证示例能跑通；
    合成数据的回测结果没有参考意义。
    """
    try:
        import akshare as ak

        df = ak.stock_zh_a_hist_sina(
            symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust
        )
        if df.empty:
            raise ValueError("返回空表")
        return pd.DataFrame(df)
    except Exception as exc:  # noqa: BLE001 - 示例容错，任何取数问题都回退
        print(f"警告: AKShare 取数失败（{type(exc).__name__}: {exc}），改用合成数据")
        return synthetic_daily(symbol, start_date, end_date)


def load_data() -> dict[str, pd.DataFrame]:
    """Load multi-symbol daily bars from AKShare."""
    df_1 = fetch_daily(symbol="sh600000", start_date="20000101", end_date="20261231")
    df_1["symbol"] = "600000"
    df_2 = fetch_daily(symbol="sh600004", start_date="20000101", end_date="20261231")
    df_2["symbol"] = "600004"
    df_3 = fetch_daily(symbol="sh600006", start_date="20000101", end_date="20261231")
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
