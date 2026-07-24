"""TA-Lib indicator playbook demo with strategy-level signal combinations."""

import argparse
from typing import cast

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, Strategy
from akquant import talib as ta
from akquant.utils import fetch_akshare_symbol


def build_demo_data(length: int = 420) -> pd.DataFrame:
    """Build deterministic OHLCV data for the playbook demo."""
    rng = np.random.default_rng(20260312)
    dates = pd.date_range(start="2024-01-01 09:30:00", periods=length, freq="1min")
    base = 100.0 + np.linspace(0.0, 8.0, length)
    wave = 1.8 * np.sin(np.linspace(0.0, 8.5 * np.pi, length))
    noise = rng.normal(0.0, 0.18, length)
    close = base + wave + noise
    open_ = close + rng.normal(0.0, 0.07, length)
    high = np.maximum(open_, close) + 0.35
    low = np.minimum(open_, close) - 0.35
    volume = 1000.0 + 120.0 * (1.0 + np.sin(np.linspace(0.0, 4.0 * np.pi, length)))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "PLAYBOOK",
        }
    )


def build_akshare_data(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> pd.DataFrame:
    """Build AKShare daily OHLCV data for the playbook demo."""
    df = fetch_akshare_symbol(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    ).copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    required = ["open", "high", "low", "close", "volume", "symbol"]
    if "date" in df.columns:
        required = ["date", *required]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"AKShare data missing required columns: {missing}")
    return cast(pd.DataFrame, df[required].dropna().reset_index(drop=True))


class TalibIndicatorPlaybookStrategy(Strategy):
    """Combine trend-following and mean-reversion templates."""

    def __init__(self) -> None:
        """Initialize strategy state."""
        super().__init__()
        self.warmup_period = 90
        self.trade_actions = 0
        self.last_signal = "none"

    def on_bar(self, bar: Bar) -> None:
        """Handle bar callback and emit orders on combined signals."""
        symbol = bar.symbol
        close = np.asarray(self.get_history(90, symbol, "close"), dtype=float)
        high = np.asarray(self.get_history(90, symbol, "high"), dtype=float)
        low = np.asarray(self.get_history(90, symbol, "low"), dtype=float)
        if close.size < 70 or high.size < 70 or low.size < 70:
            return

        ema_fast = np.asarray(
            ta.EMA(close, timeperiod=20, backend="rust"),
            dtype=float,
        )
        ema_slow = np.asarray(
            ta.EMA(close, timeperiod=60, backend="rust"),
            dtype=float,
        )
        adx = np.asarray(
            ta.ADX(high, low, close, timeperiod=14, backend="rust"),
            dtype=float,
        )
        natr = np.asarray(
            ta.NATR(high, low, close, timeperiod=14, backend="rust"),
            dtype=float,
        )
        upper_raw, middle_raw, lower_raw = ta.BBANDS(
            close,
            timeperiod=20,
            backend="rust",
        )
        upper = np.asarray(upper_raw, dtype=float)
        middle = np.asarray(middle_raw, dtype=float)
        lower = np.asarray(lower_raw, dtype=float)
        rsi = np.asarray(ta.RSI(close, timeperiod=14, backend="rust"), dtype=float)
        mom = np.asarray(ta.MOM(close, period=10, backend="rust"), dtype=float)

        latest = np.asarray(
            [
                ema_fast[-1],
                ema_slow[-1],
                adx[-1],
                natr[-1],
                upper[-1],
                middle[-1],
                lower[-1],
                rsi[-1],
                mom[-1],
            ],
            dtype=float,
        )
        if np.isnan(latest).any():
            return

        pos = self.get_position(symbol)
        trend_entry = ema_fast[-1] > ema_slow[-1] and adx[-1] >= 20.0 and natr[-1] < 4.5
        mean_reversion_entry = (
            close[-1] < lower[-1] and rsi[-1] < 35.0 and mom[-1] > 0.0
        )
        exit_signal = (
            close[-1] > middle[-1] or ema_fast[-1] < ema_slow[-1] or rsi[-1] > 72.0
        )

        if pos == 0 and (trend_entry or mean_reversion_entry):
            self.buy(symbol, 100)
            self.trade_actions += 1
            self.last_signal = "trend_entry" if trend_entry else "mean_reversion_entry"
            return

        if pos > 0 and exit_signal:
            self.sell(symbol, pos)
            self.trade_actions += 1
            self.last_signal = "exit"


def run_example(
    data_source: str,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
    synthetic_length: int,
) -> None:
    """Run playbook demo strategy and print compact summary."""
    adjust_value = "" if adjust == "none" else adjust
    if data_source == "akshare":
        df = build_akshare_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust_value,
        )
        print("playbook_data_source=akshare")
    else:
        df = build_demo_data(length=synthetic_length)
        print("playbook_data_source=synthetic")

    result = aq.run_backtest(
        data=df,
        strategy=TalibIndicatorPlaybookStrategy,
        symbols=symbol,
        initial_cash=100000.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        fill_policy=aq.CurrentClose(),
        lot_size=1,
        show_progress=False,
    )
    strategy = result.strategy
    trade_actions = getattr(strategy, "trade_actions", 0) if strategy is not None else 0
    last_signal = (
        getattr(strategy, "last_signal", "none") if strategy is not None else "none"
    )
    print(f"playbook_trade_actions={trade_actions}")
    print(f"playbook_last_signal={last_signal}")
    print(f"playbook_end_market_value={result.metrics.end_market_value:.2f}")
    print("done_talib_indicator_playbook_demo")


def main() -> None:
    """Parse CLI args and execute demo."""
    parser = argparse.ArgumentParser(
        description="TA-Lib indicator playbook demo with synthetic/AKShare data modes.",
    )
    parser.add_argument(
        "--data-source",
        choices=["synthetic", "akshare"],
        default="synthetic",
        help=(
            "Data source mode. synthetic is fully local; "
            "akshare fetches real A-share daily bars."
        ),
    )
    parser.add_argument(
        "--symbol",
        default="sh600000",
        help="Symbol for backtest and AKShare fetch.",
    )
    parser.add_argument(
        "--start-date",
        default="20240101",
        help="AKShare start date in YYYYMMDD.",
    )
    parser.add_argument(
        "--end-date",
        default="20260301",
        help="AKShare end date in YYYYMMDD.",
    )
    parser.add_argument(
        "--adjust",
        default="qfq",
        choices=["qfq", "hfq", "none"],
        help="AKShare adjust mode.",
    )
    parser.add_argument(
        "--synthetic-length",
        type=int,
        default=420,
        help="Synthetic bar count when --data-source=synthetic.",
    )
    args = parser.parse_args()
    run_example(
        data_source=args.data_source,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        adjust=args.adjust,
        synthetic_length=args.synthetic_length,
    )


if __name__ == "__main__":
    main()
