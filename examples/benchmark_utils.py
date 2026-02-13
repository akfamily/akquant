import hashlib
import time
from typing import Optional

import numpy as np
import pandas as pd


def get_benchmark_data(
    n: int = 200_000,
    symbol: str = "BENCHMARK",
    freq: str = "min",
    start_time: str = "2020-01-01",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate dummy market data using Geometric Brownian Motion.

    Ensures positive prices and consistency.
    """
    print(f"Generating {n} rows of dummy data...")
    t0 = time.time()

    dates = pd.date_range(start=start_time, periods=n, freq=freq, tz="UTC")
    # Add 15 hours to simulate market close time (15:00:00 UTC)
    dates = dates + pd.Timedelta(hours=15)

    # Random walk (Geometric Brownian Motion to ensure positive prices)
    # Use symbol hash to ensure different data for different symbols,
    # but consistent data for the same symbol.
    if seed is not None:
        seed_val = seed
    else:
        # Use stable hash (md5) instead of python's hash() which is randomized
        seed_val = int(hashlib.md5(symbol.encode("utf-8")).hexdigest(), 16) % (2**32)

    np.random.seed(seed_val)
    returns = np.random.normal(0, 0.001, n)
    price = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "date": dates,
            "open": price * (1 + np.random.normal(0, 0.0005, n)),
            "high": price * (1 + np.abs(np.random.normal(0, 0.0005, n))),
            "low": price * (1 - np.abs(np.random.normal(0, 0.0005, n))),
            "close": price,
            "volume": np.random.randint(100, 10000, n),
            "symbol": [symbol] * n,
        }
    )

    # Ensure consistency (High >= Open/Close, Low <= Open/Close)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    # Add Chinese columns for Akquant compatibility if needed,
    # but Akquant example can rename them.
    # Let's keep standard English columns here.

    print(f"Data generation took {time.time() - t0:.4f}s")
    return df


def print_report(
    engine_name: str,
    duration: float,
    total_bars: int,
    total_return_pct: float,
    total_trades: int,
) -> None:
    """Print a standardized benchmark report."""
    throughput = total_bars / duration if duration > 0 else 0

    print("-" * 50)
    print(f"Backtest Results ({engine_name})")
    print("-" * 50)
    print(f"Data Size     : {total_bars} bars")
    print(f"Execution Time: {duration:.4f} s")
    print(f"Throughput    : {throughput:,.0f} bars/sec")
    print(f"Total Return  : {total_return_pct:.2f}%")
    print(f"Total Trades  : {total_trades}")
    print("-" * 50)
