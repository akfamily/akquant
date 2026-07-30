"""Generate test data for Golden Suite."""

import hashlib
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_parquet_with_hash(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to Parquet with hash in metadata."""
    # Ensure timezone is Asia/Shanghai
    if "timestamp" in df.columns:
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Shanghai")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Shanghai")

    path = DATA_DIR / filename
    df.to_parquet(path, index=False)

    # Calculate hash of the file content
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    # Write hash to metadata (simulate via sidecar or re-write with pyarrow metadata)
    # For simplicity, we just print it here or save a sidecar .sha256 file
    with open(path.with_suffix(".parquet.sha256"), "w") as f:
        f.write(file_hash)

    print(f"Saved {filename} with SHA256: {file_hash[:8]}...")


def gen_stock_data() -> None:
    """Generate Stock Data for T+1 and Limit Up/Down tests."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="Asia/Shanghai")
    # Day 0: Normal
    # Day 1: Limit Up (+10%)
    # Day 2: Limit Down (-10%)
    # Day 3: Normal
    # Day 4: Normal

    data = []
    # base_price = 10.0  # Unused

    # Day 0
    data.append(
        {
            "timestamp": dates[0],
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.2,
            "volume": 1000,
            "symbol": "STOCK_A",
        }
    )

    # Day 1: Limit Up from 10.2 -> 11.22
    limit_up = round(10.2 * 1.10, 2)
    data.append(
        {
            "timestamp": dates[1],
            "open": limit_up,
            "high": limit_up,
            "low": limit_up,
            "close": limit_up,
            "volume": 500,  # Volume exists so trade possible
            "symbol": "STOCK_A",
        }
    )

    # Day 2: Limit Down from 11.22 -> 10.10
    limit_down = round(11.22 * 0.90, 2)
    data.append(
        {
            "timestamp": dates[2],
            "open": limit_down,
            "high": limit_down,
            "low": limit_down,
            "close": limit_down,
            "volume": 500,
            "symbol": "STOCK_A",
        }
    )

    # Day 3
    data.append(
        {
            "timestamp": dates[3],
            "open": 10.5,
            "high": 10.8,
            "low": 10.2,
            "close": 10.6,
            "volume": 1000,
            "symbol": "STOCK_A",
        }
    )

    # Day 4
    data.append(
        {
            "timestamp": dates[4],
            "open": 10.6,
            "high": 11.0,
            "low": 10.5,
            "close": 10.8,
            "volume": 1000,
            "symbol": "STOCK_A",
        }
    )

    df = pd.DataFrame(data)
    save_parquet_with_hash(df, "stock_t1.parquet")


def gen_future_data() -> None:
    """Generate Futures Data for Margin tests."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="Asia/Shanghai")

    data = []
    prices = [3500.0, 3450.0, 3400.0, 3550.0, 3600.0]

    for i, price in enumerate(prices):
        data.append(
            {
                "timestamp": dates[i],
                "open": price,
                "high": price + 10,
                "low": price - 10,
                "close": price,
                "volume": 1000,
                "symbol": "FUTURE_A",
            }
        )

    df = pd.DataFrame(data)
    save_parquet_with_hash(df, "future_margin.parquet")


def gen_option_data() -> None:
    """Generate Option Data for basic trading."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="Asia/Shanghai")

    data = []
    # Call Option prices
    prices = [50.0, 40.0, 30.0, 60.0, 70.0]

    for i, price in enumerate(prices):
        data.append(
            {
                "timestamp": dates[i],
                "open": price,
                "high": price + 5,
                "low": price - 5,
                "close": price,
                "volume": 100,
                "symbol": "OPTION_A",
            }
        )

    df = pd.DataFrame(data)
    save_parquet_with_hash(df, "option_basic.parquet")


def gen_cancel_data() -> None:
    """Generate data for the order-cancel lifecycle test.

    Flat prices with ample volume: fills are driven purely by the strategy's
    order/cancel sequence, not by price moves.
    """
    dates = pd.date_range("2024-01-01", periods=6, freq="D", tz="Asia/Shanghai")

    data = []
    for i in range(6):
        data.append(
            {
                "timestamp": dates[i],
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 100000,
                "symbol": "CANCEL_A",
            }
        )

    df = pd.DataFrame(data)
    save_parquet_with_hash(df, "order_cancel.parquet")


if __name__ == "__main__":
    gen_stock_data()
    gen_future_data()
    gen_option_data()
    gen_cancel_data()
