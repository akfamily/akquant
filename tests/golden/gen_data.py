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
    dates = pd.bdate_range("2024-01-02", periods=5, tz="Asia/Shanghai")
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
    dates = pd.bdate_range("2024-01-02", periods=5, tz="Asia/Shanghai")

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
    dates = pd.bdate_range("2024-01-02", periods=5, tz="Asia/Shanghai")

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
    # 6 个连续交易日（01-02～01-09，跳过周末）：最后一根 bar 用于撮合第 5 根
    # bar 上下的平仓单，不能落在周六（非交易日行情不撮合）。
    dates = pd.bdate_range("2024-01-02", periods=6, tz="Asia/Shanghai")

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


def _a_share_minute_times(day: str) -> list[pd.Timestamp]:
    """一个 A 股交易日的 1 分钟 bar 时间戳（区间右端点）.

    09:25 一根（区间 (09:24, 09:25] 属开盘集合竞价）、09:31～11:30、
    13:01～15:00（14:58～15:00 三根区间落在收盘集合竞价 14:57～15:00 内）。
    """
    morning = pd.date_range(f"{day} 09:31", f"{day} 11:30", freq="1min")
    afternoon = pd.date_range(f"{day} 13:01", f"{day} 15:00", freq="1min")
    return [pd.Timestamp(f"{day} 09:25"), *morning, *afternoon]


def gen_stock_minute_auction_data() -> None:
    """Generate A-share 1-minute data with opening/closing auction bars.

    两个交易日（2024-01-02 周二、01-03 周三）。价格刻意分档，便于从成交价
    认出撮合来源：开盘竞价 bar（09:25）= base；连续竞价 bar 的开盘价在
    base+0.05～base+0.11；收盘竞价末根（15:00）= base+0.30，14:58/14:59
    两根零量、价格持平。
    """
    data = []
    for day_index, day in enumerate(["2024-01-02", "2024-01-03"]):
        base = 10.0 + 0.5 * day_index
        times = _a_share_minute_times(day)
        continuous = 0
        last_close = base
        for ts in times:
            hhmm = ts.strftime("%H:%M")
            if hhmm == "09:25":
                o = h = low = c = base
                volume = 20000.0
            elif hhmm in ("14:58", "14:59"):
                o = h = low = c = last_close
                volume = 0.0
            elif hhmm == "15:00":
                o = h = low = c = round(base + 0.30, 2)
                volume = 30000.0
            else:
                o = round(base + 0.05 + 0.01 * (continuous % 7), 2)
                c = round(o + 0.01, 2)
                h = round(c + 0.01, 2)
                low = round(o - 0.01, 2)
                volume = 1000.0
                continuous += 1
                last_close = c
            data.append(
                {
                    "timestamp": ts,
                    "open": o,
                    "high": h,
                    "low": low,
                    "close": c,
                    "volume": volume,
                    "symbol": "600000.SH",
                }
            )

    df = pd.DataFrame(data)
    save_parquet_with_hash(df, "stock_minute_auction.parquet")


def gen_futures_night_weekend_data() -> None:
    """Generate sparse rb 1-minute data: Fri day + Fri night + Mon day.

    2024-01-05 周五日盘、周五夜盘 21:01～23:00、2024-01-08 周一日盘，每段
    取几根。时段表 CN_FUTURES_NIGHT_23 下周五夜盘属下周一交易日。
    """
    segments = [
        ("2024-01-05", ["09:01", "09:02", "10:00", "14:59", "15:00"], 3500.0),
        ("2024-01-05", ["21:01", "21:02", "21:30", "22:59", "23:00"], 3520.0),
        ("2024-01-08", ["09:01", "09:02", "10:00", "14:59", "15:00"], 3540.0),
    ]
    data = []
    for day, clock, base in segments:
        for i, hhmm in enumerate(clock):
            o = base + 2.0 * i
            c = o + 1.0
            data.append(
                {
                    "timestamp": pd.Timestamp(f"{day} {hhmm}"),
                    "open": o,
                    "high": c + 1.0,
                    "low": o - 1.0,
                    "close": c,
                    "volume": 500.0,
                    "symbol": "rb2405",
                }
            )

    df = pd.DataFrame(data)
    save_parquet_with_hash(df, "futures_night_weekend.parquet")


if __name__ == "__main__":
    gen_stock_data()
    gen_future_data()
    gen_option_data()
    gen_cancel_data()
    gen_stock_minute_auction_data()
    gen_futures_night_weekend_data()
