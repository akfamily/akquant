from datetime import timedelta
from typing import cast

import pandas as pd


def create_dummy_data_1m(symbol: str, start_date: str, days: int) -> pd.DataFrame:
    """Generate 1-minute dummy data."""
    timestamps: list[pd.Timestamp] = []
    prices: list[float] = []
    current_date = pd.Timestamp(start_date).tz_localize("Asia/Shanghai")

    for _ in range(days):
        rng_am = pd.date_range(
            start=current_date + timedelta(hours=9, minutes=31),
            end=current_date + timedelta(hours=11, minutes=30),
            freq="1min",
            tz="Asia/Shanghai",
        )
        rng_pm = pd.date_range(
            start=current_date + timedelta(hours=13, minutes=1),
            end=current_date + timedelta(hours=15, minutes=0),
            freq="1min",
            tz="Asia/Shanghai",
        )
        timestamps.extend(rng_am)
        timestamps.extend(rng_pm)
        current_date += timedelta(days=1)

    prices = [100.0] * len(timestamps)
    return pd.DataFrame({"close": prices}, index=timestamps)


def create_dummy_data_1d(
    symbol: str, start_date: str, days: int, df_1m: pd.DataFrame
) -> pd.DataFrame:
    """Generate daily dummy data from 1-minute data."""
    df_1m_copy = df_1m.copy()
    daily_groups = df_1m_copy.resample("1D")

    daily_index: list[pd.Timestamp] = []

    for date, group in daily_groups:
        if group.empty:
            continue
        # Cast date to Timestamp to satisfy mypy
        ts_start = cast(pd.Timestamp, date)
        ts = ts_start + timedelta(hours=15)
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Shanghai")
        daily_index.append(ts)

    return pd.DataFrame({"close": [100.0] * len(daily_index)}, index=daily_index)


df_1m = create_dummy_data_1m("S", "2023-01-01", 1)
df_1d = create_dummy_data_1d("S_1D", "2023-01-01", 1, df_1m)

print("1M Index Head:", df_1m.index[0])
print("1M Index Head UTC:", df_1m.index[0].tz_convert("UTC"))
print("1D Index Head:", df_1d.index[0])
print("1D Index Head UTC:", df_1d.index[0].tz_convert("UTC"))
