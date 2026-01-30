from typing import Dict, List, Optional, Tuple, cast

import pandas as pd

from .akquant import Bar, from_arrays


def load_akshare_bar(df: pd.DataFrame, symbol: Optional[str] = None) -> List[Bar]:
    r"""
    将 AKShare 返回的 DataFrame 转换为 akquant.Bar 列表.

    :param df: AKShare 历史行情数据
    :type df: pandas.DataFrame
    :param symbol: 标的代码；未提供时尝试使用 DataFrame 的“股票代码”列
    :type symbol: str, optional
    :return: 转换后的 Bar 对象列表
    :rtype: List[Bar]
    """
    if df.empty:
        return []

    # Check for required columns
    required_map = {
        "日期": "timestamp",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }

    # Validate columns
    missing = [col for col in required_map.keys() if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame 缺少必要列: {missing}")

    # Vectorized Preprocessing

    # 1. Handle Timestamp
    # Convert to datetime with error coercion (invalid dates becomes NaT)
    dt_series = pd.to_datetime(df["日期"], errors="coerce")
    # Fill NaT with 0 (Epoch 0) or handle appropriately
    dt_series = dt_series.fillna(pd.Timestamp(0))
    # type: ignore
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize("Asia/Shanghai")
    dt_series = dt_series.dt.tz_convert("UTC")
    timestamps = dt_series.astype("int64").tolist()

    # 2. Extract numeric columns
    # Use astype(float) to ensure correct type, fillna(0.0) for safety
    opens = df["开盘"].fillna(0.0).astype(float).tolist()
    highs = df["最高"].fillna(0.0).astype(float).tolist()
    lows = df["最低"].fillna(0.0).astype(float).tolist()
    closes = df["收盘"].fillna(0.0).astype(float).tolist()
    volumes = df["成交量"].fillna(0.0).astype(float).tolist()

    # 3. Handle Symbol
    symbols_list: Optional[List[str]] = None
    symbol_val = None

    if symbol:
        symbol_val = symbol
    elif "股票代码" in df.columns:
        # Convert to string
        symbols_list = cast(List[str], df["股票代码"].astype(str).tolist())
    else:
        symbol_val = "UNKNOWN"

    # Call Rust extension
    return from_arrays(
        timestamps, opens, highs, lows, closes, volumes, symbol_val, symbols_list
    )


def df_to_arrays(
    df: pd.DataFrame, symbol: Optional[str] = None
) -> Tuple[
    List[int],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    Optional[str],
    Optional[List[str]],
    Optional[Dict[str, List[float]]],
]:
    r"""
    将 DataFrame 转换为用于 DataFeed.add_arrays 的数组元组.

    :param df: 输入的 DataFrame
    :param symbol: 标的代码 (可选)
    :return: (timestamps, opens, highs, lows, closes, volumes, symbol, symbols, extra)
    """
    if df.empty:
        return ([], [], [], [], [], [], symbol, None, None)

    # Column Mapping Strategy
    # Priority:
    # 1. AKShare Chinese columns: "日期", "开盘", ...
    # 2. Standard English columns: "date", "open", ...
    # 3. Lowercase normalized check

    # Define targets
    targets = {
        "timestamp": ["日期", "date", "datetime", "time", "timestamp"],
        "open": ["开盘", "open"],
        "high": ["最高", "high"],
        "low": ["最低", "low"],
        "close": ["收盘", "close"],
        "volume": ["成交量", "volume", "vol"],
        "symbol": ["股票代码", "symbol", "code", "ticker"],
    }

    # Resolve columns
    df_cols = df.columns
    df_cols_lower = [str(c).lower() for c in df_cols]

    resolved = {}

    for key, candidates in targets.items():
        found = None
        for cand in candidates:
            if cand in df_cols:
                found = cand
                break

        # If not found, try case-insensitive
        if not found:
            for cand in candidates:
                if cand.lower() in df_cols_lower:
                    idx = df_cols_lower.index(cand.lower())
                    found = str(df_cols[idx])
                    break

        if found:
            resolved[key] = found

    # Check essential columns
    missing = []
    for essential in ["timestamp", "open", "high", "low", "close", "volume"]:
        if essential not in resolved:
            missing.append(essential)

    if missing:
        # If timestamp is index, handle it
        if isinstance(df.index, pd.DatetimeIndex):
            resolved["timestamp"] = "__index__"
            missing = [m for m in missing if m != "timestamp"]

        if missing:
            msg = f"Missing columns: {missing}. Available: {df.columns.tolist()}"
            raise ValueError(msg)

    # 1. Handle Timestamp
    if resolved.get("timestamp") == "__index__":
        dt_series = df.index
    else:
        dt_series = pd.to_datetime(df[resolved["timestamp"]], errors="coerce")

    if not isinstance(dt_series, pd.DatetimeIndex):
        dt_series = pd.to_datetime(dt_series)

    dt_series = dt_series.fillna(pd.Timestamp(0))

    # Handle timezone (support both Series and DatetimeIndex)
    # Convert to Series for consistent handling
    dt_series_s: pd.Series
    if isinstance(dt_series, pd.Index):
        dt_series_s = dt_series.to_series(index=dt_series)
    else:
        dt_series_s = cast(pd.Series, dt_series)

    # Help mypy know it's a Series
    if dt_series_s.dt.tz is None:
        dt_series_s = dt_series_s.dt.tz_localize("Asia/Shanghai")
    dt_series_s = dt_series_s.dt.tz_convert("UTC")

    timestamps = dt_series_s.astype("int64").values.tolist()

    # 2. Extract numeric columns
    def get_col(name: str) -> List[float]:
        return df[resolved[name]].fillna(0.0).astype(float).values.tolist()

    opens = get_col("open")
    highs = get_col("high")
    lows = get_col("low")
    closes = get_col("close")
    volumes = get_col("volume")

    # 3. Handle Symbol
    symbols_list: Optional[List[str]] = None
    symbol_val = None

    if symbol:
        symbol_val = symbol
    elif "symbol" in resolved:
        symbols_list = cast(List[str], df[resolved["symbol"]].astype(str).tolist())
    else:
        symbol_val = "UNKNOWN"

    # 4. Handle Extra Columns
    extra = {}
    used_columns = set(resolved.values())

    # Iterate over all columns to find numeric ones not in resolved
    for col in df.columns:
        if col in used_columns:
            continue

        # Try to convert to float
        try:
            # We use fillna(0.0) for safety, similar to other fields
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                extra[str(col)] = df[col].fillna(0.0).astype(float).values.tolist()
        except Exception:
            # Skip non-numeric extra columns
            pass

    return (
        timestamps,
        opens,
        highs,
        lows,
        closes,
        volumes,
        symbol_val,
        symbols_list,
        extra if extra else None,
    )


def prepare_dataframe(
    df: pd.DataFrame, date_col: Optional[str] = None, tz: str = "Asia/Shanghai"
) -> pd.DataFrame:
    r"""
    自动预处理 DataFrame，处理时区并生成标准时间戳列.

    :param df: 输入 DataFrame
    :param date_col: 日期列名 (若为 None 则自动探测)
    :param tz: 默认时区 (若数据为 Naive 时间，则假定为此时区)
    :return: 处理后的 DataFrame (包含 'timestamp' 列)
    """
    df = df.copy()

    # 1. Auto-detect date column
    if date_col is None:
        candidates = ["date", "datetime", "time", "timestamp", "日期", "时间"]
        for c in candidates:
            if c in df.columns:
                date_col = c
                break

    if date_col and date_col in df.columns:
        # 2. Convert to datetime
        dt = pd.to_datetime(df[date_col], errors="coerce")

        # 3. Handle Timezone
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(tz)

        # 4. Convert to UTC
        dt = dt.dt.tz_convert("UTC")

        # 5. Assign back
        df[date_col] = dt
        df["timestamp"] = dt.astype("int64")
    else:
        # Warn or ignore? For now silent, user might be processing non-time data?
        pass

    return df
