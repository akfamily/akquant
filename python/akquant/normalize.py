"""统一数据归一化漏斗(单一实现).

本模块是「任意输入 -> 规范表示」的唯一实现处, 取代此前散落在
``utils``/``feed_adapter``/``data`` 中重复的列名解析、时区归一与 dtype 强转逻辑.

- ``dataframe_to_arrays`` / ``dataframe_to_bars``: 与旧
  ``utils.df_to_arrays`` / ``utils.load_bar_from_df`` **行为等价**的核心实现,
  两者共享 :func:`_timestamps_to_utc_ns` 与 :mod:`akquant.schema` 常量.
- ``to_frame`` / ``normalize``: 新的多源漏斗, 把 pandas / polars / pyarrow /
  parquet 路径 / ``list[Bar]`` 收敛到一处; ``normalize`` 产出规范 ``pa.Table``.
- ``arrow_to_legacy_arrays`` / ``arrow_to_bars``: Arrow -> 现有 Rust 边界的过渡桥.

A 期原则: 不动 Rust、不改数值、golden 全绿. 详见 docs/zh/meta/columnar-rfc.md 第 15 节.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from .akquant import Bar, from_arrays
from .schema import COLUMN_ALIASES, DEFAULT_INPUT_TIMEZONE, OHLCV_FIELDS

_LegacyArrays = Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[str],
    Optional[List[str]],
    Optional[Dict[str, np.ndarray]],
]


def _empty_arrays(symbol: Optional[str]) -> _LegacyArrays:
    return (
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        symbol,
        None,
        None,
    )


def _timestamps_to_utc_ns(dt_like: Any) -> np.ndarray:
    """将时间列/索引统一转为 UTC 纳秒 int64 数组.

    naive 时间按 :data:`DEFAULT_INPUT_TIMEZONE` 本地化后转 UTC; NaT 填充为 Epoch 0.
    行为与旧 ``utils.df_to_arrays`` 时间戳处理保持一致.
    """
    dt_series: Any
    if isinstance(dt_like, pd.DatetimeIndex):
        dt_series = dt_like
    else:
        dt_series = pd.to_datetime(dt_like, errors="coerce")
        if not isinstance(dt_series, pd.DatetimeIndex):
            dt_series = pd.to_datetime(dt_series)

    # naive 情况下先确保纳秒分辨率, 再本地化
    if isinstance(dt_series, pd.DatetimeIndex):
        is_aware = dt_series.tz is not None
    else:
        is_aware = dt_series.dt.tz is not None
    if not is_aware:
        dt_series = dt_series.astype("datetime64[ns]")

    dt_series = dt_series.fillna(pd.Timestamp(0))

    if isinstance(dt_series, pd.Index):
        dt_series_s = dt_series.to_series(index=dt_series)
    else:
        dt_series_s = cast(pd.Series, dt_series)

    if dt_series_s.dt.tz is None:
        dt_series_s = dt_series_s.dt.tz_localize(DEFAULT_INPUT_TIMEZONE)
    dt_series_s = dt_series_s.dt.tz_convert("UTC")

    if not str(dt_series_s.dtype).startswith("datetime64[ns"):
        try:
            dt_series_s = dt_series_s.astype("datetime64[ns, UTC]")
        except Exception:
            pass

    return cast(np.ndarray, dt_series_s.astype("int64").values)


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """解析规范字段到实际列名(精确匹配优先, 大小写不敏感回退)."""
    df_cols = list(df.columns)
    df_cols_lower = [str(c).lower() for c in df_cols]
    resolved: Dict[str, str] = {}
    for key, candidates in COLUMN_ALIASES.items():
        found: Optional[str] = None
        for cand in candidates:
            if cand in df_cols:
                found = cand
                break
        if not found:
            for cand in candidates:
                if cand.lower() in df_cols_lower:
                    idx = df_cols_lower.index(cand.lower())
                    found = str(df_cols[idx])
                    break
        if found:
            resolved[key] = found
    return resolved


def dataframe_to_arrays(
    df: pd.DataFrame, symbol: Optional[str] = None
) -> _LegacyArrays:
    """DataFrame -> ``DataFeed.add_arrays`` 的 9 元组入参.

    与旧 ``utils.df_to_arrays`` 行为等价: 列名解析、时区归一、dtype 强转、
    extra 数值列透传均保持一致.
    """
    if df.empty:
        return _empty_arrays(symbol)

    resolved = resolve_columns(df)

    missing = [f for f in OHLCV_FIELDS if f not in resolved]
    if "timestamp" not in resolved:
        missing.append("timestamp")

    if missing:
        if isinstance(df.index, pd.DatetimeIndex):
            resolved["timestamp"] = "__index__"
            missing = [m for m in missing if m != "timestamp"]
        if missing:
            msg = f"Missing columns: {missing}. Available: {df.columns.tolist()}"
            raise ValueError(msg)

    # 1. 时间戳
    if resolved.get("timestamp") == "__index__":
        timestamps = _timestamps_to_utc_ns(df.index)
    else:
        raw = pd.to_datetime(df[resolved["timestamp"]], errors="coerce")
        timestamps = _timestamps_to_utc_ns(raw)

    # 2. OHLCV
    def get_col(name: str) -> np.ndarray:
        return cast(np.ndarray, df[resolved[name]].fillna(0.0).astype(float).values)

    opens = get_col("open")
    highs = get_col("high")
    lows = get_col("low")
    closes = get_col("close")
    volumes = get_col("volume")

    # 3. Symbol
    symbols_list: Optional[List[str]] = None
    symbol_val: Optional[str] = None
    if symbol:
        symbol_val = symbol
    elif "symbol" in resolved:
        symbols_list = cast(List[str], df[resolved["symbol"]].astype(str).tolist())
    else:
        symbol_val = "UNKNOWN"

    # 4. Extra 数值列透传
    extra: Dict[str, np.ndarray] = {}
    used_columns = set(resolved.values())
    for col in df.columns:
        if col in used_columns:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                extra[str(col)] = cast(
                    np.ndarray, df[col].fillna(0.0).astype(float).values
                )
        except Exception:
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


def dataframe_to_bars(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    column_map: Optional[Dict[str, str]] = None,
) -> List[Bar]:
    """DataFrame -> ``list[Bar]``.

    与旧 ``utils.load_bar_from_df`` 行为等价: 支持 ``column_map`` (DF 列 -> 规范字段),
    并保留秒级时间戳自动修正 (< 1e10 时 ×1e9).
    """
    if df.empty:
        return []

    required_map = {
        "date": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    if column_map:
        required_map.update(column_map)

    internal_fields = ["timestamp", "open", "high", "low", "close", "volume"]
    field_to_col: Dict[str, str] = {}
    for col, field in required_map.items():
        if field in internal_fields:
            field_to_col[field] = col

    missing_fields: List[str] = []
    for field in internal_fields:
        if field not in field_to_col:
            if field in df.columns:
                field_to_col[field] = field
            else:
                missing_fields.append(field)
        else:
            if field_to_col[field] not in df.columns:
                missing_fields.append(f"{field} (mapped to {field_to_col[field]})")

    if missing_fields:
        raise ValueError(f"DataFrame missing columns for fields: {missing_fields}")

    timestamps = _timestamps_to_utc_ns(
        pd.to_datetime(df[field_to_col["timestamp"]], errors="coerce")
    )

    opens = df[field_to_col["open"]].fillna(0.0).astype(float).values
    highs = df[field_to_col["high"]].fillna(0.0).astype(float).values
    lows = df[field_to_col["low"]].fillna(0.0).astype(float).values
    closes = df[field_to_col["close"]].fillna(0.0).astype(float).values
    volumes = df[field_to_col["volume"]].fillna(0.0).astype(float).values

    symbols_list: Optional[List[str]] = None
    symbol_val: Optional[str] = None
    if symbol:
        symbol_val = symbol
    elif "股票代码" in df.columns:
        symbols_list = cast(List[str], df["股票代码"].astype(str).tolist())
    else:
        symbol_val = "UNKNOWN"

    bars = from_arrays(
        timestamps, opens, highs, lows, closes, volumes, symbol_val, symbols_list, None
    )

    # 秒级时间戳自动修正: 有效纳秒时间戳(2023 年)约 1.6e18,
    # 10_000_000_000 秒已是 2286 年, 故 < 1e10 视为秒级, 乘 1e9.
    if bars and bars[0].timestamp < 10_000_000_000:
        for bar in bars:
            bar.timestamp = bar.timestamp * 1_000_000_000

    return bars


# --------------------------------------------------------------------------- #
# 多源漏斗 + Arrow 过渡桥
# --------------------------------------------------------------------------- #


def to_frame(source: Any) -> pd.DataFrame:
    """把任意受支持的数据源转为 pandas DataFrame(漏斗第一步, 仅负责"变成表").

    支持: pandas.DataFrame / polars.DataFrame|LazyFrame / pyarrow.Table /
    parquet|csv 路径 / ``list[Bar]``.
    """
    if isinstance(source, pd.DataFrame):
        return source

    if isinstance(source, pa.Table):
        return cast(pd.DataFrame, source.to_pandas())

    # polars (可选, 惰性导入避免强依赖顺序)
    try:
        import polars as pl

        if isinstance(source, pl.LazyFrame):
            source = source.collect()
        if isinstance(source, pl.DataFrame):
            return cast(pd.DataFrame, source.to_pandas())
    except ImportError:
        pass

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if path.suffix.lower() in (".csv", ".txt"):
            return pd.read_csv(path)
        raise ValueError(f"unsupported file extension: {path.suffix}")

    if isinstance(source, list):
        records = [
            {
                "timestamp": pd.Timestamp(b.timestamp, unit="ns", tz="UTC"),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "symbol": b.symbol,
            }
            for b in source
        ]
        return pd.DataFrame(records)

    raise TypeError(f"unsupported data source type: {type(source)!r}")


def coerce_to_pandas(data: Any) -> Any:
    """将 polars / pyarrow 输入转为 pandas DataFrame; 其余类型原样返回.

    供 ``run_backtest`` 分派入口调用: 让 polars/Arrow 输入复用既有 pandas 数据路径,
    使 ``pl.DataFrame`` 成为平权一等输入(issue #298), 且不改动下游行为.
    ``DataFeed`` / ``dict`` / ``list[Bar]`` / ``pd.DataFrame`` 等一律不动.
    """
    if isinstance(data, pa.Table):
        return data.to_pandas()
    try:
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            return data.collect().to_pandas()
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
    except ImportError:
        pass
    return data


def normalize(source: Any, symbol: Optional[str] = None) -> pa.Table:
    """任意输入 -> 规范 ``pa.Table``(RFC 漏斗对外入口).

    A 期产出全量物化的 ``pa.Table``; 流式 ``RecordBatchReader`` 留待 C 期.
    """
    df = to_frame(source)
    ts, o, h, low, c, v, sym, syms, extra = dataframe_to_arrays(df, symbol=symbol)

    columns: Dict[str, Any] = {
        "ts_event": pd.to_datetime(ts, unit="ns", utc=True),
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "volume": v,
    }
    if syms is not None:
        columns["instrument_id"] = syms
    else:
        columns["instrument_id"] = [sym or "UNKNOWN"] * len(ts)
    if extra:
        for key, arr in extra.items():
            columns[key] = arr

    return pa.table(columns)


def arrow_to_legacy_arrays(table: pa.Table) -> _LegacyArrays:
    """规范 ``pa.Table`` -> ``DataFeed.add_arrays`` 的 9 元组(Arrow 过渡桥)."""
    names = set(table.column_names)
    ts = table.column("ts_event").to_numpy().astype("int64")

    def col(name: str) -> np.ndarray:
        return cast(np.ndarray, table.column(name).to_numpy().astype("float64"))

    opens, highs, lows, closes, volumes = (
        col("open"),
        col("high"),
        col("low"),
        col("close"),
        col("volume"),
    )

    symbol_val: Optional[str] = None
    symbols_list: Optional[List[str]] = None
    if "instrument_id" in names:
        ids = [str(x) for x in table.column("instrument_id").to_pylist()]
        unique = set(ids)
        if len(unique) <= 1:
            symbol_val = next(iter(unique)) if unique else "UNKNOWN"
        else:
            symbols_list = ids

    reserved = {"ts_event", "instrument_id", *OHLCV_FIELDS}
    extra: Dict[str, np.ndarray] = {}
    for name in table.column_names:
        if name in reserved:
            continue
        arr = table.column(name)
        if pa.types.is_floating(arr.type) or pa.types.is_integer(arr.type):
            extra[name] = arr.to_numpy().astype("float64")

    return (
        ts,
        opens,
        highs,
        lows,
        closes,
        volumes,
        symbol_val,
        symbols_list,
        extra if extra else None,
    )


def arrow_to_bars(table: pa.Table, symbol: Optional[str] = None) -> List[Bar]:
    """规范 ``pa.Table`` -> ``list[Bar]``(Arrow 过渡桥)."""
    arrays = arrow_to_legacy_arrays(table)
    if symbol is not None:
        arrays = (*arrays[:6], symbol, None, arrays[8])
    bars = from_arrays(*arrays)
    if bars and bars[0].timestamp < 10_000_000_000:
        for bar in bars:
            bar.timestamp = bar.timestamp * 1_000_000_000
    return bars
