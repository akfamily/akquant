"""prepare_data: 本地文件 / 合成数据 -> 规范化 DataFrame -> Parquet 缓存.

规范列: date(带时区) open high low close volume symbol, 可直接传给
``run_backtest(data=df)``。dataset_id 是规范化 spec 的 sha1 前 12 位,
file 来源再并入绝对路径与 mtime_ns, 文件改动后自动失效。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from ...schema import COLUMN_ALIASES
from . import ServiceError
from .paths import ServiceConfig, resolve_data_file, write_json_atomic

DEFAULT_TZ = "Asia/Shanghai"
STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume", "symbol"]
#: 与 akquant.normalize 的候选时间列一致
DATE_CANDIDATES = ("date", "datetime", "time", "timestamp", "日期", "时间")
_OHLC = ["open", "high", "low", "close"]
#: 除时间列外, 各规范列认哪些列名(小写)。取自 akquant.schema 的别名表, 与
#: run_backtest(data=df) 认的列名一致, 含 AKShare 东方财富接口的中文列名
_FIELD_ALIASES = {
    field: tuple(alias.lower() for alias in COLUMN_ALIASES[field])
    for field in (*_OHLC, "volume", "symbol")
}
#: 文件里可以充当标的列的列名(小写); CSV 中这些列按文本读, 保住 000001 的前导零
_SYMBOL_COLUMNS = _FIELD_ALIASES["symbol"]
_ALLOWED_KEYS = {
    "file": {"source", "path", "symbol", "timezone"},
    "synthetic": {
        "source",
        "symbols",
        "start",
        "end",
        "freq",
        "seed",
        "start_price",
        "volatility",
    },
}
_SYNTHETIC_RANGE = {
    "1d": ("2023-01-02", "2024-12-31"),
    "1m": ("2024-01-02", "2024-01-08"),
}
#: source=akshare 已移除: 取数交给 AKShare 自己的 MCP 服务 AKTools
_AKSHARE_REMOVED = (
    'source="akshare" 已移除, AKQuant 的 MCP 不再联网取数。真实行情请用 AKTools'
    "(AKShare 的 MCP 服务)的 save_data 存成 CSV, 再把它返回的 path 连同 symbol "
    '传给 prepare_data: {"source": "file", "path": "<save_data 返回的 path>", '
    '"symbol": "600519"}; 没有接入 AKTools 时, 请先自行把数据存成 CSV 或 '
    "Parquet 放进项目目录"
)
_DATASET_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _meta_path(cfg: ServiceConfig, dataset_id: str) -> Path:
    return cfg.datasets_dir / f"{dataset_id}.meta.json"


def dataset_file(cfg: ServiceConfig, dataset_id: str) -> Path:
    """返回已准备好的数据文件; 不存在时提示先调 prepare_data."""
    if not _DATASET_ID_RE.match(dataset_id):
        raise ServiceError(f"dataset_id 格式不对: {dataset_id!r}")
    path = cfg.datasets_dir / f"{dataset_id}.parquet"
    if not path.is_file() or not _meta_path(cfg, dataset_id).is_file():
        raise ServiceError(f"找不到数据集 {dataset_id}, 请先调用 prepare_data")
    return path


def synthetic_frame(
    symbols: Sequence[str] = ("SYN001",),
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: str = "1d",
    seed: int = 0,
    start_price: float = 10.0,
    volatility: Optional[float] = None,
) -> pd.DataFrame:
    """几何随机游走 K 线, seed 固定时结果确定; 只用于验证策略能跑通."""
    if freq not in _SYNTHETIC_RANGE:
        raise ServiceError("synthetic 的 freq 只支持 1d / 1m")
    if not symbols:
        raise ServiceError("symbols 不能为空")
    if start_price <= 0:
        raise ServiceError("start_price 必须 > 0")
    vol = volatility if volatility is not None else (0.02 if freq == "1d" else 0.001)
    if not 0 < vol < 0.2:
        raise ServiceError("volatility 取值范围 (0, 0.2)")
    default_start, default_end = _SYNTHETIC_RANGE[freq]
    days = pd.bdate_range(start or default_start, end or default_end)
    if len(days) == 0:
        raise ServiceError("start~end 之间没有工作日")
    if freq == "1d":
        stamps = days.tz_localize(DEFAULT_TZ)
    else:
        offsets = [
            *pd.timedelta_range(
                pd.Timedelta(hours=9, minutes=31),
                pd.Timedelta(hours=11, minutes=30),
                freq="1min",
            ),
            *pd.timedelta_range(
                pd.Timedelta(hours=13, minutes=1),
                pd.Timedelta(hours=15),
                freq="1min",
            ),
        ]
        stamps = pd.DatetimeIndex(
            [day + offset for day in days for offset in offsets]
        ).tz_localize(DEFAULT_TZ)
    rng = np.random.default_rng(seed)
    n = len(stamps)
    frames = []
    for symbol in symbols:
        close = start_price * np.exp(np.cumsum(rng.normal(0.0, vol, n)))
        prev = np.concatenate([[start_price], close[:-1]])
        open_ = prev * (1 + rng.normal(0.0, vol / 4, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, vol / 2, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, vol / 2, n)))
        volume = rng.integers(100_000, 1_000_000, n).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "date": stamps,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "symbol": str(symbol),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _parse_dates(column: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(column):
        return pd.to_datetime(column.astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(column, errors="coerce")


def normalize_frame(
    df: pd.DataFrame, symbol: str, timezone: str
) -> Tuple[pd.DataFrame, List[str]]:
    """规范成 STANDARD_COLUMNS, 返回 (frame, issues); 阻断性问题抛 ServiceError."""
    issues: List[str] = []
    frame = df.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    date_col = next((c for c in DATE_CANDIDATES if c in frame.columns), None)
    if date_col is None and isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index(names="date")
        date_col = "date"
    if date_col is None:
        need = ", ".join(DATE_CANDIDATES)
        raise ServiceError(f"找不到时间列, 需要以下列名之一: {need}")
    for field, aliases in _FIELD_ALIASES.items():
        if field in frame.columns:
            continue  # 规范列名已在时不被别名列覆盖
        alias = next((a for a in aliases if a in frame.columns), None)
        if alias is not None:
            frame = frame.rename(columns={alias: field})
    missing = [c for c in _OHLC if c not in frame.columns]
    if missing:
        raise ServiceError(f"缺少必需列: {missing}")
    dates = _parse_dates(frame[date_col])
    if not pd.api.types.is_datetime64_any_dtype(dates) or dates.isna().all():
        raise ServiceError(f"时间列 {date_col} 无法解析(或混有不同时区)")
    if dates.dt.tz is None:
        dates = dates.dt.tz_localize(timezone)
        issues.append(f"数据不带时区, 已按 {timezone} 本地化")
    frame["date"] = dates
    for col in _OHLC:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    else:
        frame["volume"] = 0.0
        issues.append("缺少 volume 列, 已补 0")
    if "symbol" not in frame.columns:
        frame["symbol"] = symbol
    frame["symbol"] = frame["symbol"].astype(str)
    bad = frame["date"].isna() | frame[_OHLC].isna().any(axis=1)
    if bad.any():
        issues.append(f"{int(bad.sum())} 行时间或 OHLC 缺失, 已丢弃")
    frame = frame.loc[~bad, STANDARD_COLUMNS]
    frame = frame.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ServiceError("规范化后没有有效数据行")
    dup = int(frame.duplicated(["symbol", "date"]).sum())
    if dup:
        issues.append(f"{dup} 行“标的+时间”重复(未去重)")
    if bool((frame["volume"] == 0).all()):
        issues.append("成交量全为 0")
    return frame, issues


def _summary(frame: pd.DataFrame) -> Dict[str, Any]:
    dates = frame["date"]
    first = frame["symbol"].iloc[0]
    diffs = frame.loc[frame["symbol"] == first, "date"].diff().dropna()
    freq_guess: Optional[str] = None
    if not diffs.empty:
        med = diffs.median()
        seconds = int(med.total_seconds())
        if med >= pd.Timedelta(hours=20):
            freq_guess = "1d"
        elif seconds >= 60:
            freq_guess = f"{seconds // 60}m"
        else:
            freq_guess = f"{max(seconds, 1)}s"
    return {
        "rows": int(len(frame)),
        "symbols": sorted(str(s) for s in frame["symbol"].unique()),
        "start": dates.min().isoformat(),
        "end": dates.max().isoformat(),
        "freq_guess": freq_guess,
        "columns": [str(c) for c in frame.columns],
        "timezone": str(dates.dt.tz),
    }


def _normalize_spec(
    cfg: ServiceConfig, spec: Any
) -> Tuple[Dict[str, Any], Optional[Path]]:
    if not isinstance(spec, dict):
        raise ServiceError("spec 必须是对象")
    source = spec.get("source")
    if source == "akshare":
        raise ServiceError(_AKSHARE_REMOVED)
    if source not in _ALLOWED_KEYS:
        msg = f"spec.source 必须是 file / synthetic 之一, 收到 {source!r}"
        raise ServiceError(msg)
    extra = sorted(set(spec) - _ALLOWED_KEYS[source])
    if extra:
        allowed = sorted(_ALLOWED_KEYS[source])
        raise ServiceError(f"source={source} 不支持字段 {extra}; 允许: {allowed}")
    if source == "file":
        if "path" not in spec:
            raise ServiceError("source=file 需要 path")
        file, in_aktools = resolve_data_file(cfg, spec["path"])
        if not file.is_file():
            raise ServiceError(f"数据文件不存在: {spec['path']}")
        if file.suffix.lower() not in (".csv", ".parquet"):
            raise ServiceError("只支持 .csv / .parquet 文件")
        symbol = str(spec.get("symbol") or "").strip()
        if not symbol and in_aktools and not _has_symbol_column(file):
            # AKTools 的文件名是"接口名-参数哈希", 当标的名没有意义
            raise ServiceError(
                "AKTools 保存的文件里没有标的列, 文件名也不是标的代码: "
                '请在 spec 里传 symbol, 如 "symbol": "600519"'
            )
        return {
            "source": "file",
            "symbol": symbol or file.stem,
            "timezone": spec.get("timezone") or DEFAULT_TZ,
        }, file
    freq = spec.get("freq", "1d")
    default_start, default_end = _SYNTHETIC_RANGE.get(freq, (None, None))
    vol = spec.get("volatility")
    return {
        "source": "synthetic",
        "symbols": [str(s) for s in (spec.get("symbols") or ["SYN001"])],
        "start": spec.get("start") or default_start,
        "end": spec.get("end") or default_end,
        "freq": freq,
        "seed": int(spec.get("seed", 0)),
        "start_price": float(spec.get("start_price", 10.0)),
        "volatility": None if vol is None else float(vol),
    }, None


def _file_columns(file: Path) -> List[str]:
    """只读表头, 返回数据文件的列名."""
    if file.suffix.lower() == ".csv":
        return [str(c) for c in pd.read_csv(file, nrows=0).columns]
    import pyarrow.parquet as pq

    return [str(c) for c in pq.read_schema(file).names]


def _has_symbol_column(file: Path) -> bool:
    return any(c.strip().lower() in _SYMBOL_COLUMNS for c in _file_columns(file))


def _dataset_id(norm: Dict[str, Any], file: Optional[Path]) -> str:
    payload = dict(norm)
    if file is not None:
        payload["path"] = str(file)
        payload["mtime_ns"] = file.stat().st_mtime_ns
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _load(norm: Dict[str, Any], file: Optional[Path]) -> Tuple[pd.DataFrame, List[str]]:
    if norm["source"] == "file":
        assert file is not None
        if file.suffix.lower() == ".csv":
            header = pd.read_csv(file, nrows=0)
            text_cols = [
                c for c in header.columns if str(c).strip().lower() in _SYMBOL_COLUMNS
            ]
            raw = pd.read_csv(file, dtype={c: str for c in text_cols})
        else:
            raw = pd.read_parquet(file)
        return normalize_frame(raw, norm["symbol"], norm["timezone"])
    frame = synthetic_frame(
        symbols=norm["symbols"],
        start=norm["start"],
        end=norm["end"],
        freq=norm["freq"],
        seed=norm["seed"],
        start_price=norm["start_price"],
        volatility=norm["volatility"],
    )
    return frame, []


def prepare_data(cfg: ServiceConfig, spec: Dict[str, Any]) -> Dict[str, Any]:
    """准备回测数据并缓存, 返回 dataset_id 与摘要."""
    try:
        norm, file = _normalize_spec(cfg, spec)
    except ServiceError:
        raise
    except (ValueError, TypeError, KeyError) as exc:
        raise ServiceError(f"spec 字段取值不合法: {exc}") from exc
    dataset_id = _dataset_id(norm, file)
    synthetic = norm["source"] == "synthetic"
    meta_path = _meta_path(cfg, dataset_id)
    parquet = cfg.datasets_dir / f"{dataset_id}.parquet"
    if meta_path.is_file() and parquet.is_file():
        meta = cast(Dict[str, Any], json.loads(meta_path.read_text(encoding="utf-8")))
        return {
            "dataset_id": dataset_id,
            **meta["summary"],
            "issues": meta["issues"],
            "synthetic": synthetic,
            "cached": True,
        }
    try:
        frame, issues = _load(norm, file)
    except ServiceError:
        raise
    except (ValueError, TypeError, KeyError, OSError) as exc:
        # 含 pandas ParserError/EmptyDataError、坏时区、混合时区、坏日期等
        raise ServiceError(f"数据无法读取或规范化: {exc!r}") from exc
    cfg.datasets_dir.mkdir(parents=True, exist_ok=True)
    tmp = parquet.with_name(parquet.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, parquet)
    summary = _summary(frame)
    write_json_atomic(
        meta_path,
        {
            "spec": norm,
            "summary": summary,
            "issues": issues,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    return {
        "dataset_id": dataset_id,
        **summary,
        "issues": issues,
        "synthetic": synthetic,
        "cached": False,
    }
