"""把 pandas/numpy 值转成可 JSON 序列化的 Python 值."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    """递归转换值.

    时间转带时区的 ISO 8601; NaN/NaT/inf 转 None; numpy 标量与数组转原生值;
    时间差转字符串; 其他未知类型转 ``str``。
    """
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, dt.timedelta)):
        return str(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def frame_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame 转 JSON 友好的行列表."""
    return [
        {str(k): to_jsonable(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]
