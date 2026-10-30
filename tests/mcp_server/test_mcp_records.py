"""akquant.mcp.service.records: pandas/numpy 值转 JSON."""

import json
import math

import numpy as np
import pandas as pd
from akquant.mcp.service.records import frame_to_records, to_jsonable


def test_scalars() -> None:
    """NaN/NaT/inf 转 None, numpy 标量转原生值, 时间带时区."""
    ts = pd.Timestamp("2024-01-02 09:31", tz="Asia/Shanghai")
    assert to_jsonable(float("nan")) is None
    assert to_jsonable(math.inf) is None
    assert to_jsonable(pd.NaT) is None
    assert to_jsonable(np.int64(3)) == 3
    assert isinstance(to_jsonable(np.float64(1.5)), float)
    assert to_jsonable(ts) == "2024-01-02T09:31:00+08:00"
    assert to_jsonable(pd.Timedelta(minutes=5)) == "0 days 00:05:00"
    assert to_jsonable({"a": [np.int64(1), None]}) == {"a": [1, None]}
    assert to_jsonable(np.array([1, 2])) == [1, 2]


def test_frame_to_records_is_json_serializable() -> None:
    """DataFrame 行可直接 json.dumps."""
    df = pd.DataFrame(
        {
            "t": pd.to_datetime(["2024-01-02"]).tz_localize("Asia/Shanghai"),
            "x": [float("nan")],
            "d": [pd.Timedelta(days=1)],
            "s": [None],
        }
    )
    rows = frame_to_records(df)
    assert rows == [
        {"t": "2024-01-02T00:00:00+08:00", "x": None, "d": "1 days 00:00:00", "s": None}
    ]
    json.dumps(rows)
