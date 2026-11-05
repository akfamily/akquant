"""绘图对象 → LWC 复盘 payload(spec 第 7 节).

点的时间吸附到不晚于它的最近一根 bar, 这样前端 ``timeToCoordinate`` 总能拿到
坐标(LWC 只认数据里存在的 time); 早于第一根 bar 的点丢弃, 点数不够的对象整个
丢弃。晚于最后一根 bar 的点吸附到最后一根。

不复用 :func:`._payload._snap_time`: 它对早于首根 bar 的时间退回首根(交易标记
需要这个兜底), 而绘图对象要丢弃这类点。

时区: 点的 ``ts`` 是 UTC 毫秒。行情索引带时区时直接比较; 索引不带时区时按
``tz``(回测的策略时区, 引擎正是按它本地化无时区行情的)换成本地墙钟时间再比较,
缺省按 UTC。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ._payload import _bar_times

#: 各 kind 渲染所需的最少点数; 未知 kind 不渲染(spec 9.5: 未知 kind 忽略)。
_MIN_POINTS = {"line": 2, "box": 2, "label": 1, "polyline": 2, "fib": 2}
#: 表格 9 宫格位置, 也是表格的排序顺序。
_POSITION_ORDER = (
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)


def _utc_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """``DatetimeIndex`` → int64 纳秒数组(带时区时为 UTC 纳秒, 无时区时为墙钟纳秒)."""
    index = index.as_unit("ns")
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index.to_numpy(dtype="datetime64[ns]").astype("int64")


def _snap_positions(
    ts_ms: list[Any], index_ns: np.ndarray, naive: bool, tz: Optional[str]
) -> np.ndarray:
    """每个点吸附到 ``<= ts`` 的最后一根 bar 的下标; 早于首根 bar 为 ``-1``.

    :param ts_ms: 点的 UTC 毫秒.
    :param index_ns: :func:`_utc_ns` 处理后的 bar 索引(整个标的只算一次).
    :param naive: bar 索引是否无时区; 是则点先换成 ``tz`` 的墙钟时间再比较.
    """
    if len(index_ns) == 0 or not ts_ms:
        return np.full(len(ts_ms), -1, dtype="int64")
    points = pd.DatetimeIndex(
        pd.to_datetime(np.asarray(ts_ms, dtype="int64"), unit="ms", utc=True)
    )
    if naive:
        # 无时区索引是本地墙钟时间: 先换到策略时区再抹掉时区信息。
        points = points.tz_convert(tz or "UTC").tz_localize(None)
    return index_ns.searchsorted(_utc_ns(points), side="right") - 1


def to_lwc_drawings(
    drawings: list[dict[str, Any]],
    symbol: str,
    bar_index: pd.DatetimeIndex,
    intraday: bool,
    *,
    tz: Optional[str] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把 ``result.drawings`` 转成某个标的的 LWC 图形与表格.

    :param drawings: ``BacktestResult.drawings`` 的对象列表(spec 第 4 节结构).
    :param symbol: 目标标的; 图形只取该标的, 表格取该标的与全局表(``"*"``).
    :param bar_index: 该标的 K 线的时间索引(已去重、升序).
    :param intraday: 与 K 线同一口径的日内标记, 决定时间值格式.
    :param tz: 策略时区; 只在 ``bar_index`` 无时区时用于解释墙钟时间.
    :return: ``(shapes, tables)``。shape 为 ``{drawing_id, kind, text, style,
        points: [{time, price}]}``; table 为 ``{drawing_id, symbol, table,
        style}``, 按 ``(position, 全局表在前, created_seq)`` 排序。
    """
    shapes: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    naive = bar_index.tz is None
    # 索引的纳秒数组与 LWC 时间值都按标的只算一次(日内可能数万根 bar)。
    index_ns = _utc_ns(bar_index)
    bar_times: Optional[list[Any]] = None
    for d in drawings:
        sym = str(d.get("symbol", ""))
        kind = str(d.get("kind", ""))
        if kind == "table":
            table = d.get("table")
            if sym in (symbol, "*") and isinstance(table, dict):
                if table.get("position") in _POSITION_ORDER:
                    tables.append(d)
            continue
        if sym != symbol or kind not in _MIN_POINTS:
            continue
        raw_points = list(d.get("points") or [])
        positions = _snap_positions(
            [p["ts"] for p in raw_points], index_ns, naive, tz
        ).tolist()
        if bar_times is None:
            bar_times = _bar_times(bar_index, intraday)
        points = [
            {"time": bar_times[pos], "price": float(p["price"])}
            for p, pos in zip(raw_points, positions)
            if pos >= 0
        ]
        if len(points) < _MIN_POINTS[kind]:
            continue
        shapes.append(
            {
                "drawing_id": d.get("drawing_id"),
                "kind": kind,
                "text": d.get("text") or "",
                "style": dict(d.get("style") or {}),
                "points": points,
            }
        )
    tables.sort(
        key=lambda t: (
            _POSITION_ORDER.index(t["table"]["position"]),
            0 if t.get("symbol") == "*" else 1,
            int(t.get("created_seq", 0)),
        )
    )
    return shapes, [
        {
            "drawing_id": t.get("drawing_id"),
            "symbol": t.get("symbol"),
            "table": t["table"],
            "style": dict(t.get("style") or {}),
        }
        for t in tables
    ]
