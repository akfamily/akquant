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

import pandas as pd

from ._payload import _bar_time

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


def _point_ns(ts_ms: Any, bar_index: pd.DatetimeIndex, tz: Optional[str]) -> int:
    """把 UTC 毫秒换成与 ``bar_index`` 同一口径的纳秒整数(便于 searchsorted)."""
    ts = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC")
    if getattr(bar_index, "tz", None) is None:
        # 无时区索引是本地墙钟时间: 先换到策略时区再抹掉时区信息。
        ts = ts.tz_convert(tz or "UTC").tz_localize(None)
    return int(ts.as_unit("ns").value)


def _snap_positions(
    ts_list: list[Any], bar_index: pd.DatetimeIndex, tz: Optional[str]
) -> list[int]:
    """每个点吸附到 ``<= ts`` 的最后一根 bar 的下标; 早于首根 bar 为 ``-1``."""
    if len(bar_index) == 0 or not ts_list:
        return [-1] * len(ts_list)
    index_ns = bar_index.as_unit("ns")
    if index_ns.tz is not None:
        index_ns = index_ns.tz_convert("UTC").tz_localize(None)
    values = index_ns.to_numpy(dtype="datetime64[ns]").astype("int64")
    targets = [_point_ns(t, bar_index, tz) for t in ts_list]
    return [int(p) - 1 for p in values.searchsorted(targets, side="right")]


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
        positions = _snap_positions([p["ts"] for p in raw_points], bar_index, tz)
        points = [
            {
                "time": _bar_time(bar_index[pos], intraday),
                "price": float(p["price"]),
            }
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
