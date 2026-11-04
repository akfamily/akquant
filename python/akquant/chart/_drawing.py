"""策略绘图对象的归一化：记录层与所有渲染后端共享的唯一事实来源（spec 第 3、4 节）.

只做“输入 → 规范 dict”的纯函数变换；身份（owner/id/symbol）、去重、淘汰由
``akquant.strategy_drawing.DrawingApi`` 负责。
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ._normalize import timestamp_ms_from_ns

DRAWING_KINDS = ("line", "box", "label", "polyline", "fib", "table")
TABLE_POSITIONS = (
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
LINE_STYLES = ("solid", "dashed", "dotted")
LINE_EXTENDS = ("none", "right", "both")
LABEL_STYLES = ("up", "down", "left", "right", "none")
LABEL_SIZES = ("small", "normal", "large")
FIB_LEVELS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
GLOBAL_SYMBOL = "*"
MAX_TABLE_ROWS = 50
MAX_TABLE_COLS = 10
MAX_POLYLINE_POINTS = 1000
MAX_TEXT_LEN = 500

_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def _identity(ns: int) -> int:
    return ns


def _choice(value: Any, allowed: tuple[str, ...], field: str) -> str:
    text = str(value).strip().lower()
    if text not in allowed:
        raise ValueError(f"{field} 必须是 {', '.join(allowed)} 之一，当前为 {value!r}")
    return text


def normalize_color(value: Any, *, field: str) -> str:
    """Normalize a color to lowercase ``#rrggbb[aa]``; empty means renderer default."""
    if value is None or value == "":
        return ""
    text = str(value).strip()
    if not _COLOR_RE.match(text):
        raise ValueError(f"{field} 必须是 #RRGGBB 或 #RRGGBBAA，当前为 {value!r}")
    return text.lower()


def _positive_width(value: Any, field: str = "width") -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} 必须是正数，当前为 {value!r}") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} 必须是正数，当前为 {value!r}")
    return int(number) if number.is_integer() else number


def _unwrap_numpy(value: Any) -> Any:
    """Turn numpy scalars (except datetime64) into Python natives."""
    if isinstance(value, np.generic) and not isinstance(value, np.datetime64):
        return value.item()
    return value


def _to_ns(value: Any, tz: str) -> int:
    value = _unwrap_numpy(value)
    if isinstance(value, bool):
        raise ValueError("坐标时间不能是 bool")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        return int(ts.value)
    raise ValueError(
        "坐标时间须为 int(UTC 纳秒)/pd.Timestamp/datetime/np.datetime64，"
        f"当前为 {type(value).__name__}"
    )


def normalize_point(
    point: Any, *, tz: str, to_end_ns: Callable[[int], int] = _identity
) -> dict[str, Any]:
    """Normalize ``(time, price)`` to ``{"ts": ms, "price": float}``."""
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise ValueError(f"坐标点须为 (time, price)，当前为 {point!r}")
    time_value, price_value = point
    try:
        price = float(price_value)
    except (TypeError, ValueError):
        raise ValueError(f"坐标价格须为数值，当前为 {price_value!r}") from None
    if not math.isfinite(price):
        raise ValueError(f"坐标价格须为有限数，当前为 {price_value!r}")
    ns = to_end_ns(_to_ns(time_value, tz))
    return {"ts": timestamp_ms_from_ns(ns), "price": price}


def normalize_text(value: Any, *, field: str) -> tuple[str, bool]:
    """Return ``(text, truncated)``; text is cut to ``MAX_TEXT_LEN`` characters."""
    text = "" if value is None else str(value)
    if len(text) > MAX_TEXT_LEN:
        return text[:MAX_TEXT_LEN], True
    return text, False


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _normalize_cell_flag(cell: Any, precision: int) -> tuple[dict[str, Any], bool]:
    """Normalize one cell; second item says whether any text was truncated."""
    bg, fg, text_override, cut = "", "", None, False
    if isinstance(cell, dict):
        bg = normalize_color(cell.get("bg_color"), field="bg_color")
        fg = normalize_color(cell.get("text_color"), field="text_color")
        if cell.get("text") is not None:
            text_override, cut = normalize_text(cell["text"], field="text")
        value = cell.get("value")
    else:
        value = cell
    value = _unwrap_numpy(value)
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(
            f"单元格值须为 str/int/float/bool/None，当前为 {type(value).__name__}"
        )
    if isinstance(value, str) and len(value) > MAX_TEXT_LEN:
        value, cut = value[:MAX_TEXT_LEN], True
    if text_override is not None:
        text = text_override
    elif value is None or (isinstance(value, float) and math.isnan(value)):
        text = "—"
    elif isinstance(value, bool):
        text = "是" if value else "否"
    elif isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        if math.isinf(value):
            text = "∞" if value > 0 else "-∞"
        else:
            text = f"{value:,.{precision}f}"
    else:
        text = value
    out = {"value": _json_safe(value), "text": text, "bg_color": bg, "text_color": fg}
    return out, cut


def normalize_cell(cell: Any, precision: int) -> dict[str, Any]:
    """Normalize one table cell to ``{value, text, bg_color, text_color}``."""
    return _normalize_cell_flag(cell, precision)[0]


_EMPTY_CELL: dict[str, Any] = {
    "value": None,
    "text": "",
    "bg_color": "",
    "text_color": "",
}


def _normalize_table_flag(
    rows: Any, header: Any, precision: int
) -> tuple[dict[str, Any], bool]:
    if isinstance(rows, (str, bytes)) or not hasattr(rows, "__iter__"):
        raise ValueError("rows 须为二维序列（list[list]）")
    if header is not None and (
        isinstance(header, (str, bytes, dict)) or not hasattr(header, "__iter__")
    ):
        raise ValueError("header 须为序列（list），每项是一个表头单元格")
    raw_rows = [list(header)] if header is not None else []
    for row in rows:
        if isinstance(row, (str, bytes, dict)) or not hasattr(row, "__iter__"):
            raise ValueError("rows 须为二维序列（list[list]），每行是一个序列")
        raw_rows.append(list(row))
    if len(raw_rows) > MAX_TABLE_ROWS:
        raise ValueError(
            f"表格最多 {MAX_TABLE_ROWS} 行（含表头），当前 {len(raw_rows)}"
        )
    width = max((len(r) for r in raw_rows), default=0)
    if width > MAX_TABLE_COLS:
        raise ValueError(f"表格最多 {MAX_TABLE_COLS} 列，当前 {width}")
    truncated = False
    out = []
    for r in raw_rows:
        cells = []
        for c in r:
            cell, cut = _normalize_cell_flag(c, precision)
            truncated = truncated or cut
            cells.append(cell)
        cells += [dict(_EMPTY_CELL) for _ in range(width - len(r))]
        out.append(cells)
    return {"rows": out, "header_rows": 1 if header is not None else 0}, truncated


def normalize_table(rows: Any, header: Any, precision: int) -> dict[str, Any]:
    """Normalize a 2-D sequence (plus optional header) into padded cell rows."""
    return _normalize_table_flag(rows, header, precision)[0]


def _base(
    kind: str,
    points: list[dict[str, Any]],
    text: str,
    style: dict[str, Any],
    table: Optional[dict[str, Any]] = None,
    truncated: bool = False,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "pane": 0,
        "points": points,
        "text": text,
        "style": style,
        "table": table,
        "truncated": truncated,
    }


def build_line(
    p1: Any,
    p2: Any,
    *,
    tz: str,
    to_end_ns: Callable[[int], int],
    extend: str,
    color: Any,
    width: Any,
    style: str,
) -> dict[str, Any]:
    """Build a normalized ``line`` drawing."""
    return _base(
        "line",
        [normalize_point(p, tz=tz, to_end_ns=to_end_ns) for p in (p1, p2)],
        "",
        {
            "color": normalize_color(color, field="color"),
            "width": _positive_width(width),
            "line_style": _choice(style, LINE_STYLES, "style"),
            "extend": _choice(extend, LINE_EXTENDS, "extend"),
        },
    )


def build_box(
    top_left: Any,
    bottom_right: Any,
    *,
    tz: str,
    to_end_ns: Callable[[int], int],
    border_color: Any,
    fill_color: Any,
    width: Any,
    text: Any,
) -> dict[str, Any]:
    """Build a normalized ``box`` drawing."""
    body, cut = normalize_text(text, field="text")
    pts = [
        normalize_point(p, tz=tz, to_end_ns=to_end_ns) for p in (top_left, bottom_right)
    ]
    return _base(
        "box",
        pts,
        body,
        {
            "border_color": normalize_color(border_color, field="border_color"),
            "fill_color": normalize_color(fill_color, field="fill_color"),
            "width": _positive_width(width),
        },
        truncated=cut,
    )


def build_label(
    point: Any,
    text: Any,
    *,
    tz: str,
    to_end_ns: Callable[[int], int],
    style: str,
    color: Any,
    text_color: Any,
    size: str,
) -> dict[str, Any]:
    """Build a normalized ``label`` drawing."""
    body, cut = normalize_text(text, field="text")
    return _base(
        "label",
        [normalize_point(point, tz=tz, to_end_ns=to_end_ns)],
        body,
        {
            "label_style": _choice(style, LABEL_STYLES, "style"),
            "color": normalize_color(color, field="color"),
            "text_color": normalize_color(text_color, field="text_color"),
            "size": _choice(size, LABEL_SIZES, "size"),
        },
        truncated=cut,
    )


def build_polyline(
    points: Any,
    *,
    tz: str,
    to_end_ns: Callable[[int], int],
    closed: bool,
    color: Any,
    width: Any,
    fill_color: Any,
) -> dict[str, Any]:
    """Build a normalized ``polyline`` drawing (2 to 1000 points)."""
    raw = list(points)
    if not 2 <= len(raw) <= MAX_POLYLINE_POINTS:
        raise ValueError(
            f"polyline 点数须在 2～{MAX_POLYLINE_POINTS} 之间，当前 {len(raw)}"
        )
    return _base(
        "polyline",
        [normalize_point(p, tz=tz, to_end_ns=to_end_ns) for p in raw],
        "",
        {
            "color": normalize_color(color, field="color"),
            "width": _positive_width(width),
            "closed": bool(closed),
            "fill_color": normalize_color(fill_color, field="fill_color"),
        },
    )


def build_fib(
    p1: Any,
    p2: Any,
    *,
    tz: str,
    to_end_ns: Callable[[int], int],
    color: Any,
) -> dict[str, Any]:
    """Build a normalized ``fib`` drawing; levels are expanded by the renderer."""
    return _base(
        "fib",
        [normalize_point(p, tz=tz, to_end_ns=to_end_ns) for p in (p1, p2)],
        "",
        {"color": normalize_color(color, field="color")},
    )


def build_table(
    rows: Any,
    *,
    header: Any,
    position: str,
    precision: int,
    bg_color: Any,
    border_color: Any,
) -> dict[str, Any]:
    """Build a normalized ``table`` drawing (screen-fixed, no points)."""
    if (
        not isinstance(precision, int)
        or isinstance(precision, bool)
        or not 0 <= precision <= 10
    ):
        raise ValueError(f"precision 须为 0～10 的整数，当前为 {precision!r}")
    table, cut = _normalize_table_flag(rows, header, precision)
    table["position"] = _choice(position, TABLE_POSITIONS, "position")
    return _base(
        "table",
        [],
        "",
        {
            "bg_color": normalize_color(bg_color, field="bg_color"),
            "border_color": normalize_color(border_color, field="border_color"),
        },
        table=table,
        truncated=cut,
    )
