"""指标契约 → lightweight-charts v5 series 的适配层.

``chart/d3kline.py`` 的镜像: 同一份指标契约(``indicator_definitions`` 的 7 值
``render_type``、整数 ``pane``、点位 ``timestamp_ms``)面向第二个具体前端。
AKQuant 仍只是"标准化指标数据的生产者", 本模块是可选的消费端适配器。

两条与 K 线对齐的硬约束:

- **时间值与 candles 同一套**(:func:`.._payload._bar_time`: 日频 ``YYYY-MM-DD``
  / 日内 UTC 秒), 否则 LWC 画不到同一根 bar 上;
- **严格递增且唯一**: LWC ``setData`` 对乱序/重复 time 直接报错, 这里去重(留
  最后)+ 排序, 并丢掉 ``None``/``NaN``(预热期)。

pane 映射: akquant ``0`` 叠主图 → LWC pane ``0``; akquant ``k>=1`` → LWC pane
``k+1`` —— LWC pane ``1`` 已被成交量占用, 量纲不同的指标不该被成交量压扁。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Optional, Union

import pandas as pd

from ._payload import _bar_time

#: akquant render_type(封闭 7 值)→ LWC v5 series 构造名。``Marker`` 不是 LWC
#: series: signal 的契约是"主图上的标记类渲染", 前端把它交给 K 线的
#: ``createSeriesMarkers`` 而不建 series。scatter 用 Line 系列关掉连线只画点
#: (``lineVisible=false`` + ``pointMarkersVisible=true``)。
LWC_SERIES_TYPE: dict[str, str] = {
    "line": "Line",
    "area": "Area",
    "bar": "Histogram",
    "column": "Histogram",
    "histogram": "Histogram",
    "scatter": "Line",
    "signal": "Marker",
}

#: 成交量占用的 LWC pane, 指标副图从它之后排。
_VOLUME_PANE = 1

Definition = Mapping[str, Any]
Point = Mapping[str, Any]


def lwc_pane_index(pane: Any) -> int:
    """把 akquant 整数 pane 映射到 LWC pane 下标(0 叠主图, k>=1 → k+1)."""
    if pane is None or pane == "":
        return 0
    try:
        value = int(pane)
    except (TypeError, ValueError):
        return 0
    return 0 if value <= 0 else value + _VOLUME_PANE


def _as_float(value: Any) -> Optional[float]:
    """未就绪(``None``/``NaN``/不可转数)返回 ``None``, 否则返回 float."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _time_of(ts_ms: Any, intraday: bool) -> Union[str, int]:
    ts = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC")
    return _bar_time(ts, intraday)


def _price_lines(raw: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    if not isinstance(raw, (list, tuple)):
        return lines
    for entry in raw:
        if not isinstance(entry, Mapping) or "value" not in entry:
            continue
        lines.append(
            {
                "value": float(entry["value"]),
                "label": str(entry.get("label") or ""),
                "color": str(entry["color"]) if entry.get("color") else None,
            }
        )
    return lines


def to_lwc_indicator_series(
    defs: Iterable[Definition],
    points: Iterable[Point],
    symbol: str,
    intraday: bool,
) -> list[dict[str, Any]]:
    """把某个 symbol 的指标定义与点位转成 LWC 可 ``setData`` 的 series 列表.

    :param defs: 指标定义, 字段同 ``BacktestResult.indicator_definitions`` 的列.
    :param points: 点位, 字段同 ``indicator_df()`` 的行(``symbol`` /
        ``indicator_key`` / ``timestamp_ms`` / ``value``); 跨 symbol 混传可以,
        这里按 ``symbol`` 过滤.
    :param symbol: 目标标的.
    :param intraday: 与该标的 K 线同一口径的日内标记, 决定时间值格式.
    :return: 按 ``(pane, indicator_key)`` 排序的 series; 该 symbol 无点位的指标
        不输出(免得前端建空副图).
    """
    target = str(symbol).strip()
    by_key: dict[str, dict[Union[str, int], float]] = {}
    for point in points:
        if str(point.get("symbol", "")).strip() != target:
            continue
        number = _as_float(point.get("value"))
        if number is None:
            continue
        key = str(point.get("indicator_key", ""))
        if not key:
            continue
        time_value = _time_of(point.get("timestamp_ms"), intraday)
        # dict 按 time 去重, 后来者覆盖 —— 与 confirmed 语义一致
        by_key.setdefault(key, {})[time_value] = number

    out: list[dict[str, Any]] = []
    for definition in defs:
        key = str(definition.get("indicator_key", ""))
        bucket = by_key.get(key)
        if not bucket:
            continue
        render_type = str(definition.get("render_type") or "line").strip().lower()
        color = definition.get("color")
        out.append(
            {
                "indicator_key": key,
                "display_name": str(definition.get("display_name") or key),
                "pane": lwc_pane_index(definition.get("pane")),
                "render_type": render_type,
                "series_type": LWC_SERIES_TYPE.get(render_type, "Line"),
                "color": str(color) if color else None,
                "price_lines": _price_lines(definition.get("reference_lines")),
                "points": [
                    {"time": t, "value": v}
                    for t, v in sorted(bucket.items(), key=_sort_key)
                ],
            }
        )
    out.sort(key=lambda s: (s["pane"], s["indicator_key"]))
    return out


def _sort_key(item: tuple[Union[str, int], float]) -> tuple[int, Union[str, int]]:
    # 同一 series 内 time 类型一致(全字符串或全整数); 兜底把两类分开排避免比较报错
    t = item[0]
    return (0 if isinstance(t, int) else 1, t)


def to_lwc_update(message: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """把 :func:`akquant.to_indicator_message` 的 point 消息转成一条增量更新.

    产出可直接喂给前端 ``series.update({time, value})``。时间恒用 UTC 秒: 实时
    路径都是日内。``confirmed`` 原样带出, 缺失当 ``True``(早期版本与第三方
    sink 只在 bar 闭合后产点); 临时点与确认点同 time, LWC ``update()`` 按 time
    覆盖, 所以 L2 落地后前端零改动。

    :return: 增量更新 dict; 非 point 消息或未就绪(``None``/``NaN``)点返回 ``None``.
    """
    if str(message.get("type", "")) != "point":
        return None
    indicator = message.get("indicator")
    if not isinstance(indicator, Mapping):
        return None
    number = _as_float(indicator.get("value"))
    if number is None:
        return None
    render_type = str(indicator.get("render_type") or "line").strip().lower()
    color = indicator.get("color")
    confirmed_raw = indicator.get("confirmed", True)
    confirmed = (
        confirmed_raw
        if isinstance(confirmed_raw, bool)
        else str(confirmed_raw).strip().lower() in {"1", "true", "yes", "on"}
    )
    return {
        "symbol": message.get("symbol") or indicator.get("symbol"),
        "indicator_key": str(indicator.get("indicator_key", "")),
        "display_name": str(
            indicator.get("display_name") or indicator.get("indicator_key", "")
        ),
        "pane": lwc_pane_index(indicator.get("pane")),
        "series_type": LWC_SERIES_TYPE.get(render_type, "Line"),
        "color": str(color) if color else None,
        "confirmed": confirmed,
        "point": {
            "time": int(indicator["timestamp_ms"]) // 1000,
            "value": number,
        },
    }
