"""指标契约 → lightweight-charts v5 series 的适配层.

``chart/d3kline.py`` 的镜像: 同一份指标契约(``indicator_definitions`` 的 8 值
``render_type``、整数 ``pane``、点位 ``timestamp_ms``)面向第二个具体前端。
AKQuant 仍只是"标准化指标数据的生产者", 本模块是可选的消费端适配器。

两条与 K 线对齐的硬约束:

- **时间值与 candles 同一套**(:func:`.._payload._bar_time`: 日频 ``YYYY-MM-DD``
  / 日内 UTC 秒), 否则 LWC 画不到同一根 bar 上;
- **严格递增且唯一**: LWC ``setData`` 对乱序/重复 time 直接报错, 这里去重(留
  最后)+ 排序, 并丢掉 ``None``/``NaN``(预热期)。

pane 映射: akquant ``0`` 叠主图 → LWC pane ``0``; akquant ``k>=1`` → LWC pane
``k+1`` —— LWC pane ``1`` 已被成交量占用, 量纲不同的指标不该被成交量压扁。

``render_type="candle"``: 同一 ``(owner, group)`` 的四个分量合成**一个**
``Candlestick`` series, 点位为 ``{time, open, high, low, close}``; 某个时刻缺
任一分量时跳过该时刻。组的 series 键来自
:func:`akquant.indicator_identity.chart_series_keys`。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Optional, Union

import pandas as pd

from ..chart._normalize import CANDLE_ROLES
from ..indicator_identity import (
    candle_group_of,
    chart_series_keys,
    display_label,
    group_display_label,
    identity_of,
)
from ._payload import _bar_time

#: akquant render_type → LWC v5 series 构造名。``Marker`` 不是 LWC
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
    "candle": "Candlestick",
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
        ``owner_strategy_id`` / ``indicator_key`` / ``timestamp_ms`` / ``value``);
        跨 symbol 混传可以, 这里按 ``symbol`` 过滤.
    :param symbol: 目标标的.
    :param intraday: 与该标的 K 线同一口径的日内标记, 决定时间值格式.
    :return: 按 ``(pane, series_key)`` 排序的 series; 该 symbol 无点位的指标
        不输出(免得前端建空副图). 定义与点位按 ``(owner_strategy_id,
        indicator_key)`` 关联; 每条 series 带 ``owner_strategy_id`` 与
        ``series_key``(见 :func:`akquant.indicator_identity.series_keys`: 同名键
        被多个 owner 使用时为 ``"owner:key"``, 否则就是 ``indicator_key``),
        ``indicator_key`` 保持原始值. candle 组只输出一条
        ``series_type="Candlestick"`` 的 series: ``indicator_key`` 与 ``group``
        为组名, ``series_key`` 为组的 series 键, 点位为 OHLC.
    """
    target = str(symbol).strip()
    defs = list(defs)
    keys = chart_series_keys(defs)
    by_identity: dict[tuple[str, str], dict[Union[str, int], float]] = {}
    for point in points:
        if str(point.get("symbol", "")).strip() != target:
            continue
        number = _as_float(point.get("value"))
        if number is None:
            continue
        identity = identity_of(point)
        if not identity[1]:
            continue
        time_value = _time_of(point.get("timestamp_ms"), intraday)
        # 按 (owner, key) 分桶, 桶内按 time 去重、后来者覆盖 —— 与 confirmed 语义一致
        by_identity.setdefault(identity, {})[time_value] = number

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    # candle 组: (owner, group) -> {role: 该分量的定义}; 组在首个分量出现的位置输出
    groups: dict[tuple[str, str], dict[str, Definition]] = {}
    for definition in defs:
        group = candle_group_of(definition)
        if group:
            role = str(definition.get("role") or "").strip().lower()
            groups.setdefault((identity_of(definition)[0], group), {})[role] = (
                definition
            )
    seen_groups: set[tuple[str, str]] = set()
    for definition in defs:
        identity = identity_of(definition)
        group = candle_group_of(definition)
        if group:
            group_id = (identity[0], group)
            if group_id in seen_groups:
                continue
            seen_groups.add(group_id)
            candle_series = _candle_series(
                group_id, groups[group_id], keys.groups[group_id], by_identity
            )
            if candle_series is not None:
                out.append(candle_series)
            continue
        if identity in seen:
            continue
        seen.add(identity)
        bucket = by_identity.get(identity)
        if not bucket:
            continue
        owner, key = identity
        series_key = keys.lines[identity]
        render_type = str(definition.get("render_type") or "line").strip().lower()
        color = definition.get("color")
        out.append(
            {
                "owner_strategy_id": owner,
                "indicator_key": key,
                "series_key": series_key,
                "display_name": display_label(definition, series_key),
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
    out.sort(key=lambda s: (s["pane"], s["series_key"]))
    return out


def _candle_series(
    group_id: tuple[str, str],
    components: Mapping[str, Definition],
    series_key: str,
    by_identity: Mapping[tuple[str, str], Mapping[Union[str, int], float]],
) -> Optional[dict[str, Any]]:
    """把一个 candle 组的四个分量拼成一条 Candlestick series.

    只保留四个分量在同一 time 都有值的时刻(缺任一分量的时刻跳过); 定义里缺
    分量或没有完整时刻时返回 ``None``。
    """
    owner, group = group_id
    if any(role not in components for role in CANDLE_ROLES):
        return None
    buckets = [
        by_identity.get(identity_of(components[role]), {}) for role in CANDLE_ROLES
    ]
    times = set(buckets[0])
    for bucket in buckets[1:]:
        times &= set(bucket)
    if not times:
        return None
    points = []
    for time_value in sorted(times, key=lambda t: _sort_key((t, 0.0))):
        point: dict[str, Any] = {"time": time_value}
        for role, bucket in zip(CANDLE_ROLES, buckets):
            point[role] = bucket[time_value]
        points.append(point)
    first = components["close"]
    color = first.get("color")
    return {
        "owner_strategy_id": owner,
        "indicator_key": group,
        "group": group,
        "series_key": series_key,
        "display_name": group_display_label(first, series_key),
        "pane": lwc_pane_index(first.get("pane")),
        "render_type": "candle",
        "series_type": "Candlestick",
        "color": str(color) if color else None,
        "price_lines": _price_lines(first.get("reference_lines")),
        "points": points,
    }


def _sort_key(item: tuple[Union[str, int], float]) -> tuple[int, Union[str, int]]:
    # 同一 series 内 time 类型一致(全字符串或全整数); 兜底把两类分开排避免比较报错
    t = item[0]
    return (0 if isinstance(t, int) else 1, t)


def to_lwc_update(message: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """把 :func:`akquant.to_indicator_message` 的 point 消息转成一条增量更新.

    产出可直接喂给前端 ``series.update({time, value})``。时间恒用 UTC 秒: 实时
    路径都是日内。``confirmed`` 原样带出, 缺失当 ``True``(早期版本与第三方
    sink 只在 bar 闭合后产点); 临时点与确认点同 time, LWC ``update()`` 按 time
    覆盖, 所以 L2 落地后前端零改动。``phase`` 同样原样带出, 缺失当 ``"run"``:
    ``"preload"`` 是 ``run_live(preload_history=)`` 回放出的历史段点。
    ``owner_strategy_id``(缺失按 ``"_default"``)与 ``series_key``(缺失回落到
    ``indicator_key``)也一并带出(1.5): 两个 owner 同名时按 ``series_key`` 区分线。
    ``group``/``role``(1.6, 缺失为 ``""``)同样带出: candle 分量的
    ``series_type`` 为 ``"Candlestick"``, ``point`` 仍是单值; 前端按
    ``(owner_strategy_id, group, symbol, point.time)`` 把四个分量合成一根 K 线,
    凑齐 open/high/low/close 后再 ``series.update({time, open, high, low, close})``。

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
    owner, indicator_key = identity_of(indicator)
    return {
        "symbol": message.get("symbol") or indicator.get("symbol"),
        "owner_strategy_id": owner,
        "indicator_key": str(indicator.get("indicator_key", "")),
        # 与 indicator_series() 的 series_key 同规则, 前端按它找到同一条线;
        # 1.4 消息没有该字段时回落到 indicator_key。
        "series_key": str(indicator.get("series_key") or indicator_key),
        "display_name": str(
            indicator.get("display_name") or indicator.get("indicator_key", "")
        ),
        "pane": lwc_pane_index(indicator.get("pane")),
        "series_type": LWC_SERIES_TYPE.get(render_type, "Line"),
        "color": str(color) if color else None,
        "group": str(indicator.get("group") or "").strip(),
        "role": str(indicator.get("role") or "").strip().lower(),
        "confirmed": confirmed,
        "phase": str(indicator.get("phase") or "run"),
        "point": {
            "time": int(indicator["timestamp_ms"]) // 1000,
            "value": number,
        },
    }
