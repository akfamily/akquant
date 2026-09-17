"""Convert indicator definitions/points into d3Kline ``IndicatorOption`` dicts.

（可选的消费端适配器）

AKQuant 的定位是"标准化指标数据的生产者"，本模块是它面向一个**具体前端**
（金融界期魔方前端的 d3Kline 渲染器）的适配层，与 ``indicator_stream`` 产出
"前端友好消息"同一性质。放在核心包里的理由只有一个：回测服务与实盘服务两条
链路都要产出这套结构，前端用**同一套**渲染逻辑消费——若两边各写一份，任何一边
漂移都会让前端收到不一致的数据。这里是单一事实来源。

输入刻意用 ``Mapping[str, Any]`` 而非 pydantic 模型：两个消费方一个传 sqlite
Row 转的裸 dict、一个传 ``model_dump()``，最低公分母是 Mapping。缺失字段一律
按 akquant 契约的默认值处理（``pane=0``、``render_type="line"``）。
"""

from collections.abc import Iterable, Mapping
from typing import Any

# akquant render_type（封闭 7 值）→ d3Kline IndicatorSeriesOption.type（仅 line/bar）
D3KLINE_SERIES_TYPE: dict[str, str] = {
    "line": "line",
    "area": "line",  # d3Kline 无面积图
    "bar": "bar",
    "column": "bar",
    "histogram": "bar",
    "scatter": "line",  # d3Kline 无散点
    "signal": "line",  # 标记类应走 trade-markers 通道，此处兜底
}

# 副图默认高度，对齐前端 indicatorMeta.ts 内置指标惯例（多数 100）
SUB_PANE_HEIGHT = 100

Definition = Mapping[str, Any]
Point = Mapping[str, Any]


def _pane_of(definition: Definition) -> int:
    """Normalize ``pane`` to int (sqlite Row / JSON may give str or None)."""
    raw = definition.get("pane")
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _group_by_pane(defs: Iterable[Definition]) -> dict[int, list[Definition]]:
    """Group definitions by pane index."""
    panes: dict[int, list[Definition]] = {}
    for definition in defs:
        panes.setdefault(_pane_of(definition), []).append(definition)
    return panes


def to_d3kline_options(
    defs: Iterable[Definition],
    points_by_key: Mapping[str, Iterable[Point]],
) -> list[dict[str, Any]]:
    """Build d3Kline ``IndicatorOption`` dicts ready for ``IndicatorManager.create``.

    同一 pane 内的多个指标打包成**一个** option，各指标作为 ``dataList`` 的一条
    series——这样「5 个副图指标只用 2 个 pane」时前端只占 2 个 record，可缓解
    ``IndicatorManager`` 的 ``MAX_SUB_INDICATORS=3`` 静默丢弃。

    调用方需保证同一 pane 内 ``indicator_key`` 唯一（存储层的 UNIQUE 约束负责）；
    本函数不校验、不去重，重复的 key 会产出同名 series 而前端 ``_mergeIndicator``
    会按 name 合并掉其中一条。

    :param defs: 指标定义，字段同 ``BacktestResult.indicator_definitions`` 的列
        （``indicator_key`` 必填，其余可缺）。
    :param points_by_key: ``indicator_key`` → 点位序列，每个点位含
        ``timestamp_ms`` 与 ``value``。
    :return: d3Kline ``IndicatorOption`` 的 dict 列表，按 pane 升序。
    """
    options: list[dict[str, Any]] = []
    grouped = _group_by_pane(defs)
    for pane in sorted(grouped):
        is_main = pane == 0
        data_list: list[dict[str, Any]] = []
        for definition in grouped[pane]:
            key = str(definition["indicator_key"])
            render_type = str(definition.get("render_type") or "line").strip().lower()
            series_type = D3KLINE_SERIES_TYPE.get(render_type, "line")
            series: dict[str, Any] = {
                # series 身份用 indicator_key：前端 _mergeIndicator 以 name 做
                # merge，必须在 pane 内唯一
                "name": key,
                "label": definition.get("display_name") or key,
                "type": series_type,
                "data": [
                    {"timestamp": point["timestamp_ms"], "value": point["value"]}
                    for point in points_by_key.get(key, [])
                    if point.get("value") is not None
                ],
            }
            # 无颜色时**不输出 style**：前端合并是
            # {color:'#f0b90b', ...old, ...s.style}，传 None 会覆盖掉默认色
            color = definition.get("color")
            if color:
                series["style"] = {"color": color}
            if render_type != series_type:
                series["render_type"] = render_type
            data_list.append(series)

        option: dict[str, Any] = {
            "name": "STRATEGY" if is_main else f"STRATEGY_SUB{pane}",
            "isMain": is_main,
            "dataList": data_list,
            "pane": pane,
        }
        # 主图刻意不带 height，避免挤压价格图（前端默认 120）
        if not is_main:
            option["style"] = {"height": SUB_PANE_HEIGHT}
        options.append(option)
    return options


def to_raw_panes(
    defs: Iterable[Definition],
    points_by_key: Mapping[str, Iterable[Point]],
) -> list[dict[str, Any]]:
    """Group by pane while keeping akquant-native semantics (no frontend downgrade).

    供非 d3Kline 的消费方使用。点位键名保留 ``timestamp_ms``（与 ``to_d3kline_options``
    的 ``timestamp`` 不同），字段与 ``indicator_definitions`` 一一对应。
    """
    panes: list[dict[str, Any]] = []
    grouped = _group_by_pane(defs)
    for pane in sorted(grouped):
        indicators: list[dict[str, Any]] = []
        for definition in grouped[pane]:
            key = str(definition["indicator_key"])
            indicators.append(
                {
                    "indicator_key": key,
                    "display_name": definition.get("display_name") or key,
                    "render_type": definition.get("render_type") or "line",
                    "color": definition.get("color"),
                    "unit": definition.get("unit"),
                    "precision": definition.get("precision"),
                    "scale_group": definition.get("scale_group"),
                    "points": [
                        {"timestamp_ms": point["timestamp_ms"], "value": point["value"]}
                        for point in points_by_key.get(key, [])
                    ],
                }
            )
        panes.append({"pane": pane, "indicators": indicators})
    return panes
