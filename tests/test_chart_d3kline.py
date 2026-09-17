"""Contract tests for akquant.chart.d3kline (indicator -> d3Kline IndicatorOption).

这是回测服务与实盘服务**共用**的单一实现。前端一套渲染逻辑同时吃两边数据，
全靠此处输出结构一致；改动这里等于改动两个服务的对外契约。
"""

from typing import Any

from akquant.chart import SUB_PANE_HEIGHT, to_d3kline_options, to_raw_panes


def _defs() -> list[dict[str, Any]]:
    return [
        {
            "indicator_key": "ma",
            "display_name": "均线",
            "pane": 0,
            "render_type": "line",
            "color": "#2563eb",
        },
        {
            "indicator_key": "ema",
            "display_name": "EMA",
            "pane": 0,
            "render_type": "area",
        },
        {
            "indicator_key": "amp",
            "display_name": "振幅",
            "pane": 1,
            "render_type": "histogram",
            "color": "#8b5cf6",
        },
    ]


def _points() -> dict[str, list[dict[str, Any]]]:
    return {
        "ma": [{"timestamp_ms": 1704067200000, "value": 10.0}],
        "ema": [{"timestamp_ms": 1704067200000, "value": 10.1}],
        "amp": [{"timestamp_ms": 1704067200000, "value": 2.0}],
    }


def test_groups_by_pane_into_one_option_each() -> None:
    """Indicators sharing a pane pack into one option (front MAX_SUB_INDICATORS=3)."""
    options = to_d3kline_options(_defs(), _points())
    assert len(options) == 2
    assert [o["pane"] for o in options] == [0, 1]
    assert len(options[0]["dataList"]) == 2


def test_pane_zero_is_main_others_are_sub() -> None:
    """Pane 0 is main; pane>=1 is sub and carries a height."""
    options = to_d3kline_options(_defs(), _points())
    assert options[0]["isMain"] is True
    assert "style" not in options[0]  # 主图不带 height，避免挤压价格图
    assert options[1]["isMain"] is False
    assert options[1]["style"]["height"] == SUB_PANE_HEIGHT


def test_render_type_downgraded_to_line_or_bar() -> None:
    """Only line|bar exist in d3Kline; everything else downgrades."""
    options = to_d3kline_options(_defs(), _points())
    types = {s["name"]: s["type"] for o in options for s in o["dataList"]}
    assert types == {"ma": "line", "ema": "line", "amp": "bar"}


def test_downgraded_series_keeps_original_render_type() -> None:
    """Downgraded series keep the original render_type for lossless upgrade later."""
    options = to_d3kline_options(_defs(), _points())
    series = {s["name"]: s for o in options for s in o["dataList"]}
    assert series["ema"]["render_type"] == "area"
    assert series["amp"]["render_type"] == "histogram"
    assert "render_type" not in series["ma"]  # 未降级则不带


def test_missing_color_omits_style_entirely() -> None:
    """No color means no ``style`` key at all.

    前端 _mergeIndicator 的合并是 {color:'#f0b90b', ...old, ...s.style}，
    传 style:{color:None} 会覆盖掉默认色导致无色。
    """
    options = to_d3kline_options(_defs(), _points())
    series = {s["name"]: s for o in options for s in o["dataList"]}
    assert "style" not in series["ema"]
    assert series["ma"]["style"]["color"] == "#2563eb"


def test_data_items_use_timestamp_and_value() -> None:
    """Data items are {timestamp, value} with millisecond timestamps."""
    options = to_d3kline_options(_defs(), _points())
    item = options[0]["dataList"][0]["data"][0]
    assert set(item) == {"timestamp", "value"}
    assert item["timestamp"] == 1704067200000


def test_none_values_are_dropped() -> None:
    """Points whose value is None are dropped from data."""
    points: dict[str, list[dict[str, Any]]] = {
        "ma": [{"timestamp_ms": 1, "value": None}, {"timestamp_ms": 2, "value": 1.0}]
    }
    options = to_d3kline_options([_defs()[0]], points)
    assert len(options[0]["dataList"][0]["data"]) == 1


def test_option_names_are_unique() -> None:
    """Option names are unique: STRATEGY for main, STRATEGY_SUB{pane} for subs."""
    options = to_d3kline_options(_defs(), _points())
    assert {o["name"] for o in options} == {"STRATEGY", "STRATEGY_SUB1"}


def test_raw_format_keeps_full_render_type() -> None:
    """Raw format keeps akquant-native render_type without downgrade."""
    panes = to_raw_panes(_defs(), _points())
    render_types = {i["render_type"] for p in panes for i in p["indicators"]}
    assert render_types == {"line", "area", "histogram"}


def test_raw_points_keep_timestamp_ms_key() -> None:
    """Raw points keep the native ``timestamp_ms`` key (d3kline uses ``timestamp``)."""
    panes = to_raw_panes(_defs(), _points())
    point = panes[0]["indicators"][0]["points"][0]
    assert set(point) == {"timestamp_ms", "value"}


def test_empty_input_returns_empty_list() -> None:
    """Empty definitions yield an empty list instead of raising."""
    assert to_d3kline_options([], {}) == []
    assert to_raw_panes([], {}) == []


def test_unknown_render_type_falls_back_to_line() -> None:
    """Unknown render_type falls back to line and keeps the original value.

    第三方 IndicatorSink 可能写入枚举外的值。
    """
    defs = [{"indicator_key": "x", "render_type": "unknown_type", "pane": 1}]
    points = {"x": [{"timestamp_ms": 1_704_067_200_000, "value": 1.0}]}
    series = to_d3kline_options(defs, points)[0]["dataList"][0]
    assert series["type"] == "line"
    assert series["render_type"] == "unknown_type"


def test_panes_are_sorted_regardless_of_input_order() -> None:
    """Output is sorted by pane regardless of input order (frontend draws top-down)."""
    defs = [
        {"indicator_key": "c", "pane": 2},
        {"indicator_key": "a", "pane": 0},
        {"indicator_key": "b", "pane": 1},
    ]
    assert [option["pane"] for option in to_d3kline_options(defs, {})] == [0, 1, 2]


def test_definition_without_points_yields_empty_data() -> None:
    """A definition without points yields data=[] (metadata builds the skeleton)."""
    options = to_d3kline_options([{"indicator_key": "ma", "pane": 0}], {})
    assert options[0]["dataList"][0]["data"] == []


def test_missing_optional_fields_use_defaults() -> None:
    """Only indicator_key given: pane=0, render_type=line, label falls back to key.

    backtest_server 传的是 sqlite Row 转的裸 dict，字段可能为 None 或缺失。
    """
    options = to_d3kline_options([{"indicator_key": "solo"}], {})
    assert options[0]["pane"] == 0
    series = options[0]["dataList"][0]
    assert series["type"] == "line"
    assert series["label"] == "solo"
    assert "render_type" not in series


def test_pane_accepts_string_from_sqlite_rows() -> None:
    """Pane may arrive as a string (sqlite Row / JSON) and must group/sort as int."""
    defs = [{"indicator_key": "b", "pane": "1"}, {"indicator_key": "a", "pane": "0"}]
    options = to_d3kline_options(defs, {})
    assert [o["pane"] for o in options] == [0, 1]
    assert options[0]["isMain"] is True
