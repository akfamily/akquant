"""akquant.chart._drawing 归一化规则（spec 第 3、4、10 节）."""

import json
from datetime import datetime

import pandas as pd
import pytest
from akquant.chart import (
    FIB_LEVELS,
    MAX_TABLE_COLS,
    MAX_TABLE_ROWS,
    build_box,
    build_fib,
    build_label,
    build_line,
    build_polyline,
    build_table,
    normalize_cell,
    normalize_color,
    normalize_point,
)

TZ = "Asia/Shanghai"
NS = pd.Timestamp("2024-01-02 10:00:00", tz=TZ).value


def test_point_accepts_ns_timestamp_and_naive_datetime() -> None:
    """Points accept ns ints, Timestamps and naive datetimes."""
    ms = NS // 1_000_000
    assert normalize_point((NS, 10), tz=TZ) == {"ts": ms, "price": 10.0}
    assert normalize_point((pd.Timestamp(NS, tz="UTC"), 10), tz=TZ)["ts"] == ms
    # naive 时间按策略时区本地化
    assert normalize_point((datetime(2024, 1, 2, 10), 10), tz=TZ)["ts"] == ms


def test_point_applies_end_ns_hook() -> None:
    """The to_end_ns hook is applied before ms conversion."""
    p = normalize_point((NS, 1.0), tz=TZ, to_end_ns=lambda ns: ns + 1_000_000)
    assert p["ts"] == NS // 1_000_000 + 1


@pytest.mark.parametrize(
    "bad", [(NS,), (NS, float("nan")), (NS, float("inf")), "x", None]
)
def test_point_rejects_bad_input(bad: object) -> None:
    """Malformed points raise ValueError."""
    with pytest.raises(ValueError):
        normalize_point(bad, tz=TZ)


def test_color_rules() -> None:
    """Colors are #RRGGBB(AA), lowercase, empty means default."""
    assert normalize_color(None, field="c") == ""
    assert normalize_color("#E03131", field="c") == "#e03131"
    assert normalize_color("#e0313133", field="c") == "#e0313133"
    for bad in ("red", "#fff", "e03131", "#e0313"):
        with pytest.raises(ValueError, match="c"):
            normalize_color(bad, field="c")


def test_line_style_and_extend_validation() -> None:
    """Line style and extend values are validated."""
    d = build_line(
        (NS, 1),
        (NS + 60_000_000_000, 2),
        tz=TZ,
        to_end_ns=lambda n: n,
        extend="right",
        color="#e03131",
        width=2,
        style="dashed",
    )
    assert d["kind"] == "line" and len(d["points"]) == 2
    assert d["style"] == {
        "color": "#e03131",
        "width": 2,
        "line_style": "dashed",
        "extend": "right",
    }
    with pytest.raises(ValueError, match="extend"):
        build_line(
            (NS, 1),
            (NS, 2),
            tz=TZ,
            to_end_ns=lambda n: n,
            extend="left",
            color="",
            width=1,
            style="solid",
        )
    with pytest.raises(ValueError, match="style"):
        build_line(
            (NS, 1),
            (NS, 2),
            tz=TZ,
            to_end_ns=lambda n: n,
            extend="none",
            color="",
            width=1,
            style="wavy",
        )


def test_box_and_label_and_fib() -> None:
    """Box, label and fib build with expected styles."""
    b = build_box(
        (NS, 10),
        (NS + 1, 5),
        tz=TZ,
        to_end_ns=lambda n: n,
        border_color="",
        fill_color="#1971c233",
        width=1,
        text="区间",
    )
    assert b["text"] == "区间" and b["style"]["fill_color"] == "#1971c233"
    lab = build_label(
        (NS, 10),
        "突破",
        tz=TZ,
        to_end_ns=lambda n: n,
        style="down",
        color="",
        text_color="",
        size="normal",
    )
    assert lab["kind"] == "label" and lab["style"]["label_style"] == "down"
    with pytest.raises(ValueError, match="size"):
        build_label(
            (NS, 10),
            "x",
            tz=TZ,
            to_end_ns=lambda n: n,
            style="up",
            color="",
            text_color="",
            size="huge",
        )
    f = build_fib((NS, 10), (NS + 1, 20), tz=TZ, to_end_ns=lambda n: n, color="")
    assert f["kind"] == "fib" and "levels" not in f["style"]  # 档位由渲染端展开
    assert FIB_LEVELS == (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


def test_polyline_bounds() -> None:
    """Polyline point count must be within 2 to 1000."""
    with pytest.raises(ValueError, match="2"):
        build_polyline(
            [(NS, 1)],
            tz=TZ,
            to_end_ns=lambda n: n,
            closed=False,
            color="",
            width=1,
            fill_color="",
        )
    with pytest.raises(ValueError, match="1000"):
        build_polyline(
            [(NS + i, 1) for i in range(1001)],
            tz=TZ,
            to_end_ns=lambda n: n,
            closed=False,
            color="",
            width=1,
            fill_color="",
        )


def test_label_text_truncated_flag() -> None:
    """Long text is cut to 500 chars and flagged."""
    lab = build_label(
        (NS, 1),
        "长" * 600,
        tz=TZ,
        to_end_ns=lambda n: n,
        style="none",
        color="",
        text_color="",
        size="small",
    )
    assert len(lab["text"]) == 500 and lab["truncated"] is True


def test_cell_normalization() -> None:
    """Cells normalize value, text and colors."""
    assert normalize_cell(1.23456, 2) == {
        "value": 1.23456,
        "text": "1.23",
        "bg_color": "",
        "text_color": "",
    }
    assert normalize_cell(None, 2)["text"] == "—"
    assert normalize_cell(float("nan"), 2)["text"] == "—"
    assert normalize_cell(True, 2)["text"] == "是"
    assert normalize_cell(7, 2)["text"] == "7"
    c = normalize_cell({"value": 0.5, "text": "50%", "bg_color": "#e03131"}, 2)
    assert c == {"value": 0.5, "text": "50%", "bg_color": "#e03131", "text_color": ""}
    with pytest.raises(ValueError):
        normalize_cell({"value": object()}, 2)


def test_table_pads_ragged_rows_and_header() -> None:
    """Ragged rows are padded and header counted."""
    t = build_table(
        [["a", 1], ["b"]],
        header=["名称", "值"],
        position="top_right",
        precision=2,
        bg_color="",
        border_color="",
    )
    assert t["kind"] == "table" and t["points"] == []
    rows = t["table"]["rows"]
    assert t["table"]["header_rows"] == 1 and t["table"]["position"] == "top_right"
    assert [len(r) for r in rows] == [2, 2, 2]
    assert rows[2][1]["text"] == ""  # 补齐的空单元格


def test_table_padding_cells_are_independent() -> None:
    """Padded cells must not share one dict."""
    t = build_table(
        [["a"], ["b", "c", "d"]],
        header=None,
        position="top_left",
        precision=2,
        bg_color="",
        border_color="",
    )
    pad_a, pad_b = t["table"]["rows"][0][1], t["table"]["rows"][0][2]
    assert pad_a is not pad_b
    pad_a["text"] = "x"
    assert pad_b["text"] == ""


def test_table_limits_and_position() -> None:
    """Table size limits and position are validated."""
    with pytest.raises(ValueError, match="position"):
        build_table(
            [["a"]],
            header=None,
            position="top",
            precision=2,
            bg_color="",
            border_color="",
        )
    with pytest.raises(ValueError, match=str(MAX_TABLE_ROWS)):
        build_table(
            [["a"]] * (MAX_TABLE_ROWS + 1),
            header=None,
            position="top_left",
            precision=2,
            bg_color="",
            border_color="",
        )
    with pytest.raises(ValueError, match=str(MAX_TABLE_COLS)):
        build_table(
            [["a"] * (MAX_TABLE_COLS + 1)],
            header=None,
            position="top_left",
            precision=2,
            bg_color="",
            border_color="",
        )
    with pytest.raises(ValueError, match="二维"):
        build_table(
            "abc",
            header=None,
            position="top_left",
            precision=2,
            bg_color="",
            border_color="",
        )


def test_nan_safe_values_are_json_serializable() -> None:
    """NaN and inf cells serialize as strict JSON."""
    t = build_table(
        [[float("nan"), float("inf")]],
        header=None,
        position="top_left",
        precision=2,
        bg_color="",
        border_color="",
    )
    # NaN/inf 的 value 转成 None，保证 JSON 合法（前端 JSON.parse 不接受 NaN）
    json.dumps(t, allow_nan=False)
    nan_cell, inf_cell = t["table"]["rows"][0]
    assert nan_cell["value"] is None and nan_cell["text"] == "—"
    assert inf_cell["value"] is None and inf_cell["text"] == "∞"
