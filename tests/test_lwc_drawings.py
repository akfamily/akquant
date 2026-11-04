"""viz.review 的绘图对象 payload 与 HTML."""

from pathlib import Path
from typing import Any

import pandas as pd
from akquant.lwc._drawings import to_lwc_drawings

IDX = pd.date_range("2024-01-02", periods=5, freq="D", tz="Asia/Shanghai")
MS = [int(t.value // 1_000_000) for t in IDX]


def _d(
    kind: str, points: list[tuple[int, float]], symbol: str = "AAA", **kw: Any
) -> dict[str, Any]:
    return {
        "owner_strategy_id": "_default",
        "drawing_id": kw.pop("id", kind),
        "kind": kind,
        "symbol": symbol,
        "pane": 0,
        "points": [{"ts": t, "price": p} for t, p in points],
        "text": "",
        "style": {},
        "table": kw.pop("table", None),
        "created_seq": kw.pop("seq", 1),
        "updated_ts": 0,
        "phase": "run",
    }


def test_points_snap_to_bar_and_drop_before_first() -> None:
    """点吸附到不晚于它的 bar; 早于首根 bar 的点丢弃, 点数不足的对象整个丢弃."""
    mid = MS[1] + 3_600_000  # 第 2 根和第 3 根 bar 之间
    shapes, _ = to_lwc_drawings(
        [
            _d("line", [(MS[0], 1.0), (mid, 2.0)]),
            _d("label", [(MS[0] - 86_400_000, 1.0)], id="early"),
        ],
        "AAA",
        IDX,
        intraday=False,
    )
    assert [s["drawing_id"] for s in shapes] == ["line"]
    assert shapes[0]["points"][1]["time"] == "2024-01-03"


def test_other_symbol_filtered_and_global_table_included() -> None:
    """只取本标的的图形; 表格含全局表且同位置时全局表在前."""
    table = {"rows": [], "header_rows": 0, "position": "top_right"}
    items = [
        _d("line", [(MS[0], 1.0), (MS[1], 1.0)], symbol="BBB"),
        _d("table", [], symbol="*", table=table, id="g", seq=2),
        _d("table", [], symbol="AAA", table=table, id="s", seq=1),
    ]
    shapes, tables = to_lwc_drawings(items, "AAA", IDX, intraday=False)
    assert shapes == []
    assert [t["drawing_id"] for t in tables] == ["g", "s"]  # 同位置时全局表在前


def test_review_html_contains_drawing_layer(tmp_path: Path) -> None:
    """复盘页带绘图层与表格浮层; include_drawings=False 时不带."""
    from akquant import Bar, Strategy, run_backtest

    class S(Strategy):
        def on_bar(self, bar: Bar) -> None:
            self.draw.label("l", (bar.timestamp, bar.high), "突破")
            self.draw.table("perf", self.stats.as_rows(), symbol="*")

    bars = [
        Bar(
            timestamp=int(t.value),
            open=10,
            high=11,
            low=9,
            close=10,
            volume=1e5,
            symbol="AAA",
        )
        for t in IDX
    ]
    result = run_backtest(data=bars, strategy=S(), show_progress=False)
    df = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1e5},
        index=IDX,
    )
    path = result.viz.review({"AAA": df}, filename=str(tmp_path / "r.html"))
    html = open(path, encoding="utf-8").read()
    assert "akq-tables" in html and "drawings-toggle" in html
    assert "突破" in html and "累计收益" in html
    off = result.viz.review(
        {"AAA": df}, filename=str(tmp_path / "r2.html"), include_drawings=False
    )
    assert "突破" not in open(off, encoding="utf-8").read()


def test_naive_index_interpreted_in_strategy_timezone() -> None:
    """无时区行情是策略时区的墙钟时间: 首根 bar 上的点不能因 8 小时时差被丢掉."""
    naive = IDX.tz_localize(None)
    line = _d("line", [(MS[0], 1.0), (MS[2], 2.0)])
    shapes, _ = to_lwc_drawings([line], "AAA", naive, False, tz="Asia/Shanghai")
    assert [p["time"] for p in shapes[0]["points"]] == ["2024-01-02", "2024-01-04"]
    # 不给 tz 时按 UTC 解释: 首点落在首根 bar 之前被丢弃, 整条线点数不足
    assert to_lwc_drawings([line], "AAA", naive, False) == ([], [])


def test_intraday_seconds_unknown_kind_and_late_point() -> None:
    """日内用 UTC 秒; 晚于末根 bar 的点吸附到末根; 未知 kind 忽略."""
    idx = pd.date_range("2024-01-02 09:30", periods=4, freq="min", tz="Asia/Shanghai")
    ms = [int(t.value // 1_000_000) for t in idx]
    items = [
        _d("label", [(ms[1] + 30_000, 5.0)]),  # 两根 bar 之间 -> 吸附到前一根
        _d("label", [(ms[3] + 3_600_000, 5.0)], id="late"),  # 晚于末根 -> 末根
        _d("arrow", [(ms[0], 1.0)], id="unknown"),  # 未知 kind 忽略
    ]
    shapes, tables = to_lwc_drawings(items, "AAA", idx, intraday=True)
    assert [s["drawing_id"] for s in shapes] == ["label", "late"]
    assert shapes[0]["points"][0]["time"] == ms[1] // 1000
    assert shapes[1]["points"][0]["time"] == ms[3] // 1000
    assert tables == []


def test_payload_drawings_keys_and_disable() -> None:
    """Payload 每段带 drawings/tables; 关掉开关时为空列表."""
    from types import SimpleNamespace

    from akquant.lwc._payload import build_review_payload

    table = {
        "rows": [[{"value": 1, "text": "1", "bg_color": "", "text_color": ""}]],
        "header_rows": 0,
        "position": "bottom_left",
    }
    result = SimpleNamespace(
        trades_df=pd.DataFrame(),
        indicator_outputs={
            "drawings": [
                _d("label", [(MS[1], 1.0)]),
                _d("table", [], symbol="*", table=table, id="t"),
            ]
        },
        _timezone="Asia/Shanghai",
    )
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=IDX)
    seg = build_review_payload(result, {"AAA": df})["symbols"][0]
    assert [s["drawing_id"] for s in seg["drawings"]] == ["label"]
    assert [t["drawing_id"] for t in seg["tables"]] == ["t"]
    off = build_review_payload(result, {"AAA": df}, include_drawings=False)
    assert off["symbols"][0]["drawings"] == [] and off["symbols"][0]["tables"] == []
