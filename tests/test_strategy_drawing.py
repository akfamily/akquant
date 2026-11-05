"""DrawingApi：身份、去重、淘汰、kind 冲突、clear、sink 兼容（spec 3.4、4、6.1）."""

import json
import logging
from typing import Any

import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest


class FakeSink:
    """测试用 sink，记录 record_drawing 调用."""

    def __init__(self) -> None:
        """Init."""
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def record(self, **kwargs: Any) -> None:  # 指标点，忽略
        """Record."""

    def build_payload(self) -> dict[str, list[dict[str, Any]]]:
        """Record."""
        return {"definitions": [], "instances": [], "points": []}

    def set_stream_emitter(self, emitter: Any) -> None:
        """Set stream emitter."""

    def record_drawing(self, *, op: str, drawing: dict[str, Any]) -> None:
        """Record drawing."""
        self.calls.append((op, drawing))


def _strategy(sink: Any, owner: str = "_default") -> Strategy:
    """Strategy."""
    s = Strategy()
    s._indicator_recorder = sink
    s._owner_strategy_id = owner
    s._resolve_symbol = lambda symbol=None: symbol or "AAA"  # type: ignore[method-assign]
    s._resolve_record_timestamp = lambda: 1_704_160_800_000_000_000  # type: ignore[method-assign]
    return s


NS = 1_704_160_800_000_000_000


def test_upsert_dedupe_and_overwrite() -> None:
    """同 id 内容不变不发，变化则整体覆盖且保留创建序号."""
    sink = FakeSink()
    s = _strategy(sink)
    s.draw.line("t", (NS, 1), (NS + 1, 2))
    s.draw.line("t", (NS, 1), (NS + 1, 2))  # 内容相同：不发
    s.draw.line("t", (NS, 1), (NS + 1, 3))  # 整体覆盖
    ops = [op for op, _ in sink.calls]
    assert ops == ["upsert", "upsert"]
    d = sink.calls[-1][1]
    assert d["drawing_id"] == "t" and d["symbol"] == "AAA" and d["kind"] == "line"
    assert d["points"][1]["price"] == 3.0
    assert "truncated" not in d
    assert d["created_seq"] == sink.calls[0][1]["created_seq"]  # 覆盖不改创建序号


def test_kind_conflict_raises() -> None:
    """同 id 换 kind 报错并提示先 delete."""
    s = _strategy(FakeSink())
    s.draw.line("x", (NS, 1), (NS, 2))
    with pytest.raises(ValueError, match="delete"):
        s.draw.box("x", (NS, 2), (NS, 1))


def test_delete_and_clear_scope() -> None:
    """Delete 不存在静默；clear 按 kind / symbol 过滤."""
    sink = FakeSink()
    s = _strategy(sink)
    s.draw.label("a", (NS, 1), "A")
    s.draw.label("b", (NS, 1), "B", symbol="BBB")
    s.draw.line("c", (NS, 1), (NS, 2))
    s.draw.delete("missing")  # 不存在：静默
    s.draw.clear(kind="label")
    deleted = sorted(d["drawing_id"] for op, d in sink.calls if op == "delete")
    assert deleted == ["a", "b"]
    s.draw.clear(symbol="AAA")
    assert [d["drawing_id"] for op, d in sink.calls if op == "delete"][-1] == "c"


def test_eviction_oldest_first_and_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """超限淘汰最早创建的，只告警一次."""
    sink = FakeSink()
    s = _strategy(sink)
    s.max_drawings_per_kind = 3
    with caplog.at_level(logging.WARNING):
        for i in range(5):
            s.draw.label(f"l{i}", (NS, 1), str(i))
    deleted = [d["drawing_id"] for op, d in sink.calls if op == "delete"]
    assert deleted == ["l0", "l1"]
    assert sum("max_drawings_per_kind" in r.getMessage() for r in caplog.records) == 1


def test_eviction_is_per_kind_and_symbol() -> None:
    """淘汰按 (symbol, kind) 分别计数."""
    sink = FakeSink()
    s = _strategy(sink)
    s.max_drawings_per_kind = 1
    s.draw.label("a", (NS, 1), "A")
    s.draw.line("b", (NS, 1), (NS, 2))
    s.draw.label("c", (NS, 1), "C", symbol="BBB")
    assert not [1 for op, _ in sink.calls if op == "delete"]


def test_global_table_only_for_table() -> None:
    """symbol="*" 仅 table 可用."""
    sink = FakeSink()
    s = _strategy(sink)
    s.draw.table("perf", [["a", 1]], symbol="*")
    assert sink.calls[-1][1]["symbol"] == "*"
    with pytest.raises(ValueError, match="table"):
        s.draw.label("x", (NS, 1), "x", symbol="*")


def test_owner_isolation() -> None:
    """不同 owner 的同 id 互不影响."""
    sink = FakeSink()
    a, b = _strategy(sink, "_default"), _strategy(sink, "ind1")
    a.draw.label("same", (NS, 1), "A")
    b.draw.label("same", (NS, 1), "B")
    assert [d["owner_strategy_id"] for _, d in sink.calls] == ["_default", "ind1"]


def test_sink_without_record_drawing_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sink 无 record_drawing 时只记一次 info."""

    class OldSink(FakeSink):
        """旧 sink：record_drawing 为 None."""

        record_drawing = None  # type: ignore[assignment]

    s = _strategy(OldSink())
    with caplog.at_level(logging.INFO):
        s.draw.label("a", (NS, 1), "A")
        s.draw.label("b", (NS, 1), "B")
    assert sum("record_drawing" in r.getMessage() for r in caplog.records) == 1


def test_no_recorder_is_noop() -> None:
    """没有 recorder 时不报错."""
    s = Strategy()
    s._indicator_recorder = None
    s._resolve_symbol = lambda symbol=None: "AAA"  # type: ignore[method-assign]
    s.draw.label("a", (NS, 1), "A")  # 不报错


def test_empty_id_rejected() -> None:
    """空 id 被拒绝."""
    with pytest.raises(ValueError, match="id"):
        _strategy(FakeSink()).draw.label("  ", (NS, 1), "A")


def test_truncation_warns(caplog: pytest.LogCaptureFixture) -> None:
    """文字截断时告警."""
    s = _strategy(FakeSink())
    with caplog.at_level(logging.WARNING):
        s.draw.label("a", (NS, 1), "长" * 600)
    assert any("截断" in r.getMessage() for r in caplog.records)


# ---- Task 6：回测记录、result.drawings、导出、流事件 ----


def _daily_bars(n: int = 6, interval_start_offset_ns: Any = None) -> list[Bar]:
    """Daily bars; optionally set interval_start = timestamp - offset."""
    base = pd.Timestamp("2024-01-02 15:00:00", tz="Asia/Shanghai").value
    bars = []
    for i in range(n):
        ts = base + i * 86_400_000_000_000
        kwargs: dict[str, Any] = {}
        if interval_start_offset_ns is not None:
            kwargs["interval_start"] = ts - interval_start_offset_ns
        bars.append(
            Bar(
                timestamp=ts,
                open=10 + i,
                high=11 + i,
                low=9 + i,
                close=10 + i,
                volume=1e5,
                symbol="AAA",
                **kwargs,
            )
        )
    return bars


class _Drawer(Strategy):
    def on_bar(self, bar: Bar) -> None:
        self.draw.label(f"l{bar.timestamp}", (bar.timestamp, bar.high), "x")
        self.draw.line("trend", (self._first, 10.0), (bar.timestamp, bar.close))
        self.draw.table("perf", self.stats.as_rows(), symbol="*")
        if bar.close >= 13:
            self.draw.delete(f"l{bar.timestamp}")

    def on_start(self) -> None:
        self._first = _daily_bars()[0].timestamp


def test_result_drawings_hold_final_state(tmp_path: Any) -> None:
    """result.drawings 只保留最终存活状态，导出与之一致."""
    result = run_backtest(data=_daily_bars(), strategy=_Drawer(), show_progress=False)
    kinds = sorted({d["kind"] for d in result.drawings})
    assert kinds == ["label", "line", "table"]
    labels = [d for d in result.drawings if d["kind"] == "label"]
    assert len(labels) == 3  # close 10,11,12 留下；13,14,15 被删
    line = next(d for d in result.drawings if d["kind"] == "line")
    assert line["points"][1]["price"] == 15.0  # 只保留最后一次覆盖
    assert set(result.drawings_df().columns) >= {
        "owner_strategy_id",
        "drawing_id",
        "kind",
    }
    out = tmp_path / "ind.json"
    result.export_indicators(str(out))
    exported = json.loads(out.read_text(encoding="utf-8"))
    assert exported["drawings"] == result.drawings
    pq = tmp_path / "bundle"
    result.export_indicators(str(pq), format="parquet")
    assert json.loads((pq / "drawings.json").read_text(encoding="utf-8")) == (
        result.drawings
    )


def test_drawing_stream_events_emitted() -> None:
    """indicator_drawing 事件不被节流，ts 非零."""
    events: list[Any] = []
    run_backtest(
        data=_daily_bars(),
        strategy=_Drawer(),
        show_progress=False,
        on_event=events.append,
    )
    drawing_events = [e for e in events if e["event_type"] == "indicator_drawing"]
    ops = [e["payload"]["op"] for e in drawing_events]
    assert ops.count("delete") == 3
    assert all(e["ts"] > 0 for e in drawing_events)
    assert all(e["payload"]["timestamp"] == str(e["ts"]) for e in drawing_events)


class _SamePoint(Strategy):
    def on_bar(self, bar: Bar) -> None:
        self.record_indicator("px", bar.close)
        self.draw.label("m", (bar.timestamp, bar.high), "x")


@pytest.mark.parametrize("variant", ["plain", "interval_start", "legacy_end_minus_1ns"])
def test_drawing_point_ts_matches_indicator_ts(variant: str) -> None:
    """当前 bar 时间戳处的绘图点与同 bar 指标点 ts(ms) 同口径.

    legacy_end_minus_1ns：timestamp 为整秒 - 1ns 且 interval_start 更早，
    bar_end_ns 会取 timestamp + 1，用来确认两边都走了这条分支。
    """
    bars = _daily_bars(3, None if variant == "plain" else 86_400_000_000_000)
    if variant == "legacy_end_minus_1ns":
        bars = [
            Bar(
                timestamp=b.timestamp - 1,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
                symbol=b.symbol,
                interval_start=b.interval_start,
            )
            for b in bars
        ]
    result = run_backtest(data=bars, strategy=_SamePoint(), show_progress=False)
    ind_ts = int(result.indicator_df()["timestamp_ms"].iloc[-1])
    label = next(d for d in result.drawings if d["drawing_id"] == "m")
    assert label["points"][0]["ts"] == ind_ts


def test_delete_payload_keys_and_clear_order() -> None:
    """Delete 载荷键集合固定；clear 按 created_seq 顺序删除."""
    sink = FakeSink()
    s = _strategy(sink)
    s.draw.label("b", (NS, 1), "x")
    s.draw.label("a", (NS, 2), "x")
    s.draw.label("c", (NS, 3), "x")
    s.draw.delete("a")
    deletes = [d for op, d in sink.calls if op == "delete"]
    assert set(deletes[0]) == {
        "owner_strategy_id",
        "drawing_id",
        "kind",
        "symbol",
        "updated_ts",
    }
    sink.calls.clear()
    s.draw.clear()
    assert [d["drawing_id"] for op, d in sink.calls if op == "delete"] == ["b", "c"]


def test_new_object_does_not_scan_whole_store() -> None:
    """新建对象只碰目标桶，不遍历全部 symbol 的存储（O(1) 淘汰）."""
    sink = FakeSink()
    s = _strategy(sink)
    s.max_drawings_per_kind = 400
    for n in range(50):
        for i in range(400):
            s.draw.label(f"l{i}", (NS, 1), "x", symbol=f"S{n}")

    class SpyDict(dict):  # type: ignore[type-arg]
        scans = 0

        def items(self):  # type: ignore[no-untyped-def]
            SpyDict.scans += 1
            return super().items()

        def __iter__(self):  # type: ignore[no-untyped-def]
            SpyDict.scans += 1
            return super().__iter__()

    s.draw._store = SpyDict(s.draw._store)  # type: ignore[assignment]
    sink.calls.clear()
    s.draw.label("new", (NS, 1), "y", symbol="S7")
    assert SpyDict.scans == 0
    deleted = [d["drawing_id"] for op, d in sink.calls if op == "delete"]
    assert deleted == ["l0"]
    assert len(s.draw._index[("S7", "label")]) == 400


def test_eviction_order_after_delete_and_recreate() -> None:
    """同桶内 delete 后重建，淘汰顺序仍按创建先后；覆盖不改变位置."""
    sink = FakeSink()
    s = _strategy(sink)
    s.max_drawings_per_kind = 3
    s.draw.label("a", (NS, 1), "A")
    s.draw.label("b", (NS, 1), "B")
    s.draw.label("c", (NS, 1), "C")
    s.draw.delete("a")
    s.draw.label("a", (NS, 1), "A2")  # a 现在是最新创建
    s.draw.label("b", (NS, 2), "B2")  # 覆盖，b 仍是最早
    sink.calls.clear()
    s.draw.label("d", (NS, 1), "D")
    s.draw.label("e", (NS, 1), "E")
    deleted = [d["drawing_id"] for op, d in sink.calls if op == "delete"]
    assert deleted == ["b", "c"]
