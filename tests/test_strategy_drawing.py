"""DrawingApi：身份、去重、淘汰、kind 冲突、clear、sink 兼容（spec 3.4、4、6.1）."""

import logging
from typing import Any

import pytest
from akquant import Strategy


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
