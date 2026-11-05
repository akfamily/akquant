"""drawing 流消息（schema 1.7）与实盘 StreamingIndicatorSink."""

from typing import Any, cast

import akquant
import pandas as pd
from akquant import (
    STREAM_SCHEMA_VERSION,
    Bar,
    Strategy,
    is_drawing_stream_event,
    is_indicator_stream_event,
    run_backtest,
    to_drawing_message,
    to_indicator_message,
)
from akquant.live._runner import LiveRunner
from akquant.live._stream_sink import StreamingIndicatorSink

DRAWING = {
    "owner_strategy_id": "_default",
    "drawing_id": "t",
    "kind": "line",
    "symbol": "AAA",
    "pane": 0,
    "points": [{"ts": 1, "price": 1.0}, {"ts": 2, "price": 2.0}],
    "text": "",
    "style": {"color": "", "width": 1, "line_style": "solid", "extend": "none"},
    "table": None,
    "created_seq": 1,
    "updated_ts": 2,
    "phase": "run",
}


def test_schema_version_is_1_7() -> None:
    """Schema version is 1.7."""
    assert STREAM_SCHEMA_VERSION == "1.7"


def test_streaming_sink_emits_drawing_and_message_roundtrip() -> None:
    """Live sink emits drawing events that round-trip via to_drawing_message."""
    events: list[Any] = []
    sink = StreamingIndicatorSink(events.append, run_id="live-1")
    sink.record_drawing(op="upsert", drawing=DRAWING)
    sink.record_drawing(
        op="delete",
        drawing={
            k: DRAWING[k] for k in ("owner_strategy_id", "drawing_id", "kind", "symbol")
        }
        | {"updated_ts": 3},
    )
    assert [e["event_type"] for e in events] == ["indicator_drawing"] * 2
    msg = to_drawing_message(events[0])
    assert msg is not None
    assert msg["type"] == "drawing" and msg["op"] == "upsert"
    assert msg["schema_version"] == "1.7" and msg["drawing"] == DRAWING
    assert msg["symbol"] == "AAA" and msg["ts"] == 2 * 1_000_000
    delete_msg = to_drawing_message(events[1])
    assert delete_msg is not None and delete_msg["op"] == "delete"
    assert set(delete_msg["drawing"]) == {
        "owner_strategy_id",
        "drawing_id",
        "kind",
        "symbol",
    }


def test_old_entry_points_do_not_see_drawings() -> None:
    """旧版 live_server 只用 is_indicator_stream_event/to_indicator_message."""
    events: list[Any] = []
    StreamingIndicatorSink(events.append).record_drawing(op="upsert", drawing=DRAWING)
    assert is_drawing_stream_event(events[0])
    assert not is_indicator_stream_event(events[0])
    assert to_indicator_message(events[0]) is None


def test_global_table_symbol_is_none_in_envelope() -> None:
    """Global table has symbol None in envelope but '*' in drawing."""
    events: list[Any] = []
    table = dict(
        DRAWING,
        kind="table",
        symbol="*",
        points=[],
        table={"rows": [], "header_rows": 0, "position": "top_right"},
    )
    StreamingIndicatorSink(events.append).record_drawing(op="upsert", drawing=table)
    msg = to_drawing_message(events[0])
    assert msg is not None and msg["symbol"] is None and msg["drawing"]["symbol"] == "*"


def test_exports() -> None:
    """Public exports."""
    assert "to_drawing_message" in akquant.__all__
    assert "is_drawing_stream_event" in akquant.__all__


def test_to_drawing_message_rejects_non_drawing_events() -> None:
    """Non-drawing events yield None."""
    assert to_drawing_message(cast(Any, {"event_type": "indicator_point"})) is None


class _LabelStrategy(Strategy):
    def on_bar(self, bar: Bar) -> None:
        self.draw.label("m", (bar.timestamp, bar.high), "x", symbol=bar.symbol)


def test_attach_indicator_stream_live_path_emits_drawing() -> None:
    """run_live 路径: on_event 经 StreamingIndicatorSink 收到 indicator_drawing."""
    events: list[Any] = []
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_id = "_default"
    runner._indicator_recorder_override = None
    runner._stream_on_event = events.append

    strategy = _LabelStrategy()
    strategy._owner_strategy_id = "_default"
    runner._attach_indicator_stream([strategy])
    ts = int(pd.Timestamp("2024-01-02 10:00:00").value)
    strategy.on_bar(Bar(ts, 1.0, 2.0, 0.5, 1.5, 10.0, "AAA"))

    drawings = [e for e in events if e["event_type"] == "indicator_drawing"]
    assert len(drawings) == 1
    msg = to_drawing_message(drawings[0])
    assert msg is not None
    assert msg["op"] == "upsert" and msg["symbol"] == "AAA"
    assert msg["drawing"]["drawing_id"] == "m"


def test_backtest_drawing_event_roundtrips() -> None:
    """run_backtest 产生的 indicator_drawing 事件可经 to_drawing_message 转换."""
    bars = [
        Bar(
            int(pd.Timestamp(f"2024-01-0{d} 10:00:00").value),
            1.0,
            2.0,
            0.5,
            1.5,
            10.0,
            "AAA",
        )
        for d in (2, 3)
    ]
    events: list[Any] = []
    run_backtest(
        data=bars,
        strategy=_LabelStrategy(),
        symbols="AAA",
        show_progress=False,
        on_event=events.append,
    )
    drawings = [e for e in events if e["event_type"] == "indicator_drawing"]
    assert drawings
    msg = to_drawing_message(drawings[0])
    assert msg is not None
    assert msg["type"] == "drawing" and msg["schema_version"] == STREAM_SCHEMA_VERSION
    assert msg["drawing"]["kind"] == "label" and msg["symbol"] == "AAA"
    assert not is_indicator_stream_event(drawings[0])
