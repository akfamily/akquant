"""Tests for live indicator streaming (StreamingIndicatorSink + wiring)."""

import json
from typing import List, cast

import pandas as pd
import pytest
from akquant import IndicatorSink
from akquant.backtest import BacktestStreamEvent
from akquant.live._runner import LiveRunner
from akquant.live._stream_sink import StreamingIndicatorSink
from akquant.strategy import Strategy


def _ts(day: int) -> int:
    return int(pd.Timestamp(f"2024-01-0{day} 10:00:00").value)


def test_streaming_sink_satisfies_indicator_sink_protocol() -> None:
    """The live streaming sink is a valid public IndicatorSink."""
    sink = StreamingIndicatorSink(lambda event: None)
    assert isinstance(sink, IndicatorSink)


def test_streaming_sink_emits_point_and_snapshot() -> None:
    """Record emits indicator_point; flush emits indicator_snapshot."""
    events: List[BacktestStreamEvent] = []
    sink = StreamingIndicatorSink(events.append, run_id="live-1")

    sink.record(
        name="ma",
        value=10.0,
        symbol="IND",
        timestamp=_ts(1),
        owner_strategy_id="_default",
        display_name="MA",
        pane=0,
    )
    sink.record(
        name="rng",
        value=0.4,
        symbol="IND",
        timestamp=_ts(1),
        owner_strategy_id="_default",
        pane=1,
        render_type="bar",
    )
    sink.flush_stream_snapshot()

    points = [e for e in events if e["event_type"] == "indicator_point"]
    snaps = [e for e in events if e["event_type"] == "indicator_snapshot"]
    assert len(points) == 2
    assert len(snaps) == 1
    # point payload shape mirrors the backtest contract
    p0 = points[0]["payload"]
    assert p0["indicator_key"] == "ma"
    assert p0["pane"] == "0"
    assert p0["render_type"] == "line"
    assert "timestamp_ms" in p0
    # snapshot bundles both indicators for the same timestamp
    items = json.loads(snaps[0]["payload"]["items_json"])
    assert {i["indicator_key"] for i in items} == {"ma", "rng"}
    assert snaps[0]["payload"]["indicator_count"] == "2"
    # sequence numbers are monotonic
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_streaming_sink_does_not_accumulate() -> None:
    """build_payload stays empty regardless of how many points were recorded."""
    sink = StreamingIndicatorSink(lambda event: None)
    for day in range(1, 4):
        sink.record(
            name="ma",
            value=float(day),
            symbol="IND",
            timestamp=_ts(day),
            owner_strategy_id="_default",
        )
        sink.flush_stream_snapshot()
    payload = sink.build_payload()
    assert payload == {"definitions": [], "instances": [], "points": []}


def test_streaming_sink_rejects_unknown_render_type() -> None:
    """Render type validation is enforced, matching the core contract."""
    sink = StreamingIndicatorSink(lambda event: None)
    with pytest.raises(ValueError, match="render_type"):
        sink.record(
            name="x",
            value=1.0,
            symbol="IND",
            timestamp=_ts(1),
            owner_strategy_id="_default",
            render_type="candlestick",
        )


class _RecordingStrategy:
    """Minimal strategy stand-in exposing on_bar for wiring tests."""

    def __init__(self) -> None:
        self.bars_seen = 0

    def on_bar(self, bar: object) -> None:
        recorder = getattr(self, "_indicator_recorder", None)
        if recorder is not None:
            recorder.record(
                name="echo",
                value=1.0,
                symbol="IND",
                timestamp=_ts(1),
                owner_strategy_id="_default",
            )
        self.bars_seen += 1


def test_attach_indicator_stream_wires_sink_and_flush() -> None:
    """_attach_indicator_stream sets the recorder and flushes after on_bar."""
    events: List[BacktestStreamEvent] = []
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_id = "_default"
    runner._indicator_recorder_override = None
    runner._stream_on_event = events.append

    strategy = _RecordingStrategy()
    runner._attach_indicator_stream([cast(Strategy, strategy)])

    # sink attached
    assert isinstance(getattr(strategy, "_indicator_recorder"), IndicatorSink)
    # calling wrapped on_bar records a point and then flushes a snapshot
    strategy.on_bar(object())
    assert strategy.bars_seen == 1
    assert any(e["event_type"] == "indicator_point" for e in events)
    assert any(e["event_type"] == "indicator_snapshot" for e in events)


def test_attach_indicator_stream_noop_without_config() -> None:
    """With no override and no on_event, no recorder is attached (legacy)."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.strategy_id = "_default"
    runner._indicator_recorder_override = None
    runner._stream_on_event = None

    strategy = _RecordingStrategy()
    runner._attach_indicator_stream([cast(Strategy, strategy)])
    assert getattr(strategy, "_indicator_recorder", None) is None


def test_live_stream_point_is_isomorphic_with_backtest() -> None:
    """Live indicator_point payload has the same keys as the backtest path."""
    from akquant import Bar, Strategy, run_backtest

    class _S(Strategy):
        def on_bar(self, bar: Bar) -> None:
            self.record_indicator(
                name="close_echo",
                value=bar.close,
                display_name="Close Echo",
                pane=0,
                render_type="line",
                meta={"src": "close"},
            )

    bars = [
        Bar(
            timestamp=_ts(i + 1),
            open=c - 0.1,
            high=c + 0.2,
            low=c - 0.2,
            close=c,
            volume=1000.0,
            symbol="IND",
        )
        for i, c in enumerate([10.0, 10.5, 11.0])
    ]
    bt_events: List[BacktestStreamEvent] = []
    run_backtest(
        data=bars,
        strategy=_S,
        symbols="IND",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        on_event=bt_events.append,
        stream_batch_size=1,
        stream_max_buffer=128,
    )
    bt_point = next(e for e in bt_events if e["event_type"] == "indicator_point")

    live_events: List[BacktestStreamEvent] = []
    sink = StreamingIndicatorSink(live_events.append, run_id="_default")
    sink.record(
        name="close_echo",
        value=10.0,
        symbol="IND",
        timestamp=_ts(1),
        owner_strategy_id="_default",
        display_name="Close Echo",
        pane=0,
        render_type="line",
        meta={"src": "close"},
    )
    live_point = live_events[0]

    # Same event envelope keys and same payload keys → one frontend consumer.
    assert set(bt_point.keys()) == set(live_point.keys())
    assert set(bt_point["payload"].keys()) == set(live_point["payload"].keys())
