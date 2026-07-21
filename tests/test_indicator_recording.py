import json
from pathlib import Path

import akquant
import pandas as pd
import pytest
from akquant import (
    Bar,
    Strategy,
    is_indicator_stream_event,
    run_backtest,
    to_indicator_message,
    to_indicator_messages,
)


def _build_data() -> list[Bar]:
    closes = [10.0, 10.5, 11.0]
    bars: list[Bar] = []
    for i, close in enumerate(closes):
        bars.append(
            Bar(
                timestamp=pd.Timestamp(f"2024-01-0{i + 1} 10:00:00").value,
                open=close - 0.1,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=1000.0,
                symbol="IND",
            )
        )
    return bars


class IndicatorRecordingStrategy(Strategy):
    """Record one simple indicator point on every bar."""

    def on_bar(self, bar: Bar) -> None:
        """Emit two indicators so bridge and snapshot paths can be verified."""
        self.record_indicator(
            name="close_echo",
            value=bar.close,
            display_name="Close Echo",
            pane=0,
            render_type="line",
            precision=2,
            meta={"source": "close"},
        )
        self.record_indicator(
            name="range_echo",
            value=bar.high - bar.low,
            display_name="Range Echo",
            pane=1,
            render_type="signal",
            meta={"source": ["high", "low"]},
            warmup=bar.close < 10.5,
        )


class LegacyNoIndicatorStrategy(Strategy):
    """Legacy strategy that never records custom indicator outputs."""

    def on_bar(self, bar: Bar) -> None:
        """Access one field without recording any custom indicator."""
        _ = bar.close


def test_indicator_recording_round_trip(tmp_path: Path) -> None:
    """Recorded indicator points should be accessible and exportable."""
    events: list[akquant.BacktestStreamEvent] = []
    result = run_backtest(
        data=_build_data(),
        strategy=IndicatorRecordingStrategy,
        symbols="IND",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        on_event=events.append,
    )

    indicator_df = result.indicator_df(name="close_echo", symbol="IND")
    assert len(indicator_df) == 3
    assert indicator_df["indicator_key"].tolist() == ["close_echo"] * 3
    assert indicator_df["symbol"].tolist() == ["IND"] * 3
    assert indicator_df["value"].tolist() == [10.0, 10.5, 11.0]
    assert "datetime" in indicator_df.columns

    definitions = result.indicator_definitions
    assert len(definitions) == 2
    assert definitions.iloc[0]["display_name"] == "Close Echo"
    assert definitions.iloc[0]["pane"] == 0
    assert definitions.iloc[0]["render_type"] == "line"
    assert definitions.iloc[1]["display_name"] == "Range Echo"

    export_path = tmp_path / "indicators.json"
    result.export_indicators(str(export_path), format="json")
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert sorted(payload.keys()) == ["definitions", "instances", "points", "run_id"]
    assert result.stream_run_id == events[0]["run_id"]
    assert payload["run_id"] == result.stream_run_id
    assert len(payload["definitions"]) == 2
    assert len(payload["instances"]) == 2
    assert len(payload["points"]) == 6


class DefaultPaneStrategy(Strategy):
    """Record an indicator without specifying a pane (exercises the default)."""

    def on_bar(self, bar: Bar) -> None:
        """Record one point relying on the default pane value."""
        self.record_indicator(name="default_pane", value=bar.close)


def test_indicator_points_expose_millisecond_timestamp() -> None:
    """Recorded points carry a millisecond timestamp alongside the ns value."""
    result = run_backtest(
        data=_build_data(),
        strategy=IndicatorRecordingStrategy,
        symbols="IND",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
    )

    points = result.indicator_outputs["points"]
    assert points, "expected recorded indicator points"
    for point in points:
        assert "timestamp_ms" in point
        assert point["timestamp_ms"] == point["timestamp"] // 1_000_000


class CjkMetaStrategy(Strategy):
    """Record an indicator carrying CJK display name and metadata."""

    def on_bar(self, bar: Bar) -> None:
        """Record one point with non-ASCII metadata."""
        self.record_indicator(
            name="均线",
            value=bar.close,
            display_name="五日均线",
            meta={"名称": "五日线"},
        )


def test_indicator_df_and_export_agree_on_timestamp_ms(tmp_path: Path) -> None:
    """timestamp_ms must be present on the DataFrame, export JSON, and raw points."""
    result = run_backtest(
        data=_build_data(),
        strategy=IndicatorRecordingStrategy,
        symbols="IND",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
    )

    frame = result.indicator_df()
    assert "timestamp_ms" in frame.columns
    assert (frame["timestamp_ms"] == frame["timestamp"] // 1_000_000).all()

    export_path = tmp_path / "indicators.json"
    result.export_indicators(str(export_path), format="json")
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    for point in payload["points"]:
        assert "timestamp_ms" in point
        assert point["timestamp_ms"] == point["timestamp"] // 1_000_000


def test_export_indicators_keeps_cjk_readable(tmp_path: Path) -> None:
    """Exported JSON must keep CJK metadata readable, matching the stream path."""
    result = run_backtest(
        data=_build_data(),
        strategy=CjkMetaStrategy,
        symbols="IND",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
    )

    export_path = tmp_path / "indicators.json"
    result.export_indicators(str(export_path), format="json")
    raw_text = export_path.read_text(encoding="utf-8")
    assert "五日均线" in raw_text
    assert "\\u" not in raw_text


def test_default_pane_is_main() -> None:
    """Omitting pane resolves to the main pane (index 0)."""
    result = run_backtest(
        data=_build_data(),
        strategy=DefaultPaneStrategy,
        symbols="IND",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
    )

    definitions = result.indicator_definitions
    assert len(definitions) == 1
    assert definitions.iloc[0]["pane"] == 0


class BadRenderTypeStrategy(Strategy):
    """Record an indicator with a render type outside the canonical enum."""

    def on_bar(self, bar: Bar) -> None:
        """Attempt to record with an unsupported render type."""
        self.record_indicator(name="broken", value=bar.close, render_type="candlestick")


def test_record_indicator_rejects_unknown_render_type() -> None:
    """An unsupported render type fails fast rather than degrading silently."""
    with pytest.raises(ValueError, match="render_type"):
        run_backtest(
            data=_build_data(),
            strategy=BadRenderTypeStrategy,
            symbols="IND",
            initial_cash=100000.0,
            show_progress=False,
            commission_rate=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            min_commission=0.0,
            lot_size=1,
        )


def test_legacy_strategy_without_indicator_recording_stays_empty() -> None:
    """Legacy strategies should still work and expose empty indicator outputs."""
    result = run_backtest(
        data=_build_data(),
        strategy=LegacyNoIndicatorStrategy,
        symbols="IND",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
    )

    assert result.indicator_df().empty
    assert result.indicator_definitions.empty
    assert result.indicator_instances.empty


def test_indicator_recording_emits_stream_events() -> None:
    """Indicator recording should emit point and snapshot stream events."""
    if not hasattr(akquant.Engine(), "emit_stream_event_py"):
        pytest.skip("Engine bindings do not expose emit_stream_event_py yet")

    events: list[akquant.BacktestStreamEvent] = []
    run_backtest(
        data=_build_data(),
        strategy=IndicatorRecordingStrategy,
        symbols="IND",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        on_event=events.append,
        stream_progress_interval=16,
        stream_equity_interval=16,
        stream_batch_size=1,
        stream_max_buffer=128,
    )

    point_events = [
        event for event in events if event["event_type"] == "indicator_point"
    ]
    snapshot_events = [
        event for event in events if event["event_type"] == "indicator_snapshot"
    ]

    assert len(point_events) == 6
    assert len(snapshot_events) == 3

    point_values = [
        float(event["payload"]["value"])
        for event in point_events
        if "value" in event["payload"]
    ]
    assert point_values == pytest.approx([10.0, 0.4, 10.5, 0.4, 11.0, 0.4])
    assert {event["payload"]["indicator_key"] for event in point_events} == {
        "close_echo",
        "range_echo",
    }
    assert {event["payload"]["owner_strategy_id"] for event in point_events} == {
        "_default"
    }

    snapshot_values = [
        [float(item["value"]) for item in json.loads(event["payload"]["items_json"])]
        for event in snapshot_events
    ]
    assert len(snapshot_values) == 3
    assert snapshot_values[0] == pytest.approx([10.0, 0.4])
    assert snapshot_values[1] == pytest.approx([10.5, 0.4])
    assert snapshot_values[2] == pytest.approx([11.0, 0.4])
    assert {event["payload"]["indicator_count"] for event in snapshot_events} == {"2"}
    seq_values = [int(event["seq"]) for event in events]
    assert seq_values == sorted(seq_values)
    assert len(seq_values) == len(set(seq_values))
    assert len({event["run_id"] for event in events}) == 1


def test_indicator_recording_stream_sampling_controls() -> None:
    """Indicator stream intervals should reduce emitted point and snapshot events."""
    events: list[akquant.BacktestStreamEvent] = []
    run_backtest(
        data=_build_data(),
        strategy=IndicatorRecordingStrategy,
        symbols="IND",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        on_event=events.append,
        indicator_stream_point_interval=2,
        indicator_stream_snapshot_interval=2,
        stream_batch_size=1,
        stream_max_buffer=128,
    )

    point_events = [
        event for event in events if event["event_type"] == "indicator_point"
    ]
    snapshot_events = [
        event for event in events if event["event_type"] == "indicator_snapshot"
    ]

    assert len(point_events) == 3
    assert len(snapshot_events) == 1
    assert {event["payload"]["indicator_key"] for event in point_events} == {
        "range_echo"
    }
    assert [
        float(event["payload"]["value"]) for event in point_events
    ] == pytest.approx([0.4, 0.4, 0.4])
    snapshot_items = json.loads(snapshot_events[0]["payload"]["items_json"])
    assert [float(item["value"]) for item in snapshot_items] == pytest.approx(
        [10.5, 0.4]
    )


def test_indicator_stream_bridge_builds_frontend_messages() -> None:
    """Indicator stream helper should normalize point and snapshot payloads."""
    events: list[akquant.BacktestStreamEvent] = []
    run_backtest(
        data=_build_data(),
        strategy=IndicatorRecordingStrategy,
        symbols="IND",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        on_event=events.append,
        stream_batch_size=1,
        stream_max_buffer=128,
    )

    messages = to_indicator_messages(events)
    assert len(messages) == 9
    assert all(message["channel"] == "indicator" for message in messages)
    assert {message["type"] for message in messages} == {"point", "snapshot"}

    first_point = next(message for message in messages if message["type"] == "point")
    assert first_point["indicator"]["indicator_key"] == "close_echo"
    assert first_point["indicator"]["display_name"] == "Close Echo"
    assert first_point["indicator"]["pane"] == 0
    assert first_point["indicator"]["render_type"] == "line"
    assert first_point["indicator"]["value"] == 10.0
    assert first_point["indicator"]["meta"] == {"source": "close"}
    assert first_point["indicator"]["warmup"] is False
    assert is_indicator_stream_event(events[0]) is False

    first_snapshot = next(
        message for message in messages if message["type"] == "snapshot"
    )
    assert first_snapshot["snapshot"]["indicator_count"] == 2
    assert first_snapshot["snapshot"]["items"][0]["indicator_key"] == "close_echo"
    assert first_snapshot["snapshot"]["items"][0]["value"] == 10.0
    assert first_snapshot["snapshot"]["items"][0]["meta"] == {"source": "close"}
    assert first_snapshot["snapshot"]["items"][1]["indicator_key"] == "range_echo"
    assert first_snapshot["snapshot"]["items"][1]["warmup"] is True
    assert first_snapshot["snapshot"]["items"][1]["meta"] == {"source": ["high", "low"]}
    assert first_snapshot["snapshot"]["indicator_keys"] == ["close_echo", "range_echo"]
    assert first_snapshot["snapshot"]["panes"] == [0, 1]
    assert first_snapshot["snapshot"]["render_types"] == ["line", "signal"]
    assert first_snapshot["snapshot"]["value_by_key"]["close_echo"] == pytest.approx(
        10.0
    )
    assert first_snapshot["snapshot"]["value_by_key"]["range_echo"] == pytest.approx(
        0.4
    )
    assert first_snapshot["snapshot"]["items_by_key"]["range_echo"]["pane"] == 1
    assert first_snapshot["snapshot"]["warmup_count"] == 1
    assert first_snapshot["snapshot"]["has_warmup"] is True


def test_indicator_stream_bridge_ignores_non_indicator_events() -> None:
    """Non-indicator stream events should not be converted into bridge messages."""
    event = akquant.BacktestStreamEvent(
        run_id="demo",
        seq=1,
        ts=0,
        event_type="started",
        symbol=None,
        level="info",
        payload={"status": "started"},
    )

    assert is_indicator_stream_event(event) is False
    assert to_indicator_message(event) is None
    assert to_indicator_messages([event]) == []


def test_indicator_stream_bridge_normalizes_unknown_symbols() -> None:
    """Unknown symbols should be normalized to None in bridged messages."""
    point_event = akquant.BacktestStreamEvent(
        run_id="demo",
        seq=2,
        ts=1,
        event_type="indicator_point",
        symbol="_unknown",
        level="info",
        payload={
            "owner_strategy_id": "_default",
            "indicator_key": "close_echo",
            "display_name": "Close Echo",
            "pane": "0",
            "render_type": "line",
            "symbol": "_unknown",
            "timestamp": "1",
            "value": "10.0",
            "warmup": "false",
            "meta_json": "{}",
        },
    )
    snapshot_event = akquant.BacktestStreamEvent(
        run_id="demo",
        seq=3,
        ts=1,
        event_type="indicator_snapshot",
        symbol="_unknown",
        level="info",
        payload={
            "owner_strategy_id": "_default",
            "symbol": "_unknown",
            "timestamp": "1",
            "indicator_count": "1",
            "items_json": json.dumps(
                [
                    {
                        "indicator_key": "close_echo",
                        "display_name": "Close Echo",
                        "pane": "0",
                        "render_type": "line",
                        "value": 10.0,
                        "warmup": False,
                        "meta_json": "{}",
                    }
                ]
            ),
        },
    )

    point_message = to_indicator_message(point_event)
    snapshot_message = to_indicator_message(snapshot_event)

    assert point_message is not None
    assert snapshot_message is not None
    assert point_message["symbol"] is None
    assert point_message["indicator"]["symbol"] is None
    assert snapshot_message["symbol"] is None
    assert snapshot_message["snapshot"]["symbol"] is None


def test_indicator_stream_bridge_accepts_predecoded_payloads() -> None:
    """Bridge helper should also accept already-decoded list/dict payload values."""
    event = akquant.BacktestStreamEvent(
        run_id="demo",
        seq=4,
        ts=2,
        event_type="indicator_snapshot",
        symbol="IND",
        level="info",
        payload={
            "owner_strategy_id": "_default",
            "symbol": "IND",
            "timestamp": "2",
            "indicator_count": "2",
            "items_json": [
                {
                    "indicator_key": "close_echo",
                    "display_name": "Close Echo",
                    "pane": "0",
                    "render_type": "line",
                    "value": 10.5,
                    "warmup": False,
                    "meta_json": {"source": "close"},
                },
                {
                    "indicator_key": "range_echo",
                    "display_name": "Range Echo",
                    "pane": "1",
                    "render_type": "signal",
                    "value": 0.4,
                    "warmup": True,
                    "meta_json": {"source": ["high", "low"]},
                },
            ],
        },
    )

    message = to_indicator_message(event)

    assert message is not None
    assert message["snapshot"]["items"][0]["meta"] == {"source": "close"}
    assert message["snapshot"]["items"][1]["meta"] == {"source": ["high", "low"]}
    assert message["snapshot"]["items"][1]["warmup"] is True
    assert message["snapshot"]["indicator_keys"] == ["close_echo", "range_echo"]
    assert message["snapshot"]["value_by_key"]["range_echo"] == pytest.approx(0.4)
