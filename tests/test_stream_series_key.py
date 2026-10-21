"""实时指标流带上 owner 与 series_key(stream schema 1.5).

主策略与指标脚本用同名键 ``ema`` 时, 实时流里的每个 point 与 snapshot item 都要
带上与历史接口 ``series_keys`` 一致的 series 名, 让前端实时侧不再把两条线合成
一条。见 spec ``docs/superpowers/specs/2026-10-20-stream-series-key-design.md``。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Indicator, Strategy, run_backtest, run_live
from akquant.akquant import AssetType, Instrument
from akquant.indicator_identity import series_keys
from akquant.indicator_recording import IndicatorRecorder
from akquant.live._stream_sink import StreamingIndicatorSink
from akquant.lwc import to_lwc_update

SYMBOL = "SKEY"


def _md(n: int = 60) -> pd.DataFrame:
    closes = [10.0 + (i % 7) * 0.5 + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "symbol": SYMBOL,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": 100.0,
        }
    )


# ------------------------------------------------------------------ 策略


class MainEma(Strategy):
    """主策略: on_start 声明主图 ema(EMA5, 早就绪)."""

    def on_start(self) -> None:
        """声明主图 ema."""
        self.ema = self.I(aq.EMA(5), name="ema", plot=True, pane=0)

    def on_bar(self, bar: Bar) -> None:
        """不交易."""


class Sub(Indicator):
    """指标脚本: on_start 声明同名 ema(EMA20, 晚就绪), 默认附图."""

    def on_start(self) -> None:
        """声明同名 ema."""
        self.ema = self.I(aq.EMA(20), name="ema")


class MainEmaInit(Strategy):
    """主策略: 在 ``__init__`` 里声明 ema —— 早于录制器挂载."""

    def __init__(self) -> None:
        """声明主图 ema."""
        super().__init__()
        self.ema = self.I(aq.EMA(5), name="ema", plot=True, pane=0)

    def on_bar(self, bar: Bar) -> None:
        """不交易."""


class SubInit(Indicator):
    """指标脚本: 在 ``__init__`` 里声明同名 ema(晚就绪)."""

    def __init__(self) -> None:
        """声明同名 ema."""
        super().__init__()
        self.ema = self.I(aq.EMA(20), name="ema")


class SoloEma(Strategy):
    """单 owner: 只有主策略画 ema."""

    def on_start(self) -> None:
        """声明主图 ema."""
        self.ema = self.I(aq.EMA(5), name="ema", plot=True, pane=0)

    def on_bar(self, bar: Bar) -> None:
        """不交易."""


class LateWriter(Indicator):
    """指标脚本: 第 10 根 bar 起才手写一个与主策略同名的 ema."""

    def __init__(self) -> None:
        """初始化计数."""
        super().__init__()
        self.bars = 0

    def on_bar(self, bar: Bar) -> None:
        """从第 10 根起手写 record_indicator."""
        self.bars += 1
        if self.bars >= 10:
            self.record_indicator("ema", bar.close, pane=1)


# ------------------------------------------------------------------ 工具


def _messages(events: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for event in events:
        if aq.is_indicator_stream_event(event):
            message = aq.to_indicator_message(event)
            if message is not None:
                out.append(message)
    return out


def _run_bt(strategy: Any, indicators: Any, **kwargs: Any) -> Any:
    events: List[Any] = []
    result = run_backtest(
        strategy=strategy,
        data=_md(),
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
        indicators=indicators,
        on_event=events.append,
        **kwargs,
    )
    return result, _messages(events)


def _instrument() -> Instrument:
    return Instrument(
        symbol=SYMBOL,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _run_live(strategy: Any, indicators: Any) -> List[Dict[str, Any]]:
    events: List[Any] = []
    base = int(pd.Timestamp("2024-03-01 09:30", tz="Asia/Shanghai").value)
    bars = [
        Bar(
            timestamp=base + i * 60_000_000_000,
            open=c,
            high=c + 0.5,
            low=c - 0.5,
            close=float(c),
            volume=100.0,
            symbol=SYMBOL,
        )
        for i, c in enumerate(10.0 + (i % 5) * 0.3 for i in range(30))
    ]
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars},
        cash=1e6,
        show_progress=False,
        on_event=events.append,
        duration="30s",
        strategy_id="main",
        indicators=indicators,
    )
    return _messages(events)


def _assert_stream_matches(
    messages: List[Dict[str, Any]], expected: Dict[tuple[str, str], str]
) -> None:
    """每个 point 与 snapshot item 的 series_key 都等于 expected 给出的名字."""
    points = [m for m in messages if m["type"] == "point"]
    snaps = [m for m in messages if m["type"] == "snapshot"]
    assert points and snaps
    for message in points:
        ind = message["indicator"]
        identity = (ind["owner_strategy_id"], ind["indicator_key"])
        assert ind["series_key"] == expected[identity], ind
    for message in snaps:
        snap = message["snapshot"]
        assert snap["series_keys"] == [i["series_key"] for i in snap["items"]]
        for item in snap["items"]:
            assert item["owner_strategy_id"] == snap["owner_strategy_id"]
            identity = (item["owner_strategy_id"], item["indicator_key"])
            assert item["series_key"] == expected[identity], item
            assert snap["items_by_key"][item["series_key"]] is item
            assert snap["value_by_key"][item["series_key"]] == item["value"]
    # 同一时间戳的快照按 value_by_key 合并: 两个 owner 的 ema 都在, 互不覆盖
    merged: Dict[int, Dict[str, Any]] = {}
    for message in snaps:
        snap = message["snapshot"]
        merged.setdefault(snap["timestamp"], {}).update(snap["value_by_key"])
    prefixed = {k for k in expected.values() if ":" in k}
    if prefixed:
        assert any(prefixed <= set(values) for values in merged.values())


# ------------------------------------------------------------------ 端到端


def test_backtest_dual_ema_stream_matches_series_keys() -> None:
    """回测 on_event: 流里的 series_key 与对最终定义表算出的一致."""
    result, messages = _run_bt(MainEma, [Sub])
    expected = series_keys(result.indicator_definitions.to_dict("records"))
    assert set(expected.values()) == {"_default:ema", "sub:ema"}
    _assert_stream_matches(messages, expected)


def test_backtest_init_declarations_join_conflict_check() -> None:
    """``__init__`` 里的声明(早于挂载)也参与冲突判断(S2 来源 ①)."""
    result, messages = _run_bt(MainEmaInit, [SubInit])
    expected = series_keys(result.indicator_definitions.to_dict("records"))
    assert set(expected.values()) == {"_default:ema", "sub_init:ema"}
    _assert_stream_matches(messages, expected)


def test_live_dual_ema_stream_matches_series_keys() -> None:
    """实盘 replay: 主策略 + 指标脚本同名 ema, 流里两边都带 owner 前缀."""
    messages = _run_live(MainEma, [Sub])
    seen = [
        {
            "owner_strategy_id": m["indicator"]["owner_strategy_id"],
            "indicator_key": m["indicator"]["indicator_key"],
        }
        for m in messages
        if m["type"] == "point"
    ]
    expected = series_keys(seen)
    assert set(expected.values()) == {"main:ema", "sub:ema"}
    _assert_stream_matches(messages, expected)


def test_live_init_declarations_join_conflict_check() -> None:
    """实盘挂载时同样补登 ``__init__`` 里的声明."""
    messages = _run_live(MainEmaInit, [SubInit])
    expected = {("main", "ema"): "main:ema", ("sub_init", "ema"): "sub_init:ema"}
    _assert_stream_matches(messages, expected)


def test_single_owner_series_key_is_plain_key() -> None:
    """单 owner: series_key 就是 key, 原有字段不变."""
    _, messages = _run_bt(SoloEma, None)
    points = [m for m in messages if m["type"] == "point"]
    assert points
    for message in points:
        assert message["indicator"]["series_key"] == "ema"
        assert message["indicator"]["indicator_key"] == "ema"
        assert message["indicator"]["owner_strategy_id"] == "_default"
    snap = next(m for m in messages if m["type"] == "snapshot")["snapshot"]
    assert snap["indicator_keys"] == ["ema"]
    assert snap["series_keys"] == ["ema"]
    assert set(snap["items_by_key"]) == {"ema"}
    assert set(snap["value_by_key"]) == {"ema"}


@pytest.mark.parametrize("kind", ["recorder", "sink"])
def test_single_owner_payload_only_adds_series_key(kind: str) -> None:
    """单 owner 时 point payload 只多一个 series_key, 原有字段的值逐一不变."""
    payloads: List[Dict[str, str]] = []
    sink: Any
    if kind == "recorder":
        sink = IndicatorRecorder(
            stream_emitter=lambda et, sym, lvl, payload: payloads.append(payload)
        )
    else:
        sink = StreamingIndicatorSink(lambda e: payloads.append(e["payload"]))
    sink.record(
        name="ema",
        value=1.5,
        symbol=SYMBOL,
        timestamp=1_704_153_600_000_000_000,
        owner_strategy_id="_default",
    )
    sink.flush_stream_snapshot()
    assert payloads[0] == {
        "owner_strategy_id": "_default",
        "indicator_key": "ema",
        "series_key": "ema",
        "display_name": "ema",
        "pane": "0",
        "render_type": "line",
        "symbol": SYMBOL,
        "timestamp": "1704153600000000000",
        "timestamp_ms": "1704153600000",
        "value": "1.5",
        "warmup": "false",
        "confirmed": "true",
        "phase": "run",
        "scale_group": "",
        # 1.6 新增; 非 candle 为空串
        "group": "",
        "role": "",
        "meta_json": "",
    }
    item = json.loads(payloads[1]["items_json"])[0]
    assert item == {
        "owner_strategy_id": "_default",
        "indicator_key": "ema",
        "series_key": "ema",
        "display_name": "ema",
        "pane": 0,
        "render_type": "line",
        "value": 1.5,
        "warmup": False,
        "confirmed": True,
        "phase": "run",
        "group": "",
        "role": "",
        "meta_json": "",
    }


def test_backtest_mid_run_collision_upgrades_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """运行中途 record_indicator 撞名: 从此刻起两边都加前缀, 只告警一次."""
    with caplog.at_level(logging.WARNING):
        _, messages = _run_bt(SoloEma, [LateWriter])
    points = [m["indicator"] for m in messages if m["type"] == "point"]
    first_late = next(
        i for i, p in enumerate(points) if p["owner_strategy_id"] == "late_writer"
    )
    before, after = points[:first_late], points[first_late:]
    assert before and all(p["series_key"] == "ema" for p in before)
    assert after
    for p in after:
        assert p["series_key"] == f"{p['owner_strategy_id']}:ema"
    assert {p["owner_strategy_id"] for p in after} == {"_default", "late_writer"}
    warned = [r for r in caplog.records if "series_key" in r.getMessage()]
    assert len(warned) == 1, [r.getMessage() for r in warned]


class _RecordOnlySink:
    """第三方 sink: 不实现可选的 declare."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def build_payload(self) -> Dict[str, list[Dict[str, Any]]]:
        return {"definitions": [], "instances": [], "points": []}

    def flush_stream_snapshot(self) -> None:
        return None

    def set_stream_emitter(self, stream_emitter: Any) -> None:
        return None


def test_third_party_sink_without_declare_is_fine() -> None:
    """第三方 sink 只实现 record 等协议方法、没有 declare 时不报错."""
    sink = _RecordOnlySink()
    run_backtest(
        strategy=MainEmaInit,
        data=_md(),
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
        indicators=[Sub],
        indicator_recorder=sink,
    )
    owners = {c["owner_strategy_id"] for c in sink.calls}
    assert owners == {"_default", "sub"}


# ------------------------------------------------------------------ 单元


@pytest.mark.parametrize("kind", ["recorder", "sink"])
def test_sink_mid_run_collision_unit(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    """录制器与实盘 sink 同口径: 撞名前不加前缀, 撞名后全部加, 告警一次."""
    events: List[Any] = []
    sink: Any
    if kind == "recorder":
        sink = IndicatorRecorder()
        sink.set_stream_emitter(
            lambda et, sym, lvl, payload: events.append(
                {"event_type": et, "symbol": sym, "level": lvl, "payload": payload}
            )
        )
    else:
        sink = StreamingIndicatorSink(events.append)

    def rec(owner: str, ts: int) -> None:
        sink.record(
            name="ema", value=1.0, symbol=SYMBOL, timestamp=ts, owner_strategy_id=owner
        )
        sink.flush_stream_snapshot()

    with caplog.at_level(logging.WARNING):
        rec("a", 1_000_000_000)
        rec("b", 2_000_000_000)
        rec("a", 3_000_000_000)
        rec("c", 4_000_000_000)
    points = [e["payload"] for e in events if e["event_type"] == "indicator_point"]
    assert [p["series_key"] for p in points] == ["ema", "b:ema", "a:ema", "c:ema"]
    snaps = [e["payload"] for e in events if e["event_type"] == "indicator_snapshot"]
    items = [json.loads(s["items_json"])[0] for s in snaps]
    assert [i["series_key"] for i in items] == ["ema", "b:ema", "a:ema", "c:ema"]
    assert [i["owner_strategy_id"] for i in items] == ["a", "b", "a", "c"]
    warned = [r for r in caplog.records if "series_key" in r.getMessage()]
    assert len(warned) == 1


@pytest.mark.parametrize("kind", ["recorder", "sink"])
def test_declare_before_points_avoids_warning(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    """先 declare 两个 owner: 第一个点就带前缀, 不告警."""
    events: List[Any] = []
    sink: Any
    if kind == "recorder":
        sink = IndicatorRecorder(
            stream_emitter=lambda et, sym, lvl, payload: events.append(payload)
        )
    else:
        sink = StreamingIndicatorSink(lambda e: events.append(e["payload"]))
    sink.declare("a", "ema")
    sink.declare("b", "ema")
    sink.declare("b", "ema")
    with caplog.at_level(logging.WARNING):
        sink.record(
            name="ema", value=1.0, symbol=SYMBOL, timestamp=1, owner_strategy_id="a"
        )
    assert events[0]["series_key"] == "a:ema"
    assert not [r for r in caplog.records if "series_key" in r.getMessage()]


def _point_event(**extra: str) -> Dict[str, Any]:
    payload = {
        "owner_strategy_id": "sub",
        "indicator_key": "ema",
        "display_name": "ema",
        "pane": "1",
        "render_type": "line",
        "symbol": SYMBOL,
        "timestamp": "60000000000",
        "timestamp_ms": "60000",
        "value": "1.5",
        "warmup": "false",
        "confirmed": "true",
        "phase": "run",
        "scale_group": "",
        "meta_json": "{}",
    }
    payload.update(extra)
    return {
        "run_id": "r",
        "seq": 1,
        "ts": 0,
        "event_type": "indicator_point",
        "symbol": SYMBOL,
        "level": "info",
        "payload": payload,
    }


def _snapshot_event(items: List[Dict[str, Any]], owner: str = "_default") -> Any:
    return {
        "run_id": "r",
        "seq": 2,
        "ts": 0,
        "event_type": "indicator_snapshot",
        "symbol": SYMBOL,
        "level": "info",
        "payload": {
            "owner_strategy_id": owner,
            "symbol": SYMBOL,
            "timestamp": "60000000000",
            "timestamp_ms": "60000",
            "indicator_count": str(len(items)),
            "items_json": json.dumps(items),
        },
    }


def test_snapshot_items_by_key_uses_series_key() -> None:
    """同 indicator_key、不同 series_key 的两个 item 在 items_by_key 里互不覆盖."""
    items = [
        {"indicator_key": "ema", "series_key": "_default:ema", "value": 1.0},
        {"indicator_key": "ema", "series_key": "sub:ema", "value": 2.0},
    ]
    snap = aq.to_indicator_message(_snapshot_event(items))
    assert snap is not None
    body = snap["snapshot"]
    assert body["series_keys"] == ["_default:ema", "sub:ema"]
    assert body["indicator_keys"] == ["ema", "ema"]
    assert body["value_by_key"] == {"_default:ema": 1.0, "sub:ema": 2.0}
    assert set(body["items_by_key"]) == {"_default:ema", "sub:ema"}


def test_v14_events_fall_back_to_indicator_key() -> None:
    """1.4 事件没有 series_key/item owner: 回落到 indicator_key 与快照 owner."""
    point = aq.to_indicator_message(_point_event())
    assert point is not None
    assert point["indicator"]["series_key"] == "ema"
    snap = aq.to_indicator_message(
        _snapshot_event([{"indicator_key": "ema", "value": 1.0}], owner="sub")
    )
    assert snap is not None
    body = snap["snapshot"]
    assert body["series_keys"] == ["ema"]
    assert body["items"][0]["series_key"] == "ema"
    assert body["items"][0]["owner_strategy_id"] == "sub"
    assert set(body["items_by_key"]) == {"ema"}
    assert body["value_by_key"] == {"ema": 1.0}


def test_point_message_passes_series_key_through() -> None:
    """Point 消息透传 series_key."""
    message = aq.to_indicator_message(_point_event(series_key="sub:ema"))
    assert message is not None
    assert message["indicator"]["series_key"] == "sub:ema"
    assert message["schema_version"] == aq.STREAM_SCHEMA_VERSION


def test_to_lwc_update_carries_owner_and_series_key() -> None:
    """to_lwc_update 输出带 owner_strategy_id 与 series_key."""
    message = aq.to_indicator_message(_point_event(series_key="sub:ema"))
    assert message is not None
    update: Optional[Dict[str, Any]] = to_lwc_update(message)
    assert update is not None
    assert update["owner_strategy_id"] == "sub"
    assert update["series_key"] == "sub:ema"
    legacy = aq.to_indicator_message(_point_event())
    assert legacy is not None
    legacy_update = to_lwc_update(legacy)
    assert legacy_update is not None
    assert legacy_update["series_key"] == "ema"


def test_stream_schema_version_at_least_1_5() -> None:
    """契约至少是 1.5(1.6 起另加 group/role, 见 test_candle_render_type)."""
    major, _, minor = aq.STREAM_SCHEMA_VERSION.partition(".")
    assert (int(major), int(minor)) >= (1, 5), aq.STREAM_SCHEMA_VERSION


# ------------------------------------------------------------- 实盘 + 预热


class MainSma(Strategy):
    """主策略: on_start 声明主图 ema(SMA2, 预热第 2 根即就绪)."""

    def on_start(self) -> None:
        """声明主图 ema."""
        self.ema = self.I(aq.SMA(2), name="ema", plot=True, pane=0)

    def on_bar(self, bar: Bar) -> None:
        """不交易."""


class SubSma(Indicator):
    """指标脚本: on_start 声明同名 ema(SMA2)."""

    def on_start(self) -> None:
        """声明同名 ema."""
        self.ema = self.I(aq.SMA(2), name="ema")


def test_live_preload_slot_replay_waits_for_main_on_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """preload_history= + 主策略与指标脚本都在 on_start 声明同名 ema.

    slot 的 on_start 早于主策略(主策略的由 Rust 在 engine.run() 内触发); slot
    预热回放若在它自己的 on_start 末尾就跑, 预热点会先以 ``ema`` 发出, 之后再
    切成 ``owner:ema`` 并告警。回放须排在主策略 on_start 之后。
    """
    events: List[Any] = []
    closes = [10.0, 11.0, 12.0, 13.0]
    preload = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-08-20 15:00", periods=len(closes), freq="D", tz="Asia/Shanghai"
            ),
            "symbol": [SYMBOL] * len(closes),
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        }
    )
    live_bar = Bar(
        timestamp=int(pd.Timestamp("2026-08-25 15:00", tz="Asia/Shanghai").value),
        open=20.0,
        high=20.0,
        low=20.0,
        close=20.0,
        volume=1000.0,
        symbol=SYMBOL,
    )
    live_bar2 = Bar(
        timestamp=int(pd.Timestamp("2026-08-26 15:00", tz="Asia/Shanghai").value),
        open=21.0,
        high=21.0,
        low=21.0,
        close=21.0,
        volume=1000.0,
        symbol=SYMBOL,
    )
    with caplog.at_level(logging.WARNING):
        run_live(
            strategy_cls=MainSma,
            instruments=[_instrument()],
            broker="replay",
            trading_mode="paper",
            gateway_options={"bars": [live_bar, live_bar2], "freq": "1d"},
            cash=1e6,
            duration="30s",
            on_event=events.append,
            strategy_id="main",
            indicators=[SubSma],
            preload_history=preload,
        )
    messages = _messages(events)
    points = [m["indicator"] for m in messages if m["type"] == "point"]
    assert {(p["owner_strategy_id"], p["phase"]) for p in points} == {
        ("main", "preload"),
        ("main", "run"),
        ("sub_sma", "preload"),
        ("sub_sma", "run"),
    }
    for p in points:
        assert p["series_key"] == f"{p['owner_strategy_id']}:ema", p
    expected = {("main", "ema"): "main:ema", ("sub_sma", "ema"): "sub_sma:ema"}
    _assert_stream_matches(messages, expected)
    warned = [r for r in caplog.records if "series_key" in r.getMessage()]
    assert not warned, [r.getMessage() for r in warned]
