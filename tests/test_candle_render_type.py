"""K 线形态指标: render_type="candle" 声明层与协议(stream schema 1.6).

见 spec ``docs/superpowers/specs/2026-10-21-candle-render-type-design.md``
C1-C4 与 C6 的实时流部分: candle 沿用 outputs 机制展开成四条单值点, 定义行与
流消息带 ``group``(``self.I`` 的 name)和 ``role``(open/high/low/close);
类定义行 ``replaces_price`` 只属于 ``aq.Indicator``。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Indicator, Strategy, run_backtest
from akquant.akquant import Tick
from akquant.indicator_recording import IndicatorRecorder
from akquant.live._stream_sink import StreamingIndicatorSink

SYMBOL = "CNDL"
OHLC = ("open", "high", "low", "close")


def _md(n: int = 12) -> pd.DataFrame:
    closes = [10.0 + (i % 5) * 0.5 + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "symbol": SYMBOL,
            "open": [c - 0.2 for c in closes],
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": 100.0,
        }
    )


def _messages(events: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for event in events:
        if aq.is_indicator_stream_event(event):
            message = aq.to_indicator_message(event)
            if message is not None:
                out.append(message)
    return out


class HA(Indicator, title="平均K线", overlay=True, replaces_price=True):
    """主图平均 K 线, 外加一条非 candle 的 EMA 作对照."""

    def on_start(self) -> None:
        """声明 candle 四分量与对照线."""
        self.ha = self.I(
            aq.HeikinAshi(),
            name="ha",
            input_mode="ohlc",
            outputs=OHLC,
            render_type="candle",
            color="#26a69a",
            label="平均K线",
        )
        self.ema = self.I(aq.EMA(3), name="ema")


class Quiet(Strategy):
    """不交易的主策略."""

    def on_bar(self, bar: Bar) -> None:
        """不交易."""


def _run(indicators: Any = (HA,), **kwargs: Any) -> Any:
    events: List[Any] = []
    result = run_backtest(
        strategy=Quiet,
        data=_md(),
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
        indicators=list(indicators),
        on_event=events.append,
        **kwargs,
    )
    return result, _messages(events)


# ------------------------------------------------------------- 声明校验(C3)


def _declare(**kwargs: Any) -> Strategy:
    strategy = Quiet()
    strategy.I(aq.HeikinAshi(), name="ha", input_mode="ohlc", **kwargs)
    return strategy


def test_candle_is_canonical_render_type() -> None:
    """封闭枚举里有 candle."""
    assert "candle" in aq.chart.RENDER_TYPE_CANONICAL
    assert aq.chart.normalize_render_type("CANDLE") == "candle"


@pytest.mark.parametrize(
    "outputs",
    [
        ("open", "high", "low"),  # 缺一个
        ("open", "high", "low", "close", "volume"),  # 多一个
        ("open", "high", "low", "low"),  # 重复
        ("o", "h", "l", "c"),  # 名字不对
    ],
)
def test_candle_requires_exact_ohlc_outputs(outputs: tuple[str, ...]) -> None:
    """声明时 outputs 必须恰好是 open/high/low/close 四个, 否则报错."""
    with pytest.raises(ValueError, match="candle"):
        _declare(outputs=outputs, render_type="candle")


def test_candle_without_outputs_rejected() -> None:
    """不写 outputs 的 candle 也报错."""
    with pytest.raises(ValueError, match="candle"):
        _declare(render_type="candle")


def test_candle_outputs_order_free() -> None:
    """顺序不限; 每个分量的 group 是 name、role 是分量名."""
    strategy = _declare(outputs=("close", "open", "low", "high"), render_type="candle")
    declaration = strategy._incremental_indicators["ha"]
    assert declaration.render_type == "candle"
    entries = declaration.report_entries("ha", (1.0, 2.0, 3.0, 4.0))
    assert [(e.key, e.group, e.role, e.value) for e in entries] == [
        ("ha.close", "ha", "close", 1.0),
        ("ha.open", "ha", "open", 2.0),
        ("ha.low", "ha", "low", 3.0),
        ("ha.high", "ha", "high", 4.0),
    ]


def test_candle_outputs_accepts_generator() -> None:
    """传生成器作 outputs 时只遍历一次, 合法的 candle 不被误拒."""
    strategy = _declare(outputs=(o for o in OHLC), render_type="candle")
    assert strategy._incremental_indicators["ha"].outputs == OHLC


def test_plain_name_colliding_with_candle_component_rejected() -> None:
    """同一 owner 里普通指标名等于 candle 分量键: 先后两种顺序都报错."""
    strategy = _declare(outputs=OHLC, render_type="candle")
    with pytest.raises(ValueError, match="ha.open"):
        strategy.I(aq.SMA(3), name="ha.open")

    other = Quiet()
    other.I(aq.SMA(3), name="ha.close")
    with pytest.raises(ValueError, match="ha.close"):
        other.I(
            aq.HeikinAshi(),
            name="ha",
            input_mode="ohlc",
            outputs=OHLC,
            render_type="candle",
        )


def test_non_candle_outputs_have_empty_group_role() -> None:
    """非 candle 的多值指标: group/role 为空串."""
    strategy = _declare(outputs=OHLC)
    entries = strategy._incremental_indicators["ha"].report_entries(
        "ha", (1.0, 2.0, 3.0, 4.0)
    )
    assert {(e.group, e.role) for e in entries} == {("", "")}


def test_record_indicator_rejects_candle() -> None:
    """record_indicator 不能直接写 candle, 只能经 self.I(outputs=) 声明."""
    strategy = Quiet()
    strategy._indicator_recorder = IndicatorRecorder()
    with pytest.raises(ValueError, match="candle"):
        strategy.record_indicator(
            "x", 1.0, symbol=SYMBOL, timestamp=1, render_type="candle"
        )


@pytest.mark.parametrize("kind", ["recorder", "sink"])
def test_sink_rejects_inconsistent_group_role(kind: str) -> None:
    """非 candle 带 role、candle 缺 group/role 或 role 不合法, 都 fail-fast."""
    sink: Any = (
        IndicatorRecorder() if kind == "recorder" else StreamingIndicatorSink(print)
    )
    base = dict(value=1.0, symbol=SYMBOL, timestamp=1, owner_strategy_id="a")
    with pytest.raises(ValueError, match="role"):
        sink.record(name="x", render_type="line", role="open", **base)
    with pytest.raises(ValueError, match="group"):
        sink.record(name="x", render_type="line", group="g", **base)
    with pytest.raises(ValueError, match="candle"):
        sink.record(name="x.open", render_type="candle", **base)
    with pytest.raises(ValueError, match="role"):
        sink.record(name="x.v", render_type="candle", group="x", role="volume", **base)


# ------------------------------------------------------------- replaces_price(C4)


def test_replaces_price_recorded_and_inherited() -> None:
    """replaces_price 写在类定义行; 和 overlay 一样可被子类继承."""

    class Child(HA):
        pass

    assert HA.resolve_replaces_price() is True
    assert Child.resolve_replaces_price() is True
    assert Child.resolve_overlay() is True


def test_replaces_price_defaults_false() -> None:
    """省略时为 False."""

    class Plain(Indicator, overlay=True):
        pass

    assert Plain.resolve_replaces_price() is False


def test_replaces_price_requires_overlay() -> None:
    """没写 overlay=True 或写了 overlay=False, 都在定义类时报错."""
    with pytest.raises(ValueError, match="overlay"):

        class NoOverlay(Indicator, replaces_price=True):
            pass

    with pytest.raises(ValueError, match="overlay"):

        class SubPane(Indicator, overlay=False, replaces_price=True):
            pass

    with pytest.raises(ValueError, match="overlay"):

        class ChildToSub(HA, overlay=False):
            pass


def test_replaces_price_must_be_bool() -> None:
    """replaces_price 必须是 bool."""
    with pytest.raises(TypeError, match="replaces_price"):

        class Bad(Indicator, overlay=True, replaces_price="yes"):  # type: ignore[arg-type]
            pass


def test_strategy_rejects_replaces_price() -> None:
    """replaces_price 只适用于指标脚本."""
    with pytest.raises(TypeError, match="replaces_price"):

        class Bad(Strategy, replaces_price=True):  # type: ignore[call-arg]
            pass


def test_inspect_strategy_reports_replaces_price() -> None:
    """静态检查: 字面量、同文件继承、动态值、策略为 None、指标默认 False."""
    src = (
        "import akquant as aq\n"
        "class HA(aq.Indicator, title='平均K线', overlay=True, replaces_price=True):\n"
        "    pass\n"
        "class Child(HA):\n"
        "    pass\n"
        "class Plain(aq.Indicator, overlay=True):\n"
        "    pass\n"
        "class Dyn(aq.Indicator, overlay=True, replaces_price=FLAG):\n"
        "    pass\n"
        "class S(aq.Strategy):\n"
        "    def on_start(self):\n"
        "        self.x = self.I(aq.SMA(3))\n"
    )
    by_name = {c.name: c for c in aq.inspect_strategy(src).classes}
    assert by_name["HA"].replaces_price is True
    assert by_name["Child"].replaces_price is True
    assert by_name["Plain"].replaces_price is False
    assert by_name["Dyn"].replaces_price is None
    assert "replaces_price" in by_name["Dyn"].dynamic
    assert by_name["S"].replaces_price is None
    assert "replaces_price" in by_name["HA"].__dict__


# ------------------------------------------------------------- 定义行(C2)


def test_definitions_carry_group_and_role() -> None:
    """indicator_definitions 有 group/role 列; candle 四分量共用 pane/color/label."""
    result, _ = _run()
    defs = result.indicator_definitions
    assert {"group", "role"} <= set(defs.columns)
    rows = defs.set_index("indicator_key")
    for role in OHLC:
        row = rows.loc[f"ha.{role}"]
        assert (row["group"], row["role"], row["render_type"]) == (
            "ha",
            role,
            "candle",
        )
        assert (row["pane"], row["color"], row["display_name"]) == (
            0,
            "#26a69a",
            "平均K线",
        )
    assert (rows.loc["ema", "group"], rows.loc["ema", "role"]) == ("", "")
    raw = {d["indicator_key"]: d for d in result.indicator_outputs["definitions"]}
    assert (raw["ha.open"]["group"], raw["ha.open"]["role"]) == ("ha", "open")
    assert (raw["ema"]["group"], raw["ema"]["role"]) == ("", "")


def test_definitions_frame_fills_missing_group_role() -> None:
    """外部 sink 的定义行没有 group/role 时补空串."""
    result, _ = _run()
    legacy = [
        {k: v for k, v in d.items() if k not in {"group", "role"}}
        for d in result.indicator_outputs["definitions"]
    ]
    result.indicator_outputs = {**result.indicator_outputs, "definitions": legacy}
    result.__dict__.pop("indicator_definitions", None)
    defs = result.indicator_definitions
    assert set(defs["group"]) == {""} and set(defs["role"]) == {""}


def test_candle_values_equal_heikin_ashi() -> None:
    """四个分量的点位值与 HeikinAshi 逐根一致."""
    result, _ = _run()
    df = _md()
    ref = aq.HeikinAshi()
    expected = []
    for o, h, low, c in zip(df["open"], df["high"], df["low"], df["close"]):
        row = ref.update(o, h, low, c)
        assert row is not None  # 第一根就有值
        expected.append(row)
    points = result.indicator_outputs["points"]
    for idx, role in enumerate(OHLC):
        values = [p["value"] for p in points if p["indicator_key"] == f"ha.{role}"]
        assert values == pytest.approx([row[idx] for row in expected])


# ------------------------------------------------------------- 实时流(C6/C7)


def test_stream_points_and_snapshots_carry_group_role() -> None:
    """每个分量点带 group/role, 同一时刻的 snapshot 里四个分量齐全."""
    _, messages = _run()
    points = [m["indicator"] for m in messages if m["type"] == "point"]
    candle = [p for p in points if p["render_type"] == "candle"]
    assert candle
    for p in candle:
        assert p["group"] == "ha"
        assert p["indicator_key"] == f"ha.{p['role']}"
    other = [p for p in points if p["indicator_key"] == "ema"]
    assert other and all((p["group"], p["role"]) == ("", "") for p in other)

    snaps = [m["snapshot"] for m in messages if m["type"] == "snapshot"]
    assert snaps
    for snap in snaps:
        roles = sorted(i["role"] for i in snap["items"] if i["group"] == "ha")
        assert roles == sorted(OHLC)
    assert all(m["schema_version"] == "1.6" for m in messages)


def test_multi_owner_same_group_name_series_keys_prefixed() -> None:
    """两个脚本都叫 ha: 分量 series_key 带 owner 前缀, group 仍是 ha."""

    class HA2(HA, id="ha_two"):
        pass

    _, messages = _run(indicators=(HA, HA2))
    points = [m["indicator"] for m in messages if m["type"] == "point"]
    candle = [p for p in points if p["render_type"] == "candle"]
    assert {p["owner_strategy_id"] for p in candle} == {"ha", "ha_two"}
    for p in candle:
        assert p["group"] == "ha"
        assert p["series_key"] == f"{p['owner_strategy_id']}:ha.{p['role']}"


@pytest.mark.parametrize("kind", ["recorder", "sink"])
def test_raw_payload_has_group_role(kind: str) -> None:
    """原始 point payload 与 snapshot item 带 group/role(非 candle 为空串)."""
    payloads: List[Dict[str, str]] = []
    sink: Any
    if kind == "recorder":
        sink = IndicatorRecorder(
            stream_emitter=lambda et, sym, lvl, payload: payloads.append(payload)
        )
    else:
        sink = StreamingIndicatorSink(lambda e: payloads.append(e["payload"]))
    sink.record(
        name="ha.open",
        value=1.0,
        symbol=SYMBOL,
        timestamp=1,
        owner_strategy_id="a",
        render_type="candle",
        group="ha",
        role="open",
    )
    sink.record(
        name="ema", value=2.0, symbol=SYMBOL, timestamp=1, owner_strategy_id="a"
    )
    sink.flush_stream_snapshot()
    assert (payloads[0]["group"], payloads[0]["role"]) == ("ha", "open")
    assert (payloads[1]["group"], payloads[1]["role"]) == ("", "")
    items = json.loads(payloads[2]["items_json"])
    assert [(i["group"], i["role"]) for i in items] == [("ha", "open"), ("", "")]


def test_v15_events_default_group_role_to_empty() -> None:
    """1.5 事件没有 group/role: 桥接后为空串."""
    point = aq.to_indicator_message(
        {
            "run_id": "r",
            "seq": 1,
            "ts": 0,
            "event_type": "indicator_point",
            "symbol": SYMBOL,
            "level": "info",
            "payload": {"indicator_key": "ema", "value": "1.0", "timestamp": "1"},
        }
    )
    assert point is not None
    assert (point["indicator"]["group"], point["indicator"]["role"]) == ("", "")
    snap = aq.to_indicator_message(
        {
            "run_id": "r",
            "seq": 2,
            "ts": 0,
            "event_type": "indicator_snapshot",
            "symbol": SYMBOL,
            "level": "info",
            "payload": {
                "timestamp": "1",
                "items_json": json.dumps([{"indicator_key": "ema", "value": 1.0}]),
            },
        }
    )
    assert snap is not None
    item = snap["snapshot"]["items"][0]
    assert (item["group"], item["role"]) == ("", "")


def test_stream_schema_version_is_1_6() -> None:
    """契约升到 1.6."""
    assert aq.STREAM_SCHEMA_VERSION == "1.6"


# ------------------------------------------------------------- 第三方 sink


class _LegacySink:
    """1.5 协议签名的第三方 sink: record 不接受 group/role."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def record(
        self,
        *,
        name: str,
        value: Any,
        symbol: str,
        timestamp: Any,
        owner_strategy_id: str,
        display_name: Optional[str] = None,
        pane: int = 0,
        render_type: str = "line",
        unit: Optional[str] = None,
        precision: Optional[int] = None,
        color: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        reference_lines: Optional[list[Dict[str, Any]]] = None,
        scale_group: Optional[str] = None,
        warmup: bool = False,
        confirmed: bool = True,
        phase: str = "run",
    ) -> None:
        self.calls.append(
            {"name": name, "render_type": render_type, "display_name": display_name}
        )

    def build_payload(self) -> Dict[str, list[Dict[str, Any]]]:
        return {"definitions": [], "instances": [], "points": []}

    def flush_stream_snapshot(self) -> None:
        return None

    def set_stream_emitter(self, stream_emitter: Any) -> None:
        return None


class _KwargsSink(_LegacySink):
    """接受 **kwargs 的 sink: 能收到 group/role."""

    def record(self, **kwargs: Any) -> None:  # type: ignore[override]
        self.calls.append(kwargs)


class HAMain(Strategy):
    """主策略里声明 candle(plot=True)."""

    def on_start(self) -> None:
        """声明 candle."""
        self.ha = self.I(
            aq.HeikinAshi(),
            name="ha",
            input_mode="ohlc",
            outputs=OHLC,
            render_type="candle",
            plot=True,
        )

    def on_bar(self, bar: Bar) -> None:
        """不交易."""


def _run_with_sink(sink: Any) -> None:
    run_backtest(
        strategy=HAMain,
        data=_md(),
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
        indicator_recorder=sink,
    )


def test_legacy_sink_gets_line_without_group_role(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """旧 sink 不认 group/role: 降级为 line, 显示名用分量键, 只告警一次."""
    sink = _LegacySink()
    with caplog.at_level(logging.WARNING):
        _run_with_sink(sink)
    assert sink.calls
    assert {c["render_type"] for c in sink.calls} == {"line"}
    # 四条线靠分量键区分, 不共用 label
    for call in sink.calls:
        assert call["display_name"] == call["name"]
    assert {c["display_name"] for c in sink.calls} == {f"ha.{r}" for r in OHLC}
    warned = [r for r in caplog.records if "group=/role=" in r.getMessage()]
    assert len(warned) == 1


def test_kwargs_sink_receives_group_role() -> None:
    """接受 **kwargs 的 sink 收到 render_type=candle 与 group/role."""
    sink = _KwargsSink()
    _run_with_sink(sink)
    assert sink.calls
    for call in sink.calls:
        assert call["render_type"] == "candle"
        assert call["group"] == "ha"
        assert call["name"] == f"ha.{call['role']}"


# ------------------------------------------------------------- intrabar


_BASE_NS = int(pd.Timestamp("2024-01-02 09:31:00", tz="Asia/Shanghai").value)


class _HAIntrabar(Strategy):
    """基础周期 HeikinAshi 开 intrabar: tick 上试算, bar 闭合确认."""

    def __init__(self) -> None:
        super().__init__()
        self.tick_rows: List[tuple[Any, bool]] = []

    def on_start(self) -> None:
        self.ha = self.I(
            aq.HeikinAshi(),
            name="ha",
            input_mode="ohlc",
            outputs=OHLC,
            render_type="candle",
            intrabar=True,
            plot=True,
        )

    def on_tick(self, tick: Tick) -> None:
        self.tick_rows.append((self.ha.close[0], self.ha.confirmed))


def test_intrabar_heikin_ashi_candle() -> None:
    """开 intrabar 时 tick 上有临时四分量, 流里的临时点也带 group/role."""
    ticks = [
        Tick(
            timestamp=_BASE_NS + i * 30_000_000_000,
            price=10.0 + (i % 4) * 0.5,
            volume=100.0,
            symbol=SYMBOL,
        )
        for i in range(13)
    ]
    events: List[Any] = []
    result = run_backtest(
        strategy=_HAIntrabar,
        data=ticks,
        freq="1min",
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        on_event=events.append,
    )
    probe = result.strategy
    assert probe is not None
    provisional = [v for v, confirmed in probe.tick_rows if not confirmed]
    assert provisional and all(v is not None for v in provisional)
    points = [m["indicator"] for m in _messages(events) if m["type"] == "point"]
    temp = [p for p in points if not p["confirmed"]]
    assert temp
    assert {p["role"] for p in temp} == set(OHLC)
    assert {p["group"] for p in temp} == {"ha"}
