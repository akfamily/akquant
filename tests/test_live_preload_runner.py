"""run_live(preload_history=) 端到端: 历史段与实时段由同一会话连续算出.

用真实 ``Engine`` 跑 ``broker='replay'`` 的有界会话(事件放完自动结束, 无需柜台)。
设计见 ``docs/superpowers/specs/2026-08-27-live-history-preload-design.md`` 的
「2026-10-14 修订」一节。
"""

import json
import logging
from typing import Any, Dict, List, Optional

import akquant as aq
import numpy as np
import pandas as pd
import pytest
from akquant import AssetType, Engine, Instrument, Strategy, run_live
from akquant.akquant import Bar
from akquant.backtest import BacktestStreamEvent
from akquant.indicator_stream import to_indicator_message

_SYMBOL = "600000.SH"


def _instrument(symbol: str = _SYMBOL) -> Instrument:
    return Instrument(
        symbol=symbol,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _preload_frame(closes: tuple[float, ...] = (10.0, 11.0, 12.0)) -> pd.DataFrame:
    """三根历史日线, 收盘价可辨认."""
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-08-20 15:00", periods=len(closes), freq="D", tz="Asia/Shanghai"
            ),
            "symbol": [_SYMBOL] * len(closes),
            "open": list(closes),
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": list(closes),
            "volume": [1000.0] * len(closes),
        }
    )


def _bar(close: float, stamp: pd.Timestamp, symbol: str = _SYMBOL) -> Bar:
    return Bar(
        timestamp=int(stamp.value),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
        symbol=symbol,
    )


def _live_bar(close: float = 20.0, stamp: str = "2026-08-25 15:00:00") -> Bar:
    return _bar(close, pd.Timestamp(stamp, tz="Asia/Shanghai"))


def _run(
    strategy: Strategy,
    *,
    live_bars: List[Bar],
    preload: Any = None,
    freq: str = "1d",
    on_event: Any = None,
    indicator_recorder: Any = None,
) -> None:
    kwargs: Dict[str, Any] = {}
    if preload is not None:
        kwargs["preload_history"] = preload
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": live_bars, "freq": freq},
        cash=1_000_000.0,
        duration="60s",
        on_event=on_event,
        indicator_recorder=indicator_recorder,
        **kwargs,
    )


class _HistoryProbe(Strategy):
    """首个 on_bar 里抓一次历史; 刻意不调 set_history_depth."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: List[Any] = []
        self.error: Optional[Exception] = None
        self.on_bar_calls = 0

    def on_bar(self, bar: Any) -> None:
        self.on_bar_calls += 1
        if self.captured or self.error is not None:
            return
        try:
            self.captured.append(list(self.get_history(3, field="close")))
        except Exception as exc:  # noqa: BLE001 - 要把异常带出来断言
            self.error = exc


def test_preload_makes_history_available_in_first_on_bar() -> None:
    """预热后首根实盘 bar 里 get_history 就有数据, 且无需手动 set_history_depth.

    未显式设深度时 ``_history_depth`` 被抬到预热深度(3), 窗口按 3 滚动:
    首根实盘 bar 进来后是 [11, 12, 20]。
    """
    strategy = _HistoryProbe()
    _run(strategy, live_bars=[_live_bar()], preload=_preload_frame())

    assert strategy.error is None, f"get_history 抛了异常: {strategy.error!r}"
    assert strategy.captured, "策略没收到 on_bar"
    values = strategy.captured[0]
    assert not any(np.isnan(v) for v in values), f"历史里仍有 NaN: {values}"
    assert values == [11.0, 12.0, 20.0]


class _DeepHistoryProbe(_HistoryProbe):
    def __init__(self) -> None:
        super().__init__()
        self.set_history_depth(10)

    def on_bar(self, bar: Any) -> None:
        self.on_bar_calls += 1
        if not self.captured:
            self.captured.append(list(self.get_history(4, field="close")))


def test_preload_keeps_larger_explicit_history_depth() -> None:
    """用户显式设的更大深度不被下调: 预热 3 根 + 实盘 1 根全部可见."""
    strategy = _DeepHistoryProbe()
    _run(strategy, live_bars=[_live_bar()], preload=_preload_frame())

    assert strategy.captured == [[10.0, 11.0, 12.0, 20.0]]


def test_preload_does_not_drive_on_bar() -> None:
    """预热只填历史与指标, 不驱动 on_bar(零下单风险)."""
    strategy = _HistoryProbe()
    _run(strategy, live_bars=[_live_bar()], preload=_preload_frame())

    assert strategy.on_bar_calls == 1


class _WarmupProbe(_HistoryProbe):
    warmup_period = 3


def test_preload_bars_count_toward_warmup_period() -> None:
    """预热根数计入 per-symbol warmup: 攒够的策略从首根实盘 bar 就能交易."""
    strategy = _WarmupProbe()
    _run(strategy, live_bars=[_live_bar()], preload=_preload_frame())

    assert strategy.on_bar_calls == 1, "预热 3 根 + warmup_period=3, 首根应放行"


class _DeclaredSma(Strategy):
    """on_start 里声明(参数化指标的标准写法), 主策略 on_start 在 engine.run 内触发."""

    def __init__(self) -> None:
        super().__init__()
        self.prev_at_first_bar: List[Any] = []

    def on_start(self) -> None:
        self.sma = self.I(aq.SMA(2), name="sma2", pane=0, color="#e91e63", plot=True)

    def on_bar(self, bar: Any) -> None:
        if not self.prev_at_first_bar:
            self.prev_at_first_bar.append((self.sma[0], self.sma[1]))


def _indicator_points(events: List[BacktestStreamEvent]) -> List[Dict[str, Any]]:
    messages = [to_indicator_message(e) for e in events]
    return [m["indicator"] for m in messages if m is not None and "indicator" in m]


def test_preload_replays_declared_indicators_into_stream() -> None:
    """on_start 里声明的指标吃到整段预热, 历史点带 phase=preload 进流.

    实时段的首个点由同一实例接着算: SMA(2) 在实盘首根 = (12 + 20) / 2。
    若历史段与实时段是两套实例(回测+实盘拼接), 这里会是 None(未满窗)。
    """
    events: List[BacktestStreamEvent] = []
    strategy = _DeclaredSma()
    _run(
        strategy,
        live_bars=[_live_bar()],
        preload=_preload_frame(),
        on_event=events.append,
    )

    points = [p for p in _indicator_points(events) if p["indicator_key"] == "sma2"]
    assert [(p["value"], p["phase"]) for p in points] == [
        (10.5, "preload"),
        (11.5, "preload"),
        (16.0, "run"),
    ]
    assert strategy.prev_at_first_bar == [(16.0, 11.5)], "[1] 应回溯到预热末值"


def test_preload_snapshot_carries_phase() -> None:
    """Snapshot 与 point 同口径: 预热段 snapshot 也带 phase=preload."""
    events: List[BacktestStreamEvent] = []
    _run(
        _DeclaredSma(),
        live_bars=[_live_bar()],
        preload=_preload_frame(),
        on_event=events.append,
    )

    messages = [to_indicator_message(e) for e in events]
    snaps = [m["snapshot"] for m in messages if m is not None and "snapshot" in m]
    assert [s["phase"] for s in snaps] == ["preload", "preload", "run"]
    assert all(
        m["schema_version"] == aq.STREAM_SCHEMA_VERSION
        for m in messages
        if m is not None
    )
    # 每根预热 bar 一个快照周期: 前端按时间顺序增量画, 不会等预热全部回放完
    # 才一次性收到所有快照。
    kinds = [m["type"] for m in messages if m is not None]
    assert kinds == ["point", "snapshot"] * 3


def test_no_preload_keeps_existing_behaviour(caplog: Any) -> None:
    """不传 preload_history 时: 不碰 history depth, 指标点 phase 恒为 run."""
    events: List[BacktestStreamEvent] = []
    strategy = _DeclaredSma()
    with caplog.at_level(logging.INFO):
        _run(strategy, live_bars=[_live_bar()], on_event=events.append)

    assert [r for r in caplog.records if "历史预热" in r.getMessage()] == []
    assert int(getattr(strategy, "_history_depth", 0) or 0) == 0
    assert {p["phase"] for p in _indicator_points(events)} <= {"run"}


class _WindowSma(Strategy):
    """窗口周期指标: 预热期间由 Rust 聚合出的窗口 bar 驱动."""

    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars("5min")

    def on_start(self) -> None:
        self.wsma = self.I(aq.SMA(2), name="wsma", freq="5min", pane=0, plot=True)

    def on_bar(self, bar: Any) -> None:
        pass


def test_preload_replays_window_freq_indicators() -> None:
    """10 根 1 分钟预热闭合 2 根 5 分钟窗口, 窗口指标在预热期就产出点."""
    base = pd.Timestamp("2024-01-02 09:31:00", tz="Asia/Shanghai")
    closes = [10.0 + i for i in range(10)]
    preload = [_bar(c, base + pd.Timedelta(minutes=i)) for i, c in enumerate(closes)]
    live = [_bar(30.0, base + pd.Timedelta(minutes=10))]
    events: List[BacktestStreamEvent] = []
    _run(
        _WindowSma(),
        live_bars=live,
        preload=preload,
        freq="1min",
        on_event=events.append,
    )

    points = [p for p in _indicator_points(events) if p["indicator_key"] == "wsma"]
    # 预热窗口收盘 14, 19 → SMA(2) = 16.5; 会话结束时形成中的 09:41 窗口被尾部
    # flush(收盘 30), 同一实例接着算出 (19 + 30) / 2 —— 预热与实时无缝。
    assert [(p["value"], p["phase"]) for p in points] == [
        (16.5, "preload"),
        (24.5, "run"),
    ]
    label = base + pd.Timedelta(minutes=9)
    assert points[0]["timestamp"] == int(label.value)


class _LegacySink:
    """1.3 协议的第三方 sink: record 不认 phase=."""

    def __init__(self) -> None:
        self.values: List[float] = []

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
    ) -> None:
        self.values.append(float(value))

    def build_payload(self) -> Dict[str, list[Dict[str, Any]]]:
        return {"definitions": [], "instances": [], "points": []}

    def flush_stream_snapshot(self) -> None:
        return None

    def set_stream_emitter(self, stream_emitter: Any) -> None:
        return None


def test_legacy_sink_without_phase_still_receives_preload_points() -> None:
    """旧协议 sink 不认 phase=: 预热点照常送达(都是确认值), 不 TypeError."""
    sink = _LegacySink()
    _run(
        _DeclaredSma(),
        live_bars=[_live_bar()],
        preload=_preload_frame(),
        indicator_recorder=sink,
    )

    assert sink.values == [10.5, 11.5, 16.0]


def test_overlapping_first_bar_warns_but_is_not_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """实盘首根与预热末尾同戳 → 点名告警, 但照常派发(重复一根远比丢一根轻)."""
    frame = _preload_frame()
    last_stamp = pd.Timestamp(frame["date"].iloc[-1])
    strategy = _HistoryProbe()
    with caplog.at_level(logging.WARNING):
        _run(strategy, live_bars=[_bar(12.0, last_stamp)], preload=frame)

    overlap = [r for r in caplog.records if "与预热数据重叠" in r.getMessage()]
    assert len(overlap) == 1
    assert _SYMBOL in overlap[0].getMessage()
    assert strategy.on_bar_calls == 1, "重叠的实盘 bar 被吞了 —— 应当只告警不丢"


def test_bridge_defaults_missing_phase_to_run() -> None:
    """1.4 之前的事件与旧协议 sink 没有 phase: 桥接一律当 run."""
    point: BacktestStreamEvent = {
        "run_id": "r",
        "seq": 1,
        "ts": 0,
        "event_type": "indicator_point",
        "symbol": "A",
        "level": "info",
        "payload": {"indicator_key": "x", "timestamp": "1", "value": "1.0"},
    }
    snapshot: BacktestStreamEvent = {
        **point,
        "event_type": "indicator_snapshot",
        "payload": {
            "timestamp": "1",
            "items_json": json.dumps([{"indicator_key": "x", "value": 1.0}]),
        },
    }

    point_msg = to_indicator_message(point)
    snap_msg = to_indicator_message(snapshot)

    assert point_msg is not None and point_msg["indicator"]["phase"] == "run"
    assert snap_msg is not None and snap_msg["snapshot"]["phase"] == "run"
    assert snap_msg["snapshot"]["items"][0]["phase"] == "run"


def _raw_bar(ts_ns: int, symbol: str = "600000.SH", close: float = 10.0) -> Bar:
    return Bar(ts_ns, close, close, close, close, 100.0, symbol)


def test_engine_preload_history_updates_internal_state() -> None:
    """验证灌入的历史数据确实被写进引擎状态.

    历史缓冲通过 get_state_bytes 随引擎状态一起序列化(src/engine/python.rs:622-624),
    因此可以通过检查序列化后是否含特定标的来验证数据已进缓冲,
    无需为测试新增生产接口。
    """
    engine = Engine()
    before = engine.get_state_bytes()
    base = 1_000_000_000_000_000_000
    bars = [_raw_bar(base + i, symbol="600000.SH", close=10.0 + i) for i in range(3)]
    events = engine.preload_history(bars)
    after = engine.get_state_bytes()

    # 验证数据确实进了缓冲
    assert b"600000.SH" not in before
    assert b"600000.SH" in after
    # 无窗口订阅时回放序列就是输入本身
    assert [b.close for b in events] == [10.0, 11.0, 12.0]

    # 空输入是 no-op, 不该抛
    assert engine.preload_history([]) == []
