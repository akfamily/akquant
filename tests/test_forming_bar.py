"""行情源推来的未收盘 K 线(``DataFeed.add_forming_bar``).

设计见 ``docs/superpowers/specs/2026-11-24-forming-bar-preview-design.md``。
用真实 ``Engine`` 跑 ``replay`` 的有界会话; 另注册一个包装 replay 的测试 broker,
把 feed 换成代理: 推某根收盘 bar 之前(或之后)先推脚本里给它安排的未收盘 bar。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar, DataFeed
from akquant.gateway import GatewayBundle
from akquant.gateway.brokers.replay.gateway import build_replay_bundle
from akquant.gateway.registry import register_broker, unregister_broker

_SYMBOL = "600000"
_BROKER = "replay_forming_test"
_Key = Tuple[str, int]


def _instrument(symbol: str = _SYMBOL) -> Instrument:
    return Instrument(
        symbol=symbol,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bars(count: int = 8, symbol: str = _SYMBOL, base: float = 10.0) -> List[Bar]:
    """生成 count 根 1 分钟 bar: 第 i 根收盘价 base+i, 最高 +0.5, 最低 -0.5."""
    stamps = pd.date_range(
        "2024-01-02 09:31:00", periods=count, freq="1min", tz="Asia/Shanghai"
    )
    return [
        Bar(
            timestamp=int(ts.value),
            open=base + i - 0.1,
            high=base + i + 0.5,
            low=base + i - 0.5,
            close=base + i,
            volume=100.0,
            symbol=symbol,
        )
        for i, ts in enumerate(stamps)
    ]


def _forming(bar: Bar, close: Optional[float] = None) -> Bar:
    """与 ``bar`` 同一桶的未收盘 bar; 不给 close 时 OHLCV 与收盘 bar 完全相同."""
    price = float(bar.close) if close is None else float(close)
    return Bar(
        timestamp=int(bar.timestamp),
        open=float(bar.open),
        high=max(float(bar.high), price),
        low=min(float(bar.low), price),
        close=price,
        volume=float(bar.volume),
        symbol=str(bar.symbol),
    )


def _key(bar: Bar) -> _Key:
    return str(bar.symbol), int(bar.timestamp)


@dataclass
class _Script:
    """收盘 bar → 推它之前 / 之后要推的未收盘 bar."""

    before: Dict[_Key, List[Bar]] = field(default_factory=dict)
    after: Dict[_Key, List[Bar]] = field(default_factory=dict)


class _FormingFeed:
    """包住真实 feed: 只在 add_bar 前后插入脚本里的未收盘 bar."""

    def __init__(self, feed: Any, script: _Script) -> None:
        self._feed = feed
        self._script = script

    def add_bar(self, bar: Bar) -> None:
        for forming in self._script.before.get(_key(bar), []):
            self._feed.add_forming_bar(forming)
        self._feed.add_bar(bar)
        for forming in self._script.after.get(_key(bar), []):
            self._feed.add_forming_bar(forming)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._feed, name)


@pytest.fixture
def script() -> Any:
    """注册包装 replay 的测试 broker, 用后注销以免污染全局 registry."""
    current = _Script()

    def _build(
        feed: Any, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
    ) -> GatewayBundle:
        return build_replay_bundle(
            _FormingFeed(feed, current), symbols, use_aggregator, **kwargs
        )

    register_broker(_BROKER, _build)
    yield current
    unregister_broker(_BROKER)


def _run(
    strategy: Strategy,
    bars: List[Bar],
    *,
    on_event: Any = None,
    instruments: Optional[List[Instrument]] = None,
    **kwargs: Any,
) -> None:
    run_live(
        strategy_cls=strategy,
        instruments=instruments or [_instrument()],
        broker=_BROKER,
        trading_mode="paper",
        gateway_options={"bars": bars, "freq": "1min"},
        cash=1_000_000.0,
        show_progress=False,
        duration="30s",
        on_event=on_event,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 引擎: 入队、派发、与行情隔离(F1、F2)
# ---------------------------------------------------------------------------


def test_add_forming_bar_rejected_on_backtest_feed() -> None:
    """回测数据源没有未收盘的 K 线, 调用即报错."""
    with pytest.raises(RuntimeError, match="add_forming_bar"):
        DataFeed().add_forming_bar(_bars(1)[0])


class _HookProbe(Strategy):
    """把引擎派发的未收盘 bar 与 on_bar 记进同一条日志."""

    def __init__(self) -> None:
        super().__init__()
        self.records: List[Tuple[str, int, float]] = []

    def on_bar(self, bar: Bar) -> None:
        self.records.append(("bar", int(bar.timestamp), float(bar.close)))

    def _on_forming_bar_event(self, bar: Bar) -> None:
        self.records.append(("forming", int(bar.timestamp), float(bar.close)))


def test_forming_bar_reaches_strategy_in_feed_order_without_on_bar(
    script: _Script,
) -> None:
    """未收盘 bar 按入队顺序交给策略内部入口, 不触发 on_bar."""
    bars = _bars()
    script.before[_key(bars[3])] = [_forming(bars[3], 99.0), _forming(bars[3], 98.0)]
    probe = _HookProbe()

    _run(probe, bars)

    ts3 = int(bars[3].timestamp)
    expected: List[Tuple[str, int, float]] = []
    for i, bar in enumerate(bars):
        if i == 3:
            expected += [("forming", ts3, 99.0), ("forming", ts3, 98.0)]
        expected.append(("bar", int(bar.timestamp), float(bar.close)))
    assert probe.records == expected


class _Trader(Strategy):
    """第 2 根买入、第 5 根卖出, 逐根记下引擎状态."""

    def __init__(self) -> None:
        super().__init__()
        self.set_history_depth(5)
        self.seen = 0
        self.records: List[Tuple[int, float, float, float, List[float]]] = []

    def on_bar(self, bar: Bar) -> None:
        self.seen += 1
        if self.seen == 2:
            self.buy(bar.symbol, 100)
        if self.seen == 5:
            self.sell(bar.symbol, 100)
        history = self.get_history(min(self.seen, 3), bar.symbol)
        self.records.append(
            (
                int(bar.timestamp),
                float(bar.close),
                float(self.get_position(bar.symbol)),
                round(float(self.cash), 6),
                [float(value) for value in history],
            )
        )


def test_forming_bars_do_not_change_engine_state(script: _Script) -> None:
    """穿插价格离谱的未收盘 bar, 逐根的历史、持仓、现金与不穿插时完全相同."""
    bars = _bars()
    baseline = _Trader()
    _run(baseline, bars)

    for bar in bars:
        script.before[_key(bar)] = [_forming(bar, 99.0), _forming(bar, 1.0)]
    with_forming = _Trader()
    _run(with_forming, bars)

    assert len(baseline.records) == len(bars)
    assert with_forming.records == baseline.records
    # 买单确实成交过, 否则上面的相等只是两边都没交易。
    assert any(position == 100.0 for _, _, position, _, _ in baseline.records)


# ---------------------------------------------------------------------------
# 策略: intrabar 指标在未收盘 bar 上试算(F3～F7)
# ---------------------------------------------------------------------------


def _collector() -> Tuple[List[Dict[str, Any]], Any]:
    """收集指标点消息(``type == "point"``)的 on_event."""
    points: List[Dict[str, Any]] = []

    def on_event(event: Any) -> None:
        if aq.is_indicator_stream_event(event):
            message = aq.to_indicator_message(event)
            if message is not None and message["type"] == "point":
                points.append(message["indicator"])

    return points, on_event


def _confirmed(points: List[Dict[str, Any]]) -> List[Tuple[str, str, int, float]]:
    return [
        (p["symbol"], p["indicator_key"], p["timestamp"], p["value"])
        for p in points
        if p["confirmed"]
    ]


def _provisional(points: List[Dict[str, Any]]) -> List[Tuple[str, str, int, float]]:
    return [
        (p["symbol"], p["indicator_key"], p["timestamp"], p["value"])
        for p in points
        if not p["confirmed"]
    ]


def _assert_same_points(
    actual: List[Tuple[str, str, int, float]],
    expected: List[Tuple[str, str, int, float]],
) -> None:
    """标的/键/时间戳精确相等, 数值按浮点近似比较(approx 不支持含字符串的元组)."""
    assert [point[:3] for point in actual] == [point[:3] for point in expected]
    assert [point[3] for point in actual] == pytest.approx(
        [point[3] for point in expected]
    )


class _Sma(Strategy):
    period = 2
    intrabar = True

    def on_start(self) -> None:
        self.sma = self.I(
            aq.SMA(self.period), name="sma", intrabar=self.intrabar, plot=True
        )

    def on_bar(self, bar: Bar) -> None:
        pass


class _Macd(Strategy):
    def on_start(self) -> None:
        self.macd = self.I(
            aq.MACD(2, 3, 2),
            name="macd",
            outputs=("dif", "dea", "hist"),
            intrabar=True,
            plot=True,
        )

    def on_bar(self, bar: Bar) -> None:
        pass


class _Atr(Strategy):
    def on_start(self) -> None:
        self.atr = self.I(
            aq.ATR(3), name="atr", input_mode="hlc", intrabar=True, plot=True
        )

    def on_bar(self, bar: Bar) -> None:
        pass


@pytest.mark.parametrize(
    ("strategy_cls", "series_count"), [(_Sma, 1), (_Macd, 3), (_Atr, 1)]
)
def test_identical_forming_bar_previews_the_confirmed_value(
    script: _Script, strategy_cls: Any, series_count: int
) -> None:
    """未收盘 bar 与最终收盘 bar 相同时, 试算值等于随后的确认值, 且先于它到达."""
    bars = _bars(12)
    for bar in bars:
        script.before[_key(bar)] = [_forming(bar)]
    points, on_event = _collector()

    _run(strategy_cls(), bars, on_event=on_event)

    provisional = _provisional(points)
    confirmed = _confirmed(points)
    assert provisional, "应产出试算点"
    assert len({key for _, key, _, _ in provisional}) == series_count
    _assert_same_points(sorted(provisional), sorted(confirmed))
    for index, point in enumerate(points):
        if point["confirmed"]:
            continue
        later = [
            p
            for p in points[index + 1 :]
            if p["confirmed"]
            and p["indicator_key"] == point["indicator_key"]
            and p["timestamp"] == point["timestamp"]
        ]
        assert later, f"试算点之后没有同时间戳的确认点: {point}"


def test_forming_bars_do_not_pollute_confirmed_values(script: _Script) -> None:
    """穿插任意未收盘 bar, 确认值序列与不穿插时逐点相同; 试算值按未收盘价算."""
    bars = _bars()
    baseline_points, baseline_on_event = _collector()
    _run(_Sma(), bars, on_event=baseline_on_event)

    for bar in bars:
        script.before[_key(bar)] = [_forming(bar, 50.0), _forming(bar, 99.0)]
    points, on_event = _collector()
    _run(_Sma(), bars, on_event=on_event)

    assert _confirmed(baseline_points), "基线应有确认点"
    _assert_same_points(_confirmed(points), _confirmed(baseline_points))
    assert not _provisional(baseline_points)
    # SMA(2): 第 i 根(i>=1)上的试算值 = (上一根收盘 + 未收盘价) / 2。
    expected = []
    for i in range(1, len(bars)):
        for price in (50.0, 99.0):
            expected.append(
                (
                    _SYMBOL,
                    "sma",
                    int(bars[i].timestamp),
                    (bars[i - 1].close + price) / 2,
                )
            )
    _assert_same_points(_provisional(points), expected)


def test_preview_appears_before_first_confirmed_value(script: _Script) -> None:
    """确认 bar 还差一根就绪时, 未收盘 bar 上就能出首个试算点."""

    class _Sma3(_Sma):
        period = 3

    bars = _bars()
    script.before[_key(bars[2])] = [_forming(bars[2])]
    points, on_event = _collector()

    _run(_Sma3(), bars, on_event=on_event)

    assert points[0]["confirmed"] is False
    assert points[0]["timestamp"] == int(bars[2].timestamp)
    assert points[0]["value"] == pytest.approx(11.0)


def test_indicator_without_intrabar_ignores_forming_bars(script: _Script) -> None:
    """未声明 intrabar 的指标不出试算点."""

    class _Plain(_Sma):
        intrabar = False

    bars = _bars()
    for bar in bars:
        script.before[_key(bar)] = [_forming(bar, 99.0)]
    points, on_event = _collector()

    _run(_Plain(), bars, on_event=on_event)

    assert _confirmed(points)
    assert not _provisional(points)


def test_stale_forming_bar_is_dropped(script: _Script, caplog: Any) -> None:
    """时间戳不晚于最近确认 bar 的未收盘 bar 直接丢弃."""
    bars = _bars()
    script.after[_key(bars[3])] = [_forming(bars[3], 99.0)]
    points, on_event = _collector()

    with caplog.at_level("INFO"):
        _run(_Sma(), bars, on_event=on_event)

    assert _confirmed(points)
    assert not _provisional(points)
    stale_logs = [r for r in caplog.records if "已丢弃" in r.getMessage()]
    assert len(stale_logs) == 1
    assert _SYMBOL in stale_logs[0].getMessage()


def test_preloaded_history_counts_for_the_stale_rule(script: _Script) -> None:
    """预热回放的历史 bar 也算已确认, 与预热末根同戳的未收盘 bar 被丢弃."""
    bars = _bars()
    history = [
        Bar(
            timestamp=int(pd.Timestamp(stamp, tz="Asia/Shanghai").value),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=100.0,
            symbol=_SYMBOL,
        )
        for stamp, close in (
            ("2024-01-02 09:28:00", 1.0),
            ("2024-01-02 09:29:00", 2.0),
            ("2024-01-02 09:30:00", 3.0),
        )
    ]
    script.before[_key(bars[0])] = [_forming(history[-1], 99.0), _forming(bars[0])]
    points, on_event = _collector()

    _run(_Sma(), bars, on_event=on_event, preload_history=history)

    # SMA(2): 预热末根收盘 3.0, 首个实盘桶收盘 10.0 → 试算值 6.5。
    _assert_same_points(
        _provisional(points), [(_SYMBOL, "sma", int(bars[0].timestamp), 6.5)]
    )


class _ReadProbe(_Sma):
    """在未收盘 bar 入口与 on_bar 里读 [0] / [1] / confirmed."""

    def __init__(self) -> None:
        super().__init__()
        self.forming_reads: List[Tuple[float, float, bool]] = []
        self.bar_reads: List[Tuple[int, float, bool]] = []

    def _on_forming_bar_event(self, bar: Bar) -> None:
        super()._on_forming_bar_event(bar)
        view = self.sma.of(str(bar.symbol))
        self.forming_reads.append((view[0], view[1], view.confirmed))

    def on_bar(self, bar: Bar) -> None:
        if self.sma[0] is not None:
            self.bar_reads.append((int(bar.timestamp), self.sma[0], self.sma.confirmed))


def test_provisional_value_occupies_index_zero_until_bar_closes(
    script: _Script,
) -> None:
    """有临时值时 [0] 是临时值、[1] 是上一根确认值; bar 闭合后恢复为确认值."""
    bars = _bars()
    script.before[_key(bars[3])] = [_forming(bars[3], 99.0)]
    probe = _ReadProbe()

    _run(probe, bars)

    # 第 2 根确认值 (11+12)/2=11.5; 第 3 根上的试算 (12+99)/2=55.5, 确认 12.5。
    assert probe.forming_reads == [(pytest.approx(55.5), pytest.approx(11.5), False)]
    read3 = [r for r in probe.bar_reads if r[0] == int(bars[3].timestamp)]
    assert read3 == [(int(bars[3].timestamp), pytest.approx(12.5), True)]


def test_forming_bar_only_affects_its_own_symbol(script: _Script) -> None:
    """多标的: 一个标的的未收盘 bar 不影响另一个标的."""

    class _Multi(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(
                factory=lambda: aq.SMA(2), name="sma", intrabar=True, plot=True
            )

        def on_bar(self, bar: Bar) -> None:
            pass

    other = "600001"
    bars = _bars() + _bars(symbol=other, base=20.0)
    baseline_points, baseline_on_event = _collector()
    instruments = [_instrument(), _instrument(other)]
    _run(_Multi(), bars, on_event=baseline_on_event, instruments=instruments)

    for bar in _bars():
        script.before[_key(bar)] = [_forming(bar, 99.0)]
    points, on_event = _collector()
    _run(_Multi(), bars, on_event=on_event, instruments=instruments)

    assert {symbol for symbol, _, _, _ in _provisional(points)} == {_SYMBOL}
    _assert_same_points(sorted(_confirmed(points)), sorted(_confirmed(baseline_points)))


def test_slot_strategies_each_preview_forming_bars(script: _Script) -> None:
    """主策略与槽位策略各自试算, 试算点带各自的 owner."""
    bars = _bars()
    for bar in bars:
        script.before[_key(bar)] = [_forming(bar, 99.0)]
    points, on_event = _collector()

    _run(_Sma(), bars, on_event=on_event, strategies_by_slot={"slot_a": _Sma()})

    owners = {p["owner_strategy_id"] for p in points if not p["confirmed"]}
    assert len(owners) == 2, owners


def test_failing_preview_does_not_stop_the_session(
    script: _Script, caplog: Any
) -> None:
    """试算抛异常只告警一次并跳过, 确认值照常产出."""

    class _Fragile:
        """价格超过 50 就抛错的自写指标."""

        def __init__(self) -> None:
            self.last: Optional[float] = None

        def update(self, value: float) -> None:
            if value > 50.0:
                raise RuntimeError("boom")
            self.last = float(value)

        @property
        def value(self) -> Optional[float]:
            return self.last

    class _UsesFragile(Strategy):
        def on_start(self) -> None:
            self.fragile = self.I(_Fragile(), name="fragile", intrabar=True, plot=True)

        def on_bar(self, bar: Bar) -> None:
            pass

    bars = _bars()
    for bar in bars:
        script.before[_key(bar)] = [_forming(bar, 99.0)]
    points, on_event = _collector()

    with caplog.at_level("WARNING"):
        _run(_UsesFragile(), bars, on_event=on_event)

    assert [value for _, _, _, value in _confirmed(points)] == pytest.approx(
        [float(bar.close) for bar in bars]
    )
    assert not _provisional(points)
    warnings = [r for r in caplog.records if "未收盘 K 线" in r.getMessage()]
    assert len(warnings) == 1
    assert "fragile" in warnings[0].getMessage()
