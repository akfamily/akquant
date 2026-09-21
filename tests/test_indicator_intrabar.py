"""L2 Tier A: 窗口周期指标在未闭合窗口上的实时值(intrabar).

设计见 docs/zh/meta/indicator-tradingview-rfc.md §2.3 / §5.1。声明
``self.I(..., freq="5min", intrabar=True)`` 后, 每根基础 bar 闭合时用
``ctx.current_window()`` 的未闭合窗口快照**试算**一个临时值(``confirmed=False``),
窗口闭合时真 ``update()``(``confirmed=True``)。``[0]`` 语义对齐 Pine: 有临时值时
``[0]`` 是临时值、``[1]`` 是上一根已确认。

试算不能污染增量状态: Rust 内建指标经 ``akquant.clone_indicator`` 复制,
用户自写 Python 指标经 ``copy.deepcopy``。
"""

from __future__ import annotations

from collections import deque
from typing import Any, List, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest
from akquant.akquant import Tick
from akquant.indicator_declaration import peek_indicator

SYMBOL = "INTRA"
_BASE_NS = int(pd.Timestamp("2024-01-02 09:31:00", tz="Asia/Shanghai").value)
_MIN_NS = 60_000_000_000


# --------------------------------------------------------------------------
# peek: 试算不提交
# --------------------------------------------------------------------------
def test_clone_indicator_copies_rust_state_independently() -> None:
    """clone_indicator 得到状态相同的新对象, 副本 update 不影响原对象."""
    sma = aq.SMA(2)
    sma.update(10.0)
    sma.update(20.0)

    clone = aq.clone_indicator(sma)
    assert clone is not sma
    assert clone.value == sma.value == 15.0
    clone.update(30.0)
    assert clone.value == 25.0
    assert sma.value == 15.0


def test_clone_indicator_rejects_non_indicator() -> None:
    """非内建指标 fail-fast, 文案指向 copy.deepcopy."""
    with pytest.raises(TypeError, match="deepcopy"):
        aq.clone_indicator(object())


def test_peek_leaves_rust_indicator_untouched() -> None:
    """试算返回"喂入这个值后的结果", 原指标状态不变."""
    ema = aq.EMA(3)
    for v in (10.0, 11.0, 12.0):
        ema.update(v)
    before = ema.value

    peeked = peek_indicator(ema, (20.0,))

    assert peeked is not None and peeked > before
    assert ema.value == before


def test_peek_supports_python_indicator_via_deepcopy() -> None:
    """用户自写指标没有 clone_indicator, 走 copy.deepcopy, 同样不污染."""

    class Mom:
        def __init__(self) -> None:
            self.buf: deque[float] = deque(maxlen=2)

        def update(self, v: float) -> None:
            self.buf.append(v)

        @property
        def value(self) -> Any:
            return None if len(self.buf) < 2 else self.buf[-1] - self.buf[0]

    mom = Mom()
    mom.update(1.0)
    mom.update(3.0)

    assert peek_indicator(mom, (10.0,)) == 7.0
    assert mom.value == 2.0
    assert list(mom.buf) == [1.0, 3.0]


def test_peek_multi_output_indicator() -> None:
    """多值指标 peek 出元组, 原状态不变."""
    macd = aq.MACD(2, 3, 2)
    for v in (10.0, 11.0, 12.0, 11.0, 13.0):
        macd.update(v)
    before = macd.value

    peeked = peek_indicator(macd, (15.0,))

    assert isinstance(peeked, tuple) and len(peeked) == 3
    assert macd.value == before


# --------------------------------------------------------------------------
# 声明校验
# --------------------------------------------------------------------------
def _daily_md_3() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "symbol": SYMBOL,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )


def test_intrabar_on_base_freq_is_allowed() -> None:
    """Tier B: intrabar=True 不再要求 freq= —— 基础周期指标也能在 tick 上试算."""

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), intrabar=True)

        def on_bar(self, bar: Bar) -> None:
            pass

    run_backtest(
        strategy=Probe,
        data=_daily_md_3(),
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
    )


def test_intrabar_rejects_precomputed_indicator() -> None:
    """向量化预计算指标没有增量状态可试算, intrabar=True 直接报错."""

    class Probe(Strategy):
        def on_start(self) -> None:
            self.mom = self.I(
                aq.Indicator("m", lambda df: df["close"].diff()), intrabar=True
            )

        def on_bar(self, bar: Bar) -> None:
            pass

    with pytest.raises(ValueError, match="intrabar"):
        run_backtest(
            strategy=Probe,
            data=_daily_md_3(),
            symbols=[SYMBOL],
            initial_cash=1e5,
            show_progress=False,
            timezone="UTC",
        )


# --------------------------------------------------------------------------
# 端到端: tick 聚合成 1min 基础 bar(基础周期已知) + 5min 窗口
# --------------------------------------------------------------------------
def _ticks(minutes: int) -> List[Tick]:
    """每分钟一笔 tick, 价格线性递增; run_backtest(freq='1min') 把它们聚成基础 bar."""
    return [
        Tick(
            timestamp=_BASE_NS + i * _MIN_NS,
            price=10.0 + i,
            volume=100.0,
            symbol=SYMBOL,
        )
        for i in range(minutes)
    ]


class _IntrabarProbe(Strategy):
    """记录每根基础 bar 上 5min 指标的 [0] / [1] / confirmed."""

    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars("5min")
        # (bar_ts, [0], [1], confirmed, 此刻已确认的窗口个数)
        self.seen: List[Tuple[int, Any, Any, bool, int]] = []
        self.window_closes: List[Tuple[int, Any]] = []

    def on_start(self) -> None:
        self.sma5 = self.I(
            aq.SMA(2), name="sma5", freq="5min", intrabar=True, plot=True
        )

    def on_bar(self, bar: Bar) -> None:
        self.seen.append(
            (
                int(bar.timestamp),
                self.sma5[0],
                self.sma5[1],
                self.sma5.confirmed,
                len(self.window_closes),
            )
        )

    def on_window_bar(self, bar: Bar) -> None:
        self.window_closes.append((int(bar.timestamp), self.sma5[0]))


def _run_intrabar(minutes: int = 16, **kwargs: Any) -> Any:
    return run_backtest(
        strategy=_IntrabarProbe,
        data=_ticks(minutes),
        freq="1min",
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        **kwargs,
    )


def test_intrabar_gives_provisional_value_between_window_closes() -> None:
    """窗口未闭合期间 [0] 是用未闭合快照试算的临时值, confirmed=False."""
    result = _run_intrabar()
    probe = result.strategy

    # 第一个 5min 窗口(09:31-09:35)闭合前 SMA(2) 一个窗口都没攒够, 临时值也拿不到
    # 第二个窗口形成期间(09:36..09:39)SMA(2) 有 1 个确认窗口 + 1 个临时窗口 → 有值
    provisional = [row for row in probe.seen if not row[3] and row[1] is not None]
    assert provisional, "应存在未确认的临时值"
    for ts, v0, v1, _, n_closed in provisional:
        # 价格单调上升 → 试算窗口的 close 高于上一窗口, SMA(2) 临时值必大于上一
        # 根已确认值。(不能断言"≠任何确认值": 闭合前最后一根基础 bar 的快照已含
        # 最终 close, 临时值恰好等于即将确认的值。)
        assert v0 is not None
        if v1 is not None:
            assert v0 > v1, (ts, v0, v1)
        # [1] 必须正好是 on_bar 发生时最近一根已确认窗口的值。按"当时已确认的窗口
        # 个数"取, 不按时间戳比: tick 聚合出的基础 bar 时间戳(09:40:59.999…)与窗口
        # 标签(09:40:00)不对齐, 而 on_bar 先于 on_window_bar, 闭合 bar 自己那根还不算。
        earlier = [v for _, v in probe.window_closes[:n_closed]]
        assert v1 == (earlier[-1] if earlier else None), (ts, v1, earlier)


def test_intrabar_index_one_is_last_confirmed_value() -> None:
    """有临时值时 [1] 是上一根已确认窗口的值 —— 与 Pine 的 realtime bar 语义一致."""
    result = _run_intrabar()
    probe = result.strategy

    confirmed_values = [v for _, v in probe.window_closes if v is not None]
    assert len(confirmed_values) >= 2
    # 找一个临时行, 它的 [1] 必须是某个此前的确认值
    rows = [row for row in probe.seen if not row[3] and row[1] is not None]
    assert rows
    for _, _, v1, _, _ in rows:
        if v1 is not None:
            assert v1 in confirmed_values


def test_intrabar_never_peeks_next_window_before_confirmation() -> None:
    """闭合那根基础 bar 上不拿下一窗口的快照试算 —— 那会跳过刚闭合的整根窗口.

    引擎顺序是 on_bar 先、on_window_bar 后: 在闭合 bar 上 current_window() 已是
    下一窗口, 而上一窗口的确认值还没入账。流里的表现是"标签更大的临时点出现在
    上一标签的确认点之前", 这里断言它不发生。
    """
    messages: List[Any] = []

    def on_event(event: Any) -> None:
        if aq.is_indicator_stream_event(event):
            m = aq.to_indicator_message(event)
            if m is not None and m["type"] == "point":
                messages.append(m["indicator"])

    _run_intrabar(on_event=on_event, stream_batch_size=1)

    seen_confirmed_labels: set[int] = set()
    for m in messages:
        ts = m["timestamp"]
        if m["confirmed"]:
            seen_confirmed_labels.add(ts)
            continue
        # 一个临时点的标签若大于某个尚未确认的更早标签, 就是跳窗
        earlier_unconfirmed = [
            c["timestamp"]
            for c in messages
            if c["confirmed"]
            and c["timestamp"] < ts
            and c["timestamp"] not in seen_confirmed_labels
        ]
        assert not earlier_unconfirmed, (ts, earlier_unconfirmed)


def test_intrabar_does_not_pollute_indicator_state() -> None:
    """临时值不进增量状态: 确认序列与不开 intrabar 时逐点相同."""

    class Plain(_IntrabarProbe):
        def on_start(self) -> None:
            self.sma5 = self.I(aq.SMA(2), name="sma5", freq="5min", plot=True)

    intra_probe = _run_intrabar().strategy
    assert intra_probe is not None
    intra = intra_probe.window_closes
    plain_probe = run_backtest(
        strategy=Plain,
        data=_ticks(16),
        freq="1min",
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
    ).strategy
    assert plain_probe is not None
    plain = plain_probe.window_closes

    assert intra == plain


def test_intrabar_stream_emits_unconfirmed_points_with_window_label() -> None:
    """临时点走流事件 confirmed=false, 时间戳等于窗口闭合标签; 确认点同 time 覆盖."""
    messages: List[Any] = []

    def on_event(event: Any) -> None:
        if aq.is_indicator_stream_event(event):
            m = aq.to_indicator_message(event)
            if m is not None and m["type"] == "point":
                messages.append(m["indicator"])

    _run_intrabar(on_event=on_event, stream_batch_size=1)

    provisional = [m for m in messages if m["confirmed"] is False]
    confirmed = [m for m in messages if m["confirmed"] is True]
    assert provisional and confirmed
    # 每个临时点的 timestamp 都必须能在确认点里找到同 time 的"后来者"
    # (包括数据末尾 flush 的未满窗口: 它的确认点也打窗口标签, 不是基础 bar 时间)
    confirmed_ts = {m["timestamp"] for m in confirmed}
    assert all(m["timestamp"] in confirmed_ts for m in provisional)


def test_window_indicator_confirmed_point_uses_window_label_not_base_bar() -> None:
    """窗口指标的确认点时间戳是窗口标签 —— 尾部 flush 时它与最后一根基础 bar 不同."""
    messages: List[Any] = []

    def on_event(event: Any) -> None:
        if aq.is_indicator_stream_event(event):
            m = aq.to_indicator_message(event)
            if m is not None and m["type"] == "point" and m["indicator"]["confirmed"]:
                messages.append(m["indicator"]["timestamp"])

    result = _run_intrabar(on_event=on_event, stream_batch_size=1)
    probe = result.strategy
    assert probe is not None
    window_labels = [ts for ts, v in probe.window_closes if v is not None]

    assert messages == window_labels


def test_intrabar_provisional_points_stay_out_of_indicator_df() -> None:
    """DataFrame / export 出口只含确认点 —— 临时点只走流, 不累积."""
    result = _run_intrabar()
    frame = result.indicator_df(name="sma5")

    assert not frame.empty
    assert len(frame) == len(
        [v for _, v in result.strategy.window_closes if v is not None]
    )


def test_intrabar_off_by_default_keeps_legacy_behaviour() -> None:
    """默认 intrabar=False: 窗口未闭合时 [0] 仍是上一根确认值, confirmed 恒 True."""

    class Plain(_IntrabarProbe):
        def on_start(self) -> None:
            self.sma5 = self.I(aq.SMA(2), name="sma5", freq="5min")

    probe = run_backtest(
        strategy=Plain,
        data=_ticks(16),
        freq="1min",
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
    ).strategy
    assert probe is not None

    assert all(row[3] for row in probe.seen)
    confirmed_values = {v for _, v in probe.window_closes if v is not None}
    for _, v0, _, _, _ in probe.seen:
        if v0 is not None:
            assert v0 in confirmed_values


def test_live_replay_intrabar_streams_unconfirmed_points() -> None:
    """实盘 replay(网关声明 freq)同样发 confirmed=false 临时点, 后有同 time 确认点."""
    from akquant import run_live
    from akquant.akquant import AssetType, Instrument

    messages: List[Any] = []

    def on_event(event: Any) -> None:
        if aq.is_indicator_stream_event(event):
            m = aq.to_indicator_message(event)
            if m is not None and m["type"] == "point":
                messages.append(m["indicator"])

    idx = pd.date_range(
        "2024-01-02 09:31:00", periods=12, freq="1min", tz="Asia/Shanghai"
    )
    bars = [
        Bar(int(ts.value), 10.0 + i, 10.5 + i, 9.5 + i, 10.1 + i, 100.0, "600000")
        for i, ts in enumerate(idx)
    ]

    class LiveIntrabar(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.subscribe_bars("5min")

        def on_start(self) -> None:
            self.sma5 = self.I(
                aq.SMA(2), name="sma5", freq="5min", intrabar=True, plot=True
            )

        def on_bar(self, bar: Bar) -> None:
            pass

    run_live(
        strategy_cls=LiveIntrabar,
        instruments=[
            Instrument(
                symbol="600000",
                asset_type=AssetType.Stock,
                multiplier=1.0,
                margin_ratio=1.0,
                tick_size=0.01,
                lot_size=100,
                option_type=None,
                strike_price=None,
                expiry_date=None,
            )
        ],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars, "freq": "1min"},
        cash=1e6,
        show_progress=False,
        on_event=on_event,
        duration="30s",
    )

    provisional = [m for m in messages if m["confirmed"] is False]
    confirmed_ts = {m["timestamp"] for m in messages if m["confirmed"] is True}
    assert provisional, "实盘路径应产出临时点"
    assert all(m["timestamp"] in confirmed_ts for m in provisional)


# --------------------------------------------------------------------------
# Tier B: 基础周期指标在 tick 上的实时值(每 30s 一笔 tick, 1min 基础 bar)
# --------------------------------------------------------------------------
from akquant.indicator_declaration import end_stamp_label  # noqa: E402

_HALF_MIN_NS = 30_000_000_000


def _bar_close_of_minute(k: int) -> float:
    """第 k 分钟(0 起)基础 bar 的 close = 该分钟第二笔 tick 的价格."""
    return 10.0 + 0.5 * (2 * k + 1)


def _minute_index(ts: int) -> int:
    return int((ts - _BASE_NS) // 60_000_000_000)


def _half_minute_ticks(count: int) -> List[Tick]:
    """每 30 秒一笔 tick, 价格每笔 +0.5; run_backtest(freq='1min') 聚成基础 bar."""
    return [
        Tick(
            timestamp=_BASE_NS + i * _HALF_MIN_NS,
            price=10.0 + 0.5 * i,
            volume=100.0,
            symbol=SYMBOL,
        )
        for i in range(count)
    ]


def test_end_stamp_label_matches_rust_aggregator_convention() -> None:
    """标签公式 = Rust BarAggregator 的区间末 -1ns 打戳: (ts // 间隔 + 1) * 间隔 - 1."""
    minute = 60_000_000_000
    base = int(pd.Timestamp("2024-01-02 09:31:00", tz="Asia/Shanghai").value)
    expected = base + minute - 1  # 09:31:59.999999999

    assert end_stamp_label(base, minute) == expected
    assert end_stamp_label(base + 30_000_000_000, minute) == expected
    assert end_stamp_label(base + minute, minute) == expected + minute


class _TickIntrabarProbe(Strategy):
    """基础周期 SMA(2) 开 intrabar: on_tick 看临时值, on_bar 看确认值."""

    def __init__(self) -> None:
        super().__init__()
        self.tick_rows: List[Tuple[int, float, Any, Any, bool]] = []
        self.bar_rows: List[Tuple[int, float, Any, bool]] = []

    def on_start(self) -> None:
        self.sma = self.I(aq.SMA(2), name="sma", intrabar=True, plot=True)

    def on_tick(self, tick: Tick) -> None:
        self.tick_rows.append(
            (
                int(tick.timestamp),
                tick.price,
                self.sma[0],
                self.sma[1],
                self.sma.confirmed,
            )
        )

    def on_bar(self, bar: Bar) -> None:
        self.bar_rows.append(
            (int(bar.timestamp), bar.close, self.sma[0], self.sma.confirmed)
        )


def _run_tick_intrabar(ticks: int = 13, **kwargs: Any) -> Any:
    return run_backtest(
        strategy=_TickIntrabarProbe,
        data=_half_minute_ticks(ticks),
        freq="1min",
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        **kwargs,
    )


def test_tick_intrabar_provisional_uses_forming_bar_close() -> None:
    """on_tick 里 [0] = SMA(上一根确认 bar 的 close, 形成中 bar 最新价), 未确认."""
    probe = _run_tick_intrabar().strategy
    assert probe is not None

    # 注意: 自动推断的 warmup 会吞掉首根 bar 的 on_bar(指标照常更新), 所以
    # "上一根 bar 的 close" 从 tick 序列推导, 不从 bar_rows 反推。
    checked = 0
    for ts, price, v0, v1, confirmed in probe.tick_rows:
        k = _minute_index(ts)
        if k == 0:
            assert v0 is None  # 第一根 bar 尚未闭合, SMA(2) 一根都没攒够
            continue
        assert confirmed is False
        prev_close = _bar_close_of_minute(k - 1)
        assert v0 == pytest.approx((prev_close + price) / 2), (ts, v0, prev_close)
        checked += 1
    assert checked >= 8


def test_tick_intrabar_bar_close_confirms_and_state_not_polluted() -> None:
    """on_bar 里 [0] 是确认值 = SMA(前两根 bar close), 即 tick 试算未污染增量状态."""
    probe = _run_tick_intrabar().strategy
    assert probe is not None

    assert len(probe.bar_rows) >= 4
    for ts, close, v0, confirmed in probe.bar_rows:
        assert confirmed is True
        k = _minute_index(ts)
        assert close == pytest.approx(_bar_close_of_minute(k))
        if k == 0:
            assert v0 is None
        else:
            expected = (_bar_close_of_minute(k - 1) + close) / 2
            assert v0 == pytest.approx(expected), (ts, v0, expected)


def test_tick_intrabar_index_one_is_last_confirmed_bar_value() -> None:
    """on_tick 里 [1] 是上一根已确认 bar 的 SMA 值."""
    probe = _run_tick_intrabar().strategy
    assert probe is not None
    confirmed_by_ts = {ts: v0 for ts, _, v0, _ in probe.bar_rows}

    checked = 0
    for ts, _, _, v1, _ in probe.tick_rows:
        prior = [confirmed_by_ts[b] for b in sorted(confirmed_by_ts) if b < ts]
        if len(prior) >= 2:
            assert v1 == prior[-1]
            checked += 1
    assert checked >= 4


def test_tick_intrabar_stream_labels_match_closing_bar_timestamps() -> None:
    """临时点时间戳按 end-stamp 公式预测, 必须与随后闭合 bar 的确认点同 time."""
    messages: List[Any] = []

    def on_event(event: Any) -> None:
        if aq.is_indicator_stream_event(event):
            m = aq.to_indicator_message(event)
            if m is not None and m["type"] == "point":
                messages.append(m["indicator"])

    _run_tick_intrabar(on_event=on_event, stream_batch_size=1)

    provisional = [m for m in messages if m["confirmed"] is False]
    confirmed_ts = {m["timestamp"] for m in messages if m["confirmed"] is True}
    assert provisional and confirmed_ts
    # 数据末尾最后一笔 tick 开出的 bar 永远不会闭合(没有下一笔 tick 触发),
    # 它的临时点天然没有确认点 —— 这是回测尾部的固有现象, 实盘下一笔 tick 就会
    # 闭合。排除它, 其余每个临时点都必须有同 time 的确认点。
    last_confirmed = max(confirmed_ts)
    in_scope = [m for m in provisional if m["timestamp"] <= last_confirmed]
    assert in_scope
    assert all(m["timestamp"] in confirmed_ts for m in in_scope)


def test_tick_intrabar_dataframe_has_only_confirmed_points() -> None:
    """DataFrame 出口只含确认点: 行数 = SMA 就绪的 bar 数, 与 tick 数无关."""
    result = _run_tick_intrabar()
    frame = result.indicator_df(name="sma")
    probe = result.strategy
    assert probe is not None

    ready_bars = [row for row in probe.bar_rows if row[2] is not None]
    assert len(frame) == len(ready_bars)


def test_tick_intrabar_hl_input_mode_gets_real_running_high_low() -> None:
    """HL 类指标(ATR)在 tick 上原本只能跳过; intrabar 用形成中 bar 真实 H/L 试算."""

    class AtrProbe(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.tick_values: List[Any] = []

        def on_start(self) -> None:
            self.atr = self.I(aq.ATR(2), name="atr", input_mode="hlc", intrabar=True)

        def on_tick(self, tick: Tick) -> None:
            self.tick_values.append(self.atr[0])

        def on_bar(self, bar: Bar) -> None:
            pass

    probe = run_backtest(
        strategy=AtrProbe,
        data=_half_minute_ticks(13),
        freq="1min",
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
    ).strategy
    assert probe is not None

    assert any(v is not None for v in probe.tick_values)


def test_tick_intrabar_off_keeps_ticks_driving_base_indicator() -> None:
    """默认 intrabar=False: tick 照旧 update() 基础指标(纯 tick 策略依赖), 不变."""

    class Legacy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.seen: List[Any] = []

        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), name="sma")

        def on_tick(self, tick: Tick) -> None:
            self.seen.append((self.sma[0], self.sma.confirmed))

        def on_bar(self, bar: Bar) -> None:
            pass

    probe = run_backtest(
        strategy=Legacy,
        data=_half_minute_ticks(6),
        freq="1min",
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
    ).strategy
    assert probe is not None

    # 第二笔 tick 时 SMA(2) 已被两笔 tick 推进 → 有值且是"确认"(没有临时值概念)
    assert probe.seen[1][0] == pytest.approx((10.0 + 10.5) / 2)
    assert all(c is True for _, c in probe.seen)


def test_tick_intrabar_peek_receives_forming_bar_high_low_close() -> None:
    """Peek 收到形成中 bar 的真实 H/L/C: low=本分钟首笔, high=最新价, 逐 bar 重置.

    只断言"有值"抓不住两类缺陷: bar 闭合后 partial 不重置(low 永远是全天首笔),
    以及 partial 不跟踪 H/L(高低恒等于最新价)。这里用一个记录 update 参数的
    spy 指标把 peek 收到的三元组抓出来逐笔核对。价格单调上升, 所以本分钟
    首笔就是 low、最新笔就是 high。
    """
    seen: List[Tuple[int, float, float, float]] = []

    class Spy:
        def __init__(self) -> None:
            self.calls = 0

        def update(self, high: float, low: float, close: float) -> None:
            self.calls += 1
            seen.append((self.calls, high, low, close))

        @property
        def value(self) -> float:
            return float(self.calls)

    class Probe(Strategy):
        def on_start(self) -> None:
            self.spy = self.I(Spy(), name="spy", input_mode="hlc", intrabar=True)

        def on_bar(self, bar: Bar) -> None:
            pass

    ticks = _half_minute_ticks(9)  # 4 根闭合 bar + 1 笔开出第 5 根
    run_backtest(
        strategy=Probe,
        data=ticks,
        freq="1min",
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
    )

    # 每笔 tick 一次 peek(在副本上, 但 seen 是外层列表, 记得到); bar 闭合一次真 update。
    # 只看 peek: 它收到的 (h, l, c) 必须等于该分钟迄今的 (最新价, 首笔价, 最新价)。
    prices = [t.price for t in ticks]
    hlc = [(high, low, close) for _, high, low, close in seen]
    expected = []
    for i, price in enumerate(prices):
        k = i // 2
        first_of_minute = prices[2 * k]
        expected.append((price, first_of_minute, price))  # peek on tick i
        if i % 2 == 1:
            expected.append((price, first_of_minute, price))  # bar close -> real update
    assert hlc == expected, (hlc[:6], expected[:6])
