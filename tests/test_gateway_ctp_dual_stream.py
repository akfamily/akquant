"""CTP 行情网关的双流语义(不连真实柜台, 只验分发决策).

``openctp-ctp`` 在本机通常未安装, ``native.py`` 顶部已用 ``try/except`` 把
``mdapi``/``tdapi`` 替换成占位类(见 ``HAS_OPENCTP``), 模块本身可以正常导入。
但 ``CTPMarketGateway.__init__`` 会在 ``not HAS_OPENCTP`` 时主动
``raise ImportError`` 拒绝构造, 这与 CTP SDK 是否真的装了无关, 纯粹是可用性
守卫。为了测参数解析(``emit_ticks``/``emit_bars`` 回退规则)与推送顺序,
本文件把该模块级 ``HAS_OPENCTP`` monkeypatch 成 ``True`` 后再构造 ——
``CTPMarketGateway`` 的基类在模块导入时已经固定为占位 ``MockSpi``(空类, 用
``object.__init__``), 与是否装了真 SDK 无关, 所以这样 monkeypatch 不会引入
对真实 CTP SDK 的依赖, 只是跳过一道与本测试无关的可用性检查。这与文件里
``test_gateway_ctp_native.py`` 用 ``CTPMarketGateway.__new__`` 绕开
``__init__`` 的既有做法是同一类思路的另一种取舍: 这里恰恰需要跑通
``__init__`` 内的参数解析逻辑, 所以选择放行 ``__init__`` 而非跳过它。
"""

from __future__ import annotations

from typing import Any

import pytest
from akquant import DataFeed
from akquant.gateway.brokers.builtins import _build_ctp
from akquant.gateway.brokers.ctp import native as ctp_native
from akquant.gateway.brokers.ctp.adapter import CTPMarketAdapter
from akquant.gateway.brokers.ctp.native import CTPMarketGateway


class FakeFeed:
    """记录被推入的 tick, 不依赖真实 DataFeed."""

    def __init__(self) -> None:
        """初始化收集列表."""
        self.ticks: list[Any] = []

    def add_tick(self, tick: Any) -> None:
        """记录被推入的 tick."""
        self.ticks.append(tick)


@pytest.fixture(autouse=True)
def _allow_ctp_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """本文件全程绕开"未装 openctp-ctp"这道与双流语义无关的可用性检查."""
    monkeypatch.setattr(ctp_native, "HAS_OPENCTP", True)


def _gateway(feed: Any = None, **kwargs: Any) -> CTPMarketGateway:
    if feed is None:
        feed = DataFeed()
    kwargs.setdefault("front_url", "tcp://test")
    return CTPMarketGateway(feed=feed, symbols=["rb2401"], **kwargs)


def _md(symbol: str = "rb2401", price: float = 100.0, volume: float = 10.0) -> Any:
    class _MarketData:
        InstrumentID = symbol
        LastPrice = price
        Volume = volume

    return _MarketData()


def test_ctp_dual_stream_flags_default_to_bar_only() -> None:
    """默认保持既有行为: 只出 bar, 不推 tick."""
    gateway = _gateway()
    assert gateway.emit_bars is True
    assert gateway.emit_ticks is False
    assert gateway.aggregator is not None


def test_ctp_can_emit_both_tick_and_bar() -> None:
    """显式开启双流后两个标志都为真, 且聚合器存在."""
    gateway = _gateway(emit_ticks=True, emit_bars=True)
    assert gateway.emit_ticks is True
    assert gateway.emit_bars is True
    assert gateway.aggregator is not None


def test_ctp_use_aggregator_false_defaults_to_tick_only() -> None:
    """use_aggregator=False 单独传入时应等价于只出 tick(不再是假 bar)."""
    gateway = _gateway(use_aggregator=False)
    assert gateway.emit_ticks is True
    assert gateway.emit_bars is False
    assert gateway.aggregator is None


def test_ctp_emit_ticks_alone_enables_dual_stream_without_disabling_bar() -> None:
    """只显式传 emit_ticks=True(不传 emit_bars)必须是双流, 不能悄悄关掉 on_bar.

    回退逻辑是逐参数的: 只显式表态一个参数时, 另一个仍按 use_aggregator 别名
    推导(默认 True -> 只出 bar), 不会被 ``bool(None)`` 压成 False。
    """
    gateway = _gateway(emit_ticks=True)
    assert gateway.emit_ticks is True
    assert gateway.emit_bars is True
    assert gateway.aggregator is not None


def test_ctp_emit_ticks_true_emit_bars_false_stays_tick_only() -> None:
    """emit_ticks=True, emit_bars=False 显式组合应保持只出 tick."""
    gateway = _gateway(emit_ticks=True, emit_bars=False)
    assert gateway.emit_ticks is True
    assert gateway.emit_bars is False
    assert gateway.aggregator is None


def test_ctp_emit_ticks_and_emit_bars_both_false_raises() -> None:
    """两者都显式为 False 时应报错: 那样网关不会推任何行情."""
    with pytest.raises(ValueError, match="emit_ticks"):
        _gateway(emit_ticks=False, emit_bars=False)


def test_ctp_dual_stream_feeds_aggregator_before_pushing_tick() -> None:
    """双流下必须先喂聚合器再推 Tick, 否则 bar 会排在更晚的 tick 之后造成时间戳倒退.

    实盘无 sort 兜底(``RealtimeDataClient::sort`` 是空实现), 顺序反了是**静默**
    的, 属性断言抓不到顺序回归, 故此处用调用序列把它锁住。

    构造期仍传真 ``DataFeed``(``BarAggregator`` 只接受它, 不吃鸭子类型), 构造
    完成后再把 ``aggregator``/``feed`` 换成记录器 —— 只关心
    ``OnRtnDepthMarketData`` 调用两者的先后顺序, 不关心聚合器真实合成逻辑。
    """
    calls: list[str] = []

    class _RecordingAggregator:
        def on_tick(self, *_args: object) -> None:
            calls.append("aggregator")

    class _RecordingFeed(FakeFeed):
        def add_tick(self, tick: object) -> None:
            calls.append("add_tick")
            super().add_tick(tick)

    gateway = _gateway(emit_ticks=True, emit_bars=True)
    gateway.aggregator = _RecordingAggregator()  # type: ignore[assignment]
    gateway.feed = _RecordingFeed()  # type: ignore[assignment]
    gateway.OnRtnDepthMarketData(_md())
    assert calls == ["aggregator", "add_tick"]


# ---------------------------------------------------------------------------
# Builder 层(``_build_ctp`` / ``run_live(broker="ctp", gateway_options={...})``
# 的真实入口): 终审 finding I5 —— 上面所有测试都直接构造 ``CTPMarketGateway``,
# 完全绕开了 ``_build_ctp`` 未转发 ``emit_ticks``/``emit_bars`` 的 bug。这里从
# ``gateway_options`` 这一层出发, 断言参数真的能捅到底层 gateway。
# ---------------------------------------------------------------------------


def _build(**gateway_options: Any) -> CTPMarketGateway:
    feed = DataFeed()
    gateway_options.setdefault("md_front", "tcp://test")
    bundle = _build_ctp(
        feed=feed,
        symbols=["rb2401"],
        use_aggregator=gateway_options.pop("use_aggregator", True),
        **gateway_options,
    )
    assert isinstance(bundle.market_gateway, CTPMarketAdapter)
    return bundle.market_gateway.gateway


def test_build_ctp_declares_dual_stream_in_metadata() -> None:
    """Builder 在 ``metadata["dual_stream"]`` 声明是否双流, 供 runner 注入策略.

    双流下声明式指标只由 bar 推进(见 ``tests/test_indicator_dual_stream.py``);
    漏声明则 tick 与 bar 都会推进基础指标, 值静默算错。
    """
    for options, expected in (
        ({}, False),
        ({"emit_ticks": True, "emit_bars": True}, True),
        ({"use_aggregator": False}, False),
    ):
        opts: dict[str, Any] = {"md_front": "tcp://test", **options}
        bundle = _build_ctp(
            feed=DataFeed(),
            symbols=["rb2401"],
            use_aggregator=opts.pop("use_aggregator", True),
            **opts,
        )
        assert bundle.metadata is not None
        assert bundle.metadata["dual_stream"] is expected, options


def test_build_ctp_default_stays_bar_only() -> None:
    """未传 emit_ticks/emit_bars 时经 builder 层仍是默认只出 bar."""
    gateway = _build()
    assert gateway.emit_bars is True
    assert gateway.emit_ticks is False


def test_build_ctp_forwards_emit_ticks_and_emit_bars() -> None:
    """``gateway_options={"emit_ticks": True, "emit_bars": True}`` 必须捅到底层 gateway.

    这是终审 finding 的核心场景: ``run_live(broker="ctp", gateway_options={...})``
    在修复前会静默丢弃这两个键, 底层仍是默认的 emit_ticks=False。
    """
    gateway = _build(emit_ticks=True, emit_bars=True)
    assert gateway.emit_ticks is True
    assert gateway.emit_bars is True


def test_build_ctp_emit_ticks_alone_stays_dual_stream_via_builder() -> None:
    """经 builder 层, 只传 emit_ticks=True 时逐参数回退, 不悄悄关掉 on_bar."""
    gateway = _build(emit_ticks=True)
    assert gateway.emit_ticks is True
    assert gateway.emit_bars is True


def test_build_ctp_use_aggregator_false_forwards_to_tick_only() -> None:
    """经 builder 层, 顶层 use_aggregator=False 仍能推导出只出 tick."""
    gateway = _build(use_aggregator=False)
    assert gateway.emit_ticks is True
    assert gateway.emit_bars is False


def test_build_ctp_emit_ticks_and_emit_bars_both_false_raises_via_builder() -> None:
    """经 builder 层, 两者都显式为 False 时构造期报错, 不静默吞掉."""
    with pytest.raises(ValueError, match="emit_ticks"):
        _build(emit_ticks=False, emit_bars=False)


# ---------------------------------------------------------------------------
# Tick.volume 累计量 -> 单笔量: CTP 的 pDepthMarketData.Volume 是当日累计成交
# 量, 而回测里 Tick.volume 是单笔量。三条规则见
# ``CTPMarketGateway._cumulative_volume_to_delta`` 的 docstring: 正常差分 /
# 首帧记 0(不是 cum_volume, 理由见该方法注释) / 负差分(跨日重置或重连)时
# delta = cum_volume。这里只用 emit_ticks=True, emit_bars=False(纯 tick, 不
# 触发聚合器)以避免依赖真实 ``DataFeed`` 的 aggregator 合成路径, 构造后把
# ``feed`` 换成 ``FakeFeed`` 只记录被推入的 Tick。
# ---------------------------------------------------------------------------


def test_ctp_tick_volume_first_frame_is_zero_not_cumulative() -> None:
    """首帧还没有该 symbol 的前值时, 推出的 Tick.volume 必须是 0, 不是原始累计量.

    进程可能盘中启动, 此时累计量可能是几十万手, 当成单笔量会严重失真。
    """
    gateway = _gateway(emit_ticks=True, emit_bars=False)
    fake_feed = FakeFeed()
    gateway.feed = fake_feed  # type: ignore[assignment]
    gateway.OnRtnDepthMarketData(_md(volume=123456.0))
    assert len(fake_feed.ticks) == 1
    assert fake_feed.ticks[0].volume == 0.0


def test_ctp_tick_volume_normal_frame_is_delta() -> None:
    """非首帧的正常情况下, Tick.volume 是与上一帧的累计量差分."""
    gateway = _gateway(emit_ticks=True, emit_bars=False)
    fake_feed = FakeFeed()
    gateway.feed = fake_feed  # type: ignore[assignment]
    gateway.OnRtnDepthMarketData(_md(volume=100.0))
    gateway.OnRtnDepthMarketData(_md(volume=150.0))
    gateway.OnRtnDepthMarketData(_md(volume=170.0))
    assert [t.volume for t in fake_feed.ticks] == [0.0, 50.0, 20.0]


def test_ctp_tick_volume_negative_diff_uses_raw_volume_as_delta() -> None:
    """累计量比上一帧还小(跨日重置/重连归零)时, delta 直接取当前累计量.

    此时该累计量本身就约等于新一天/新连接下首笔的单笔量。
    """
    gateway = _gateway(emit_ticks=True, emit_bars=False)
    fake_feed = FakeFeed()
    gateway.feed = fake_feed  # type: ignore[assignment]
    gateway.OnRtnDepthMarketData(_md(volume=100.0))
    gateway.OnRtnDepthMarketData(_md(volume=150.0))
    gateway.OnRtnDepthMarketData(_md(volume=30.0))  # 归零重来, 比上一帧的 150 小
    assert [t.volume for t in fake_feed.ticks] == [0.0, 50.0, 30.0]


def test_ctp_tick_volume_state_is_isolated_per_symbol() -> None:
    """两个 symbol 的累计量状态互不影响, 各自独立算首帧/差分."""
    gateway = _gateway(emit_ticks=True, emit_bars=False)
    fake_feed = FakeFeed()
    gateway.feed = fake_feed  # type: ignore[assignment]
    gateway.OnRtnDepthMarketData(_md(symbol="rb2401", volume=100.0))
    gateway.OnRtnDepthMarketData(_md(symbol="au2406", volume=999.0))
    gateway.OnRtnDepthMarketData(_md(symbol="rb2401", volume=130.0))
    gateway.OnRtnDepthMarketData(_md(symbol="au2406", volume=1010.0))
    assert [t.volume for t in fake_feed.ticks] == [0.0, 0.0, 30.0, 11.0]


def test_ctp_tick_volume_delta_does_not_affect_aggregator_cumulative_input() -> None:
    """add_tick 推的是单笔量, 但喂给 aggregator.on_tick 的仍是原始累计量.

    aggregator 的 volume_is_cumulative=True(BarAggregator 默认值, native.py
    构造时未覆盖)会自己差分, 这里如果误把换算后的单笔量喂给它会造成二次差分。
    """
    calls: list[float] = []

    class _RecordingAggregator:
        def on_tick(self, symbol: str, price: float, volume: float, ts: int) -> None:
            calls.append(volume)

    gateway = _gateway(emit_ticks=True, emit_bars=True)
    gateway.aggregator = _RecordingAggregator()  # type: ignore[assignment]
    fake_feed = FakeFeed()
    gateway.feed = fake_feed  # type: ignore[assignment]
    gateway.OnRtnDepthMarketData(_md(volume=100.0))
    gateway.OnRtnDepthMarketData(_md(volume=150.0))
    assert calls == [100.0, 150.0]
    assert [t.volume for t in fake_feed.ticks] == [0.0, 50.0]


# ---------------------------------------------------------------------------
# 行情源交易日: 登录回调与逐笔行情里的 TradingDay 经 feed.add_trading_day_hint
# 送入引擎(交易时段重构阶段 7)。
# ---------------------------------------------------------------------------


class _HintFeed(FakeFeed):
    """记录 tick 与交易日 hint 的推送顺序."""

    def __init__(self) -> None:
        """初始化收集列表."""
        super().__init__()
        self.hints: list[tuple[int, Any, str]] = []
        self.calls: list[str] = []

    def add_tick(self, tick: Any) -> None:
        """记录被推入的 tick."""
        self.calls.append("tick")
        super().add_tick(tick)

    def add_trading_day_hint(self, timestamp_ns: int, day: Any, source: str) -> None:
        """记录被推入的交易日 hint."""
        self.calls.append("hint")
        self.hints.append((timestamp_ns, day, source))


def _md_day(trading_day: str, symbol: str = "rb2401") -> Any:
    class _MarketData:
        InstrumentID = symbol
        LastPrice = 100.0
        Volume = 10.0
        TradingDay = trading_day

    return _MarketData()


def _hint_gateway() -> tuple[CTPMarketGateway, _HintFeed]:
    gateway = _gateway(emit_ticks=True, emit_bars=False)
    feed = _HintFeed()
    gateway.feed = feed  # type: ignore[assignment]
    return gateway, feed


def test_ctp_tick_trading_day_change_pushes_one_hint() -> None:
    """TradingDay 由 20241008 变 20241009 时恰好多一个 hint, 同值不重复推."""
    import datetime as dt

    gateway, feed = _hint_gateway()
    gateway.OnRtnDepthMarketData(_md_day("20241008"))
    gateway.OnRtnDepthMarketData(_md_day("20241008"))
    assert [(d, s) for _, d, s in feed.hints] == [(dt.date(2024, 10, 8), "ctp")]
    gateway.OnRtnDepthMarketData(_md_day("20241009"))
    gateway.OnRtnDepthMarketData(_md_day("20241009"))
    assert [d for _, d, _ in feed.hints] == [
        dt.date(2024, 10, 8),
        dt.date(2024, 10, 9),
    ]
    # hint 先于同一笔 tick 入 feed, 时间戳与该 tick 相同(本地接收时刻)
    assert feed.calls[:2] == ["hint", "tick"]
    assert feed.hints[0][0] == int(feed.ticks[0].timestamp)


def test_ctp_trading_day_deduped_and_bad_values_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """同一交易日跨品种只推一次; 空值/非法值忽略, 原始值不变时不重复记 debug."""
    gateway, feed = _hint_gateway()
    gateway.OnRtnDepthMarketData(_md_day("20241008", "rb2401"))
    gateway.OnRtnDepthMarketData(_md_day("20241008", "au2412"))
    assert len(feed.hints) == 1
    with caplog.at_level("DEBUG", logger="akquant"):
        for _ in range(3):
            gateway.OnRtnDepthMarketData(_md_day("", "rb2401"))
        for _ in range(3):
            gateway.OnRtnDepthMarketData(_md_day("2024-13-xx", "rb2401"))
    assert len(feed.hints) == 1
    assert len(feed.ticks) == 8
    debug = [r for r in caplog.records if "TradingDay" in r.getMessage()]
    assert len(debug) == 2, [r.getMessage() for r in debug]
    # 恢复为合法的新交易日时照常推
    gateway.OnRtnDepthMarketData(_md_day("20241009", "rb2401"))
    assert [d.day for _, d, _ in feed.hints] == [8, 9]


def test_ctp_relogin_same_trading_day_does_not_push_again() -> None:
    """重新登录后 TradingDay 变化只推一次: 登录推, 之后各品种首笔同值不再推."""
    gateway, feed = _hint_gateway()

    class _Login:
        TradingDay = "20241008"
        BrokerID = "9999"
        UserID = "u"

    gateway.OnRspUserLogin(_Login(), None, 1, True)
    gateway.OnRtnDepthMarketData(_md_day("20241008", "rb2401"))
    _Login.TradingDay = "20241009"
    gateway.OnRspUserLogin(_Login(), None, 2, True)
    gateway.OnRtnDepthMarketData(_md_day("20241009", "rb2401"))
    gateway.OnRtnDepthMarketData(_md_day("20241009", "au2412"))
    assert [d.day for _, d, _ in feed.hints] == [8, 9]


def test_ctp_login_trading_day_pushes_hint() -> None:
    """行情登录回调的 TradingDay 推一个 hint; 之后同值的逐笔行情不重复推."""
    import datetime as dt

    gateway, feed = _hint_gateway()

    class _Login:
        TradingDay = "20241008"
        BrokerID = "9999"
        UserID = "u"

    gateway.OnRspUserLogin(_Login(), None, 1, True)
    assert [(d, s) for _, d, s in feed.hints] == [(dt.date(2024, 10, 8), "ctp")]
    gateway.OnRtnDepthMarketData(_md_day("20241008"))
    assert len(feed.hints) == 1


# ---------------------------------------------------------------------------
# 收盘宽限钳制钩子: runner 经 bundle.market_gateway(适配器)下发
# ---------------------------------------------------------------------------


def test_ctp_configure_session_clamp_merges_grace_tick_into_close_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """适配器的 configure_session_clamp 转给聚合器: 15:00:20 的 tick 并入收盘 bar."""
    import pandas as pd
    from akquant import AssetType, Bar, SessionResolver, Strategy, run_backtest

    def at(text: str) -> int:
        return int(pd.Timestamp(f"2024-01-02 {text}", tz="Asia/Shanghai").value)

    clock = iter([at("14:59:30"), at("15:00:20")])
    monkeypatch.setattr(ctp_native.time, "time_ns", lambda: next(clock))
    feed = DataFeed()
    bundle = _build_ctp(
        feed=feed, symbols=["rb2401"], use_aggregator=True, md_front="tcp://test"
    )
    adapter = bundle.market_gateway
    assert isinstance(adapter, CTPMarketAdapter)
    resolver = SessionResolver()
    resolver.set_active([("rb2401", AssetType.Futures)])
    adapter.configure_session_clamp(resolver, 55 * 10**9)

    adapter.gateway.OnRtnDepthMarketData(_md(price=100.0, volume=10.0))
    adapter.gateway.OnRtnDepthMarketData(_md(price=101.0, volume=15.0))
    aggregator = adapter.gateway.aggregator
    assert aggregator is not None
    assert aggregator.flush_due(at("15:00:57") - 1) == 0
    assert aggregator.flush_due(at("15:00:57")) == 1

    class _Recorder(Strategy):
        bars: list[tuple[int, float, dict[str, float]]] = []

        def on_bar(self, bar: Bar) -> None:
            type(self).bars.append((bar.timestamp, bar.close, dict(bar.extra)))

    _Recorder.bars = []
    run_backtest(strategy=_Recorder, data=feed, symbols=["rb2401"], initial_cash=1e5)
    assert _Recorder.bars == [(at("15:00:00"), 101.0, {"__akq_close_price": 100.0})]


def test_ctp_configure_session_clamp_without_aggregator_is_noop() -> None:
    """只出 tick(无聚合器)时钩子什么也不做."""
    from akquant import SessionResolver

    gateway = _gateway(use_aggregator=False)
    assert gateway.aggregator is None
    gateway.configure_session_clamp(SessionResolver(), 0)
