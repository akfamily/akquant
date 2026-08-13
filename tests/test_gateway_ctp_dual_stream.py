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
from akquant.gateway.brokers.ctp import native as ctp_native
from akquant.gateway.brokers.ctp.native import CTPMarketGateway


class FakeFeed:
    """记录被推入的 bar/tick, 不依赖真实 DataFeed."""

    def __init__(self) -> None:
        """初始化收集列表."""
        self.bars: list[Any] = []
        self.ticks: list[Any] = []

    def add_bar(self, bar: Any) -> None:
        """记录被推入的 bar."""
        self.bars.append(bar)

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
    assert gateway.stamp_bar_at_interval_end is True


def test_ctp_use_aggregator_false_defaults_to_tick_only() -> None:
    """use_aggregator=False 单独传入时应等价于只出 tick(不再是假 bar)."""
    gateway = _gateway(use_aggregator=False)
    assert gateway.emit_ticks is True
    assert gateway.emit_bars is False
    assert gateway.aggregator is None
    assert gateway.stamp_bar_at_interval_end is False


def test_ctp_emit_ticks_alone_enables_dual_stream_without_disabling_bar() -> None:
    """只显式传 emit_ticks=True(不传 emit_bars)必须是双流, 不能悄悄关掉 on_bar.

    回退逻辑是逐参数的: 只显式表态一个参数时, 另一个仍按 use_aggregator 别名
    推导(默认 True -> 只出 bar), 不会被 ``bool(None)`` 压成 False。
    """
    gateway = _gateway(emit_ticks=True)
    assert gateway.emit_ticks is True
    assert gateway.emit_bars is True
    assert gateway.aggregator is not None
    assert gateway.stamp_bar_at_interval_end is True


def test_ctp_emit_ticks_true_emit_bars_false_stays_tick_only() -> None:
    """emit_ticks=True, emit_bars=False 显式组合应保持只出 tick."""
    gateway = _gateway(emit_ticks=True, emit_bars=False)
    assert gateway.emit_ticks is True
    assert gateway.emit_bars is False
    assert gateway.aggregator is None
    assert gateway.stamp_bar_at_interval_end is False


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


def test_ctp_fallback_fake_bar_branch_survives_for_defensive_case() -> None:
    """既不出 tick 也没有聚合器时, 保留旧的"包装成假 bar"兜底分支.

    正常构造路径下 ``emit_ticks``/``emit_bars`` 不会同时为 False(构造函数已用
    ``ValueError`` 挡掉), 所以这个分支理论上不会被合法构造出的实例触发。这里
    用 ``__new__`` 绕开 ``__init__`` 校验、直接摆出该组合, 只是为了证明分支
    代码本身还在、没有被误删, 而不是主张这是受支持的构造方式。
    """
    gateway = CTPMarketGateway.__new__(CTPMarketGateway)
    fake_feed = FakeFeed()
    gateway.feed = fake_feed  # type: ignore[assignment]
    gateway.emit_ticks = False
    gateway.emit_bars = False
    gateway.aggregator = None
    gateway.last_volume = {}

    gateway.OnRtnDepthMarketData(_md(volume=10.0))
    gateway.OnRtnDepthMarketData(_md(volume=15.0))

    assert len(fake_feed.bars) == 2
    assert fake_feed.bars[-1].volume == 5.0
