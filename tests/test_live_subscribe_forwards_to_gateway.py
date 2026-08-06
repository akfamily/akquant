"""实盘 subscribe() 必须真正下发到行情网关.

``MarketGateway`` 协议早已定义 ``subscribe(symbols)``, 各 broker 网关(ctp /
replay / miniqmt / ptrade)也都实现了它, 但 ``LiveRunner`` 从不调用——订阅链路
只接了一半: ``instruments`` 进得去, ``Strategy.subscribe()`` 出不来。于是用户在
``on_start`` 里 subscribe 一个标的时既订不到行情, 也只能收到一条告警。

注意两个约束:

1. **时序**: 行情网关在 ``run()`` 里先启动, ``on_start`` 稍后才在 ``engine.run()``
   内触发, 因此只能在 subscribe 发生的当刻**运行期转发**, 无法在装配期收集。
2. **替换语义**: 四个网关的 ``subscribe()`` 都是 ``self.symbols = list(symbols)``
   (整集替换, 非追加)。故必须下发 ``instruments ∪ 各 slot 的订阅``的并集,
   只下发新标的会把其余标的静默退订。
"""

import logging
from typing import Any, Callable, Dict, List, Sequence, cast

import pytest
from akquant import AssetType, Instrument, Strategy
from akquant.akquant import Bar
from akquant.live._runner import LiveRunner


def _instrument(symbol: str) -> Instrument:
    """构造一个股票标的."""
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


class _RecordingMarketGateway:
    """记录每次 subscribe 下发的订阅集, 复刻真实网关的整集替换语义."""

    def __init__(self) -> None:
        """初始化订阅集与调用流水."""
        self.symbols: List[str] = []
        self.calls: List[List[str]] = []

    def connect(self) -> None:
        """no-op."""

    def disconnect(self) -> None:
        """no-op."""

    def subscribe(self, symbols: Sequence[str]) -> None:
        """整集替换当前订阅集(与 ctp/replay/miniqmt/ptrade 一致)."""
        self.symbols = [str(s) for s in symbols]
        self.calls.append(list(self.symbols))

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        """从订阅集移除给定标的."""
        removed = {str(s) for s in symbols}
        self.symbols = [s for s in self.symbols if s not in removed]

    def on_tick(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """记录 tick 回调引用."""

    def on_bar(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """记录 bar 回调引用."""

    def start(self) -> None:
        """no-op."""


class _StubEngine:
    """吸收 _configure_strategy_slots 对引擎的可选 set_* 调用."""

    def __getattr__(self, name: str) -> Any:
        """任何属性都返回一个接受任意入参的空函数."""

        def _accept(*args: Any, **kwargs: Any) -> None:
            return None

        return _accept


class _Sub(Strategy):
    """不在 on_start 里自动订阅的最简策略(由测试显式调用 subscribe)."""

    def on_bar(self, bar: Bar) -> None:
        """不做任何事."""


def _make_runner(
    instruments: List[Instrument],
    market_gateway: Any,
) -> LiveRunner:
    """装配一个仅够跑 _configure_strategy_slots 的 runner."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.trading_mode = "paper"
    runner.context = cast(Any, None)
    runner.instruments = instruments
    runner.engine = cast(Any, _StubEngine())
    runner._market_gateway = market_gateway
    return runner


def test_live_subscribe_forwards_union_of_instruments_and_subscriptions() -> None:
    """subscribe() 应把 instruments 与订阅的并集下发到网关.

    只下发新标的(["600001"])会因替换语义把 600000 静默退订; 完全不下发则
    订阅永远不生效——两种写法都必须让本测试失败。
    """
    gateway = _RecordingMarketGateway()
    runner = _make_runner([_instrument("600000")], gateway)
    strategy = _Sub()
    runner._configure_strategy_slots(strategy, {}, "alpha")

    strategy.subscribe("600001")

    assert sorted(gateway.symbols) == ["600000", "600001"]


def test_live_subscribe_does_not_warn_when_forwarded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """订阅真正下发后, 不应再告警"subscribe 在实盘不生效"."""
    gateway = _RecordingMarketGateway()
    runner = _make_runner([_instrument("600000")], gateway)
    strategy = _Sub()
    runner._configure_strategy_slots(strategy, {}, "alpha")

    with caplog.at_level(logging.WARNING):
        strategy.subscribe("600001")

    messages = [r.getMessage() for r in caplog.records]
    assert not any("subscribe" in m for m in messages), (
        f"订阅已生效却仍告警: {messages}"
    )


def test_live_subscribe_still_warns_without_market_gateway(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """网关缺失时(如 broker='qmf' 的 market_gateway=None)必须保留告警.

    这类 broker 压根没有行情通道可供下发, 静默会让用户继续误以为订阅成功。
    """
    runner = _make_runner([_instrument("600000")], None)
    strategy = _Sub()
    runner._configure_strategy_slots(strategy, {}, "alpha")

    with caplog.at_level(logging.WARNING):
        strategy.subscribe("600001")

    messages = [r.getMessage() for r in caplog.records]
    assert any("subscribe" in m for m in messages), f"无行情网关却未告警: {messages}"


def test_slot_subscription_preserves_other_slots_subscriptions() -> None:
    """某个 slot 订阅时, 其他 slot 已有的订阅不得被替换掉.

    并集只算调用方自己的 _subscriptions 时, 先订阅的 600001 会被后一次
    下发覆盖丢失。
    """
    gateway = _RecordingMarketGateway()
    runner = _make_runner([_instrument("600000")], gateway)
    primary = _Sub()
    slot = _Sub()
    runner._configure_strategy_slots(primary, {"beta": slot}, "alpha")

    primary.subscribe("600001")
    slot.subscribe("600002")

    assert sorted(gateway.symbols) == ["600000", "600001", "600002"]


def test_backtest_subscribe_never_touches_gateway() -> None:
    """无实盘标记的策略(纯回测)不得触发任何下发."""
    gateway = _RecordingMarketGateway()
    strategy = _Sub()

    strategy.subscribe("600001")

    assert gateway.calls == []
    assert strategy._subscriptions == ["600001"], "回测语义必须保持不变"
