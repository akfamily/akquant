"""行情源与交易源可分别指定(``market_broker`` + ``broker``).

``GatewayBundle`` 早已把 ``market_gateway`` / ``trader_gateway`` 拆成两个独立可
选字段, 但 ``create_gateway_bundle`` 只接受**一个** broker 名, 于是两者只能来自
同一个 builder。现实里这个组合是缺的:

- ``broker='qmf'`` 只有交易通道(``market_gateway=None``) → 收不到任何行情;
- ``broker='replay'`` 只有行情通道(``trader_gateway=None``) → 无法下单。

两者都存在却无法拼在一起。本模块锁定"行情用 A、交易用 B"的混搭契约。
"""

from typing import Any, Callable, Dict, List, Sequence, cast

import pytest
from akquant import DataFeed
from akquant.gateway import create_gateway_bundle
from akquant.gateway.protocols import GatewayBundle, MarketGateway, TraderGateway
from akquant.gateway.registry import register_broker, unregister_broker

MARKET_ONLY = "test_market_only"
TRADER_ONLY = "test_trader_only"


class _MarketOnlyGateway:
    """只提供行情的网关替身."""

    def __init__(self, symbols: Sequence[str]) -> None:
        """记录订阅集."""
        self.symbols: List[str] = [str(s) for s in symbols]

    def connect(self) -> None:
        """no-op."""

    def disconnect(self) -> None:
        """no-op."""

    def subscribe(self, symbols: Sequence[str]) -> None:
        """整集替换订阅集."""
        self.symbols = [str(s) for s in symbols]

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        """移除给定标的."""
        removed = {str(s) for s in symbols}
        self.symbols = [s for s in self.symbols if s not in removed]

    def on_tick(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """记录 tick 回调."""

    def on_bar(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """记录 bar 回调."""

    def start(self) -> None:
        """no-op."""


class _TraderOnlyGateway:
    """只提供交易的网关替身."""

    def connect(self) -> None:
        """no-op."""

    def disconnect(self) -> None:
        """no-op."""

    def start(self) -> None:
        """no-op."""

    def heartbeat(self) -> bool:
        """始终就绪."""
        return True


@pytest.fixture(autouse=True)
def _register_test_brokers() -> Any:
    """注册两个互补的测试 broker, 用后注销以免污染全局 registry."""

    def _build_market_only(
        feed: Any, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
    ) -> GatewayBundle:
        _ = (feed, use_aggregator, kwargs)
        return GatewayBundle(
            market_gateway=cast(MarketGateway, _MarketOnlyGateway(symbols)),
            trader_gateway=None,
            trader_capabilities=None,
            metadata={"broker": MARKET_ONLY},
        )

    def _build_trader_only(
        feed: Any, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
    ) -> GatewayBundle:
        _ = (feed, symbols, use_aggregator, kwargs)
        return GatewayBundle(
            market_gateway=None,
            trader_gateway=cast(TraderGateway, _TraderOnlyGateway()),
            trader_capabilities=None,
            metadata={"broker": TRADER_ONLY},
        )

    register_broker(MARKET_ONLY, _build_market_only)
    register_broker(TRADER_ONLY, _build_trader_only)
    yield
    unregister_broker(MARKET_ONLY)
    unregister_broker(TRADER_ONLY)


def test_market_broker_supplies_market_gateway() -> None:
    """指定 market_broker 后, 行情网关应来自该 broker.

    未实现时 create_gateway_bundle 不认识 market_broker 这个参数。
    """
    bundle = create_gateway_bundle(
        broker=TRADER_ONLY,
        market_broker=MARKET_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.market_gateway is not None


def test_market_broker_keeps_trader_gateway_from_broker() -> None:
    """混搭时交易网关必须仍来自 broker, 不能被行情 broker 的 None 覆盖."""
    bundle = create_gateway_bundle(
        broker=TRADER_ONLY,
        market_broker=MARKET_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.trader_gateway is not None


def test_market_gateway_receives_requested_symbols() -> None:
    """行情 broker 应收到本次会话的标的集, 而不是空集."""
    bundle = create_gateway_bundle(
        broker=TRADER_ONLY,
        market_broker=MARKET_ONLY,
        feed=DataFeed(),
        symbols=["600000", "600001"],
    )

    assert getattr(bundle.market_gateway, "symbols", None) == ["600000", "600001"]


def test_metadata_records_both_brokers() -> None:
    """混搭时应能从 metadata 看出行情与交易各自来自谁(便于日志与排障)."""
    bundle = create_gateway_bundle(
        broker=TRADER_ONLY,
        market_broker=MARKET_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.metadata is not None
    assert bundle.metadata.get("broker") == TRADER_ONLY
    assert bundle.metadata.get("market_broker") == MARKET_ONLY


def test_unknown_market_broker_raises_with_supported_list() -> None:
    """未注册的 market_broker 应报错并列出可选项, 而非静默无行情."""
    with pytest.raises(ValueError, match="market_broker"):
        create_gateway_bundle(
            broker=TRADER_ONLY,
            market_broker="no_such_broker",
            feed=DataFeed(),
            symbols=["600000"],
        )


def test_without_market_broker_bundle_is_unchanged() -> None:
    """不传 market_broker 时行为完全不变(单 broker 语义保持)."""
    bundle = create_gateway_bundle(
        broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.market_gateway is None
    assert bundle.trader_gateway is not None
    assert bundle.metadata is not None
    assert bundle.metadata.get("broker") == TRADER_ONLY
    assert "market_broker" not in bundle.metadata


def test_trader_broker_supplies_trader_gateway() -> None:
    """指定 trader_broker 后, 交易网关应来自该 broker.

    与 market_broker 对称: ``broker`` 是两者的默认源, ``market_broker`` /
    ``trader_broker`` 各自覆盖一侧。只有 market_broker 而无 trader_broker 时,
    "行情用 A、交易用 B"只能从行情侧表达, 反方向写不出来。
    """
    bundle = create_gateway_bundle(
        broker=MARKET_ONLY,
        trader_broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.trader_gateway is not None


def test_trader_broker_keeps_market_gateway_from_broker() -> None:
    """指定 trader_broker 时行情网关仍来自 broker."""
    bundle = create_gateway_bundle(
        broker=MARKET_ONLY,
        trader_broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.market_gateway is not None


def test_trader_broker_and_market_broker_are_equivalent_spellings() -> None:
    """两种写法应得到同构的 bundle(对称性的核心断言).

    ``broker=A, trader_broker=B`` 与 ``broker=B, market_broker=A`` 描述的是同一
    件事: 行情来自 A、交易来自 B。二者不等价就说明这组参数语义错位。
    """
    via_trader = create_gateway_bundle(
        broker=MARKET_ONLY,
        trader_broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )
    via_market = create_gateway_bundle(
        broker=TRADER_ONLY,
        market_broker=MARKET_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert type(via_trader.market_gateway) is type(via_market.market_gateway)
    assert type(via_trader.trader_gateway) is type(via_market.trader_gateway)


def test_metadata_records_trader_broker() -> None:
    """混搭时应能从 metadata 看出交易源来自谁(与 market_broker 对称)."""
    bundle = create_gateway_bundle(
        broker=MARKET_ONLY,
        trader_broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.metadata is not None
    assert bundle.metadata.get("trader_broker") == TRADER_ONLY


def test_unknown_trader_broker_raises_with_supported_list() -> None:
    """未注册的 trader_broker 应报错并点名该参数, 而非静默无交易通道."""
    with pytest.raises(ValueError, match="trader_broker"):
        create_gateway_bundle(
            broker=MARKET_ONLY,
            trader_broker="no_such_broker",
            feed=DataFeed(),
            symbols=["600000"],
        )


def test_both_overrides_together_bypass_default_broker() -> None:
    """两侧都覆盖时, broker 只作默认源不再贡献网关."""
    bundle = create_gateway_bundle(
        broker=TRADER_ONLY,
        market_broker=MARKET_ONLY,
        trader_broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.market_gateway is not None
    assert bundle.trader_gateway is not None


def test_market_broker_equal_to_broker_is_accepted() -> None:
    """market_broker 与 broker 同名时应等价于单 broker, 不重复构建."""
    bundle = create_gateway_bundle(
        broker=MARKET_ONLY,
        market_broker=MARKET_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.market_gateway is not None
    assert bundle.trader_gateway is None
