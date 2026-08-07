"""行情源与交易源分开指定: 要么只给 broker, 要么两侧都写明.

``GatewayBundle`` 的 ``market_gateway`` / ``trader_gateway`` 是两个独立可选字段,
因此一个 broker 可以只提供其中一侧(``replay`` 只有行情, 某些券商插件只有交易)。
把两者拼起来的入口是 ``market_broker`` + ``trader_broker``:

- 只给 ``broker``: 单 broker 供两侧(原语义, 不变);
- 同时给 ``market_broker`` 与 ``trader_broker``: 各自供一侧, ``broker`` **不参与**;
- 只给其中一个: 报错。此时另一侧的来源无从确定, 而让 ``broker`` 去兼任会使它
  一词双义——读 ``broker='qmf', market_broker='replay'`` 必须先知道 "qmf 只有交易
  通道" 才能推出 broker 在这里指交易源, 参数名本身没有表达这件事。
"""

from typing import Any, Callable, Dict, List, Sequence, cast

import pytest
from akquant import DataFeed
from akquant.gateway import create_gateway_bundle
from akquant.gateway.protocols import GatewayBundle, MarketGateway, TraderGateway
from akquant.gateway.registry import register_broker, unregister_broker

MARKET_ONLY = "test_market_only"
TRADER_ONLY = "test_trader_only"
BOTH_SIDES = "test_both_sides"

# BOTH_SIDES builder 的构建次数, 用于验证同名 broker 只构建一次。
_build_counts: Dict[str, int] = {}


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
    """注册测试 broker, 用后注销以免污染全局 registry."""
    _build_counts.clear()

    def _build_market_only(
        feed: Any, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
    ) -> GatewayBundle:
        _ = (feed, use_aggregator, kwargs)
        return GatewayBundle(
            market_gateway=cast(MarketGateway, _MarketOnlyGateway(symbols)),
            trader_gateway=None,
            trader_capabilities=None,
            # market_note 代表"行情侧声明的会话级信息", 真实场景是 replay 的
            # bounded_event_total(runner 据此在事件放完后结束会话)。
            metadata={"broker": MARKET_ONLY, "market_note": "from-market-side"},
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

    def _build_both_sides(
        feed: Any, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
    ) -> GatewayBundle:
        _ = (feed, use_aggregator, kwargs)
        _build_counts[BOTH_SIDES] = _build_counts.get(BOTH_SIDES, 0) + 1
        return GatewayBundle(
            market_gateway=cast(MarketGateway, _MarketOnlyGateway(symbols)),
            trader_gateway=cast(TraderGateway, _TraderOnlyGateway()),
            trader_capabilities=None,
            metadata={"broker": BOTH_SIDES},
        )

    register_broker(MARKET_ONLY, _build_market_only)
    register_broker(TRADER_ONLY, _build_trader_only)
    register_broker(BOTH_SIDES, _build_both_sides)
    yield
    unregister_broker(MARKET_ONLY)
    unregister_broker(TRADER_ONLY)
    unregister_broker(BOTH_SIDES)


def test_split_mode_does_not_build_broker() -> None:
    """两侧都指定时 ``broker`` 完全不参与构建.

    用 ``broker='ctp'`` 作探针: 真去构建它会因缺 ``md_front`` 抛 ValueError。这正是
    ``run_live`` 的实际处境——``broker`` 默认值是 ``'ctp'``, 用户即使把两侧都写明,
    未被跳过的 ctp 也会报一句莫名其妙的 "md_front is required"。
    """
    bundle = create_gateway_bundle(
        broker="ctp",
        market_broker=MARKET_ONLY,
        trader_broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.market_gateway is not None
    assert bundle.trader_gateway is not None


def test_market_broker_alone_raises_asking_for_trader_broker() -> None:
    """只给 market_broker 应报错并要求写明 trader_broker.

    旧行为是让 ``broker`` 兼任交易侧, 那使 ``broker`` 一词双义。
    """
    with pytest.raises(ValueError, match="trader_broker"):
        create_gateway_bundle(
            broker=TRADER_ONLY,
            market_broker=MARKET_ONLY,
            feed=DataFeed(),
            symbols=["600000"],
        )


def test_trader_broker_alone_raises_asking_for_market_broker() -> None:
    """只给 trader_broker 应报错并要求写明 market_broker(与上一条对称)."""
    with pytest.raises(ValueError, match="market_broker"):
        create_gateway_bundle(
            broker=MARKET_ONLY,
            trader_broker=TRADER_ONLY,
            feed=DataFeed(),
            symbols=["600000"],
        )


def test_split_mode_market_gateway_comes_from_market_broker() -> None:
    """行情网关应来自 market_broker(而非 trader_broker 的 None)."""
    bundle = create_gateway_bundle(
        market_broker=MARKET_ONLY,
        trader_broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert isinstance(bundle.market_gateway, _MarketOnlyGateway)


def test_split_mode_trader_gateway_comes_from_trader_broker() -> None:
    """交易网关应来自 trader_broker(而非 market_broker 的 None)."""
    bundle = create_gateway_bundle(
        market_broker=MARKET_ONLY,
        trader_broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert isinstance(bundle.trader_gateway, _TraderOnlyGateway)


def test_split_mode_market_gateway_receives_requested_symbols() -> None:
    """行情 broker 应收到本次会话的标的集, 而不是空集."""
    bundle = create_gateway_bundle(
        market_broker=MARKET_ONLY,
        trader_broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000", "600001"],
    )

    assert getattr(bundle.market_gateway, "symbols", None) == ["600000", "600001"]


def test_split_mode_metadata_records_both_sides() -> None:
    """分开指定时应能从 metadata 看出两侧各自来自谁(便于日志与排障)."""
    bundle = create_gateway_bundle(
        market_broker=MARKET_ONLY,
        trader_broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.metadata is not None
    assert bundle.metadata.get("market_broker") == MARKET_ONLY
    assert bundle.metadata.get("trader_broker") == TRADER_ONLY


def test_split_mode_preserves_market_side_metadata() -> None:
    """行情侧声明的会话级信息不得因混搭而丢失.

    真实场景: ``replay`` 用 metadata 声明 ``bounded_event_total``, runner 据此在事件
    放完后结束会话。只保留交易侧 metadata 会丢掉它, 会话便只能等 duration 超时。
    """
    bundle = create_gateway_bundle(
        market_broker=MARKET_ONLY,
        trader_broker=TRADER_ONLY,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.metadata is not None
    assert bundle.metadata.get("market_note") == "from-market-side"


def test_same_broker_on_both_sides_is_built_once() -> None:
    """两侧同名时只构建一次: builder 可能连柜台/起线程, 建两次有副作用."""
    bundle = create_gateway_bundle(
        market_broker=BOTH_SIDES,
        trader_broker=BOTH_SIDES,
        feed=DataFeed(),
        symbols=["600000"],
    )

    assert bundle.market_gateway is not None
    assert bundle.trader_gateway is not None
    assert _build_counts.get(BOTH_SIDES) == 1


def test_unknown_market_broker_raises_naming_the_parameter() -> None:
    """未注册的 market_broker 应报错并点名该参数, 而非静默无行情."""
    with pytest.raises(ValueError, match="market_broker"):
        create_gateway_bundle(
            market_broker="no_such_broker",
            trader_broker=TRADER_ONLY,
            feed=DataFeed(),
            symbols=["600000"],
        )


def test_unknown_trader_broker_raises_naming_the_parameter() -> None:
    """未注册的 trader_broker 应报错并点名该参数, 而非静默无交易通道."""
    with pytest.raises(ValueError, match="trader_broker"):
        create_gateway_bundle(
            market_broker=MARKET_ONLY,
            trader_broker="no_such_broker",
            feed=DataFeed(),
            symbols=["600000"],
        )


def test_single_broker_mode_is_unchanged() -> None:
    """只给 broker 时行为完全不变(含 metadata 不多出覆盖键)."""
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
    assert "trader_broker" not in bundle.metadata


def test_missing_feed_raises_instead_of_reaching_builder() -> None:
    """``feed`` 缺失应直接报错, 而非把 None 传进 builder.

    ``feed`` 形式上可选只是为了让 ``broker`` 能有默认值(Python 不允许有默认值的参数
    排在无默认值参数之前), 实际仍必填。少了这道校验, None 会一路传到 builder, 报出
    的是各 broker 内部五花八门的 AttributeError 而非一句明确的缺参提示。
    """
    with pytest.raises(ValueError, match="feed"):
        create_gateway_bundle(broker=TRADER_ONLY, symbols=["600000"])


def test_neither_broker_nor_split_pair_raises() -> None:
    """既没有 broker 也没有两侧覆盖时应报错(而非返回空 bundle)."""
    with pytest.raises(ValueError, match="broker"):
        create_gateway_bundle(feed=DataFeed(), symbols=["600000"])


def test_unknown_broker_still_raises_naming_broker() -> None:
    """单 broker 模式下未注册的名字仍点名 broker 参数."""
    with pytest.raises(ValueError, match="broker"):
        create_gateway_bundle(
            broker="no_such_broker",
            feed=DataFeed(),
            symbols=["600000"],
        )
