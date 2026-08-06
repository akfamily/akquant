from typing import Any, Dict, Optional, Sequence

from ..akquant import DataFeed
from .protocols import GatewayBundle
from .registry import create_registered_gateway_bundle, list_registered_brokers


def create_gateway_bundle(
    broker: str,
    feed: DataFeed,
    symbols: Sequence[str],
    use_aggregator: bool = True,
    market_broker: Optional[str] = None,
    trader_broker: Optional[str] = None,
    **kwargs: Any,
) -> GatewayBundle:
    """Create a market/trader gateway bundle by broker key (registry-based).

    ``broker`` 是两侧的**默认源**; ``market_broker`` / ``trader_broker`` 各自
    覆盖一侧, 用于把行情源与交易源分开指定。这补齐了两类单边 broker 的组合——
    ``broker='qmf'`` 只有交易通道、``broker='replay'`` 只有行情通道, 此前二者
    无法拼接。以下两种写法等价(都是"行情来自 replay、交易来自 qmf")::

        create_gateway_bundle(broker="qmf", market_broker="replay", ...)
        create_gateway_bundle(broker="replay", trader_broker="qmf", ...)

    两侧都不覆盖时, 返回的 bundle 与单 broker 时完全一致(含 metadata)。
    """
    broker_key = broker.strip().lower()
    market_key = (market_broker or "").strip().lower()
    trader_key = (trader_broker or "").strip().lower()

    # 同名视为未覆盖: builder 可能连柜台/起线程, 同一 broker 不应构建两次。
    if market_key == broker_key:
        market_key = ""
    if trader_key == broker_key:
        trader_key = ""

    built: Dict[str, GatewayBundle] = {}

    def _build(key: str, param_name: str) -> GatewayBundle:
        """按 registry 构建并缓存; 未注册则报错并点名出错的参数."""
        if key in built:
            return built[key]
        bundle = create_registered_gateway_bundle(
            name=key,
            feed=feed,
            symbols=symbols,
            use_aggregator=use_aggregator,
            **kwargs,
        )
        if bundle is None:
            supported = ", ".join(list_registered_brokers())
            raise ValueError(f"{param_name} must be one of: {supported}")
        built[key] = bundle
        return bundle

    base_bundle = _build(broker_key, "broker")
    if not market_key and not trader_key:
        return base_bundle

    market_bundle = _build(market_key, "market_broker") if market_key else base_bundle
    trader_bundle = _build(trader_key, "trader_broker") if trader_key else base_bundle

    # 行情侧 metadata 作底、交易侧覆盖其上: 行情 broker 可能声明会话级信息
    # (如 replay 的 ``bounded_event_total`` —— runner 据此在事件放完后结束会话),
    # 只保留交易侧会丢掉它, 混搭会话便只能等 duration 墙钟超时。``broker`` 键由
    # 交易侧覆盖胜出, 与单 broker 时一致。
    metadata = {**(market_bundle.metadata or {}), **(trader_bundle.metadata or {})}
    if market_key:
        metadata["market_broker"] = market_key
    if trader_key:
        metadata["trader_broker"] = trader_key
    return GatewayBundle(
        market_gateway=market_bundle.market_gateway,
        trader_gateway=trader_bundle.trader_gateway,
        trader_capabilities=trader_bundle.trader_capabilities,
        metadata=metadata,
    )
