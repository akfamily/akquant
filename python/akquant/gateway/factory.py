from typing import Any, Dict, Optional, Sequence

from ..akquant import DataFeed
from .protocols import GatewayBundle
from .registry import create_registered_gateway_bundle, list_registered_brokers


def create_gateway_bundle(
    broker: Optional[str] = None,
    feed: Optional[DataFeed] = None,
    symbols: Optional[Sequence[str]] = None,
    use_aggregator: bool = True,
    market_broker: Optional[str] = None,
    trader_broker: Optional[str] = None,
    **kwargs: Any,
) -> GatewayBundle:
    """Create a market/trader gateway bundle by broker key (registry-based).

    两种模式, 二选一:

    - **单 broker**: 只传 ``broker``, 由它同时提供行情与交易两侧(原语义, 不变)。
    - **分开指定**: 同时传 ``market_broker`` 与 ``trader_broker``, 各供一侧;
      此时 ``broker`` **完全不参与构建**。

    分开指定用于把两类"单边 broker"拼起来——``GatewayBundle`` 的 ``market_gateway``
    / ``trader_gateway`` 本就是独立可选字段, ``replay`` 只有行情(不能下单), 而某些
    券商/柜台插件只有交易通道(收不到行情)::

        create_gateway_bundle(market_broker="replay", trader_broker="qmf", ...)

    只传其中一个会报错: 另一侧的来源无从确定。让 ``broker`` 去兼任另一侧会使它
    一词双义——读 ``broker='qmf', market_broker='replay'`` 必须先知道"qmf 只有交易
    通道"才能推出 ``broker`` 在此指交易源, 参数名本身没有表达这件事。
    """
    # feed / symbols 形式上可选只为让 broker 能有默认值(Python 不允许有默认值的
    # 参数排在无默认值参数之前), 实际仍必填。
    if feed is None:
        raise ValueError("feed is required")

    market_key = (market_broker or "").strip().lower()
    trader_key = (trader_broker or "").strip().lower()

    if bool(market_key) != bool(trader_key):
        missing = "trader_broker" if market_key else "market_broker"
        given = "market_broker" if market_key else "trader_broker"
        raise ValueError(
            f"{given} 已指定, 但缺少 {missing}: 行情源与交易源必须同时写明"
            f"(或改为只传 broker 让单个 broker 供两侧)"
        )

    built: Dict[str, GatewayBundle] = {}

    def _build(key: str, param_name: str) -> GatewayBundle:
        """按 registry 构建并缓存; 未注册则报错并点名出错的参数."""
        if key in built:
            return built[key]
        bundle = create_registered_gateway_bundle(
            name=key,
            feed=feed,
            symbols=symbols or [],
            use_aggregator=use_aggregator,
            **kwargs,
        )
        if bundle is None:
            supported = ", ".join(list_registered_brokers())
            raise ValueError(f"{param_name} must be one of: {supported}")
        built[key] = bundle
        return bundle

    # 单 broker 模式。
    if not market_key:
        broker_key = (broker or "").strip().lower()
        if not broker_key:
            raise ValueError(
                "broker is required (或同时指定 market_broker 与 trader_broker)"
            )
        return _build(broker_key, "broker")

    # 分开指定: broker 不参与——它可能只是 run_live 的默认值 'ctp', 构建它会因
    # 缺少 md_front 等参数而报出与本次配置无关的错误。同名时只构建一次(builder
    # 可能连柜台/起线程, 建两次有副作用)。
    market_bundle = _build(market_key, "market_broker")
    trader_bundle = _build(trader_key, "trader_broker")

    # 行情侧 metadata 作底、交易侧覆盖其上: 行情 broker 可能声明会话级信息
    # (如 replay 的 ``bounded_event_total`` —— runner 据此在事件放完后结束会话),
    # 只保留交易侧会丢掉它, 混搭会话便只能等 duration 墙钟超时。
    metadata = {**(market_bundle.metadata or {}), **(trader_bundle.metadata or {})}
    metadata["market_broker"] = market_key
    metadata["trader_broker"] = trader_key
    return GatewayBundle(
        market_gateway=market_bundle.market_gateway,
        trader_gateway=trader_bundle.trader_gateway,
        trader_capabilities=trader_bundle.trader_capabilities,
        metadata=metadata,
    )
