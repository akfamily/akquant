from typing import Any, Sequence

from ..akquant import DataFeed
from .protocols import GatewayBundle
from .registry import create_registered_gateway_bundle, list_registered_brokers


def create_gateway_bundle(
    broker: str,
    feed: DataFeed,
    symbols: Sequence[str],
    use_aggregator: bool = True,
    **kwargs: Any,
) -> GatewayBundle:
    """Create a market/trader gateway bundle by broker key (registry-based)."""
    broker_key = broker.strip().lower()
    bundle = create_registered_gateway_bundle(
        name=broker_key,
        feed=feed,
        symbols=symbols,
        use_aggregator=use_aggregator,
        **kwargs,
    )
    if bundle is not None:
        return bundle
    supported = ", ".join(list_registered_brokers())
    raise ValueError(f"broker must be one of: {supported}")
