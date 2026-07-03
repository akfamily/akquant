from akquant.gateway.protocols import GatewayBundle


def test_bundle_allows_omitting_market_gateway() -> None:
    """A trade-only bundle can omit the market gateway entirely."""
    # Omitting market_gateway would raise TypeError before it gained a default;
    # this is the real RED/GREEN for making market data optional.
    bundle = GatewayBundle(trader_gateway=None)
    assert bundle.market_gateway is None


def test_bundle_accepts_explicit_none_market_gateway() -> None:
    """Passing market_gateway=None is also accepted."""
    bundle = GatewayBundle(market_gateway=None, trader_gateway=None)
    assert bundle.market_gateway is None
