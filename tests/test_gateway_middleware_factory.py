import pytest

pytest.importorskip("httpx")

from akquant.gateway.brokers.builtins import register_builtin_brokers
from akquant.gateway.brokers.middleware.adapter import MiddlewareTraderGateway
from akquant.gateway.registry import (
    create_registered_gateway_bundle,
    list_registered_brokers,
)


def _kwargs() -> dict:
    return {
        "base_url": "http://gw.test/api/v1",
        "broker_id": "hengsheng",
        "fund_account": "20432166",
        "password": "pw",
        "ws_url": "ws://gw.test/api/v1/ws",
    }


def test_middleware_broker_is_registered() -> None:
    """register_builtin_brokers registers the middleware broker."""
    register_builtin_brokers()
    assert "middleware" in list_registered_brokers()


def test_build_bundle_produces_middleware_trader_gateway() -> None:
    """Building the bundle yields a MiddlewareTraderGateway with capabilities."""
    register_builtin_brokers()
    bundle = create_registered_gateway_bundle(
        "middleware",
        feed=None,
        symbols=[],
        use_aggregator=False,
        **_kwargs(),
    )
    assert bundle is not None
    assert isinstance(bundle.trader_gateway, MiddlewareTraderGateway)
    assert bundle.trader_capabilities.broker_name == "middleware"
    assert bundle.metadata["broker"] == "middleware"


def test_missing_required_kwargs_raises() -> None:
    """Missing required config raises a clear ValueError."""
    register_builtin_brokers()
    kwargs = _kwargs()
    del kwargs["fund_account"]
    with pytest.raises(ValueError, match="fund_account"):
        create_registered_gateway_bundle(
            "middleware",
            feed=None,
            symbols=[],
            use_aggregator=False,
            **kwargs,
        )
