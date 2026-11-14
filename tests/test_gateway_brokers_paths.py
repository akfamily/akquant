from akquant.gateway import (
    BrokerEventBridge,
    BrokerOrderSubmitter,
    BrokerRecovery,
    BrokerRuntime,
)
from akquant.gateway.brokers.ctp.adapter import CTPMarketAdapter, CTPTraderAdapter
from akquant.gateway.brokers.ctp.native import CTPMarketGateway, CTPTraderGateway


def test_gateway_top_level_exports_new_runtime_components() -> None:
    """Top-level gateway package should expose the new runtime building blocks."""
    assert BrokerRuntime.__name__ == "BrokerRuntime"
    assert BrokerEventBridge.__name__ == "BrokerEventBridge"
    assert BrokerRecovery.__name__ == "BrokerRecovery"
    assert BrokerOrderSubmitter.__name__ == "BrokerOrderSubmitter"


def test_new_ctp_paths_remain_importable() -> None:
    """Direct imports from brokers/ctp should expose adapter/native classes."""
    assert CTPMarketAdapter.__name__ == "CTPMarketAdapter"
    assert CTPTraderAdapter.__name__ == "CTPTraderAdapter"
    assert CTPMarketGateway.__name__ == "CTPMarketGateway"
    assert CTPTraderGateway.__name__ == "CTPTraderGateway"
