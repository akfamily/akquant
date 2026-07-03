from .broker_event_bridge import BrokerEventBridge
from .broker_event_mapper import BrokerEventMapper, create_default_mapper
from .broker_models import (
    BrokerCapability,
    UnifiedAccount,
    UnifiedErrorType,
    UnifiedExecutionReport,
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedPosition,
    UnifiedTrade,
    normalize_position_effect,
    validate_execution_semantics,
)
from .broker_recovery import BrokerRecovery
from .broker_runtime import BrokerRuntime
from .brokers.ctp.adapter import CTPMarketAdapter, CTPTraderAdapter
from .brokers.ctp.native import CTPMarketGateway, CTPTraderGateway
from .brokers.miniqmt.stub import MiniQMTMarketGateway, MiniQMTTraderGateway
from .brokers.ptrade.stub import PTradeMarketGateway, PTradeTraderGateway
from .factory import create_gateway_bundle
from .order_submitter import BrokerOrderSubmitter
from .protocols import GatewayBundle, MarketGateway, TraderGateway
from .registry import (
    get_broker_builder,
    list_registered_brokers,
    register_broker,
    unregister_broker,
)

__all__ = [
    "MarketGateway",
    "TraderGateway",
    "GatewayBundle",
    "BrokerCapability",
    "BrokerEventBridge",
    "BrokerRecovery",
    "BrokerRuntime",
    "BrokerOrderSubmitter",
    "UnifiedOrderStatus",
    "UnifiedErrorType",
    "UnifiedExecutionReport",
    "UnifiedOrderRequest",
    "UnifiedOrderSnapshot",
    "UnifiedTrade",
    "UnifiedAccount",
    "UnifiedPosition",
    "normalize_position_effect",
    "validate_execution_semantics",
    "BrokerEventMapper",
    "create_default_mapper",
    "CTPMarketGateway",
    "CTPTraderGateway",
    "CTPMarketAdapter",
    "CTPTraderAdapter",
    "MiniQMTMarketGateway",
    "MiniQMTTraderGateway",
    "PTradeMarketGateway",
    "PTradeTraderGateway",
    "create_gateway_bundle",
    "register_broker",
    "unregister_broker",
    "get_broker_builder",
    "list_registered_brokers",
    "register_builtin_brokers",
]

from .brokers.builtins import register_builtin_brokers

register_builtin_brokers()
