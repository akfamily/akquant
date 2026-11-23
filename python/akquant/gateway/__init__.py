from typing import TYPE_CHECKING, Any

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
    "create_gateway_bundle",
    "register_broker",
    "unregister_broker",
    "get_broker_builder",
    "list_registered_brokers",
    "register_builtin_brokers",
    "register_plugin_brokers",
]

# 以下名字按需加载: 柜台适配器只在用到时导入; 内置网关与插件的注册推迟到第一次
# 用到注册表时(见 registry._ensure_default_brokers), 不在 import akquant 时做。
_LAZY_EXPORTS = {
    "CTPMarketAdapter": ".brokers.ctp.adapter",
    "CTPTraderAdapter": ".brokers.ctp.adapter",
    "CTPMarketGateway": ".brokers.ctp.native",
    "CTPTraderGateway": ".brokers.ctp.native",
    "register_builtin_brokers": ".brokers.builtins",
    "register_plugin_brokers": ".brokers.plugins",
}

if TYPE_CHECKING:
    from .brokers.builtins import register_builtin_brokers
    from .brokers.ctp.adapter import CTPMarketAdapter, CTPTraderAdapter
    from .brokers.ctp.native import CTPMarketGateway, CTPTraderGateway
    from .brokers.plugins import register_plugin_brokers


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
