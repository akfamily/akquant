import threading
from typing import Any, Callable, Dict, Sequence

from ..akquant import DataFeed
from .protocols import GatewayBundle

GatewayBuilder = Callable[[DataFeed, Sequence[str], bool], GatewayBundle]

_BROKER_BUILDERS: Dict[str, Callable[..., GatewayBundle]] = {}

# 内置网关与插件的注册状态: 0 未开始, 1 进行中, 2 已完成。
_defaults_state = 0
_defaults_lock = threading.RLock()


def _ensure_default_brokers() -> None:
    """第一次用到注册表时, 登记内置网关并发现、登记 broker 插件(只做一次).

    不放在 ``import akquant`` 时做: 插件一导入就会带进它的 HTTP、WebSocket 依赖,
    只做回测的进程用不上。本模块每个对外函数开头都先调用它, 所以默认网关总是先于
    调用方自己的登记、注销生效, 先后顺序与导入时注册相同(用户登记的同名网关覆盖
    内置的)。

    登记过程自身会调用 ``register_broker``: 同一线程重入时状态已是"进行中", 直接
    返回; 其他线程在锁上等到登记完成。
    """
    global _defaults_state
    if _defaults_state == 2:
        return
    with _defaults_lock:
        if _defaults_state != 0:
            return
        _defaults_state = 1
        try:
            from .brokers.builtins import register_builtin_brokers
            from .brokers.plugins import register_plugin_brokers

            register_builtin_brokers()
            register_plugin_brokers()
        finally:
            _defaults_state = 2


def register_broker(name: str, builder: Callable[..., GatewayBundle]) -> None:
    """Register a custom broker builder."""
    _ensure_default_brokers()
    broker_key = name.strip().lower()
    if not broker_key:
        raise ValueError("broker name cannot be empty")
    _BROKER_BUILDERS[broker_key] = builder


def unregister_broker(name: str) -> None:
    """Unregister a custom broker builder."""
    _ensure_default_brokers()
    broker_key = name.strip().lower()
    _BROKER_BUILDERS.pop(broker_key, None)


def get_broker_builder(name: str) -> Callable[..., GatewayBundle] | None:
    """Get custom broker builder by key."""
    _ensure_default_brokers()
    broker_key = name.strip().lower()
    return _BROKER_BUILDERS.get(broker_key)


def list_registered_brokers() -> list[str]:
    """List registered custom broker keys."""
    _ensure_default_brokers()
    return sorted(_BROKER_BUILDERS.keys())


def create_registered_gateway_bundle(
    name: str,
    feed: DataFeed,
    symbols: Sequence[str],
    use_aggregator: bool,
    **kwargs: Any,
) -> GatewayBundle | None:
    """Create gateway bundle from custom registry if broker exists."""
    builder = get_broker_builder(name)
    if builder is None:
        return None
    return builder(
        feed=feed,
        symbols=symbols,
        use_aggregator=use_aggregator,
        **kwargs,
    )
