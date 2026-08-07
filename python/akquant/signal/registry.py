"""信号源注册表与 entry-point 插件发现.

让第三方/私有仓库提供自己的信号源实现, 用法与 broker 插件对称
(见 ``gateway/brokers/plugins.py``):

```toml
# 插件仓的 pyproject.toml
[project.entry-points."akquant.signal_sources"]
my_platform = "my_pkg.signal:register"
```

```python
# my_pkg/signal.py
from akquant.signal import register_signal_source

def register() -> None:
    register_signal_source("my_platform", MySignalSource)
```

之后 ``create_signal_source("my_platform", **options)`` 即可构造。
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable, Dict, Optional

from ..log import build_log_extra, get_logger

logger = get_logger("signal.registry")

ENTRY_POINT_GROUP = "akquant.signal_sources"

_FACTORIES: Dict[str, Callable[..., Any]] = {}
_PLUGINS_LOADED = False


def register_signal_source(name: str, factory: Callable[..., Any]) -> None:
    """注册一个信号源工厂(可调用, 接受关键字参数并返回 SignalSource)."""
    key = name.strip().lower()
    if not key:
        raise ValueError("信号源名称不能为空")
    _FACTORIES[key] = factory


def unregister_signal_source(name: str) -> None:
    """注销(便于测试与热替换)."""
    _FACTORIES.pop(name.strip().lower(), None)


def get_signal_source_factory(name: str) -> Optional[Callable[..., Any]]:
    """按名取工厂; 未注册返回 None."""
    return _FACTORIES.get(name.strip().lower())


def list_signal_sources() -> list[str]:
    """列出已注册的信号源名称."""
    register_plugin_signal_sources()
    return sorted(_FACTORIES.keys())


def create_signal_source(name: str, **options: Any) -> Any:
    """按名构造信号源; 未注册则报错并列出可用项."""
    register_plugin_signal_sources()
    factory = get_signal_source_factory(name)
    if factory is None:
        available = ", ".join(sorted(_FACTORIES.keys())) or "(无)"
        raise ValueError(f"未知信号源 {name!r}; 可用: {available}")
    return factory(**options)


def register_builtin_signal_sources() -> None:
    """注册内置信号源(幂等)."""
    from .sources import HttpSignalSource, QueueSignalSource, RedisSignalSource

    register_signal_source("queue", QueueSignalSource)
    register_signal_source("http", HttpSignalSource)
    register_signal_source("redis", RedisSignalSource)


def register_plugin_signal_sources() -> None:
    """发现并注册 entry-point 插件(幂等、失败隔离).

    单个插件加载失败只记警告, 不拖垮其余插件与 ``import akquant`` —— 与
    ``gateway/brokers/plugins.py`` 同一范式。
    """
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 — 发现阶段失败也不能拖垮 import
        logger.warning(
            "发现信号源插件失败, 已跳过",
            exc_info=True,
            extra=build_log_extra(phase="signal"),
        )
        return
    for entry in eps:
        try:
            entry.load()()
        except Exception:  # noqa: BLE001 — 单插件失败不拖垮其余
            logger.warning(
                "加载信号源插件 %r 失败, 已跳过",
                entry.name,
                exc_info=True,
                extra=build_log_extra(phase="signal"),
            )
