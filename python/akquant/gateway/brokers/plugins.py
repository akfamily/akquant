"""通过 entry-points 发现并注册第三方/私有 broker 插件."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "akquant.brokers"
_PLUGINS_LOADED = False


def register_plugin_brokers() -> None:
    """发现并注册所有 akquant.brokers 插件（幂等、失败隔离）."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 — 发现阶段失败也不能拖垮 import akquant
        logger.warning("发现 broker 插件失败，已跳过", exc_info=True)
        return
    for ep in eps:
        try:
            register = ep.load()
            register()
        except Exception:  # noqa: BLE001 — 单插件失败不拖垮其余与 akquant
            logger.warning("加载 broker 插件 %r 失败，已跳过", ep.name, exc_info=True)
