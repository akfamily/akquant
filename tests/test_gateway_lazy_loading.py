"""网关包按需加载: 只导入 akquant 不加载柜台适配器, 也不扫描、导入 broker 插件.

内置网关与插件的注册推迟到第一次用到注册表时。每个场景在子进程里跑, 不受本进程
已导入模块的影响。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_import_akquant_does_not_load_broker_adapters_or_plugins() -> None:
    """Import akquant 之后, 柜台适配器与插件模块都还没有加载."""
    out = _run(
        """
        import sys
        import akquant
        from importlib.metadata import entry_points

        loaded = [m for m in sys.modules if m.startswith("akquant.gateway.brokers.")]
        plugins = {ep.value.split(":")[0].split(".")[0]
                   for ep in entry_points(group="akquant.brokers")}
        print(loaded, sorted(plugins & set(sys.modules)))
        """
    )
    assert out == "[] []"


def test_registry_lookup_registers_builtin_brokers_on_demand() -> None:
    """第一次查注册表时内置网关已就位; 建网关的入口同样可用."""
    out = _run(
        """
        import akquant
        from akquant.gateway import registry

        names = registry.list_registered_brokers()
        replay = registry.get_broker_builder("replay")
        print("ctp" in names, "replay" in names, replay is not None)
        """
    )
    assert out == "True True True"


def test_user_registration_still_overrides_builtin_broker() -> None:
    """用户在任何查询之前登记同名网关, 结果仍是用户的(内置的不会后来居上)."""
    out = _run(
        """
        import akquant
        from akquant.gateway import (
            get_broker_builder,
            register_broker,
            unregister_broker,
        )

        def mine(**kwargs):
            return None

        register_broker("ctp", mine)
        print(get_broker_builder("ctp") is mine)
        unregister_broker("replay")
        print(get_broker_builder("replay") is None)
        """
    )
    assert out.split() == ["True", "True"]


def test_lazy_names_are_still_importable_from_gateway_package() -> None:
    """按需加载的名字仍能从 akquant.gateway 直接导入."""
    out = _run(
        """
        from akquant.gateway import (
            CTPMarketAdapter,
            CTPMarketGateway,
            CTPTraderAdapter,
            CTPTraderGateway,
            register_builtin_brokers,
            register_plugin_brokers,
        )
        import akquant.gateway as gateway

        print(all(name in dir(gateway) for name in gateway.__all__))
        """
    )
    assert out == "True"
