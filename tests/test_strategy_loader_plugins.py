"""策略加载器的 entry-point 插件发现测试.

与 ``tests/test_broker_plugins.py`` 同构 —— broker / signal source 都有
entry-point 自动发现, 本文件覆盖 strategy loader 的那一份。
"""

import logging
from typing import Any, Callable

import akquant.strategy_loader as loader_mod
import pytest
from akquant.strategy_loader import (
    get_strategy_loader,
    register_strategy_loader,
)


class _FakeEP:
    """伪 entry point."""

    def __init__(self, name: str, register: Callable[[], None]) -> None:
        self.name = name
        self._register = register

    def load(self) -> Callable[[], None]:
        return self._register


def _patch_eps(monkeypatch: Any, eps: list[_FakeEP]) -> None:
    """把 entry_points 换成测试用的固定列表."""
    monkeypatch.setattr(
        loader_mod,
        "entry_points",
        lambda group: list(eps) if group == loader_mod.ENTRY_POINT_GROUP else [],
    )


def _unregister(name: str) -> None:
    """清理注册表, 避免污染其他测试."""
    loader_mod._LOADERS.pop(name, None)


def test_register_plugin_strategy_loaders_registers_from_entry_points(
    monkeypatch: Any,
) -> None:
    """entry-point 声明的加载器应被注册进来."""
    loader_mod._PLUGINS_LOADED = False

    def fake_register() -> None:
        register_strategy_loader("faketest_loader", lambda src, opts: lambda s, b: None)

    _patch_eps(monkeypatch, [_FakeEP("faketest_loader", fake_register)])
    try:
        loader_mod.register_plugin_strategy_loaders()
        assert get_strategy_loader("faketest_loader") is not None
    finally:
        _unregister("faketest_loader")


def test_register_plugin_strategy_loaders_isolates_failures(
    monkeypatch: Any, caplog: Any
) -> None:
    """单个插件加载失败只记警告, 不得抛出."""
    loader_mod._PLUGINS_LOADED = False

    def boom() -> None:
        raise RuntimeError("plugin broke")

    _patch_eps(monkeypatch, [_FakeEP("bad_loader", boom)])
    with caplog.at_level(logging.WARNING):
        loader_mod.register_plugin_strategy_loaders()
    assert "bad_loader" in caplog.text


def test_register_plugin_strategy_loaders_is_idempotent(monkeypatch: Any) -> None:
    """重复调用只发现一次."""
    loader_mod._PLUGINS_LOADED = False
    count = {"n": 0}

    def once() -> None:
        count["n"] += 1

    _patch_eps(monkeypatch, [_FakeEP("countep_loader", once)])
    loader_mod.register_plugin_strategy_loaders()
    loader_mod.register_plugin_strategy_loaders()
    assert count["n"] == 1


def test_register_plugin_strategy_loaders_survives_entry_points_error(
    monkeypatch: Any, caplog: Any
) -> None:
    """entry_points() 本身炸掉也不能拖垮 import akquant."""
    loader_mod._PLUGINS_LOADED = False

    def boom(group: str) -> list[object]:
        raise RuntimeError("corrupt dist metadata")

    monkeypatch.setattr(loader_mod, "entry_points", boom)
    with caplog.at_level(logging.WARNING):
        loader_mod.register_plugin_strategy_loaders()
    assert "发现策略加载器插件失败" in caplog.text


def test_get_strategy_loader_triggers_plugin_discovery(monkeypatch: Any) -> None:
    """按名取加载器时应自动触发插件发现, 用户无需手动 import 插件包."""
    loader_mod._PLUGINS_LOADED = False

    def fake_register() -> None:
        register_strategy_loader("lazy_discovered", lambda src, opts: lambda s, b: None)

    _patch_eps(monkeypatch, [_FakeEP("lazy_discovered", fake_register)])
    try:
        # 未手动调 register_plugin_strategy_loaders, 直接按名取
        assert get_strategy_loader("lazy_discovered") is not None
    finally:
        _unregister("lazy_discovered")


def test_plugin_discovery_is_lazy_not_at_import(monkeypatch: Any) -> None:
    """发现必须是懒的: import akquant 时不得执行第三方插件代码."""
    # import akquant 已在文件顶部完成; 若发现是 import 期做的,
    # 此刻 flag 应已为 True 且无法观测到"首次调用才发现"的行为。
    loader_mod._PLUGINS_LOADED = False
    called = {"n": 0}

    def spy() -> None:
        called["n"] += 1

    _patch_eps(monkeypatch, [_FakeEP("spy_loader", spy)])
    assert called["n"] == 0, "打补丁后、调用前不得已执行插件"
    loader_mod.register_plugin_strategy_loaders()
    assert called["n"] == 1


def test_unknown_strategy_loader_error_lists_available(monkeypatch: Any) -> None:
    """未知加载器的报错应列出可用项, 否则插件没装时用户无从判断."""
    loader_mod._PLUGINS_LOADED = True  # 跳过发现, 只看内置项
    with pytest.raises(ValueError) as exc:
        get_strategy_loader("no_such_loader_xyz")
    message = str(exc.value)
    assert "no_such_loader_xyz" in message
    assert "python_plain" in message, "报错须列出已注册的可用加载器"
