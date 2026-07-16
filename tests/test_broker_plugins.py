import logging

import akquant.gateway.brokers.plugins as plugins
from akquant.gateway.registry import (
    get_broker_builder,
    register_broker,
    unregister_broker,
)


class _FakeEP:
    """Fake entry point for testing."""

    def __init__(self, name, register):
        self.name = name
        self._register = register

    def load(self):
        return self._register


def _patch_eps(monkeypatch, eps):
    """Patch entry_points to return test entry points."""
    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda group: list(eps) if group == plugins.ENTRY_POINT_GROUP else [],
    )


def test_register_plugin_brokers_registers_from_entry_points(monkeypatch):
    """Test that brokers are registered from entry points."""
    plugins._PLUGINS_LOADED = False

    def fake_register():
        register_broker("faketest", lambda **kw: None)

    _patch_eps(monkeypatch, [_FakeEP("faketest", fake_register)])
    plugins.register_plugin_brokers()
    assert get_broker_builder("faketest") is not None
    unregister_broker("faketest")


def test_register_plugin_brokers_isolates_failures(monkeypatch, caplog):
    """Test that plugin loading failures are isolated and logged."""
    plugins._PLUGINS_LOADED = False

    def boom():
        raise RuntimeError("plugin broke")

    _patch_eps(monkeypatch, [_FakeEP("bad", boom)])
    with caplog.at_level(logging.WARNING):
        plugins.register_plugin_brokers()  # 不得抛出
    assert "bad" in caplog.text


def test_register_plugin_brokers_is_idempotent(monkeypatch):
    """Test that register_plugin_brokers is idempotent."""
    plugins._PLUGINS_LOADED = False
    count = {"n": 0}

    def once():
        count["n"] += 1

    _patch_eps(monkeypatch, [_FakeEP("countep", once)])
    plugins.register_plugin_brokers()
    plugins.register_plugin_brokers()
    assert count["n"] == 1


def test_register_plugin_brokers_survives_entry_points_error(monkeypatch, caplog):
    """Test that a broken entry_points() discovery does not propagate."""
    plugins._PLUGINS_LOADED = False

    def boom(group):
        raise RuntimeError("corrupt dist metadata")

    monkeypatch.setattr(plugins, "entry_points", boom)
    with caplog.at_level(logging.WARNING):
        plugins.register_plugin_brokers()  # must not raise
    assert "发现 broker 插件失败" in caplog.text
