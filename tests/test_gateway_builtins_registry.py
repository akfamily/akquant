import pytest
from akquant import DataFeed
from akquant.gateway import create_gateway_bundle, list_registered_brokers


def test_builtins_registered_on_import() -> None:
    """Importing akquant.gateway registers all built-in brokers."""
    names = set(list_registered_brokers())
    assert {"ctp", "miniqmt", "ptrade"} <= names


def test_unknown_broker_lists_builtins() -> None:
    """Unknown broker error names the registered brokers."""
    with pytest.raises(ValueError) as exc:
        create_gateway_bundle(broker="nope", feed=DataFeed(), symbols=[])
    msg = str(exc.value)
    assert "ctp" in msg


def test_ctp_requires_md_front_unchanged() -> None:
    """Built-in ctp keeps its md_front requirement (behavior parity)."""
    with pytest.raises(ValueError):
        create_gateway_bundle(broker="ctp", feed=DataFeed(), symbols=["x"])
