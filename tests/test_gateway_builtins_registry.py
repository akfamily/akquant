import base64

import pytest
from akquant import DataFeed
from akquant.gateway import create_gateway_bundle, list_registered_brokers


def test_builtins_registered_on_import() -> None:
    """Importing akquant.gateway registers all built-in brokers."""
    names = set(list_registered_brokers())
    assert {"ctp", "miniqmt", "ptrade", "qmf"} <= names


def test_unknown_broker_lists_builtins() -> None:
    """Unknown broker error names the registered brokers."""
    with pytest.raises(ValueError) as exc:
        create_gateway_bundle(broker="nope", feed=DataFeed(), symbols=[])
    msg = str(exc.value)
    assert "ctp" in msg and "qmf" in msg


def test_ctp_requires_md_front_unchanged() -> None:
    """Built-in ctp keeps its md_front requirement (behavior parity)."""
    with pytest.raises(ValueError):
        create_gateway_bundle(broker="ctp", feed=DataFeed(), symbols=["x"])


def test_qmf_still_builds() -> None:
    """Qmf builder still assembles a trader bundle after registry-ization."""
    pytest.importorskip("httpx")
    pytest.importorskip("cryptography")
    bundle = create_gateway_bundle(
        broker="qmf",
        feed=DataFeed(),
        symbols=["600000.SH"],
        base_url="http://gw.test",
        ws_url="ws://gw.test/api/v1/stream",
        qmf_user_id="u",
        account_content="8888000001",
        password="pw",
        input_content="1",
        content_type="1",
        password_key=base64.b64encode(b"0" * 32).decode("ascii"),
    )
    assert bundle.metadata["broker"] == "qmf"
    assert bundle.trader_gateway is not None
