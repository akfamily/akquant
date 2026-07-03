import base64

import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

from akquant import DataFeed
from akquant.gateway import create_gateway_bundle


def _options() -> dict:
    """Build the gateway_options for a qmf bundle."""
    return dict(
        base_url="http://gw.test",
        ws_url="ws://gw.test/api/v1/stream",
        qmf_user_id="u",
        account_content="8888000001",
        password="pw",
        input_content="1",
        content_type="1",
        password_key=base64.b64encode(b"0" * 32).decode("ascii"),
    )


def test_factory_builds_qmf_bundle() -> None:
    """The factory wires a qmf bundle with trader gateway + capabilities."""
    bundle = create_gateway_bundle(
        broker="qmf", feed=DataFeed(), symbols=["600000.SH"], **_options()
    )
    assert bundle.metadata["broker"] == "qmf"
    assert bundle.trader_gateway is not None
    assert bundle.trader_capabilities.broker_name == "qmf"


def test_factory_qmf_missing_option_raises() -> None:
    """A missing required option raises ValueError."""
    opts = _options()
    del opts["password_key"]
    with pytest.raises((ValueError, TypeError)):
        create_gateway_bundle(
            broker="qmf", feed=DataFeed(), symbols=["600000.SH"], **opts
        )


def test_unknown_broker_lists_qmf() -> None:
    """Unknown broker error message includes qmf."""
    with pytest.raises(ValueError) as exc:
        create_gateway_bundle(broker="nope", feed=DataFeed(), symbols=[])
    assert "qmf" in str(exc.value)
