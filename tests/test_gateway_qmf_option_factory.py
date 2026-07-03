import base64

import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

from akquant import DataFeed
from akquant.gateway import create_gateway_bundle


def _opts(**over) -> dict:
    """Build base qmf gateway_options; override via kwargs."""
    base = dict(
        base_url="http://gw.test",
        ws_url="ws://gw.test/api/v1/stream",
        qmf_user_id="u",
        account_content="8888000001",
        password="pw",
        input_content="1",
        content_type="1",
        password_key=base64.b64encode(b"0" * 32).decode("ascii"),
    )
    base.update(over)
    return base


def test_qmf_without_options_has_no_option_features() -> None:
    """Default (enable_options absent) keeps Phase 1 securities capability."""
    bundle = create_gateway_bundle(
        broker="qmf", feed=DataFeed(), symbols=["600000.SH"], **_opts()
    )
    assert bundle.trader_capabilities.features == frozenset()
    assert bundle.trader_gateway._option_client is None


def test_qmf_with_options_declares_capability_and_session() -> None:
    """enable_options wires an option session and option capability."""
    bundle = create_gateway_bundle(
        broker="qmf",
        feed=DataFeed(),
        symbols=["600000.SH"],
        **_opts(enable_options=True),
    )
    assert "options" in bundle.trader_capabilities.features
    assert "entrust_oc" in bundle.trader_capabilities.broker_extra_fields
    assert bundle.trader_gateway._option_client is not None
    assert bundle.trader_gateway._option_client._config.asset_prop == "B"
