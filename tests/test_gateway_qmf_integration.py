"""对 chibi_quant --mock 的端到端联调，默认跳过.

启用：先起 mock 网关，再设置 QMF_BASE_URL 指向它后运行 pytest。
"""

import base64
import os

import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

_BASE = os.getenv("QMF_BASE_URL")


@pytest.mark.skipif(not _BASE, reason="需设置 QMF_BASE_URL 指向运行中的 mock 网关")
def test_end_to_end_login_order_query() -> None:
    """Login, place an order and query account against a running gateway."""
    from akquant import DataFeed
    from akquant.gateway import create_gateway_bundle
    from akquant.gateway.broker_models import UnifiedOrderRequest

    bundle = create_gateway_bundle(
        broker="qmf",
        feed=DataFeed(),
        symbols=["600000.SH"],
        base_url=_BASE,
        ws_url=os.getenv("QMF_WS_URL", _BASE.replace("http", "ws") + "/api/v1/stream"),
        qmf_user_id="u",
        account_content="8888000001",
        password="pw",
        input_content="1",
        content_type="1",
        password_key=base64.b64encode(b"0" * 32).decode("ascii"),
    )
    trader = bundle.trader_gateway
    assert trader is not None
    trader.connect()
    assert trader.query_account() is not None
    order_id = trader.place_order(
        UnifiedOrderRequest(
            client_order_id="it-1",
            symbol="600000.SH",
            side="Buy",
            quantity=100,
            price=10.5,
            order_type="Limit",
        )
    )
    assert order_id
    trader.disconnect()
