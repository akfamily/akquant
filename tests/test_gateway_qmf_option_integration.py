"""对 chibi_quant --mock 的期权端到端联调，默认跳过（需 QMF_BASE_URL）."""

import base64
import os

import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

_BASE = os.getenv("QMF_BASE_URL")


@pytest.mark.skipif(not _BASE, reason="需设置 QMF_BASE_URL 指向运行中的 mock 网关")
def test_option_end_to_end() -> None:
    """Enable options, place an option order and query positions live."""
    from akquant import DataFeed
    from akquant.gateway import create_gateway_bundle
    from akquant.gateway.broker_models import UnifiedOrderRequest

    bundle = create_gateway_bundle(
        broker="qmf",
        feed=DataFeed(),
        symbols=["10003456.SH"],
        base_url=_BASE,
        ws_url=os.getenv("QMF_WS_URL", _BASE.replace("http", "ws") + "/api/v1/stream"),
        qmf_user_id="u",
        account_content="8888000001",
        password="pw",
        input_content="1",
        content_type="1",
        password_key=base64.b64encode(b"0" * 32).decode("ascii"),
        enable_options=True,
    )
    trader = bundle.trader_gateway
    trader.connect()
    order_id = trader.place_order(
        UnifiedOrderRequest(
            client_order_id="it-opt-1",
            symbol="10003456.SH",
            side="Buy",
            quantity=1,
            price=0.05,
            order_type="Limit",
            asset_type="option",
            extra={"entrust_oc": "O", "covered_flag": "0", "entrust_prop": "F0"},
        )
    )
    assert order_id
    assert any(p.symbol == "10003456.SH" for p in trader.query_positions())
    trader.disconnect()


class _WsFake:
    """Client stub exposing a token and a no-op close for WS wiring tests."""

    def __init__(self, token: str) -> None:
        """Record the session token."""
        self.token = token
        self.fund_account = "8888000001"

    def close(self) -> None:
        """Ignore close in wiring tests."""


def test_start_opens_second_ws_bound_to_option_token(monkeypatch) -> None:
    """Open a second push client bound to the option token; disconnect stops both."""
    pytest.importorskip("websocket")
    from akquant.gateway.brokers.qmf import adapter as adapter_mod
    from akquant.gateway.brokers.qmf.adapter import QMFTraderGateway

    started: list = []
    stopped: list = []
    monkeypatch.setattr(
        adapter_mod.QMFPushClient, "start", lambda self: started.append(self._token)
    )
    monkeypatch.setattr(
        adapter_mod.QMFPushClient, "stop", lambda self: stopped.append(self._token)
    )

    gw = QMFTraderGateway(
        client=_WsFake("gw-sec"),
        ws_url="ws://gw.test/api/v1/stream",
        option_client=_WsFake("gw-opt"),
    )
    gw.start()
    assert set(started) == {"gw-sec", "gw-opt"}
    assert gw._option_push is not None
    assert gw._option_push._token == "gw-opt"

    gw.disconnect()
    assert set(stopped) == {"gw-sec", "gw-opt"}


def test_start_without_option_client_has_no_second_ws(monkeypatch) -> None:
    """Without an option client, start() opens only the securities push client."""
    pytest.importorskip("websocket")
    from akquant.gateway.brokers.qmf import adapter as adapter_mod
    from akquant.gateway.brokers.qmf.adapter import QMFTraderGateway

    started: list = []
    monkeypatch.setattr(
        adapter_mod.QMFPushClient, "start", lambda self: started.append(self._token)
    )

    gw = QMFTraderGateway(client=_WsFake("gw-sec"), ws_url="ws://gw.test/api/v1/stream")
    gw.start()
    assert started == ["gw-sec"]
    assert gw._option_push is None
