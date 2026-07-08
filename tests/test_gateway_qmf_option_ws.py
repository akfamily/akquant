import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

from akquant.gateway.brokers.qmf.adapter import QMFTraderGateway


class _FakeClient:
    """Minimal client stub exposing a token and fund account."""

    def __init__(self, token: str) -> None:
        """Record the session token."""
        self.token = token
        self.fund_account = "8888000001"


def _make_gateway() -> QMFTraderGateway:
    """Build a gateway with securities + option client stubs."""
    return QMFTraderGateway(
        client=_FakeClient("gw-sec"),
        ws_url="ws://gw.test/api/v1/stream",
        option_client=_FakeClient("gw-opt"),
    )


def test_dispatch_option_push_trade_emits_option_trade() -> None:
    """A trade_update option frame emits a parsed option trade with mapped client id."""
    gw = _make_gateway()
    gw.record_broker_order("9000000001", "cli-1")
    trades: list = []
    gw.on_trade(lambda t: trades.append(t))

    gw._dispatch_option_push(
        "trade_update",
        {
            "serial_no": "T0000001",
            "entrust_no": "9000000001",
            "exchange_type": "1",
            "option_code": "10003456",
            "entrust_bs": "1",
            "business_amount": "1",
            "opt_business_price": "0.0500",
        },
    )

    assert len(trades) == 1
    assert trades[0].trade_id == "T0000001"
    assert trades[0].client_order_id == "cli-1"
    assert trades[0].symbol == "10003456.SH"
    assert trades[0].price == 0.05


def test_dispatch_option_push_order_emits_order_and_exec() -> None:
    """An order_update option frame emits both an order snapshot and an exec report."""
    gw = _make_gateway()
    gw.record_broker_order("9000000001", "cli-1")
    orders: list = []
    execs: list = []
    gw.on_order(lambda s: orders.append(s))
    gw.on_execution_report(lambda e: execs.append(e))

    gw._dispatch_option_push(
        "order_update",
        {
            "entrust_no": "9000000001",
            "exchange_type": "1",
            "option_code": "10003456",
            "entrust_status": "2",
            "business_amount": "0",
            "opt_business_price": "0.0000",
            "error_no": "0",
        },
    )

    assert len(orders) == 1
    assert orders[0].broker_order_id == "9000000001"
    assert orders[0].client_order_id == "cli-1"
    assert len(execs) == 1
