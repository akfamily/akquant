import pytest

pytest.importorskip("httpx")
pytest.importorskip("cryptography")

from akquant.gateway.broker_models import UnifiedOrderRequest
from akquant.gateway.brokers.qmf.adapter import QMFTraderGateway, default_capability


class _SecClient:
    """Fake securities client capturing cancels and serving fixed rows."""

    def __init__(self) -> None:
        """Init fund account and cancel log."""
        self.fund_account = "8888000001"
        self.cancelled: list = []

    def place_order(self, fields):
        """Return a fixed securities entrust_no."""
        return {"entrust_no": "100000001"}

    def cancel_order(self, entrust_no, exchange_type=None):
        """Record a securities cancel."""
        self.cancelled.append(entrust_no)
        return {"error_no": "0"}

    def query_orders(self):
        """Return one securities order row."""
        return [
            {
                "entrust_no": "100000001",
                "exchange_type": "1",
                "stock_code": "600000",
                "entrust_status": "1",
                "business_amount": "0",
                "business_price": "0",
                "error_no": "0",
            }
        ]

    def query_trades(self):
        """Return no securities trades."""
        return []

    def query_positions(self):
        """Return one securities position row."""
        return [
            {
                "exchange_type": "1",
                "stock_code": "600000",
                "current_amount": "1000",
                "enable_amount": "1000",
                "cost_price": "10.2",
            }
        ]

    def query_funds(self):
        """Return a minimal funds payload."""
        return {
            "fund_account": "8888000001",
            "asset_balance": "1",
            "current_balance": "1",
            "enable_balance": "1",
        }

    def auth_status(self, keepalive=True):
        """Report session alive."""
        return True

    def login(self):
        """No-op login."""
        return {}

    def close(self):
        """No-op close."""
        return None


class _OptClient(_SecClient):
    """Fake option client capturing option order/cancel and serving rows."""

    def place_option_order(self, fields):
        """Record option order fields and return a fixed entrust_no."""
        self.last_option_fields = fields
        return {"entrust_no": "9000000001"}

    def cancel_option_order(self, entrust_no, exchange_type=None):
        """Record an option cancel."""
        self.cancelled.append(("opt", entrust_no))
        return {"error_no": "0"}

    def query_option_orders(self):
        """Return one option order row."""
        return [
            {
                "entrust_no": "9000000001",
                "exchange_type": "1",
                "option_code": "10003456",
                "entrust_status": "2",
                "business_amount": "0",
                "opt_business_price": "0",
                "error_no": "0",
            }
        ]

    def query_option_trades(self):
        """Return no option trades."""
        return []

    def query_option_positions(self):
        """Return one option position row."""
        return [
            {
                "exchange_type": "1",
                "option_code": "10003456",
                "current_amount": "1",
                "enable_amount": "1",
                "opt_cost_price": "0.05",
            }
        ]


def _opt_req() -> UnifiedOrderRequest:
    """Build an option order request with required extra."""
    return UnifiedOrderRequest(
        client_order_id="oc1",
        symbol="10003456.SH",
        side="Buy",
        quantity=1,
        price=0.05,
        order_type="Limit",
        asset_type="option",
        extra={"entrust_oc": "O", "covered_flag": "0", "entrust_prop": "F0"},
    )


def test_capability_reflects_options() -> None:
    """Enabling options declares the option extra fields and features."""
    cap = default_capability(enable_options=True)
    assert "entrust_oc" in cap.broker_extra_fields
    assert "options" in cap.features
    assert default_capability(enable_options=False).features == frozenset()


def test_option_order_routes_to_option_client() -> None:
    """asset_type=option routes to the option client and records the id."""
    opt = _OptClient()
    gw = QMFTraderGateway(client=_SecClient(), ws_url="ws://x", option_client=opt)
    bid = gw.place_order(_opt_req())
    assert bid == "9000000001"
    assert opt.last_option_fields["option_code"] == "10003456"
    assert gw._client_id_by_broker["9000000001"] == "oc1"


def test_option_order_without_client_raises() -> None:
    """Option order without an option session raises a clear error."""
    gw = QMFTraderGateway(client=_SecClient(), ws_url="ws://x")
    with pytest.raises(RuntimeError):
        gw.place_order(_opt_req())


def test_cancel_routes_by_source() -> None:
    """Cancel routes option ids to the option client, others to securities."""
    opt = _OptClient()
    sec = _SecClient()
    gw = QMFTraderGateway(client=sec, ws_url="ws://x", option_client=opt)
    gw.place_order(_opt_req())
    gw.cancel_order("9000000001")
    gw.cancel_order("100000001")
    assert ("opt", "9000000001") in opt.cancelled
    assert "100000001" in sec.cancelled


def test_queries_merge_securities_and_options() -> None:
    """Positions/orders merge securities and option rows."""
    gw = QMFTraderGateway(
        client=_SecClient(), ws_url="ws://x", option_client=_OptClient()
    )
    syms = {p.symbol for p in gw.query_positions()}
    assert syms == {"600000.SH", "10003456.SH"}
    ids = {s.broker_order_id for s in gw.sync_open_orders()}
    assert ids == {"100000001", "9000000001"}


def test_cancel_after_recovery_routes_to_option_client() -> None:
    """A recovered (not locally placed) option order still cancels via options."""
    opt = _OptClient()
    gw = QMFTraderGateway(client=_SecClient(), ws_url="ws://x", option_client=opt)
    # Simulate fresh process: no place_order call, recover via sync_open_orders.
    gw.sync_open_orders()
    gw.cancel_order("9000000001")
    assert ("opt", "9000000001") in opt.cancelled
