"""公共写自由函数经 strategy.execution."""

from akquant import strategy_trading_api as api


class _FakeExec:
    def __init__(self):
        self.submitted = None
        self.canceled = None

    def submit_order(self, **kwargs):
        self.submitted = kwargs
        return "OID-1"

    def cancel_order(self, order_id):
        self.canceled = order_id


class _S:
    def __init__(self):
        self.execution = _FakeExec()
        self.ctx = None


def test_submit_order_delegates() -> None:
    """公共 submit_order 自由函数应转发到 strategy.execution.submit_order."""
    s = _S()
    oid = api.submit_order(s, symbol="600000.SH", side="Buy", quantity=100)
    assert oid == "OID-1"
    assert s.execution.submitted["symbol"] == "600000.SH"


def test_cancel_order_delegates() -> None:
    """公共 cancel_order 自由函数应转发到 strategy.execution.cancel_order."""
    s = _S()
    api.cancel_order(s, "OID-9")
    assert s.execution.canceled == "OID-9"
