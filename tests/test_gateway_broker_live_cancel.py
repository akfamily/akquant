from akquant.gateway.broker_models import UnifiedOrderSnapshot, UnifiedOrderStatus
from akquant.gateway.broker_strategy_api import install_broker_cancel


class _Gw:
    """Fake gateway recording cancels + serving open orders."""

    def __init__(self) -> None:
        """Init cancel log."""
        self.cancelled: list = []

    def cancel_order(self, broker_order_id: str) -> None:
        """Record a broker cancel."""
        self.cancelled.append(str(broker_order_id))

    def sync_open_orders(self):
        """Return one open option/stock order."""
        return [
            UnifiedOrderSnapshot(
                client_order_id="c1",
                broker_order_id="9000000001",
                symbol="600000.SH",
                status=UnifiedOrderStatus.SUBMITTED,
            )
        ]


class _Strategy:
    """Bare strategy target."""


def test_cancel_order_forwards_broker_id() -> None:
    """cancel_order forwards the broker_order_id straight to the gateway."""
    gw = _Gw()
    strategy = _Strategy()
    install_broker_cancel(strategy, gw)
    strategy.cancel_order("9000000001")  # 就是 submit_order 返回的 broker id
    assert gw.cancelled == ["9000000001"]


def test_cancel_all_orders_cancels_open() -> None:
    """cancel_all_orders cancels every open order's broker_order_id."""
    gw = _Gw()
    strategy = _Strategy()
    install_broker_cancel(strategy, gw)
    strategy.cancel_all_orders()
    assert gw.cancelled == ["9000000001"]
