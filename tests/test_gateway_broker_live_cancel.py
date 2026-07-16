from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_models import UnifiedOrderSnapshot, UnifiedOrderStatus
from akquant.gateway.broker_state_cache import BrokerStateCache
from akquant.gateway.order_receipt import OrderReceipt
from akquant.strategy_trading_api import cancel_group, cancel_order


class _Gw:
    """Fake gateway recording cancels + serving open orders."""

    def __init__(self) -> None:
        """Init cancel log."""
        self.cancelled: list[str] = []

    def cancel_order(self, broker_order_id: str) -> None:
        """Record a broker cancel."""
        self.cancelled.append(str(broker_order_id))

    def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
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


def _make_execution(gw: _Gw) -> BrokerExecution:
    return BrokerExecution(_Strategy(), gw, BrokerStateCache(gw), None)


def test_cancel_order_forwards_broker_id() -> None:
    """cancel_order forwards the broker_order_id straight to the gateway."""
    gw = _Gw()
    ex = _make_execution(gw)
    ex.cancel_order("9000000001")  # 就是 submit_order 返回的 broker id
    assert gw.cancelled == ["9000000001"]


def test_cancel_all_orders_cancels_open() -> None:
    """cancel_all_orders cancels every open order's broker_order_id."""
    gw = _Gw()
    ex = _make_execution(gw)
    ex.cancel_all_orders()
    assert gw.cancelled == ["9000000001"]


def test_cancel_all_orders_symbol_filter() -> None:
    """cancel_all_orders(symbol=...) cancels only that symbol's broker ids."""

    class _MultiGw(_Gw):
        def sync_open_orders(self) -> list[UnifiedOrderSnapshot]:
            """Return two open orders on different symbols."""
            return [
                UnifiedOrderSnapshot(
                    client_order_id="c1",
                    broker_order_id="9000000001",
                    symbol="600000.SH",
                    status=UnifiedOrderStatus.SUBMITTED,
                ),
                UnifiedOrderSnapshot(
                    client_order_id="c2",
                    broker_order_id="9000000002",
                    symbol="000001.SZ",
                    status=UnifiedOrderStatus.SUBMITTED,
                ),
            ]

    gw = _MultiGw()
    ex = _make_execution(gw)
    ex.cancel_all_orders(symbol="000001.SZ")
    assert gw.cancelled == ["9000000002"]


def _make_execution_with_group(gw: _Gw) -> BrokerExecution:
    legs = {"g1": ["b1", "b2"]}
    return BrokerExecution(
        _Strategy(),
        gw,
        BrokerStateCache(gw),
        None,
        group_broker_ids=lambda gid: legs.get(gid, []),
    )


def test_cancel_group_cancels_all_legs() -> None:
    """cancel_group cancels every broker id returned by group_broker_ids(group_id)."""
    gw = _Gw()
    ex = _make_execution_with_group(gw)
    ex.cancel_group("g1")
    assert set(gw.cancelled) == {"b1", "b2"}


def test_cancel_group_noop_without_callback() -> None:
    """cancel_group is a no-op when no group_broker_ids callback was injected."""
    gw = _Gw()
    ex = _make_execution(gw)
    ex.cancel_group("g1")
    assert gw.cancelled == []


def test_cancel_order_accepts_order_receipt() -> None:
    """strategy_trading_api.cancel_order tolerates an OrderReceipt (uses .primary)."""

    class _Execution:
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        def cancel_order(self, order_id: str) -> None:
            self.cancelled.append(order_id)

    class _Strat:
        def __init__(self) -> None:
            self.execution = _Execution()

    strat = _Strat()
    receipt = OrderReceipt.single(group_id="g1", broker_order_id="b1")
    cancel_order(strat, receipt)
    assert strat.execution.cancelled == ["b1"]


def test_cancel_group_resolves_order_receipt_group_id() -> None:
    """strategy_trading_api.cancel_group tolerates an OrderReceipt (uses .group_id)."""

    class _Execution:
        def __init__(self) -> None:
            self.cancelled_groups: list[str] = []

        def cancel_group(self, group_id: str) -> None:
            self.cancelled_groups.append(group_id)

    class _Strat:
        def __init__(self) -> None:
            self.execution = _Execution()

    strat = _Strat()
    receipt = OrderReceipt.single(group_id="g1", broker_order_id="b1")
    cancel_group(strat, receipt)
    assert strat.execution.cancelled_groups == ["g1"]
