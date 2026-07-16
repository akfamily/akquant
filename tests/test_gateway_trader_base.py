from akquant.gateway.broker_models import (
    UnifiedExecutionReport,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
    UnifiedTrade,
)
from akquant.gateway.trader_base import TraderGatewayBase


class _Gw(TraderGatewayBase):
    """Minimal concrete subclass for testing the base helpers."""


def _snapshot() -> UnifiedOrderSnapshot:
    """Build a filled snapshot."""
    return UnifiedOrderSnapshot(
        client_order_id="c1",
        broker_order_id="b1",
        symbol="600000.SH",
        status=UnifiedOrderStatus.FILLED,
        filled_quantity=100.0,
        avg_fill_price=10.5,
    )


def test_callbacks_and_emit_none_safe() -> None:
    """Emit helpers no-op before registration and dispatch after."""
    gw = _Gw()
    gw._emit_order(_snapshot())  # 未注册，不抛
    seen: list = []
    gw.on_order(seen.append)
    gw._emit_order(_snapshot())
    assert len(seen) == 1


def test_id_reverse_map() -> None:
    """record_broker_order enables client_order_id_for lookup."""
    gw = _Gw()
    gw.record_broker_order("b1", "c1")
    assert gw.client_order_id_for("b1") == "c1"
    assert gw.client_order_id_for("unknown") == ""


def test_emit_exec_from_order() -> None:
    """_emit_exec_from_order derives an execution report from a snapshot."""
    gw = _Gw()
    reports: list = []
    gw.on_execution_report(reports.append)
    gw._emit_exec_from_order(_snapshot())
    assert len(reports) == 1
    r = reports[0]
    assert isinstance(r, UnifiedExecutionReport)
    assert r.broker_order_id == "b1"
    assert r.status == UnifiedOrderStatus.FILLED


def test_defaults() -> None:
    """Default heartbeat/sync implementations are safe no-ops."""
    gw = _Gw()
    assert gw.heartbeat() is True
    assert gw.sync_open_orders() == []
    assert gw.sync_today_trades() == []
    assert isinstance(UnifiedTrade, type)  # import used
