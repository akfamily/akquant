"""LiveRunner 提交请求缓存：记录 + 查找 + 终态清理."""

from akquant.gateway.broker_models import (
    UnifiedOrderRequest,
    UnifiedOrderSnapshot,
    UnifiedOrderStatus,
)
from akquant.live import LiveRunner


def _runner() -> LiveRunner:
    # 不做完整构造；只准备缓存/映射所需的最小状态（这些方法在 LiveRunner 上真实存在）。
    r = LiveRunner.__new__(LiveRunner)
    r._order_requests = {}
    r._client_to_broker_order_ids = {}
    r._broker_to_client_order_ids = {}
    r._client_to_strategy_ids = {}
    r._broker_to_strategy_ids = {}
    r._broker_order_states = {}
    r._closed_broker_order_ids = set()
    return r


def test_record_and_lookup_request() -> None:
    """Record 后可通过 client_order_id 反查同一 request 对象."""
    r = _runner()
    req = UnifiedOrderRequest(
        client_order_id="c1", symbol="600000.SH", side="Buy", quantity=100.0
    )
    r._record_order_request("c1", req)
    r._sync_order_id_mapping("c1", "B1")
    snap = UnifiedOrderSnapshot(
        client_order_id="c1",
        broker_order_id="B1",
        symbol="600000.SH",
        status=UnifiedOrderStatus.FILLED,
        filled_quantity=100.0,
        avg_fill_price=10.0,
    )
    assert r._lookup_order_request(snap) is req


def test_terminal_status_clears_request() -> None:
    """终态 order 事件应清理请求缓存，避免 lookup 命中已完结订单."""
    r = _runner()
    req = UnifiedOrderRequest(
        client_order_id="c1", symbol="600000.SH", side="Buy", quantity=100.0
    )
    r._record_order_request("c1", req)
    r._sync_order_id_mapping("c1", "B1")
    snap = UnifiedOrderSnapshot(
        client_order_id="c1",
        broker_order_id="B1",
        symbol="600000.SH",
        status=UnifiedOrderStatus.FILLED,
        filled_quantity=100.0,
        avg_fill_price=10.0,
    )
    r._update_broker_state("order", snap)  # 终态应清理映射与请求缓存
    assert r._lookup_order_request(snap) is None
