"""LiveRunner 止损 remap: 记录/查找/终态清理 + 适配用 local id."""

from collections import deque

from akquant.gateway.broker_models import UnifiedTrade
from akquant.live._runner import LiveRunner


def _runner() -> LiveRunner:
    r = LiveRunner.__new__(LiveRunner)
    r._broker_to_local_stop_id = {}
    r._broker_to_client_order_ids = {}
    r._client_to_broker_order_ids = {}
    r._client_to_strategy_ids = {}
    r._broker_to_strategy_ids = {}
    r._client_to_group_ids = {}
    r._closed_broker_order_ids = set()
    r._closed_order_id_fifo = deque()
    r._order_requests = {}
    return r


def test_record_and_lookup_and_adapt() -> None:
    """Record 后可反查 local id，且 trade 适配用 local id 覆盖 order_id."""
    r = _runner()
    r._record_stop_remap("LSTOP-1", "B9")
    trade = UnifiedTrade(
        trade_id="T1",
        broker_order_id="B9",
        client_order_id="c1",
        symbol="X",
        side="Sell",
        quantity=100.0,
        price=9.4,
        timestamp_ns=1,
    )
    assert r._lookup_stop_local_id(trade) == "LSTOP-1"
    adapted = r._adapt_strategy_payload("trade", trade)
    assert adapted.order_id == "LSTOP-1"


def test_terminal_cleanup_pops_remap() -> None:
    """终态清理应同步弹出 broker_order_id -> local stop id 的映射."""
    r = _runner()
    r._record_stop_remap("LSTOP-1", "B9")
    r._close_order_mapping("c1", "B9")
    assert r._broker_to_local_stop_id == {}
