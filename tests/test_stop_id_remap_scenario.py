"""端到端: 触发止损→底层单成交推送→策略 on_trade 收到 order_id=LSTOP-n."""

from typing import Any, cast

from akquant.gateway.broker_event_adapter import map_trade
from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_models import UnifiedTrade
from akquant.gateway.broker_state_cache import BrokerStateCache


class _Cache:
    def positions(self) -> dict[str, float]:
        return {}

    def available_positions(self) -> dict[str, float]:
        return {}

    def open_orders(self) -> list[object]:
        return []

    def account(self) -> None:
        return None


class _Gw:
    def cancel_order(self, bid: str) -> None:
        return None

    def sync_open_orders(self) -> list[object]:
        return []


class _Sub:
    def submit_order(self, **kw: Any) -> str:
        return "B77"


class _S:
    current_bar = None
    current_tick = None


def test_triggered_stop_trade_reports_local_id() -> None:
    """触发止损后成交推送经 remap 还原为策略持有的本地 LSTOP-n id."""
    remap: dict[str, str] = {}
    ex = BrokerExecution(
        _S(),
        _Gw(),
        cast(BrokerStateCache, _Cache()),
        _Sub(),
        record_stop_remap=lambda lid, bid: remap.__setitem__(bid, lid),
    )
    oid = ex.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    ex.check_stop_triggers("X", last=9.4, high=9.6, low=9.3)
    assert remap == {"B77": oid}
    # 柜台成交推送该底层单 → 适配时用 remap 得 local id
    trade = UnifiedTrade(
        trade_id="T1",
        broker_order_id="B77",
        client_order_id="c1",
        symbol="X",
        side="Sell",
        quantity=100.0,
        price=9.4,
        timestamp_ns=1,
    )
    local_id = remap.get("B77")
    assert map_trade(trade, local_id=local_id).order_id == oid
