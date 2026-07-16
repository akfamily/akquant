"""端到端: broker_live 提交 stop → 经 strategy_events 钩子喂价 → 触发提交底层单."""

from types import SimpleNamespace
from typing import Any, cast

from akquant import strategy_events
from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.broker_state_cache import BrokerStateCache
from akquant.gateway.order_receipt import OrderReceipt


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


class _Submitter:
    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []

    def submit_order(self, **kw: Any) -> OrderReceipt:
        self.orders.append(kw)
        return OrderReceipt.single(group_id="BID-1", broker_order_id="BID-1")

    def _get_execution_capabilities(self) -> dict[str, bool]:
        return {"broker_live": True}


def test_stop_fires_via_bar_hook_and_submits_underlying() -> None:
    """提交 stop 单 → 经 _drive_local_stops 喂 bar 价 → 触发提交底层单."""
    strat = SimpleNamespace()
    strat.execution = BrokerExecution(
        strat,
        _Gw(),
        cast(BrokerStateCache, _Cache()),
        _Submitter(),
    )
    # 挂一个卖出止损 @9.5
    oid = strat.execution.submit_order(
        symbol="X",
        side="Sell",
        quantity=100,
        order_type="StopMarket",
        trigger_price=9.5,
    )
    assert oid.startswith("LSTOP-")
    # 未触发的一根 bar
    strategy_events._drive_local_stops(strat, "X", 9.8, high=9.9, low=9.6)
    assert strat.execution._submitter.orders == []
    # 触发的一根 bar (low 9.3 <= 9.5)
    strategy_events._drive_local_stops(strat, "X", 9.4, high=9.7, low=9.3)
    assert len(strat.execution._submitter.orders) == 1
    assert strat.execution._submitter.orders[0]["order_type"] == "Market"
    assert strat.execution.get_open_orders() == []
