"""端到端: broker_live 提交 stop → 经 strategy_events 钩子喂价 → 触发提交底层单."""

from types import SimpleNamespace

from akquant import strategy_events
from akquant.gateway.broker_execution import BrokerExecution


class _Cache:
    def positions(self):
        return {}

    def available_positions(self):
        return {}

    def open_orders(self):
        return []

    def account(self):
        return None


class _Gw:
    def cancel_order(self, bid):
        pass

    def sync_open_orders(self):
        return []


class _Submitter:
    def __init__(self):
        self.orders = []

    def submit_order(self, **kw):
        self.orders.append(kw)
        return "BID-1"

    def _get_execution_capabilities(self):
        return {"broker_live": True}


class _Bar:
    def __init__(self, symbol, o, h, lo, c):
        self.symbol, self.open, self.high, self.low, self.close = symbol, o, h, lo, c


def test_stop_fires_via_bar_hook_and_submits_underlying() -> None:
    """提交 stop 单 → 经 _drive_local_stops 喂 bar 价 → 触发提交底层单."""
    strat = SimpleNamespace()
    strat.execution = BrokerExecution(strat, _Gw(), _Cache(), _Submitter())
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
