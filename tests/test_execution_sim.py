"""SimExecution：回测后端读/写委托给 ctx（独立可测）."""

from types import SimpleNamespace

from akquant.execution.sim import SimExecution


class _Ctx:
    """Fake StrategyContext."""

    def __init__(self) -> None:
        self.cash = 1000.0
        self.positions = {"600000.SH": 500.0}
        self.available_positions = {"600000.SH": 400.0}
        self.active_orders: list = []
        self.canceled_order_ids: list = []
        self.risk_config = SimpleNamespace(account_mode="cash", enable_short_sell=False)

    def get_position(self, symbol):
        return self.positions.get(symbol, 0.0)

    def get_available_position(self, symbol):
        return self.available_positions.get(symbol, 0.0)


class _Strategy:
    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self._last_event_type = ""
        self.current_bar = SimpleNamespace(symbol="600000.SH")
        self.current_tick = None
        self._hold_bars = {"600000.SH": 3}
        self._known_orders: dict = {}


def test_sim_execution_reads_route_to_ctx() -> None:
    """SimExecution 的读操作应委托给 strategy.ctx."""
    strategy = _Strategy(_Ctx())
    ex = SimExecution(strategy)
    assert ex.get_position("600000.SH") == 500.0
    assert ex.get_available_position("600000.SH") == 400.0
    assert ex.get_cash() == 1000.0
    assert ex.get_positions() == {"600000.SH": 500.0}
    assert ex.get_position() == 500.0  # 缺省用 current_bar


def test_sim_execution_capabilities_defaults_broker_live_false() -> None:
    """回测后端 capabilities 应标记 broker_live=False 并透传 account_mode."""
    strategy = _Strategy(_Ctx())
    caps = SimExecution(strategy).capabilities()
    assert caps["broker_live"] is False
    assert caps["account_mode"] == "cash"
