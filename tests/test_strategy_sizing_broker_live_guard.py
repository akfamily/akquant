"""组合目标类下单(order_target*) 在 broker_live 下按柜台持仓 sizing 并真下单.

历史上这些函数在 broker_live 下会抛 RuntimeError
(见 _reject_target_orders_in_broker_live, Task 7 前); 现在它们的持仓/现金/组合价值
读取全部改走 strategy.execution, 因此在 broker_live 下也能正确按柜台真实状态
sizing 并下单, 不再需要守卫拒单。

（Task v2-T3 硬删了「全仓买入」薄封装；其专属的 cash-based `int(cash/price)`
sizing 用例不做等价迁移——迁移目标 order_target_percent(1.0) 按 portfolio_value
定价的 sizing 已由本文件 test_order_target_percent_sizes_off_execution_in_broker_live
覆盖。）
"""

from akquant import strategy_trading_api as api


class _CapExecution:
    """Fake execution backend：固定 position/cash/portfolio_value + submit 记录."""

    def __init__(
        self,
        broker_live: bool = True,
        position: float = 0.0,
        cash: float = 0.0,
        portfolio_value: float = 0.0,
    ) -> None:
        """Bind fixed state to report from the execution reads."""
        self._broker_live = broker_live
        self._position = position
        self._cash = cash
        self._portfolio_value = portfolio_value
        self.orders: list = []

    def capabilities(self) -> dict:
        """Report the fixed broker_live flag."""
        return {"broker_live": self._broker_live, "supports_short_sell": False}

    def get_position(self, symbol=None) -> float:
        """Return the fixed position regardless of symbol."""
        return self._position

    def get_positions(self) -> dict:
        """Return an empty position map (unused by these tests)."""
        return {}

    def get_cash(self) -> float:
        """Return the fixed cash balance."""
        return self._cash

    def get_portfolio_value(self) -> float:
        """Return the fixed portfolio value."""
        return self._portfolio_value

    def cancel_all_orders(self, symbol=None) -> None:
        """No-op: no open orders to cancel in these tests."""
        return None

    def submit_order(self, **kwargs) -> str:
        """Record the submitted order and return a fixed order id."""
        self.orders.append(kwargs)
        return "OID"


class _Strategy:
    """Strategy stub whose submit_order mirrors real Strategy: forwards to execution."""

    def __init__(self, execution: _CapExecution) -> None:
        """Bind the fake execution backend; ctx stays None like real broker_live."""
        self.ctx = None
        self.execution = execution
        self.current_bar = None
        self.current_tick = None
        self.lot_size = 1
        self._last_prices: dict = {}

    def submit_order(self, **kwargs):
        """Mirror real Strategy.submit_order: forward unconditionally to execution."""
        return self.execution.submit_order(**kwargs)


def test_order_target_sizes_off_execution_in_broker_live() -> None:
    """order_target 在 broker_live 下按 execution 持仓算 delta 并真下单."""
    execution = _CapExecution(broker_live=True, position=300.0)
    strategy = _Strategy(execution)

    api.order_target(strategy, symbol="600000.SH", target=1000.0, price=10.0)

    assert len(execution.orders) == 1
    assert execution.orders[0]["side"].lower() == "buy"
    assert execution.orders[0]["quantity"] == 700.0  # 1000 - 300


def test_order_target_value_sizes_off_execution_in_broker_live() -> None:
    """order_target_value 在 broker_live 下按 execution 持仓算 delta 并真下单."""
    execution = _CapExecution(broker_live=True, position=100.0)
    strategy = _Strategy(execution)

    api.order_target_value(
        strategy, symbol="600000.SH", target_value=2000.0, price=10.0
    )

    assert len(execution.orders) == 1
    assert execution.orders[0]["side"].lower() == "buy"
    assert execution.orders[0]["quantity"] == 100.0  # 2000/10 - 100


def test_order_target_percent_sizes_off_execution_in_broker_live() -> None:
    """order_target_percent 在 broker_live 下按 execution 组合价值 sizing 并真下单."""
    execution = _CapExecution(broker_live=True, portfolio_value=10000.0)
    strategy = _Strategy(execution)

    api.order_target_percent(
        strategy, symbol="600000.SH", target_percent=0.5, price=10.0
    )

    assert len(execution.orders) == 1
    assert execution.orders[0]["side"].lower() == "buy"
    assert execution.orders[0]["quantity"] == 500.0  # 10000*0.5/10


def test_rebalance_weights_sizes_off_execution_in_broker_live() -> None:
    """rebalance_weights 在 broker_live 下按 execution 组合价值/持仓 sizing."""
    execution = _CapExecution(broker_live=True, portfolio_value=10000.0)
    strategy = _Strategy(execution)

    api.rebalance_weights(
        strategy,
        target_weights={"600000.SH": 0.5},
        price_map={"600000.SH": 10.0},
    )

    assert len(execution.orders) == 1
    assert execution.orders[0]["side"].lower() == "buy"
    assert execution.orders[0]["quantity"] == 500.0  # 10000*0.5/10


def test_order_target_not_rejected_when_broker_live_absent() -> None:
    """order_target 对没有 capabilities() 的普通 execution 依然正常工作."""

    class _PlainExecution:
        def get_position(self, symbol: str) -> float:
            return 0.0

    class _PlainStrategy:
        def __init__(self) -> None:
            self.ctx = None
            self.execution = _PlainExecution()
            self.submit_order_calls: list = []
            self.lot_size = 1

        def submit_order(self, **kwargs):
            self.submit_order_calls.append(kwargs)
            return "order-1"

    strategy = _PlainStrategy()
    order_id = api.order_target(strategy, symbol="600000.SH", target=100)
    assert order_id == "order-1"
