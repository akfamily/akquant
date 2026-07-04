"""组合类下单(order_target_weights/order_target_positions) 的 ctx 就绪校验.

Task 7 移除了这两个函数里 `if strategy.ctx is None: raise RuntimeError(...)` 的
硬编码守卫, 使它们能在 broker_live 下工作 (broker_live 时 ctx 恒为 None, 但
strategy.execution 是已就绪的 BrokerExecution)。但这也让"回测中 ctx 尚未绑定就
调用这些函数"(例如在 __init__/on_init 里误用) 从原本的 fail-fast 报错退化成了
静默 no-op。

（Task v2-T3 硬删了「全仓买入」薄封装，其自带的 _require_execution_ready 守卫
用例未做等价迁移：迁移目标 order_target_percent(1.0)（以及 order_target/
order_target_value）从未调用过 _require_execution_ready，该守卫场景不再适用
于迁移后的调用点，故随薄封装一并移除。）

本文件覆盖修复后的 `_require_execution_ready`: 只有当 ctx 为 None 且执行后端不
具备 broker_live 能力时才报 "Context not ready"; ctx 为 None 但 execution 是
broker_live 就绪的场景应正常按 execution 状态下单，不受影响。
"""

import pytest
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


# ---------------------------------------------------------------------------
# ctx 为 None 且非 broker_live -> 明确报错 "Context not ready"
# ---------------------------------------------------------------------------


def test_order_target_weights_raises_when_ctx_not_ready_and_not_broker_live() -> None:
    """order_target_weights: ctx 未绑定且非 broker_live 时应 fail-fast."""
    execution = _CapExecution(broker_live=False, portfolio_value=10000.0)
    strategy = _Strategy(execution)

    with pytest.raises(RuntimeError, match="Context not ready"):
        api.order_target_weights(
            strategy,
            target_weights={"600000.SH": 0.5},
            price_map={"600000.SH": 10.0},
        )

    assert execution.orders == []


def test_order_target_positions_raises_when_ctx_not_ready_and_not_broker_live() -> None:
    """order_target_positions: ctx 未绑定且非 broker_live 时应 fail-fast."""
    execution = _CapExecution(broker_live=False, position=0.0)
    strategy = _Strategy(execution)

    with pytest.raises(RuntimeError, match="Context not ready"):
        api.order_target_positions(
            strategy,
            target_positions={"600000.SH": 100.0},
        )

    assert execution.orders == []


# ---------------------------------------------------------------------------
# ctx 为 None 但 execution 是 broker_live 就绪 -> 不报错，正常按 execution sizing 下单
# ---------------------------------------------------------------------------


def test_order_target_weights_still_works_when_ctx_none_and_broker_live() -> None:
    """order_target_weights: ctx 为 None 但 broker_live 就绪时不应报错."""
    execution = _CapExecution(broker_live=True, portfolio_value=10000.0)
    strategy = _Strategy(execution)

    order_ids = api.order_target_weights(
        strategy,
        target_weights={"600000.SH": 0.5},
        price_map={"600000.SH": 10.0},
    )

    assert order_ids == ["OID"]
    assert len(execution.orders) == 1
    assert execution.orders[0]["side"].lower() == "buy"
    assert execution.orders[0]["quantity"] == 500.0  # 10000*0.5/10


def test_order_target_positions_still_works_when_ctx_none_and_broker_live() -> None:
    """order_target_positions: ctx 为 None 但 broker_live 就绪时不应报错."""
    execution = _CapExecution(broker_live=True, position=0.0)
    strategy = _Strategy(execution)

    order_ids = api.order_target_positions(
        strategy,
        target_positions={"600000.SH": 100.0},
    )

    assert order_ids == ["OID"]
    assert len(execution.orders) == 1
    assert execution.orders[0]["side"].lower() == "buy"
    assert execution.orders[0]["quantity"] == 100.0
