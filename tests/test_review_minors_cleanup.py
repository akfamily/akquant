"""P4b/P1 评审 minors 清理: #1 on_error 自身抛错隔离; #3 get_order 经 execution 路由."""

from typing import Any

from akquant.gateway.broker_execution import BrokerExecution
from akquant.live import LiveRunner
from akquant.strategy import Strategy

# ---- #1: _safe_strategy_callback 里 on_error 自身抛错不得逃出 ----


def test_on_error_handler_raising_does_not_propagate(caplog: Any) -> None:
    """回调抛错 + on_error 自身也抛错 → _safe_strategy_callback 不上抛, 记 warning."""

    class _BadStrategy:
        def on_trade(self, trade: Any) -> None:
            raise RuntimeError("trade callback failed")

        def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
            raise ValueError("on_error itself blew up")

    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "ptrade"
    runner._init_broker_bridge_state()

    with caplog.at_level("WARNING", logger="akquant.gateway.live"):
        # 不得抛出(否则会杀掉 broker drain 循环)
        runner._safe_strategy_callback(
            _BadStrategy(), "on_trade", {"id": "t1", "symbol": "X"}
        )

    assert any(
        r.getMessage() == "Strategy on_error handler raised" for r in caplog.records
    )


# ---- #3: Strategy.get_order 经 self.execution 路由 ----


def test_get_order_routes_through_execution() -> None:
    """Strategy.get_order 委托给 self.execution.get_order(与 get_position 一致)."""

    class _Exec:
        def __init__(self) -> None:
            self.asked: list = []

        def get_order(self, order_id: str) -> Any:
            self.asked.append(order_id)
            return f"ORDER::{order_id}"

    s = Strategy.__new__(Strategy)
    s.execution = _Exec()
    assert s.get_order("abc") == "ORDER::abc"
    assert s.execution.asked == ["abc"]


def test_get_order_broker_live_scans_open_orders() -> None:
    """broker_live: 经 BrokerExecution 按 broker_order_id 扫柜台挂单(否则恒 None)."""

    class _Order:
        def __init__(self, bid: str) -> None:
            self.broker_order_id = bid

    class _Cache:
        def open_orders(self) -> list:
            return [_Order("B1"), _Order("B2")]

    s = Strategy.__new__(Strategy)
    s.execution = BrokerExecution(s, object(), _Cache(), object())
    found = s.get_order("B2")
    assert getattr(found, "broker_order_id", None) == "B2"
    assert s.get_order("nope") is None
