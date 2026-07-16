"""order_target 在 broker_live (execution 支撑) 下按后端持仓算 delta 并下单."""

from typing import Any

from akquant import strategy_trading_api as api


class _RecExec:
    """记录 submit 的 fake 后端."""

    def __init__(self, pos: float) -> None:
        self._pos = pos
        self.orders: list[dict[str, Any]] = []

    def get_position(self, symbol: str | None = None) -> float:
        return self._pos

    def submit_order(self, **kwargs: Any) -> str:
        self.orders.append(kwargs)
        return "OID"

    def capabilities(self) -> dict[str, bool]:
        return {"broker_live": True, "supports_short_sell": False}


class _S:
    def __init__(self, pos: float) -> None:
        self.execution = _RecExec(pos)
        self.ctx = None
        self.current_bar: Any | None = None
        self.current_tick: Any | None = None
        self.lot_size = 1

    def submit_order(self, **kwargs: Any) -> str:
        """镜像真实 Strategy.submit_order：统一转发到 execution.submit_order."""
        return self.execution.submit_order(**kwargs)


def test_order_target_sizes_off_execution_in_broker_live() -> None:
    """order_target 在 broker_live (execution 支撑) 下按后端持仓算 delta 并下单."""
    s = _S(pos=300.0)
    api.order_target(s, symbol="600000.SH", target=1000.0, price=10.0)
    assert len(s.execution.orders) == 1
    assert s.execution.orders[0]["side"].lower() == "buy"
    assert s.execution.orders[0]["quantity"] == 700.0  # 1000 - 300
