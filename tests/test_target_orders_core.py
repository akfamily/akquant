"""_target_to_orders：统一 delta 取整、不撤单（fake execution + submit spy）."""

from typing import Any

from akquant import strategy_trading_api as api


class _Exec:
    def __init__(self, pos: float) -> None:
        self._pos = pos
        self.orders: list[dict[str, Any]] = []
        self.canceled = 0

    def get_position(self, symbol: str | None = None) -> float:
        return self._pos

    def submit_order(self, **kw: Any) -> str:
        self.orders.append(kw)
        return "OID"

    def cancel_all_orders(self, symbol: str | None = None) -> None:
        self.canceled += 1

    def capabilities(self) -> dict[str, bool]:
        return {"broker_live": False}


class _S:
    def __init__(self, pos: float, lot: int = 100) -> None:
        self.execution = _Exec(pos)
        self.ctx = object()
        self.current_bar: Any | None = None
        self.current_tick: Any | None = None
        self.lot_size = lot
        self._last_prices: dict[str, float] = {}

    def submit_order(self, **kwargs: Any) -> str:
        """Mirror real Strategy.submit_order: forward unconditionally to execution."""
        return self.execution.submit_order(**kwargs)


def test_order_target_rounds_delta_to_lot() -> None:
    """order_target 经共享核心按 lot_size 向下取整 delta."""
    s = _S(pos=0, lot=100)
    api.order_target(s, symbol="600000.SH", target=137, price=10.0)
    # 137 → 向下取整到 100
    assert s.execution.orders[0]["quantity"] == 100.0
    assert s.execution.orders[0]["side"].lower() == "buy"


def test_order_target_value_no_autocancel() -> None:
    """order_target_value 不再自动撤单（统一交由共享核心处理）."""
    s = _S(pos=0, lot=1)
    api.order_target_value(s, symbol="600000.SH", target_value=1000.0, price=10.0)
    assert s.execution.canceled == 0  # 不再自动撤单
    assert s.execution.orders[0]["quantity"] == 100.0


def test_rebalance_positions_rounds_delta_to_lot() -> None:
    """rebalance_positions 对每个 leg 的 delta 也应按 lot_size 向下取整."""
    s = _S(pos=0, lot=100)
    api.rebalance_positions(s, {"600000.SH": 137})
    # 137 → 向下取整到 100（与 order_target 共用 _target_to_orders 核心）
    assert s.execution.orders[0]["quantity"] == 100.0
    assert s.execution.orders[0]["side"].lower() == "buy"
