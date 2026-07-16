"""close_position 全平（含零股，不按手数取整），order_target 仍取整."""

from typing import Any

from akquant import strategy_trading_api as api


class _Exec:
    def __init__(self, pos: float) -> None:
        self._pos = pos
        self.orders: list[dict[str, Any]] = []

    def get_position(self, symbol: str | None = None) -> float:
        return self._pos

    def submit_order(self, **kw: Any) -> str:
        self.orders.append(kw)
        return "OID"


class _S:
    def __init__(self, pos: float, lot_size: int = 100) -> None:
        self.execution = _Exec(pos)
        self.ctx = object()
        self.current_bar: Any | None = None
        self.current_tick: Any | None = None
        self.lot_size = lot_size
        self._last_prices: dict[str, float] = {}

    def submit_order(self, **kwargs: Any) -> str:
        """Mirror real Strategy.submit_order: forward unconditionally to execution."""
        return self.execution.submit_order(**kwargs)


def test_close_long_position_sells_all() -> None:
    """close_position 对多头持仓应下卖单清空全部仓位."""
    s = _S(pos=300)
    api.close_position(s, symbol="600000.SH")
    assert s.execution.orders[0]["side"].lower() == "sell"
    assert s.execution.orders[0]["quantity"] == 300.0


def test_close_flat_is_noop() -> None:
    """close_position 对空仓应为 no-op，不下单."""
    s = _S(pos=0)
    api.close_position(s, symbol="600000.SH")
    assert s.execution.orders == []


def test_close_odd_lot_position_sells_exact_quantity() -> None:
    """close_position 对零股持仓(150, lot_size=100)应全平，不按手数取整到 100."""
    s = _S(pos=150, lot_size=100)
    api.close_position(s, symbol="600000.SH")
    assert s.execution.orders[0]["side"].lower() == "sell"
    assert s.execution.orders[0]["quantity"] == 150.0


def test_order_target_still_rounds_to_lot() -> None:
    """order_target 仍按 lot_size 取整（round_to_lot 默认 True，不受本次改动影响）."""
    s = _S(pos=0, lot_size=100)
    api.order_target(s, symbol="600000.SH", target=137, price=10.0)
    assert s.execution.orders[0]["side"].lower() == "buy"
    assert s.execution.orders[0]["quantity"] == 100.0


class _SNoLotSize:
    """无 lot_size 属性的 strategy-like 对象（模拟精简/自定义 strategy）."""

    def __init__(self, pos: float) -> None:
        self.execution = _Exec(pos)
        self.ctx = object()
        self.current_bar: Any | None = None
        self.current_tick: Any | None = None
        self._last_prices: dict[str, float] = {}
        # 有意不设置 self.lot_size

    def submit_order(self, **kwargs: Any) -> str:
        return self.execution.submit_order(**kwargs)


def test_close_position_without_lot_size_attr_does_not_raise() -> None:
    """close_position 走 round_to_lot=False 分支，不应因缺失 lot_size 属性报错."""
    s = _SNoLotSize(pos=150)
    api.close_position(s, symbol="600000.SH")
    assert s.execution.orders[0]["side"].lower() == "sell"
    assert s.execution.orders[0]["quantity"] == 150.0
