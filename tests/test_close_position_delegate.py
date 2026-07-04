"""close_position 全平（含零股，不按手数取整），order_target 仍取整."""

from akquant import strategy_trading_api as api


class _Exec:
    def __init__(self, pos):
        self._pos = pos
        self.orders = []

    def get_position(self, symbol=None):
        return self._pos

    def submit_order(self, **kw):
        self.orders.append(kw)
        return "OID"


class _S:
    def __init__(self, pos, lot_size=100):
        self.execution = _Exec(pos)
        self.ctx = object()
        self.current_bar = None
        self.current_tick = None
        self.lot_size = lot_size
        self._last_prices = {}

    def submit_order(self, **kwargs):
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
