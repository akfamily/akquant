"""close_position 改为 order_target(symbol, 0) delegate."""

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
    def __init__(self, pos):
        self.execution = _Exec(pos)
        self.ctx = object()
        self.current_bar = None
        self.current_tick = None
        self.lot_size = 100

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
