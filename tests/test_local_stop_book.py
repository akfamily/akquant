"""LocalStopBook：盯价触发方向 + 追踪 + 撤单/列单（独立可测）."""

from akquant.gateway.local_stop_book import (
    LocalStopBook,
    LocalStopOrder,
    is_stop_order_type,
    underlying_order_type,
)


def _book_with(order):
    """构造仅含单个挂单的止损簿."""
    b = LocalStopBook()
    b.register(order)
    return b


def test_buy_stop_triggers_when_high_ge_trigger() -> None:
    """买入止损单应在 high>=trigger 时触发并从簿中移除."""
    b = _book_with(
        LocalStopOrder("L1", "X", "Buy", 100, "stopmarket", trigger_price=110.0)
    )
    assert b.check("X", last=105.0, high=105.0, low=104.0) == []  # not yet
    fired = b.check("X", last=112.0, high=115.0, low=108.0)
    assert [o.local_id for o in fired] == ["L1"]
    assert b.open_orders() == []  # removed after firing


def test_sell_stop_triggers_when_low_le_trigger() -> None:
    """卖出止损单应在 low<=trigger 时触发."""
    b = _book_with(
        LocalStopOrder("L2", "X", "Sell", 100, "stopmarket", trigger_price=9.5)
    )
    assert b.check("X", last=9.8, high=9.9, low=9.6) == []
    fired = b.check("X", last=9.4, high=9.7, low=9.3)
    assert [o.local_id for o in fired] == ["L2"]


def test_tick_mode_uses_last_when_no_high_low() -> None:
    """未提供 high/low 时(tick 模式)应以 last 判断触发."""
    b = _book_with(
        LocalStopOrder("L3", "X", "Buy", 100, "stopmarket", trigger_price=110.0)
    )
    assert b.check("X", last=109.0) == []
    assert [o.local_id for o in b.check("X", last=110.0)] == ["L3"]


def test_trailing_sell_tracks_high_and_triggers_on_pullback() -> None:
    """跟踪止损(卖)应随 high 上移 trigger, 并在回落触及时触发."""
    b = _book_with(
        LocalStopOrder("L4", "X", "Sell", 100, "stoptrail", trail_offset=1.0)
    )
    b.check("X", last=100.0, high=100.0, low=99.5)  # ref=100, trigger=99
    b.check("X", last=105.0, high=105.0, low=104.0)  # ref=105, trigger=104
    assert b.open_orders() and b.open_orders()[0].trigger_price == 104.0
    fired = b.check("X", last=103.5, high=104.5, low=103.5)  # low 103.5 <= 104 → fire
    assert [o.local_id for o in fired] == ["L4"]


def test_cancel_and_symbol_isolation() -> None:
    """撤单按 local_id 生效; 触发检查应按 symbol 隔离."""
    b = LocalStopBook()
    b.register(LocalStopOrder("A", "X", "Buy", 1, "stopmarket", trigger_price=10.0))
    b.register(LocalStopOrder("B", "Y", "Buy", 1, "stopmarket", trigger_price=10.0))
    fired_x = b.check("X", last=11.0, high=11.0, low=10.5)
    fired_y = [o.local_id for o in b.check("Y", last=9.0, high=9.0, low=8.0)]
    assert fired_x and fired_y == []
    assert b.cancel("B") is True
    assert b.cancel("B") is False
    assert b.open_orders() == []


def test_helpers() -> None:
    """is_stop_order_type/underlying_order_type 的归一与映射规则."""
    assert is_stop_order_type("StopMarket") and is_stop_order_type("stoptrail")
    assert not is_stop_order_type("Limit") and not is_stop_order_type(None)
    assert underlying_order_type("stopmarket") == "Market"
    assert underlying_order_type("stoplimit") == "Limit"
    assert underlying_order_type("stoptraillimit") == "Limit"
    assert underlying_order_type("stoptrail") == "Market"
