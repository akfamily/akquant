"""wrap_state_invalidation：trade 叠总持仓 delta；order 仅失效委托/资金."""

from akquant.gateway.broker_strategy_api import wrap_state_invalidation


class _Cache:
    """Fake cache 记录各类调用."""

    def __init__(self) -> None:
        self.fills: list = []
        self.inv_available = 0
        self.inv_account = 0
        self.inv_open_orders = 0
        self.inv_all = 0

    def apply_fill(self, symbol, signed_qty) -> None:
        self.fills.append((symbol, signed_qty))

    def invalidate_available(self) -> None:
        self.inv_available += 1

    def invalidate_account(self) -> None:
        self.inv_account += 1

    def invalidate_open_orders(self) -> None:
        self.inv_open_orders += 1

    def invalidate(self) -> None:
        self.inv_all += 1


def _wrap(cache, calls):
    return wrap_state_invalidation(lambda n, p: calls.append((n, p)), lambda: [cache])


def test_trade_applies_signed_fill_and_invalidates_others_not_total() -> None:
    """Trade 事件叠总持仓 delta, 失效可用/资金/委托, 不全量失效."""
    cache, calls = _Cache(), []
    wrapped = _wrap(cache, calls)
    payload = {
        "trade_id": "t1",
        "symbol": "600000.SH",
        "side": "Buy",
        "quantity": 100.0,
    }
    wrapped("trade", payload)
    assert calls == [("trade", payload)]
    assert cache.fills == [("600000.SH", 100.0)]  # Buy -> +
    assert cache.inv_available == 1
    assert cache.inv_account == 1
    assert cache.inv_open_orders == 1
    assert cache.inv_all == 0  # 不全量失效(总持仓保留)


def test_trade_sell_is_negative_delta() -> None:
    """Sell 成交对应负 delta."""
    cache = _Cache()
    wrap_state_invalidation(lambda n, p: None, lambda: [cache])(
        "trade", {"trade_id": "t2", "symbol": "X", "side": "Sell", "quantity": 30.0}
    )
    assert cache.fills == [("X", -30.0)]


def test_duplicate_trade_id_applied_once() -> None:
    """同一 trade_id 重放(恢复循环重放当日成交)只叠一次 delta, 防漂移."""
    cache = _Cache()
    wrapped = wrap_state_invalidation(lambda n, p: None, lambda: [cache])
    payload = {"trade_id": "dup", "symbol": "X", "side": "Buy", "quantity": 50.0}
    wrapped("trade", payload)
    wrapped("trade", dict(payload))  # 重放同一 trade_id
    wrapped("trade", dict(payload))
    assert cache.fills == [("X", 50.0)]  # 仅一次


def test_trade_without_id_falls_back_to_invalidate() -> None:
    """无 trade_id 无法去重 → 退回幂等 invalidate(重查), 不叠 delta."""
    cache = _Cache()
    wrap_state_invalidation(lambda n, p: None, lambda: [cache])(
        "trade", {"symbol": "X", "side": "Buy", "quantity": 10.0}
    )
    assert cache.fills == []  # 不叠 delta
    assert cache.inv_all == 1  # 退回全量失效重查


def test_unrecognized_side_is_zero_delta() -> None:
    """方向无法识别 → 0 delta(不臆测为卖出)."""
    cache = _Cache()
    wrap_state_invalidation(lambda n, p: None, lambda: [cache])(
        "trade", {"trade_id": "t3", "symbol": "X", "side": "???", "quantity": 10.0}
    )
    assert cache.fills == [("X", 0.0)]


def test_order_invalidates_open_orders_and_account_not_positions() -> None:
    """Order 事件只失效委托/资金, 不动持仓/可用."""
    cache = _Cache()
    wrap_state_invalidation(lambda n, p: None, lambda: [cache])("order", {})
    assert cache.fills == []
    assert cache.inv_open_orders == 1
    assert cache.inv_account == 1
    assert cache.inv_available == 0  # 不动持仓/可用
    assert cache.inv_all == 0


def test_bar_event_touches_nothing() -> None:
    """非 order/trade 事件（如 bar）不触碰缓存."""
    cache = _Cache()
    wrap_state_invalidation(lambda n, p: None, lambda: [cache])("bar", {})
    assert cache.fills == [] and cache.inv_open_orders == 0


def test_all_caches_get_fill_on_trade() -> None:
    """多 slot broker_live: 每个 target 缓存都叠同一笔成交 delta."""
    caches = [_Cache(), _Cache(), _Cache()]
    wrap_state_invalidation(lambda n, p: None, lambda: caches)(
        "trade", {"trade_id": "t4", "symbol": "X", "side": "Buy", "quantity": 10.0}
    )
    assert all(c.fills == [("X", 10.0)] for c in caches)


def test_empty_or_none_caches_do_not_crash() -> None:
    """install_submitter 运行前 get_caches() 为空/None——不崩."""
    wrap_state_invalidation(lambda n, p: None, lambda: [])("trade", {"symbol": "X"})
    wrap_state_invalidation(lambda n, p: None, lambda: None)("trade", {"symbol": "X"})
    wrap_state_invalidation(lambda n, p: None, lambda: [None])("order", {})
