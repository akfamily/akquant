"""实盘持仓同步: 成交叠 delta 同步准、防双计、可用重查不覆盖总持仓."""

from akquant.gateway.broker_models import UnifiedPosition
from akquant.gateway.broker_state_cache import BrokerStateCache


class _Gw:
    def __init__(self, rows: list[UnifiedPosition]) -> None:
        self.rows = rows
        self.pos_calls = 0

    def query_positions(self) -> list[UnifiedPosition]:
        self.pos_calls += 1
        return list(self.rows)


def _rows(
    qty: float = 1000.0, avail: float = 800.0, symbol: str = "X"
) -> list[UnifiedPosition]:
    return [UnifiedPosition(symbol=symbol, quantity=qty, available_quantity=avail)]


def test_apply_fill_updates_total_synchronously_no_requery() -> None:
    """apply_fill 后总持仓同步 +delta, 不触发再查柜台."""
    gw = _Gw(_rows(1000.0))
    c = BrokerStateCache(gw)
    assert c.positions()["X"] == 1000.0  # seed, query #1
    c.apply_fill("X", 100.0)  # Buy 100
    assert c.positions()["X"] == 1100.0  # 同步 +100
    assert gw.pos_calls == 1  # 未再查柜台


def test_apply_fill_before_seed_is_noop_then_seed_no_double_count() -> None:
    """未 seed 时 apply_fill 是 no-op, 避免柜台快照 + delta 双计."""
    # 柜台快照已含该笔(1100), apply_fill 在未 seed 时不得再叠
    gw = _Gw(_rows(1100.0))
    c = BrokerStateCache(gw)
    c.apply_fill("X", 100.0)  # 未 seed -> no-op
    # 直接钉住守卫: 未 seed 时不得写入(否则守卫失效也会被随后的整快照 seed 掩盖)
    assert c._total_loaded is False
    assert c._positions == {}
    assert c.positions()["X"] == 1100.0  # seed 得柜台值, 不双计


def test_available_requery_does_not_clobber_event_sourced_total() -> None:
    """可用重查(invalidate_available)不覆盖事件溯源的总持仓."""
    gw = _Gw(_rows(1000.0, 800.0))
    c = BrokerStateCache(gw)
    assert c.positions()["X"] == 1000.0  # seed total+available
    c.apply_fill("X", 100.0)  # total -> 1100 (event-sourced)
    c.invalidate_available()
    gw.rows = _rows(1000.0, 700.0)  # 柜台可用变
    assert c.available_positions()["X"] == 700.0  # 可用重查
    assert c.positions()["X"] == 1100.0  # 总持仓 delta 未被覆盖


def test_full_invalidate_reseeds_total_reconcile() -> None:
    """invalidate() 全量对账, 柜台权威值覆盖会话内累积的 delta 漂移."""
    gw = _Gw(_rows(1000.0))
    c = BrokerStateCache(gw)
    c.positions()
    c.apply_fill("X", 100.0)
    assert c.positions()["X"] == 1100.0
    gw.rows = _rows(1000.0)  # 柜台权威(未含漂移)
    c.invalidate()  # 恢复/对账
    assert c.positions()["X"] == 1000.0  # 权威覆盖


def test_granular_invalidate_methods() -> None:
    """invalidate_open_orders/invalidate_account 不影响事件溯源的总持仓."""
    gw = _Gw(_rows())
    c = BrokerStateCache(gw)
    c.positions()
    c.apply_fill("X", 50.0)
    c.invalidate_open_orders()  # 不动总持仓
    assert c.positions()["X"] == 1050.0
    assert c._open_orders_loaded is False  # 正向确认标志翻转
    c.invalidate_account()
    assert c.positions()["X"] == 1050.0
    assert c._account_loaded is False  # 正向确认标志翻转
    assert c._total_loaded is True  # 总持仓不受影响


def test_trade_event_through_wrap_keeps_get_position_synchronous() -> None:
    """经 wrap_state_invalidation 发 trade: positions() 同步反映 delta, 不重查."""
    from akquant.gateway.broker_strategy_api import wrap_state_invalidation

    gw = _Gw(_rows(1000.0))
    c = BrokerStateCache(gw)
    assert c.positions()["X"] == 1000.0  # seed, query #1
    wrapped = wrap_state_invalidation(lambda n, p: None, lambda: [c])
    wrapped(
        "trade", {"trade_id": "t1", "symbol": "X", "side": "Buy", "quantity": 100.0}
    )
    assert c.positions()["X"] == 1100.0  # 同步 +100
    assert gw.pos_calls == 1  # 总持仓未重查
    # 可用被失效 -> 下次读会重查
    gw.rows = _rows(1000.0, 700.0)
    assert c.available_positions()["X"] == 700.0


def test_recovery_replay_of_same_trade_does_not_drift_position() -> None:
    """恢复循环重放同一 trade_id: 经 wrap 去重, 总持仓不漂移(仅叠一次)."""
    from akquant.gateway.broker_strategy_api import wrap_state_invalidation

    gw = _Gw(_rows(1000.0))
    c = BrokerStateCache(gw)
    assert c.positions()["X"] == 1000.0  # seed
    wrapped = wrap_state_invalidation(lambda n, p: None, lambda: [c])
    fill = {"trade_id": "F1", "symbol": "X", "side": "Buy", "quantity": 100.0}
    wrapped("trade", fill)  # 实盘推送
    wrapped("trade", dict(fill))  # 恢复重放同一笔
    wrapped("trade", dict(fill))  # 再一周期重放
    assert c.positions()["X"] == 1100.0  # 只叠一次, 不是 1300
    assert gw.pos_calls == 1
